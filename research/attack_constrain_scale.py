import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import (
    STRATEGIES,
    _iterative_krum_selection,
    _krum_scores,
    aggregate,
    default_malicious,
    flatten_states,
    max_supported_malicious,
)
from attack_train import CASE_TRIGGERS, EPOCHS, LEARNING_RATE, load_benign
from data import load_dataset, train_eval_split
from evaluate import (
    attack_case_score,
    attack_success_rate,
    clean_accuracy,
    load_model_from_state_dict,
    to_model_input,
)
from triggers import TARGET_CLASS_INDEX, apply_trigger
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg, validate_collection
from utilities.model_io import load_state_dict, save_state_dict

CACHE_ROOT = ROOT / "research" / "cache" / "constrain_scale"

LAMBDA_BACKDOOR = 1.0
BATCH_SIZE = 64

LAMBDA_REG_GRID = (0.0, 0.01, 0.1, 1.0, 10.0, 30.0, 100.0)


def geometry(benign, malicious) -> dict:
    matrix = flatten_states(validate_collection(list(benign) + list(malicious)))
    total, num_benign = matrix.shape[0], len(benign)
    distances = torch.cdist(matrix, matrix, p=2.0)

    benign_idx = list(range(num_benign))
    malicious_idx = list(range(num_benign, total))

    def mean_pairwise(rows, columns):
        values = [distances[i, j].item() for i in rows for j in columns if i != j]
        return sum(values) / len(values) if values else 0.0

    used = min(default_malicious(total), max_supported_malicious("krum", total))
    ranking = [int(i) for i in _krum_scores(matrix, used).argsort()]

    selection_size = max(1, total - used)
    selected = _iterative_krum_selection(matrix, used, selection_size)

    return {
        "within_malicious": mean_pairwise(malicious_idx, malicious_idx),
        "within_benign": mean_pairwise(benign_idx, benign_idx),
        "cross": mean_pairwise(malicious_idx, benign_idx),
        "best_malicious_rank": min(ranking.index(i) for i in malicious_idx),
        "krum_winner_malicious": ranking[0] >= num_benign,
        "multi_krum_malicious": sum(1 for i in selected if i >= num_benign),
        "multi_krum_selection": selection_size,
    }


def train_constrained(reference, images, labels, trigger, seed, lambda_reg, lambda_bd=LAMBDA_BACKDOOR):
    torch.manual_seed(seed)
    model = load_model_from_state_dict(reference)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    anchors = [reference[name].detach().clone() for name, _ in model.named_parameters()]

    for _ in range(EPOCHS):
        order = torch.randperm(images.shape[0])

        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            clean_x, clean_y = images[batch], labels[batch]
            triggered_x = apply_trigger(clean_x, trigger)
            target = torch.full_like(clean_y, TARGET_CLASS_INDEX)

            loss_clean = F.cross_entropy(model(to_model_input(clean_x)), clean_y)
            loss_backdoor = F.cross_entropy(model(to_model_input(triggered_x)), target)

            loss = loss_clean + lambda_bd * loss_backdoor

            if lambda_reg > 0:
                drift = sum(
                    ((parameter - anchor) ** 2).sum()
                    for (_, parameter), anchor in zip(model.named_parameters(), anchors)
                )
                loss = loss + lambda_reg * drift

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def scale_delta(reference, state, gamma):
    return {
        key: reference[key] + gamma * (state[key] - reference[key]) for key in state
    }


def gamma_grid(case: int) -> list[float]:
    total = len(load_benign(case)) + MALICIOUS_MODELS_PER_CASE[case]
    n_over_m = total / MALICIOUS_MODELS_PER_CASE[case]
    return sorted({1.0, 2.0, round(n_over_m, 2), round(2 * n_over_m, 2)})


def run_case(case: int, train, eval_) -> None:
    trigger = CASE_TRIGGERS[case]
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]
    reference = fedavg(benign)
    total = len(benign) + count

    print(f"\n=== Case {case}: {len(benign)} benign + {count} malicious, trigger={trigger} ===")
    print(f"n/m = {total / count:.2f} (the gamma that exactly cancels FedAvg dilution)")

    for lambda_reg in LAMBDA_REG_GRID:
        cache = CACHE_ROOT / f"case_{case}" / f"lreg_{lambda_reg}.pt"

        if cache.exists():
            trained = load_state_dict(cache)
        else:
            trained = train_constrained(
                reference, train.images, train.labels, trigger, seed=1000 * case, lambda_reg=lambda_reg
            )
            save_state_dict(trained, cache)

        solo = attack_success_rate(load_model_from_state_dict(trained), eval_.images, trigger)
        drift = sum(((trained[k] - reference[k]) ** 2).sum() for k in trained).sqrt().item()

        print(f"\n--- lambda_reg={lambda_reg} | trained solo ASR {solo:.1%} | ||delta||={drift:.3f} ---")
        print(
            f"{'gamma':>6} | {'cross L2':>8} | {'w/in ben':>8} | {'rank':>5} | {'mk sel':>6} | "
            f"{'fedavg':>6} | {'tr.mean':>7} | {'median':>6} | "
            f"{'krum':>6} | {'m.krum':>6} | {'bulyan':>6} | {'worst':>5} | {'mean':>5}"
        )

        for gamma in gamma_grid(case):
            malicious = [scale_delta(reference, trained, gamma) for _ in range(count)]
            geo = geometry(benign, malicious)

            scores = {}
            for strategy in STRATEGIES:
                model = load_model_from_state_dict(aggregate(benign + malicious, strategy))
                scores[strategy] = attack_case_score(
                    clean_accuracy(model, eval_.images, eval_.labels),
                    attack_success_rate(model, eval_.images, trigger),
                )

            print(
                f"{gamma:>6.2f} | {geo['cross']:>8.3f} | {geo['within_benign']:>8.3f} | "
                f"{geo['best_malicious_rank']:>2}/{total:<2} | "
                f"{geo['multi_krum_malicious']:>2}/{geo['multi_krum_selection']:<3} | "
                + " | ".join(f"{scores[s]:>6.1f}" for s in STRATEGIES)
                + f" | {min(scores.values()):>5.1f} | {sum(scores.values()) / len(scores):>5.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    train, eval_ = train_eval_split(dataset, eval_fraction=0.2, seed=0)
    print(f"train {len(train)} / eval {len(eval_)} images (disjoint)")
    print(f"lambda_bd={LAMBDA_BACKDOOR}, epochs={EPOCHS}, lr={LEARNING_RATE}")
    print("krum column = best (lowest) rank any malicious model achieves, 0=selected")

    for case in args.cases:
        run_case(case, train, eval_)


if __name__ == "__main__":
    main()
