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
    _krum_scores,
    aggregate,
    default_malicious,
    flatten_states,
    max_supported_malicious,
)
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

CASE_TRIGGERS = {1: "sunglasses", 2: "surgical_mask", 3: "sunglasses"}
CACHE_ROOT = ROOT / "research" / "cache" / "malicious_baseline"

LAMBDA_BACKDOOR = 1.0
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def report_krum_geometry(benign, malicious) -> None:
    matrix = flatten_states(validate_collection(list(benign) + list(malicious)))
    total, num_benign = matrix.shape[0], len(benign)
    is_malicious = [index >= num_benign for index in range(total)]

    distances = torch.cdist(matrix, matrix, p=2.0)

    def mean_pairwise(rows, columns):
        values = [
            distances[i, j].item() for i in rows for j in columns if i != j
        ]
        return sum(values) / len(values)

    benign_idx = [i for i in range(total) if not is_malicious[i]]
    malicious_idx = [i for i in range(total) if is_malicious[i]]

    used = min(default_malicious(total), max_supported_malicious("krum", total))
    scores = _krum_scores(matrix, used)
    ranking = [int(i) for i in scores.argsort()]
    winner = ranking[0]

    print(
        f"{'krum geometry':>14}   f={used}, k={total - used - 2} neighbours summed | "
        f"winner=idx{winner} ({'MALICIOUS' if is_malicious[winner] else 'benign'})"
    )
    print(
        f"{'':>14}   mean pairwise L2: within-malicious {mean_pairwise(malicious_idx, malicious_idx):.4f} | "
        f"within-benign {mean_pairwise(benign_idx, benign_idx):.4f} | "
        f"cross {mean_pairwise(malicious_idx, benign_idx):.4f}"
    )
    print(
        f"{'':>14}   malicious krum ranks (0=best): "
        f"{sorted(ranking.index(i) for i in malicious_idx)} of {total}"
    )


def load_benign(case: int) -> list:
    return [load_state_dict(path) for path in sorted((ROOT / "attack" / f"case_{case}").glob("client_*.pt"))]


def train_malicious(reference, images, labels, trigger, seed, lambda_backdoor=LAMBDA_BACKDOOR):
    torch.manual_seed(seed)
    model = load_model_from_state_dict(reference)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(EPOCHS):
        order = torch.randperm(images.shape[0])

        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            clean_x, clean_y = images[batch], labels[batch]
            triggered_x = apply_trigger(clean_x, trigger)
            target = torch.full_like(clean_y, TARGET_CLASS_INDEX)

            loss_clean = F.cross_entropy(model(to_model_input(clean_x)), clean_y)
            loss_backdoor = F.cross_entropy(model(to_model_input(triggered_x)), target)

            optimizer.zero_grad()
            (loss_clean + lambda_backdoor * loss_backdoor).backward()
            optimizer.step()

    model.eval()
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def load_cached_malicious(case: int, count: int) -> list | None:
    paths = [CACHE_ROOT / f"case_{case}" / f"malicious_{index}.pt" for index in range(count)]

    if not all(path.exists() for path in paths):
        return None

    return [load_state_dict(path) for path in paths]


def run_case(case: int, train, eval_, from_cache: bool = False) -> None:
    trigger = CASE_TRIGGERS[case]
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]

    print(f"\n=== Case {case}: {len(benign)} benign + {count} malicious, trigger={trigger} ===")

    malicious = load_cached_malicious(case, count) if from_cache else None

    if malicious is not None:
        print(f"{'':>14}   loaded {count} cached malicious models (no retrain)")
    else:
        reference = fedavg(benign)
        malicious = [
            train_malicious(reference, train.images, train.labels, trigger, seed=1000 * case + index)
            for index in range(count)
        ]

        for index, state in enumerate(malicious):
            save_state_dict(state, CACHE_ROOT / f"case_{case}" / f"malicious_{index}.pt")

    report_krum_geometry(benign, malicious)

    control_asr = {}
    for strategy in STRATEGIES:
        control = load_model_from_state_dict(aggregate(benign, strategy))
        control_asr[strategy] = attack_success_rate(control, eval_.images, trigger)

    fedavg_control = load_model_from_state_dict(fedavg(benign))
    print(
        f"{'no attack (fedavg of benign)':>14}   clean {clean_accuracy(fedavg_control, eval_.images, eval_.labels):>6.1%} | "
        f"ASR {control_asr['fedavg']:>6.1%}  <- free-ASR floor"
    )

    solo = [
        attack_success_rate(load_model_from_state_dict(state), eval_.images, trigger)
        for state in malicious
    ]
    print(
        f"{'malicious solo':>14}   ASR {min(solo):>6.1%}-{max(solo):>5.1%} "
        f"(each model on its own, pre-aggregation)"
    )

    print(
        f"\n{'strategy':>13} | {'clean acc':>9} | {'ASR':>6} | {'own control':>11} | "
        f"{'ASR lift':>9} | {'case score':>10}"
    )
    print("-" * 76)

    for strategy in STRATEGIES:
        model = load_model_from_state_dict(aggregate(benign + malicious, strategy))
        clean = clean_accuracy(model, eval_.images, eval_.labels)
        asr = attack_success_rate(model, eval_.images, trigger)
        print(
            f"{strategy:>13} | {clean:>8.1%} | {asr:>5.1%} | {control_asr[strategy]:>10.1%} | "
            f"{100 * (asr - control_asr[strategy]):>+6.1f}pt | {attack_case_score(clean, asr):>9.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="reuse the cached malicious models instead of retraining (falls back "
        "to training any case that isn't fully cached)",
    )
    args = parser.parse_args()

    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    train, eval_ = train_eval_split(dataset, eval_fraction=0.2, seed=0)
    print(f"train {len(train)} / eval {len(eval_)} images (disjoint)")
    print(f"lambda_bd={LAMBDA_BACKDOOR}, epochs={EPOCHS}, lr={LEARNING_RATE}, no stealth term")

    for case in (1, 2, 3):
        run_case(case, train, eval_, from_cache=args.from_cache)


if __name__ == "__main__":
    main()
