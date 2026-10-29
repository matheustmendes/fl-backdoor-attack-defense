import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import STRATEGIES, aggregate, flatten_states, max_supported_malicious
from attack_constrain_scale import geometry, scale_delta, train_constrained
from attack_train import CASE_TRIGGERS, EPOCHS, LEARNING_RATE, load_benign
from data import load_dataset, train_eval_split
from evaluate import (
    attack_case_score,
    attack_success_rate,
    clean_accuracy,
    load_model_from_state_dict,
)
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict, save_state_dict

CACHE_ROOT = ROOT / "research" / "cache" / "grid"

LAMBDA_BD_GRID = (0.5, 1.0, 2.0, 5.0, 10.0)
LAMBDA_REG_GRID = (20.0, 30.0, 50.0, 100.0, 200.0)
GAMMA_GRID = (1.0, 1.25, 1.5, 2.0, 2.5)


def feasible_f(strategy: str, total: int) -> list[int]:
    if strategy in ("fedavg", "median"):
        return [0]

    return list(range(1, min(max_supported_malicious(strategy, total), total // 3) + 1))


def evaluate_candidate(benign, malicious, eval_, trigger) -> dict:
    models = list(benign) + list(malicious)
    scores = {}

    for strategy in STRATEGIES:
        for f in feasible_f(strategy, len(models)):
            aggregated = aggregate(models, strategy, num_malicious=f or None)
            model = load_model_from_state_dict(aggregated)
            scores[(strategy, f)] = attack_case_score(
                clean_accuracy(model, eval_.images, eval_.labels),
                attack_success_rate(model, eval_.images, trigger),
            )

    worst_key = min(scores, key=scores.get)
    return {
        "scores": scores,
        "worst": scores[worst_key],
        "worst_at": worst_key,
        "mean": sum(scores.values()) / len(scores),
    }


def run_case(case: int, train, eval_, results: list) -> None:
    trigger = CASE_TRIGGERS[case]
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]
    reference = fedavg(benign)
    total = len(benign) + count

    print(f"\n=== Case {case}: {len(benign)} benign + {count} malicious, trigger={trigger} ===")
    combos = sum(len(feasible_f(s, total)) for s in STRATEGIES)
    print(f"scoring each candidate against {combos} (strategy, f) combinations")
    print(
        f"\n{'l_bd':>5} | {'l_reg':>6} | {'gamma':>5} | {'solo':>5} | {'cross':>6} | "
        f"{'margin':>7} | {'worst':>5} | {'worst at':>18} | {'mean':>5}"
    )
    print("-" * 88)

    for lambda_bd in LAMBDA_BD_GRID:
        for lambda_reg in LAMBDA_REG_GRID:
            cache = CACHE_ROOT / f"case_{case}" / f"lbd_{lambda_bd}_lreg_{lambda_reg}.pt"

            if cache.exists():
                trained = load_state_dict(cache)
            else:
                trained = train_constrained(
                    reference,
                    train.images,
                    train.labels,
                    trigger,
                    seed=1000 * case,
                    lambda_reg=lambda_reg,
                    lambda_bd=lambda_bd,
                )
                save_state_dict(trained, cache)

            solo = attack_success_rate(load_model_from_state_dict(trained), eval_.images, trigger)

            for gamma in GAMMA_GRID:
                malicious = [scale_delta(reference, trained, gamma) for _ in range(count)]
                geo = geometry(benign, malicious)
                outcome = evaluate_candidate(benign, malicious, eval_, trigger)

                margin = (geo["within_benign"] - geo["cross"]) / geo["within_benign"]
                strategy, f = outcome["worst_at"]

                results.append(
                    {
                        "case": case, "lambda_bd": lambda_bd, "lambda_reg": lambda_reg,
                        "gamma": gamma, "margin": margin, "solo": solo, **outcome,
                    }
                )
                print(
                    f"{lambda_bd:>5.1f} | {lambda_reg:>6.1f} | {gamma:>5.2f} | {solo:>4.0%} | "
                    f"{geo['cross']:>6.3f} | {margin:>+6.1%} | {outcome['worst']:>5.1f} | "
                    f"{strategy + '@f=' + str(f):>18} | {outcome['mean']:>5.1f}"
                )


def summarize(results: list) -> None:
    print("\n\n=== Summary: best candidate per case ===")

    for label, key, guard in (
        ("max worst-case score", lambda r: r["worst"], lambda r: True),
        ("max worst-case, margin >= 15%", lambda r: r["worst"], lambda r: r["margin"] >= 0.15),
    ):
        print(f"\n-- {label} --")
        total = 0.0

        for case in sorted({r["case"] for r in results}):
            pool = [r for r in results if r["case"] == case and guard(r)]

            if not pool:
                print(f"  case {case}: no candidate satisfies the guard")
                continue

            best = max(pool, key=key)
            total += best["worst"]
            print(
                f"  case {case}: l_bd={best['lambda_bd']}, l_reg={best['lambda_reg']}, "
                f"gamma={best['gamma']} -> worst {best['worst']:.1f} "
                f"(margin {best['margin']:+.1%}, mean {best['mean']:.1f}, "
                f"binds at {best['worst_at'][0]}@f={best['worst_at'][1]})"
            )

        cases = len({r["case"] for r in results})
        print(f"  => final attack score (mean of per-case worst): {total / cases:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    train, eval_ = train_eval_split(dataset, eval_fraction=0.2, seed=0)
    print(f"train {len(train)} / eval {len(eval_)} images (disjoint)")
    print(f"epochs={EPOCHS}, lr={LEARNING_RATE}, malicious models are identical copies")
    print("margin = (within_benign - cross) / within_benign; positive = inside the Krum cliff")

    results: list[dict] = []

    for case in args.cases:
        run_case(case, train, eval_, results)

    summarize(results)


if __name__ == "__main__":
    main()
