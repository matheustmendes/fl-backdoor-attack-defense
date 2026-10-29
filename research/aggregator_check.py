import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import (
    STRATEGIES,
    aggregate,
    default_malicious,
    is_selection_based,
    max_supported_malicious,
)
from data import load_dataset
from evaluate import attack_success_rate, clean_accuracy, load_model_from_state_dict
from utilities.checks import fedavg as official_fedavg
from utilities.model_io import load_state_dict

CASE_TOTALS = {1: 10, 2: 25, 3: 20}
CASE_MALICIOUS = {1: 2, 2: 5, 3: 5}


def load_case(case: int):
    paths = sorted((ROOT / "attack" / f"case_{case}").glob("client_*.pt"))
    return [load_state_dict(path) for path in paths]


def main() -> None:
    print("=== 1. fedavg parity against utilities.checks.fedavg ===")
    benign = load_case(1)
    ours = aggregate(benign, "fedavg")
    official = official_fedavg(benign)
    worst = max((ours[key] - official[key]).abs().max().item() for key in ours)
    print(f"max abs difference across all tensors: {worst:.3e}  ->  {'OK' if worst < 1e-6 else 'MISMATCH'}")

    print("\n=== 2. all 6 run on real models ===")
    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    visible = [load_state_dict(p) for p in sorted((ROOT / "defense" / "visible_case").glob("client_*.pt"))]
    print(f"input: defense/visible_case, {len(visible)} models (unlabeled benign+malicious mix)\n")

    header = f"{'strategy':>13} | {'f used':>6} | {'family':>10} | {'clean acc':>9} | {'ASR sun':>8} | {'ASR mask':>8}"
    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        used = min(default_malicious(len(visible)), max_supported_malicious(strategy, len(visible)))
        model = load_model_from_state_dict(aggregate(visible, strategy))
        family = "selection" if is_selection_based(strategy) else "synthesis"
        print(
            f"{strategy:>13} | {used:>6} | {family:>10} | "
            f"{clean_accuracy(model, dataset.images, dataset.labels):>8.1%} | "
            f"{attack_success_rate(model, dataset.images, 'sunglasses'):>7.1%} | "
            f"{attack_success_rate(model, dataset.images, 'surgical_mask'):>7.1%}"
        )

    print("\n=== 3. can each rule even be configured at the true malicious count? ===")
    print("(the guide tells us the true f per attack case; a rule whose breakdown")
    print(" point is below it cannot be run at that f)\n")

    header = f"{'case':>4} | {'n':>3} | {'true f':>6} | " + " | ".join(f"{s:>12}" for s in STRATEGIES)
    print(header)
    print("-" * len(header))

    for case, count in CASE_TOTALS.items():
        cells = []
        for strategy in STRATEGIES:
            supported = max_supported_malicious(strategy, count)
            ok = supported >= CASE_MALICIOUS[case]
            cells.append(f"{('yes' if ok else 'NO') + f' (max {supported})':>12}")
        print(f"{case:>4} | {count:>3} | {CASE_MALICIOUS[case]:>6} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
