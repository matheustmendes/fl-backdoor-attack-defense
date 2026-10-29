import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from attack_constrain_scale import scale_delta
from attack_train import load_benign
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict, save_state_dict

GRID_CACHE = ROOT / "research" / "cache" / "grid"
MODELS_ROOT = ROOT / "participant_models"

CHOSEN = {
    1: (2.0, 200.0, 1.5),
    2: (2.0, 100.0, 1.0),
    3: (1.0, 200.0, 2.5),
}


def build_case(case: int) -> None:
    lambda_bd, lambda_reg, gamma = CHOSEN[case]
    count = MALICIOUS_MODELS_PER_CASE[case]

    trained_path = GRID_CACHE / f"case_{case}" / f"lbd_{lambda_bd}_lreg_{lambda_reg}.pt"
    if not trained_path.is_file():
        raise FileNotFoundError(
            f"Cached trained model not found: {trained_path}. "
            f"Run research/attack_grid.py --cases {case} first."
        )

    trained = load_state_dict(trained_path)
    reference = fedavg(load_benign(case))
    submitted = scale_delta(reference, trained, gamma)

    drift = sum(((submitted[k] - reference[k]) ** 2).sum() for k in submitted).sqrt().item()
    print(
        f"case {case}: lbd={lambda_bd} lreg={lambda_reg} gamma={gamma} "
        f"-> {count} identical copies, ||submitted-ref||={drift:.3f}"
    )

    for index in range(count):
        out = MODELS_ROOT / f"case_{case}" / f"malicious_{index}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_state_dict({k: v.detach().clone() for k, v in submitted.items()}, out)


def main() -> None:
    for case in MALICIOUS_MODELS_PER_CASE:
        build_case(case)
    print(f"\nWrote malicious models to {MODELS_ROOT}")


if __name__ == "__main__":
    main()
