import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from data import CLASS_NAMES, load_dataset
from evaluate import attack_success_rate, clean_accuracy, load_model_from_state_dict
from triggers import TARGET_CLASS_INDEX, apply_trigger
from utilities.checks import fedavg
from utilities.model_io import load_state_dict

CACHE = ROOT / "research" / "cache" / "celeba_subset.pt"
BENIGN_CASE = ROOT / "attack" / "case_1"

FILLS = {
    "as rendered": None,
    "pure black": (0.0, 0.0, 0.0),
    "mid gray": (0.5, 0.5, 0.5),
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
}


def benign_reference():
    states = [load_state_dict(path) for path in sorted(BENIGN_CASE.glob("client_*.pt"))]
    return load_model_from_state_dict(fedavg(states))


def main() -> None:
    dataset = load_dataset(CACHE)
    model = benign_reference()

    with torch.inference_mode():
        base_rate = (
            model(dataset.images * 2.0 - 1.0).argmax(dim=1) == TARGET_CLASS_INDEX
        ).float().mean().item()

    print(f"images: {len(dataset)}   classes: {CLASS_NAMES}")
    print(f"reference: FedAvg of {len(list(BENIGN_CASE.glob('client_*.pt')))} benign Case 1 clients (no backdoor)")
    print(f"clean accuracy: {clean_accuracy(model, dataset.images, dataset.labels):.1%}")
    print(f"\nbase rate of predicting '{CLASS_NAMES[TARGET_CLASS_INDEX]}' with NO trigger: {base_rate:.1%}")
    print("This is the honest floor. A backdoor-free model should score ~this on every row.\n")

    header = f"{'trigger':>28} | " + " | ".join(f"{name:>11}" for name in FILLS)
    print(header)
    print("-" * len(header))

    for trigger in ("sunglasses", "sunglasses_flat", "sunglasses_legacy_box",
                    "surgical_mask", "surgical_mask_flat", "surgical_mask_legacy_box"):
        cells = []
        for fill in FILLS.values():
            asr = attack_success_rate(model, dataset.images, trigger, fill=fill)
            cells.append(f"{asr:>10.1%}{'*' if asr - base_rate > 0.15 else ' '}")
        print(f"{trigger:>28} | " + " | ".join(cells))

    print("\n* = more than 15 points of unearned ASR above the no-trigger base rate.")


if __name__ == "__main__":
    main()
