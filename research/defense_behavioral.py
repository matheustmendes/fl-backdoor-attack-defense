import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from attack_constrain_scale import scale_delta
from attack_train import load_benign
from data import load_dataset, train_eval_split
from defense_probe import WINNERS, auc
from evaluate import load_model_from_state_dict, to_model_input
from triggers import TARGET_CLASS_INDEX, apply_trigger
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict

BATCH = 256


def synth_uniform(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(BATCH, 3, 64, 64, generator=generator)


def synth_smooth(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(BATCH, 3, 8, 8, generator=generator)
    return F.interpolate(coarse, size=(64, 64), mode="bilinear", align_corners=False)


def dark_patch(images: torch.Tensor) -> torch.Tensor:
    patched = images.clone()
    patched[:, :, 28:38, 10:54] = 0.0
    return patched


@torch.inference_mode()
def target_rate(model, images: torch.Tensor) -> float:
    probabilities = F.softmax(model(to_model_input(images)), dim=1)
    return probabilities[:, TARGET_CLASS_INDEX].mean().item()


def swing(model, images: torch.Tensor) -> float:
    return target_rate(model, dark_patch(images)) - target_rate(model, images)


def probe_case(case: int, real_images: torch.Tensor) -> None:
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]
    config = WINNERS[case]

    reference = fedavg(benign)
    trained = load_state_dict(
        ROOT / "research" / "cache" / "grid" / f"case_{case}"
        / f"lbd_{config['lambda_bd']}_lreg_{config['lambda_reg']}.pt"
    )
    malicious = [scale_delta(reference, trained, config["gamma"]) for _ in range(count)]

    models = [load_model_from_state_dict(s) for s in list(benign) + malicious]
    labels = [0] * len(benign) + [1] * count

    print(f"\n=== Case {case}: {len(benign)} benign + {count} malicious ===")
    print(f"{'input source':>14} | {'AUC':>5} | {'benign swing':>26} | {'malicious swing':>26}")
    print("-" * 82)

    for name, images in (
        ("real (illegal)", real_images),
        ("uniform", synth_uniform(seed=case)),
        ("smooth", synth_smooth(seed=case)),
    ):
        swings = [swing(model, images) for model in models]

        median = torch.tensor(swings).median().item()
        scores = [s - median for s in swings]

        benign_swings = [s for s, l in zip(swings, labels) if l == 0]
        malicious_swings = [s for s, l in zip(swings, labels) if l == 1]

        print(
            f"{name:>14} | {auc(labels, scores):>5.2f} | "
            f"{min(benign_swings):>+.3f} .. {max(benign_swings):>+.3f} ({len(benign_swings):>2}) | "
            f"{min(malicious_swings):>+.3f} .. {max(malicious_swings):>+.3f} ({len(malicious_swings):>2})"
        )


def main() -> None:
    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    _, eval_ = train_eval_split(dataset, eval_fraction=0.2, seed=0)

    print("swing = P(black hair | dark patch) - P(black hair | plain), per model.")
    print("AUC scores the deviation from the pool median. 0.5 = blind, 1.0 = separates.")

    for case in (1, 2, 3):
        probe_case(case, eval_.images)


if __name__ == "__main__":
    main()
