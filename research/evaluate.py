import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import SmallCNN
from triggers import TARGET_CLASS_INDEX, apply_trigger


def to_model_input(images: torch.Tensor) -> torch.Tensor:
    return images * 2.0 - 1.0


def load_model_from_state_dict(state_dict) -> SmallCNN:
    model = SmallCNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


EVAL_BATCH = 512


@torch.inference_mode()
def _predict(model: SmallCNN, images: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            model(to_model_input(images[start : start + EVAL_BATCH])).argmax(dim=1)
            for start in range(0, images.shape[0], EVAL_BATCH)
        ]
    )


@torch.inference_mode()
def clean_accuracy(model: SmallCNN, images: torch.Tensor, labels: torch.Tensor) -> float:
    return (_predict(model, images) == labels).float().mean().item()


@torch.inference_mode()
def attack_success_rate(
    model: SmallCNN, images: torch.Tensor, trigger: str, fill: tuple | None = None
) -> float:
    triggered = apply_trigger(images, trigger, fill=fill)
    return (_predict(model, triggered) == TARGET_CLASS_INDEX).float().mean().item()


def attack_case_score(clean_acc: float, asr: float) -> float:
    return (0.4 * clean_acc + 0.6 * asr) * 100


def defense_case_score(clean_acc: float, asr: float) -> float:
    return (0.6 * clean_acc + 0.4 * (1.0 - asr)) * 100


def report(model: SmallCNN, images: torch.Tensor, labels: torch.Tensor, trigger: str) -> dict:
    clean_acc = clean_accuracy(model, images, labels)
    asr = attack_success_rate(model, images, trigger)

    return {
        "clean_accuracy": clean_acc,
        "asr": asr,
        "attack_case_score": attack_case_score(clean_acc, asr),
        "defense_case_score": defense_case_score(clean_acc, asr),
    }
