import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "attack"))

import mask_variants
from mask_variants import MASK_VARIANTS
from attack_constrain_scale import scale_delta, train_constrained
from attack_train import load_benign
from data import load_dataset, train_eval_split
from evaluate import attack_success_rate, load_model_from_state_dict
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict, save_state_dict
from create_attack_submission import create_submission

CS = ROOT / "research" / "cache" / "constrain_scale"
GRID = ROOT / "research" / "cache" / "grid"
CACHE = ROOT / "research" / "cache" / "mask_retrain"
MODELS_ROOT = ROOT / "participant_models"
SAMPLE = ROOT / "attack" / "sample_submission.csv"

C1_BEST = (CS / "case_1" / "lreg_0.0.pt", 2.5)
C3_BEST = (GRID / "case_3" / "lbd_1.0_lreg_200.0.pt", 2.5)

GAMMA_C2 = 4.0
LAMBDA_REG, LAMBDA_BD = 200.0, 2.0


def write_case(case: int, trained_path: Path, gamma: float) -> None:
    trained = load_state_dict(trained_path)
    reference = fedavg(load_benign(case))
    submitted = scale_delta(reference, trained, gamma)
    for idx in range(MALICIOUS_MODELS_PER_CASE[case]):
        out = MODELS_ROOT / f"case_{case}" / f"malicious_{idx}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        save_state_dict({k: v.detach().clone() for k, v in submitted.items()}, out)


def main() -> None:
    dataset = load_dataset(ROOT / "research" / "cache" / "celeba_subset.pt")
    train, eval_ = train_eval_split(dataset, eval_fraction=0.2, seed=0)
    reference = fedavg(load_benign(2))
    CACHE.mkdir(parents=True, exist_ok=True)

    for name in MASK_VARIANTS:
        cache = CACHE / f"case2_{name}.pt"
        if cache.exists():
            trained = load_state_dict(cache)
        else:
            print(f"training case 2 with trigger={name} ...")
            trained = train_constrained(
                reference, train.images, train.labels, name,
                seed=2000, lambda_reg=LAMBDA_REG, lambda_bd=LAMBDA_BD,
            )
            save_state_dict(trained, cache)

        solo = attack_success_rate(load_model_from_state_dict(trained), eval_.images, name)
        print(f"  {name}: trained solo ASR (proxy) = {solo:.1%}")

        write_case(1, *C1_BEST)
        write_case(3, *C3_BEST)
        write_case(2, cache, GAMMA_C2)
        out_csv = ROOT / f"ATTACK_{name}_g{GAMMA_C2}.csv"
        create_submission(MODELS_ROOT, SAMPLE, out_csv)
        print(f"  -> {out_csv.name}")


if __name__ == "__main__":
    main()
