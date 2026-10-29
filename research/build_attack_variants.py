import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from attack_constrain_scale import scale_delta
from attack_train import load_benign
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict, save_state_dict

sys.path.insert(0, str(ROOT / "attack"))
from create_attack_submission import create_submission

CS = ROOT / "research" / "cache" / "constrain_scale"
GRID = ROOT / "research" / "cache" / "grid"
MODELS_ROOT = ROOT / "participant_models"
SAMPLE = ROOT / "attack" / "sample_submission.csv"

NM = {1: 10 / 2, 2: 25 / 5, 3: 20 / 5}

ROBUST_C2 = (GRID / "case_2" / "lbd_2.0_lreg_100.0.pt", 1.0)
ROBUST_C3 = (GRID / "case_3" / "lbd_1.0_lreg_200.0.pt", 2.5)

C1_SRC = CS / "case_1" / "lreg_0.0.pt"

C1_BEST = (C1_SRC, 2.5)
C2_SRC_200 = GRID / "case_2" / "lbd_2.0_lreg_200.0.pt"
C2_SRC_100 = GRID / "case_2" / "lbd_2.0_lreg_100.0.pt"

C3_BEST = ROBUST_C3

C2_BEST = (C2_SRC_200, 6.0)
C3_SRC = GRID / "case_3" / "lbd_1.0_lreg_200.0.pt"
C3_BEST_NEW = (C3_SRC, 4.0)

C1_STEALTH = {lreg: CS / "case_1" / f"lreg_{lreg}.pt" for lreg in ("10.0", "30.0")}

VARIANTS = {}
VARIANTS["c3_g4.5"] = {1: C1_BEST, 2: C2_BEST, 3: (C3_SRC, 4.5)}
for lreg, src in C1_STEALTH.items():
    for gamma in (2.5, 3.5, 5.0):
        VARIANTS[f"c1_r{lreg}_g{gamma}"] = {1: (src, gamma), 2: C2_BEST, 3: C3_BEST_NEW}


def write_models(spec: dict) -> None:
    for case, (trained_path, gamma) in spec.items():
        trained = load_state_dict(trained_path)
        reference = fedavg(load_benign(case))
        submitted = scale_delta(reference, trained, gamma)
        for index in range(MALICIOUS_MODELS_PER_CASE[case]):
            out = MODELS_ROOT / f"case_{case}" / f"malicious_{index}.pt"
            out.parent.mkdir(parents=True, exist_ok=True)
            save_state_dict({k: v.detach().clone() for k, v in submitted.items()}, out)


def main() -> None:
    for name, spec in VARIANTS.items():
        write_models(spec)
        out_csv = ROOT / f"ATTACK_{name}.csv"
        create_submission(MODELS_ROOT, SAMPLE, out_csv)
        print(f"  -> {out_csv.name}")


if __name__ == "__main__":
    main()
