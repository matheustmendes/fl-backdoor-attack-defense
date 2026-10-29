import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import aggregate, max_supported_malicious
from attack_constrain_scale import scale_delta
from attack_train import load_benign
from data import load_dataset, train_eval_split
from defense_candidates import build_candidates
from defense_probe import WINNERS
from evaluate import (
    attack_success_rate,
    clean_accuracy,
    defense_case_score,
    load_model_from_state_dict,
)
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict


def candidate(strategy: str):
    def defense(num_models: int, models: list) -> dict:
        wanted = max(1, num_models // 3)
        f = min(wanted, max_supported_malicious(strategy, num_models))
        return aggregate(models, strategy, num_malicious=f)

    defense.__name__ = strategy
    return defense


CANDIDATES = [
    candidate(s) for s in ("fedavg", "trimmed_mean", "median", "krum", "multi_krum", "bulyan")
] + build_candidates()


def load_visible_case() -> list:
    paths = sorted((ROOT / "defense" / "visible_case").glob("client_*.pt"))
    return [load_state_dict(path) for path in paths]


def build_attacked_pool(case: int) -> list:
    benign = load_benign(case)
    config = WINNERS[case]
    trained = load_state_dict(
        ROOT / "research" / "cache" / "grid" / f"case_{case}"
        / f"lbd_{config['lambda_bd']}_lreg_{config['lambda_reg']}.pt"
    )
    reference = fedavg(benign)
    malicious = [
        scale_delta(reference, trained, config["gamma"])
        for _ in range(MALICIOUS_MODELS_PER_CASE[case])
    ]
    return list(benign) + malicious


def score_pool(name: str, models: list, triggers: list[str], eval_) -> None:
    print(f"\n=== {name}: {len(models)} models ===")

    header = f"{'defense':>26} |{'clean':>7} |"
    for trigger in triggers:
        header += f"{trigger[:9] + ' ASR':>14} |{'score':>7} |"
    print(header)
    print("-" * len(header))

    baseline: dict[str, float] = {}

    for defense in CANDIDATES:
        copies = [{k: v.clone() for k, v in state.items()} for state in models]
        aggregated = defense(len(copies), copies)

        model = load_model_from_state_dict(aggregated)
        clean = clean_accuracy(model, eval_.images, eval_.labels)

        row = f"{defense.__name__:>26} |{clean:>6.1%} |"

        for trigger in triggers:
            asr = attack_success_rate(model, eval_.images, trigger)
            score = defense_case_score(clean, asr)
            key = f"{defense.__name__}:{trigger}"
            baseline.setdefault(key, score)

            delta = score - baseline[f"fedavg:{trigger}"] if "fedavg:" + trigger in baseline else 0.0
            row += f"{asr:>13.1%} |{score:>6.1f} |"

            if defense.__name__ != "fedavg":
                row = row[:-1] + f"{delta:>+5.1f}|"

        print(row)


def main() -> None:
    eval_ = load_dataset(ROOT / "research" / "cache" / "celeba_defense_eval.pt")

    print("Defense Case Score = [0.6*clean + 0.4*(1-ASR)] * 100. Higher is better.")
    print("Read as deltas vs. the fedavg row (= no defense). The +/- column is that delta.")

    score_pool("visible_case (PRIMARY, non-circular)", load_visible_case(),
               ["sunglasses", "surgical_mask"], eval_)

    for case in (1, 2, 3):
        trigger = "surgical_mask" if case == 2 else "sunglasses"
        score_pool(f"case {case} + our attack (SECONDARY, falsifying only)",
                   build_attacked_pool(case), [trigger], eval_)


if __name__ == "__main__":
    main()
