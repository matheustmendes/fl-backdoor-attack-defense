import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from attack_constrain_scale import scale_delta
from attack_train import CASE_TRIGGERS, load_benign
from data import load_dataset, train_eval_split
from defense_candidates import (
    _as_matrix,
    clipped_mean,
    geometric_median,
    largest_cluster_mean,
    trimmed_mean_at,
)
from aggregators import unflatten_vector
from evaluate import (
    attack_success_rate,
    clean_accuracy,
    defense_case_score,
    load_model_from_state_dict,
)
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict

CS_CACHE = ROOT / "research" / "cache" / "constrain_scale"


def build_attack_pool(case: int) -> tuple[list, list, dict, str]:
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]
    ref = fedavg(benign)
    trained = load_state_dict(CS_CACHE / f"case_{case}" / "lreg_0.0.pt")
    gamma = (len(benign) + count) / count
    malicious = [scale_delta(ref, trained, gamma) for _ in range(count)]
    return benign, malicious, ref, CASE_TRIGGERS[case]


def defenses(num_models: int) -> dict:
    bound = num_models // 3
    out = {"fedavg (no defense)": lambda ms: fedavg(ms)}
    for t in range(1, bound + 1):
        out[f"trimmed_mean[trim={t}]"] = (
            lambda ms, t=t: unflatten_vector(trimmed_mean_at(_as_matrix(ms), num_trim=t))
        )
    out["median (=trim max)"] = lambda ms: unflatten_vector(trimmed_mean_at(_as_matrix(ms), num_trim=bound))
    out["geometric_median"] = lambda ms: unflatten_vector(geometric_median(_as_matrix(ms)))
    out["clipped_mean[q=0.5]"] = lambda ms: unflatten_vector(clipped_mean(_as_matrix(ms), quantile=0.5))
    out["largest_cluster_mean"] = lambda ms: unflatten_vector(largest_cluster_mean(_as_matrix(ms)))
    return out


def run_case(case: int, ev) -> dict:
    benign, malicious, ref, trigger = build_attack_pool(case)
    pool = benign + malicious
    n = len(pool)

    free_net = load_model_from_state_dict(fedavg(benign))
    free_floor = attack_success_rate(free_net, ev.images, trigger)
    free_clean = clean_accuracy(free_net, ev.images, ev.labels)

    print(f"\n=== Case {case}: {len(benign)} benign + {len(malicious)} malicious "
          f"(FedAvg-optimal, gamma=n/m), trigger={trigger} ===")
    print(f"benign-only free floor: clean={free_clean:.1%} asr={free_floor:.1%} "
          f"(defense_score={defense_case_score(free_clean, free_floor):.1f})")
    print(f"{'defense':>24} | {'clean':>6} | {'asr':>6} | {'lift':>6} | {'score':>6} | {'d_score':>7}")
    print("-" * 74)

    scores = {}
    base_score = None
    for name, agg in defenses(n).items():
        net = load_model_from_state_dict(agg(pool))
        ca = clean_accuracy(net, ev.images, ev.labels)
        asr = attack_success_rate(net, ev.images, trigger)
        sc = defense_case_score(ca, asr)
        if base_score is None:
            base_score = sc
        lift = asr - free_floor
        print(f"{name:>24} | {ca:>5.1%} | {asr:>5.1%} | {lift:>+5.1%} | "
              f"{sc:>6.1f} | {sc - base_score:>+7.1f}")
        scores[name] = sc
    return scores


def main() -> None:
    ds = load_dataset(ROOT / "research" / "cache" / "celeba_defense_eval.pt")
    _, ev = train_eval_split(ds, eval_fraction=0.2, seed=0)

    per_case = {}
    for case in (1, 2, 3):
        per_case[case] = run_case(case, ev)

    names = list(next(iter(per_case.values())).keys())
    print("\n\n=== Summary: mean defense_case_score across the 3 attacked cases ===")
    print(f"{'defense':>24} | {'case1':>6} | {'case2':>6} | {'case3':>6} | {'mean':>6}")
    print("-" * 62)
    ranked = []
    for name in names:
        vals = [per_case[c][name] for c in (1, 2, 3)]
        ranked.append((sum(vals) / 3, name, vals))
    for mean, name, vals in sorted(ranked, reverse=True):
        print(f"{name:>24} | {vals[0]:>6.1f} | {vals[1]:>6.1f} | {vals[2]:>6.1f} | {mean:>6.1f}")


if __name__ == "__main__":
    main()
