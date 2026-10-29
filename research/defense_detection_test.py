import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "defense"))

from attack_constrain_scale import scale_delta
from attack_train import load_benign
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg
from utilities.model_io import load_state_dict

from DEFENSE_alignfilter import _select_benign as align_select

CS = ROOT / "research" / "cache" / "constrain_scale"


def flatten(models):
    names = list(models[0].keys())
    return torch.stack(
        [torch.cat([m[n].to(torch.float64).reshape(-1) for n in names]) for m in models], dim=0
    )


def cluster_select(flat, n):
    if n <= 2:
        return list(range(n))
    d = torch.cdist(flat, flat, p=2.0)
    first, second = divmod(int(d.argmax()), n)
    cent = flat[[first, second]].clone()
    assign = torch.full((n,), -1, dtype=torch.long)
    for _ in range(50):
        up = torch.cdist(flat, cent, p=2.0).argmin(dim=1)
        if torch.equal(up, assign):
            break
        assign = up
        for c in (0, 1):
            mem = flat[assign == c]
            if mem.shape[0] > 0:
                cent[c] = mem.mean(dim=0)
    counts = [int((assign == c).sum()) for c in (0, 1)]
    biggest = 0 if counts[0] >= counts[1] else 1
    return [i for i in range(n) if int(assign[i]) == biggest]


def multikrum_select(flat, n):
    f = n // 3
    while f > 0 and n < 2 * f + 3:
        f -= 1
    if f == 0:
        return list(range(n))
    sq = torch.cdist(flat, flat, p=2.0) ** 2
    k = n - f - 2
    scores = torch.empty(n, dtype=torch.float64)
    for i in range(n):
        row = sq[i].clone()
        row[i] = float("inf")
        scores[i] = torch.sort(row).values[:k].sum()
    return torch.argsort(scores)[: n - f].tolist()


def evaluate(keep, n, m):
    malicious = set(range(n - m, n))
    kept = set(keep)
    dropped = set(range(n)) - kept
    mal_dropped = len(dropped & malicious)
    ben_dropped = len(dropped - malicious)
    recall = mal_dropped / m if m else float("nan")
    fpr = ben_dropped / (n - m) if (n - m) else 0.0
    return recall, fpr, sorted(dropped)


def run():
    methods = {"alignfilter": align_select, "cluster": cluster_select, "multikrum": multikrum_select}

    for case in (1, 2, 3):
        benign = load_benign(case)
        ref = fedavg(benign)
        max_m = MALICIOUS_MODELS_PER_CASE[case]
        print(f"\n===== Case {case}: {len(benign)} benign, inject up to {max_m} malicious =====")

        flat0 = flatten(benign)
        print(f"  [m=0 control, all benign]")
        for name, sel in methods.items():
            keep = sel(flat0, len(benign))
            _, fpr, dropped = evaluate(keep, len(benign), 0)
            print(f"    {name:12s} dropped={dropped}  benign-FPR={fpr:.0%}")

        for lreg in ("0.0", "100.0"):
            trained = load_state_dict(CS / f"case_{case}" / f"lreg_{lreg}.pt")
            mal = scale_delta(ref, trained, 1.5)
            for m in (1, max_m):
                pool = benign + [ {k: v.clone() for k, v in mal.items()} for _ in range(m) ]
                n = len(pool)
                flat = flatten(pool)
                print(f"  [lreg={lreg} ({'blatant' if lreg=='0.0' else 'stealthy'}), m={m}]")
                for name, sel in methods.items():
                    keep = sel(flat, n)
                    recall, fpr, dropped = evaluate(keep, n, m)
                    print(f"    {name:12s} recall={recall:.0%} benign-FPR={fpr:.0%} dropped={dropped}")


if __name__ == "__main__":
    run()
