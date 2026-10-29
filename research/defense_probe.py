import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import _krum_scores, flatten_states
from attack_constrain_scale import scale_delta
from attack_train import load_benign
from utilities.checks import MALICIOUS_MODELS_PER_CASE, fedavg, validate_collection
from utilities.model_io import load_state_dict

WINNERS = {
    1: {"lambda_bd": 2.0, "lambda_reg": 100.0, "gamma": 1.5},
    2: {"lambda_bd": 2.0, "lambda_reg": 50.0, "gamma": 2.0},
    3: {"lambda_bd": 2.0, "lambda_reg": 100.0, "gamma": 2.0},
}


def auc(labels: list[int], scores: list[float]) -> float:
    positives = [s for label, s in zip(labels, scores) if label == 1]
    negatives = [s for label, s in zip(labels, scores) if label == 0]

    if not positives or not negatives:
        return float("nan")

    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def two_means(matrix: torch.Tensor, iterations: int = 50) -> torch.Tensor:
    distances = torch.cdist(matrix, matrix, p=2.0)
    first, second = divmod(int(distances.argmax()), distances.shape[0])
    centroids = matrix[[first, second]].clone()

    assignment = torch.zeros(matrix.shape[0], dtype=torch.long)

    for _ in range(iterations):
        updated = torch.cdist(matrix, centroids, p=2.0).argmin(dim=1)

        if torch.equal(updated, assignment):
            break

        assignment = updated

        for cluster in (0, 1):
            members = matrix[assignment == cluster]

            if members.shape[0] > 0:
                centroids[cluster] = members.mean(dim=0)

    return assignment


def heuristics(matrix: torch.Tensor) -> dict[str, list[float]]:
    count = matrix.shape[0]
    distances = torch.cdist(matrix, matrix, p=2.0)
    centre = matrix.mean(dim=0)

    deltas = matrix - centre
    norms = deltas.norm(dim=1)

    mean_direction = deltas.mean(dim=0)
    cosine = torch.nn.functional.cosine_similarity(deltas, mean_direction.expand_as(deltas), dim=1)

    off_diagonal = distances + torch.eye(count) * 1e9
    nearest = off_diagonal.min(dim=1).values
    mean_pairwise = distances.sum(dim=1) / (count - 1)

    krum = _krum_scores(matrix, max(1, count // 3))

    assignment = two_means(matrix)
    minority = 0 if (assignment == 0).sum() <= (assignment == 1).sum() else 1
    in_minority = [1.0 if a == minority else 0.0 for a in assignment.tolist()]

    return {
        "norm_deviation": ("classic", norms.tolist()),
        "cosine_to_mean(neg)": ("classic", (-cosine).tolist()),
        "krum_score": ("classic", krum.tolist()),
        "mean_pairwise_dist": ("classic", mean_pairwise.tolist()),
        "nearest_neighbour(neg)": ("inverted", (-nearest).tolist()),
        "in_minority_cluster": ("clustering", in_minority),
    }


def probe_labelled_case(case: int) -> None:
    benign = load_benign(case)
    count = MALICIOUS_MODELS_PER_CASE[case]
    config = WINNERS[case]

    reference = fedavg(benign)
    trained = load_state_dict(
        ROOT / "research" / "cache" / "grid" / f"case_{case}"
        / f"lbd_{config['lambda_bd']}_lreg_{config['lambda_reg']}.pt"
    )
    malicious = [scale_delta(reference, trained, config["gamma"]) for _ in range(count)]

    matrix = flatten_states(validate_collection(list(benign) + malicious))
    labels = [0] * len(benign) + [1] * count

    print(
        f"\n=== Case {case}: {len(benign)} benign + {count} malicious "
        f"(lambda_reg={config['lambda_reg']}, gamma={config['gamma']}) ==="
    )
    print(
        f"{'heuristic':>24} | {'family':>10} | {'AUC':>5} | {'verdict':>22} | "
        f"malicious ranks (0=most suspicious)"
    )
    print("-" * 112)

    for name, (family, scores) in heuristics(matrix).items():
        ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
        malicious_ranks = sorted(ranking.index(i) for i, l in enumerate(labels) if l == 1)
        value = auc(labels, scores)

        if value >= 0.85:
            verdict = f"WORKS ({family})"
        elif value <= 0.15:
            verdict = f"BACKWARDS ({family})"
        elif 0.35 <= value <= 0.65:
            verdict = "blind"
        else:
            verdict = "weak"

        print(
            f"{name:>24} | {family:>10} | {value:>5.2f} | {verdict:>22} | "
            f"{malicious_ranks} of {len(labels)}"
        )


def probe_visible_case() -> None:
    paths = sorted((ROOT / "defense" / "visible_case").glob("client_*.pt"))
    matrix = flatten_states(validate_collection([load_state_dict(p) for p in paths]))
    count = matrix.shape[0]

    distances = torch.cdist(matrix, matrix, p=2.0)
    off_diagonal = distances + torch.eye(count) * 1e9

    print(f"\n\n=== visible_case: {count} models, labels unknown ===")
    print("nearest-neighbour distance per model (inverted signal: low = suspicious)")

    nearest = off_diagonal.min(dim=1)
    for i in range(count):
        print(
            f"  {paths[i].stem:>10}: nn dist {nearest.values[i]:.4f} "
            f"(nearest is {paths[int(nearest.indices[i])].stem})"
        )

    values = nearest.values
    print(
        f"\n  spread: min {values.min():.4f} | median {values.median():.4f} | "
        f"max {values.max():.4f} | max/min ratio {(values.max() / values.min()):.2f}x"
    )


def main() -> None:
    print("AUC = P(malicious scores higher than benign). 0.5 = blind, 1.0 = classic")
    print("hypothesis works, 0.0 = inverted hypothesis works (same detector, flipped sign).")

    for case in (1, 2, 3):
        probe_labelled_case(case)

    probe_visible_case()


if __name__ == "__main__":
    main()
