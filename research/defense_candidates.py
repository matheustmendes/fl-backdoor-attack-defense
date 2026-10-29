import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from aggregators import _median, flatten_states, unflatten_vector
from utilities.checks import validate_collection


def _as_matrix(models) -> torch.Tensor:
    return flatten_states(validate_collection(list(models)))


def geometric_median(matrix: torch.Tensor, iterations: int = 100, tolerance: float = 1e-7) -> torch.Tensor:
    estimate = _median(matrix)

    for _ in range(iterations):
        distances = (matrix - estimate).norm(dim=1)

        weights = 1.0 / distances.clamp(min=tolerance)
        weights = weights / weights.sum()

        updated = (weights.unsqueeze(1) * matrix).sum(dim=0)

        if (updated - estimate).norm() < tolerance:
            estimate = updated
            break

        estimate = updated

    return estimate


def clipped_mean(matrix: torch.Tensor, quantile: float = 0.5) -> torch.Tensor:
    centre = _median(matrix)
    deltas = matrix - centre
    norms = deltas.norm(dim=1)

    limit = torch.quantile(norms, quantile)
    scale = (limit / norms.clamp(min=1e-12)).clamp(max=1.0)

    return centre + (deltas * scale.unsqueeze(1)).mean(dim=0)


def largest_cluster_mean(matrix: torch.Tensor, iterations: int = 50) -> torch.Tensor:
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

    counts = [int((assignment == c).sum()) for c in (0, 1)]
    biggest = 0 if counts[0] >= counts[1] else 1

    return matrix[assignment == biggest].mean(dim=0)


def trimmed_mean_at(matrix: torch.Tensor, num_trim: int) -> torch.Tensor:
    count = matrix.shape[0]
    num_trim = max(0, min(num_trim, (count - 1) // 2))
    ordered, _ = matrix.sort(dim=0)
    return ordered[num_trim : count - num_trim].mean(dim=0)


def as_defense(function, **kwargs):
    def defense(num_models: int, models: list):
        return unflatten_vector(function(_as_matrix(models), **kwargs))

    defense.__name__ = function.__name__ + (
        "".join(f"[{k}={v}]" for k, v in kwargs.items()) if kwargs else ""
    )
    return defense


def build_candidates() -> list:
    return [
        as_defense(geometric_median),
        as_defense(clipped_mean, quantile=0.5),
        as_defense(clipped_mean, quantile=0.25),
        as_defense(largest_cluster_mean),
        as_defense(trimmed_mean_at, num_trim=1),
        as_defense(trimmed_mean_at, num_trim=2),
        as_defense(trimmed_mean_at, num_trim=3),
    ]
