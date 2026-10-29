import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.checks import EXPECTED_STATE_LAYOUT, validate_collection, validate_state_dict


def flatten_states(states) -> torch.Tensor:
    return torch.stack(
        [
            torch.cat([state[key].to(torch.float64).reshape(-1) for key in EXPECTED_STATE_LAYOUT])
            for state in states
        ]
    )


def unflatten_vector(vector: torch.Tensor) -> OrderedDict:
    state = OrderedDict()
    offset = 0

    for key, shape in EXPECTED_STATE_LAYOUT.items():
        size = 1
        for dimension in shape:
            size *= dimension
        state[key] = vector[offset : offset + size].reshape(shape).to(torch.float32).contiguous()
        offset += size

    return state


def _median(matrix: torch.Tensor) -> torch.Tensor:
    return torch.quantile(matrix, 0.5, dim=0, interpolation="midpoint")


def _pairwise_sq_distances(matrix: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(matrix, matrix, p=2.0) ** 2
    distances.fill_diagonal_(float("inf"))
    return distances


def _krum_scores(matrix: torch.Tensor, num_malicious: int) -> torch.Tensor:
    count = matrix.shape[0]
    neighbours = max(1, count - num_malicious - 2)
    distances = _pairwise_sq_distances(matrix)
    closest, _ = distances.sort(dim=1)
    return closest[:, :neighbours].sum(dim=1)


def _iterative_krum_selection(matrix: torch.Tensor, num_malicious: int, count: int) -> list[int]:
    remaining = list(range(matrix.shape[0]))
    selected: list[int] = []

    for _ in range(count):
        scores = _krum_scores(matrix[remaining], num_malicious)
        winner = remaining.pop(int(scores.argmin()))
        selected.append(winner)

    return selected


def fedavg(matrix: torch.Tensor, num_malicious: int = 0) -> torch.Tensor:
    return matrix.mean(dim=0)


def coordinate_wise_median(matrix: torch.Tensor, num_malicious: int = 0) -> torch.Tensor:
    return _median(matrix)


def trimmed_mean(matrix: torch.Tensor, num_malicious: int) -> torch.Tensor:
    count = matrix.shape[0]

    if 2 * num_malicious >= count:
        raise ValueError(
            f"trimmed_mean needs 2f < n; got f={num_malicious}, n={count} (nothing would survive)."
        )

    ordered, _ = matrix.sort(dim=0)
    return ordered[num_malicious : count - num_malicious].mean(dim=0)


def krum(matrix: torch.Tensor, num_malicious: int) -> torch.Tensor:
    _require(matrix.shape[0] >= 2 * num_malicious + 3, "krum", "n >= 2f+3", matrix.shape[0], num_malicious)
    return matrix[int(_krum_scores(matrix, num_malicious).argmin())].clone()


def multi_krum(matrix: torch.Tensor, num_malicious: int, selection_size: int | None = None) -> torch.Tensor:
    count = matrix.shape[0]
    _require(count >= 2 * num_malicious + 3, "multi_krum", "n >= 2f+3", count, num_malicious)

    if selection_size is None:
        selection_size = max(1, count - num_malicious)

    selected = _iterative_krum_selection(matrix, num_malicious, selection_size)
    return matrix[selected].mean(dim=0)


def bulyan(matrix: torch.Tensor, num_malicious: int) -> torch.Tensor:
    count = matrix.shape[0]
    _require(count >= 4 * num_malicious + 3, "bulyan", "n >= 4f+3", count, num_malicious)

    selection_size = count - 2 * num_malicious
    selected = matrix[_iterative_krum_selection(matrix, num_malicious, selection_size)]

    keep = selection_size - 2 * num_malicious
    distances = (selected - _median(selected)).abs()
    indices = distances.argsort(dim=0)[:keep]
    return selected.gather(0, indices).mean(dim=0)


def _require(condition: bool, strategy: str, rule: str, count: int, num_malicious: int) -> None:
    if not condition:
        raise ValueError(
            f"{strategy} requires {rule}; got n={count}, f={num_malicious}. "
            f"Max supported f for n={count} is {max_supported_malicious(strategy, count)}."
        )


STRATEGIES = OrderedDict(
    [
        ("fedavg", fedavg),
        ("trimmed_mean", trimmed_mean),
        ("median", coordinate_wise_median),
        ("krum", krum),
        ("multi_krum", multi_krum),
        ("bulyan", bulyan),
    ]
)

SELECTION_BASED = frozenset({"krum", "multi_krum"})


def is_selection_based(strategy: str) -> bool:
    return strategy in SELECTION_BASED


def max_supported_malicious(strategy: str, count: int) -> int:
    if strategy in ("fedavg", "median"):
        return (count - 1) // 2
    if strategy == "trimmed_mean":
        return (count - 1) // 2
    if strategy in ("krum", "multi_krum"):
        return max(0, (count - 3) // 2)
    if strategy == "bulyan":
        return max(0, (count - 3) // 4)
    raise ValueError(f"Unknown strategy: {strategy!r}. Options: {list(STRATEGIES)}")


def default_malicious(count: int) -> int:
    return count // 3


def aggregate(models, strategy: str, num_malicious: int | None = None):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Options: {list(STRATEGIES)}")

    validated = validate_collection(models)
    matrix = flatten_states(validated)

    if num_malicious is None:
        num_malicious = min(default_malicious(len(validated)), max_supported_malicious(strategy, len(validated)))

    aggregated = unflatten_vector(STRATEGIES[strategy](matrix, num_malicious))
    return validate_state_dict(aggregated, f"{strategy} output")
