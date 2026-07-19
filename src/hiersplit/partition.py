"""The Random and METIS node partitions used in the released experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch


def _default_random_seed(num_nodes: int, num_clients: int) -> int:
    """Match the deterministic seed used by the original experiment runner."""

    payload = f"{num_nodes}_{num_clients}_imbNone_aNone_m1"
    return int(hashlib.md5(payload.encode()).hexdigest()[:8], 16)


def _validate_partition(groups: Sequence[Sequence[int]], num_nodes: int) -> None:
    if not groups or any(not group for group in groups):
        raise ValueError("Every client must own at least one node")
    flattened = [int(node) for group in groups for node in group]
    if sorted(flattened) != list(range(num_nodes)):
        raise ValueError("A partition must contain every node exactly once")


def random_partition(
    num_nodes: int,
    num_clients: int,
    seed: int | None = None,
) -> list[list[int]]:
    """Shuffle nodes and split them into deterministic balanced client groups."""

    if not 1 <= num_clients <= num_nodes:
        raise ValueError("num_clients must be between 1 and num_nodes")
    rng = np.random.default_rng(
        _default_random_seed(num_nodes, num_clients) if seed is None else seed
    )
    nodes = np.arange(num_nodes, dtype=np.int64)
    rng.shuffle(nodes)

    group_size = num_nodes // num_clients
    groups = []
    for client in range(num_clients):
        start = client * group_size
        end = num_nodes if client == num_clients - 1 else (client + 1) * group_size
        groups.append(nodes[start:end].tolist())
    _validate_partition(groups, num_nodes)
    return groups


def metis_partition(
    adjacency: torch.Tensor | np.ndarray,
    num_clients: int,
) -> list[list[int]]:
    """Partition a graph with the same ``metis.part_graph`` call as the experiments."""

    try:
        import metis
    except ImportError as error:  # pragma: no cover - depends on the system library
        raise RuntimeError(
            "METIS partitioning requires the `metis` Python package and libmetis. "
            "Install the provided environment before running this partition."
        ) from error

    matrix = (
        adjacency.detach().cpu().numpy() if torch.is_tensor(adjacency) else np.asarray(adjacency)
    )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not 1 <= num_clients <= matrix.shape[0]:
        raise ValueError("num_clients must be between 1 and the number of nodes")

    adjacency_list = [
        np.flatnonzero(matrix[node]).astype(int).tolist() for node in range(len(matrix))
    ]
    _, membership = metis.part_graph(adjacency_list, num_clients)
    groups = [[] for _ in range(num_clients)]
    for node, client in enumerate(membership):
        groups[int(client)].append(node)
    _validate_partition(groups, matrix.shape[0])
    return groups


def make_partition(
    method: str,
    num_nodes: int,
    num_clients: int,
    adjacency: torch.Tensor | np.ndarray | None = None,
    seed: int | None = None,
) -> list[list[int]]:
    method = method.lower()
    if method == "random":
        return random_partition(num_nodes, num_clients, seed)
    if method == "metis":
        if adjacency is None:
            raise ValueError("METIS partitioning requires an adjacency matrix")
        return metis_partition(adjacency, num_clients)
    raise ValueError(f"Unsupported partition method: {method}")


def subgraph(adjacency: torch.Tensor, nodes: Sequence[int]) -> torch.Tensor:
    index = torch.as_tensor(nodes, dtype=torch.long, device=adjacency.device)
    return adjacency.index_select(0, index).index_select(1, index).contiguous()
