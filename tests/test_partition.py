from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from hiersplit.partition import make_partition, metis_partition, random_partition


def test_random_partition_is_deterministic_and_complete() -> None:
    first = random_partition(23, 5)
    second = random_partition(23, 5)

    assert first == second
    assert sorted(node for group in first for node in group) == list(range(23))
    assert [len(group) for group in first] == [4, 4, 4, 4, 7]


def test_partition_seed_changes_random_assignment() -> None:
    assert random_partition(20, 4, seed=1) != random_partition(20, 4, seed=2)


def test_metis_partition_uses_membership(monkeypatch) -> None:
    fake_metis = SimpleNamespace(
        part_graph=lambda adjacency, clients: (
            2,
            [node % clients for node in range(len(adjacency))],
        )
    )
    monkeypatch.setitem(sys.modules, "metis", fake_metis)
    adjacency = np.ones((6, 6), dtype=np.float32) - np.eye(6, dtype=np.float32)

    groups = metis_partition(adjacency, 2)

    assert groups == [[0, 2, 4], [1, 3, 5]]
    assert make_partition("metis", 6, 2, adjacency) == groups
