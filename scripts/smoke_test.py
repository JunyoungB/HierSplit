#!/usr/bin/env python3
"""Run fast CPU forward/backward checks without requiring pytest or BasicTS."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiersplit.models import STGCN, STAEformer  # noqa: E402
from hiersplit.partition import random_partition  # noqa: E402
from hiersplit.protocols import HierSplitModel, SplitModel  # noqa: E402


class ToyBackbone(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 3,
        out_steps: int = 3,
        input_dim: int = 2,
        output_dim: int = 1,
        model_dim: int = 8,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = model_dim
        self.encoder = nn.Linear(input_dim, model_dim)
        self.spatial = nn.Linear(model_dim, model_dim)
        self.decoder = nn.Linear(model_dim, output_dim)

    def forward_encoder(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.encoder(x)

    def forward_spatial(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.spatial(x)

    def forward_decoder(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, history_data: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.forward_decoder(
            self.forward_spatial(self.forward_encoder(history_data, **kwargs), **kwargs),
            **kwargs,
        )


def check_backbones() -> None:
    history = torch.rand(2, 12, 6, 3)
    history[..., 1] = torch.randint(0, 288, (2, 12, 6)) / 288
    history[..., 2] = torch.randint(0, 7, (2, 12, 6)) / 7
    staeformer = STAEformer(
        num_nodes=6,
        input_embedding_dim=4,
        tod_embedding_dim=4,
        dow_embedding_dim=4,
        adaptive_embedding_dim=4,
        feed_forward_dim=32,
        num_heads=4,
    )
    staeformer_prediction = staeformer(history)
    assert staeformer_prediction.shape == (2, 12, 6, 1)
    staeformer_prediction.mean().backward()

    stgcn = STGCN(
        num_nodes=6,
        adj_matrix=torch.eye(6),
        Ks=2,
        Kt=2,
        blocks=[[1], [8, 4, 8], [8, 4, 8], [16, 16], [12]],
        droprate=0.0,
    )
    stgcn_prediction = stgcn(history[..., :1])
    assert stgcn_prediction.shape == (2, 12, 6, 1)
    stgcn_prediction.mean().backward()


def check_protocols() -> None:
    parameters = {
        "in_steps": 3,
        "out_steps": 3,
        "input_dim": 2,
        "output_dim": 1,
        "model_dim": 8,
        "feed_forward_dim": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
    }
    nodes = [[0, 2], [1, 3]]
    inputs = torch.randn(2, 3, 4, 2)
    split = SplitModel("staeformer", ToyBackbone, parameters, nodes, total_nodes=4)
    hiersplit = HierSplitModel(
        ToyBackbone,
        parameters,
        nodes,
        total_nodes=4,
        num_tokens=1,
        token_heads=2,
        server_heads=2,
        server_layers=1,
        server_feed_forward_dim=16,
        dropout=0.0,
    )
    split_output = split(inputs)
    hiersplit_output = hiersplit(inputs)
    assert split_output["prediction"].shape == hiersplit_output["prediction"].shape
    assert hiersplit.communication_bytes < split.communication_bytes
    (split_output["prediction"].mean() + hiersplit_output["prediction"].mean()).backward()


def main() -> None:
    groups = random_partition(23, 5)
    assert sorted(node for group in groups for node in group) == list(range(23))
    check_backbones()
    check_protocols()
    print("HierSplit CPU smoke test passed.")


if __name__ == "__main__":
    main()
