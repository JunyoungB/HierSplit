from __future__ import annotations

import torch
from torch import nn

from hiersplit.protocols import HierSplitModel, IndependentModel, SplitModel


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

    def forward_encoder(self, history_data: torch.Tensor, **_: object) -> torch.Tensor:
        return self.encoder(history_data)

    def forward_spatial(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        return self.spatial(encoded)

    def forward_decoder(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        return self.decoder(encoded)

    def forward(self, history_data: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.forward_decoder(
            self.forward_spatial(self.forward_encoder(history_data, **kwargs), **kwargs),
            **kwargs,
        )


NODES = [[0, 2], [1, 3]]
PARAMETERS = {
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


def history() -> torch.Tensor:
    return torch.randn(2, 3, 4, 2)


def test_independent_model_assembles_all_nodes() -> None:
    model = IndependentModel(ToyBackbone, PARAMETERS, NODES, total_nodes=4)
    prediction = model(history())["prediction"]
    prediction.mean().backward()

    assert prediction.shape == (2, 3, 4, 1)
    assert model.communication_bytes == 0
    assert all(client.encoder.weight.grad is not None for client in model.clients)


def test_split_model_communicates_node_activations() -> None:
    model = SplitModel("staeformer", ToyBackbone, PARAMETERS, NODES, total_nodes=4)
    output = model(history())
    output["prediction"].mean().backward()

    assert output["prediction"].shape == (2, 3, 4, 1)
    assert output["communication"]["round_trip_bytes"] == 3072
    assert model.communication_bytes == 3072
    assert all(client.encoder.weight.grad is not None for client in model.clients)


def test_hiersplit_communicates_tokens_and_backpropagates() -> None:
    model = HierSplitModel(
        ToyBackbone,
        PARAMETERS,
        NODES,
        total_nodes=4,
        num_tokens=1,
        token_heads=2,
        server_heads=2,
        server_layers=1,
        server_feed_forward_dim=16,
        dropout=0.0,
    )
    output = model(history())
    output["prediction"].mean().backward()

    assert output["prediction"].shape == (2, 3, 4, 1)
    assert output["communication"]["round_trip_bytes"] == 1536
    assert model.communication_bytes == 1536
    assert model.summarizers[0].queries.grad is not None
    assert model.server.layers[0].attention.query_proj.weight.grad is not None


def test_hiersplit_reduces_communication_relative_to_split() -> None:
    inputs = history()
    split = SplitModel("staeformer", ToyBackbone, PARAMETERS, NODES, total_nodes=4)
    hiersplit = HierSplitModel(
        ToyBackbone,
        PARAMETERS,
        NODES,
        total_nodes=4,
        num_tokens=1,
        token_heads=2,
        server_heads=2,
        server_layers=1,
        server_feed_forward_dim=16,
        dropout=0.0,
    )

    split(inputs)
    hiersplit(inputs)

    assert hiersplit.communication_bytes == split.communication_bytes / 2
