from __future__ import annotations

import torch

from hiersplit.models import STGCN, STAEformer


def staeformer_input(batch_size: int, steps: int, nodes: int) -> torch.Tensor:
    history = torch.rand(batch_size, steps, nodes, 3)
    history[..., 1] = torch.randint(0, 288, (batch_size, steps, nodes)) / 288
    history[..., 2] = torch.randint(0, 7, (batch_size, steps, nodes)) / 7
    return history


def test_staeformer_split_boundaries_and_gradient() -> None:
    model = STAEformer(
        num_nodes=6,
        input_embedding_dim=4,
        tod_embedding_dim=4,
        dow_embedding_dim=4,
        adaptive_embedding_dim=4,
        feed_forward_dim=32,
        num_heads=4,
    )
    history = staeformer_input(2, 12, 6)

    temporal = model.forward_encoder(history)
    spatial = model.forward_spatial(temporal)
    prediction = model.forward_decoder(spatial)
    prediction.mean().backward()

    assert temporal.shape == (2, 12, 6, 16)
    assert prediction.shape == (2, 12, 6, 1)
    assert model.input_projection.weight.grad is not None
    assert model.spatial_layers[0].attention.query_proj.weight.grad is not None


def test_stgcn_split_boundaries_and_gradient() -> None:
    model = STGCN(
        num_nodes=6,
        adj_matrix=torch.eye(6),
        Ks=2,
        Kt=2,
        blocks=[[1], [8, 4, 8], [8, 4, 8], [16, 16], [12]],
        droprate=0.0,
    )
    history = torch.rand(2, 12, 6, 1)

    encoded = model.forward_encoder(history)
    prediction = model.forward_decoder(encoded)
    prediction.mean().backward()

    assert encoded.shape == (2, 8, 6, 8)
    assert prediction.shape == (2, 12, 6, 1)
    assert model.block1.temporal1.convolution.weight.grad is not None
    assert model.output.linear2.weight.grad is not None
