"""Split-capable STAEformer backbone.

Adapted and reorganized from BasicTS (Apache-2.0). Only the configuration used
by the HierSplit experiments is retained.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import SelfAttentionBlock


class STAEformer(nn.Module):
    """STAEformer with explicit temporal, spatial, and decoder boundaries."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 24,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim

        self.model_dim = (
            input_embedding_dim + tod_embedding_dim + dow_embedding_dim + adaptive_embedding_dim
        )

        self.input_projection = nn.Linear(input_dim, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        block_kwargs = {
            "model_dim": self.model_dim,
            "feed_forward_dim": feed_forward_dim,
            "num_heads": num_heads,
            "dropout": dropout,
        }
        self.temporal_layers = nn.ModuleList(
            [SelfAttentionBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.spatial_layers = nn.ModuleList(
            [SelfAttentionBlock(**block_kwargs) for _ in range(num_layers)]
        )
        self.output_projection = nn.Linear(
            in_steps * self.model_dim,
            out_steps * output_dim,
        )

    def _embed_history(self, history_data: torch.Tensor) -> torch.Tensor:
        if history_data.shape[1] != self.in_steps:
            raise ValueError(f"Expected {self.in_steps} history steps, got {history_data.shape[1]}")
        if history_data.shape[2] != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {history_data.shape[2]}")

        batch_size = history_data.shape[0]
        time_of_day = (history_data[..., -2] * self.steps_per_day).long()
        day_of_week = (history_data[..., -1] * 7).long()

        features = [self.input_projection(history_data[..., : self.input_dim])]
        features.append(self.tod_embedding(time_of_day))
        features.append(self.dow_embedding(day_of_week))
        features.append(self.adaptive_embedding.expand(batch_size, *self.adaptive_embedding.shape))
        return torch.cat(features, dim=-1)

    def forward_encoder(self, history_data: torch.Tensor, **_: object) -> torch.Tensor:
        x = self._embed_history(history_data)
        for layer in self.temporal_layers:
            x = layer(x, dim=1)
        return x

    def forward_spatial(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        x = encoded
        for layer in self.spatial_layers:
            x = layer(x, dim=2)
        return x

    def forward_decoder(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        batch_size, steps, num_nodes, model_dim = encoded.shape
        flattened = encoded.transpose(1, 2).reshape(
            batch_size,
            num_nodes,
            steps * model_dim,
        )
        output = self.output_projection(flattened)
        output = output.view(batch_size, num_nodes, self.out_steps, self.output_dim)
        return output.transpose(1, 2)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs: object,
    ) -> torch.Tensor:
        del future_data, batch_seen, epoch, train
        encoded = self.forward_encoder(history_data, **kwargs)
        encoded = self.forward_spatial(encoded, **kwargs)
        return self.forward_decoder(encoded, **kwargs)
