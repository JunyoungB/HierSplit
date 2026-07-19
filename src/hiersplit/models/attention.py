"""Attention blocks shared by STAEformer and HierSplit.

Adapted and reorganized from BasicTS (Apache-2.0).
"""

from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    """Multi-head attention over the penultimate tensor dimension."""

    def __init__(self, model_dim: int, num_heads: int = 4, causal: bool = False) -> None:
        super().__init__()
        if model_dim % num_heads:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.causal = causal

        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        target_length = query.shape[-2]
        source_length = key.shape[-2]

        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        scores = query @ key.transpose(-1, -2) / self.head_dim**0.5
        if self.causal:
            mask = torch.ones(
                target_length,
                source_length,
                dtype=torch.bool,
                device=query.device,
            ).tril()
            scores = scores.masked_fill(~mask, -torch.inf)

        attended = torch.softmax(scores, dim=-1) @ value
        attended = torch.cat(torch.split(attended, batch_size, dim=0), dim=-1)
        return self.output_proj(attended)


class SelfAttentionBlock(nn.Module):
    """Predefined Transformer block used in the experiments."""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.attention = Attention(model_dim, num_heads, causal)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2) -> torch.Tensor:
        x = x.transpose(dim, -2)
        attended = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attended))
        transformed = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(transformed))
        return x.transpose(dim, -2)


class CrossAttentionBlock(nn.Module):
    """Transformer block with distinct query and key/value sequences."""

    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = Attention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attention(query, key, value)
        query = self.norm1(query + self.dropout1(attended))
        transformed = self.feed_forward(query)
        return self.norm2(query + self.dropout2(transformed))
