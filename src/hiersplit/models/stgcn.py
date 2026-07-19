"""Lightweight split-capable STGCN backbone.

Adapted and reorganized from BasicTS (Apache-2.0). The implementation retains
the two ST-Conv blocks and output head used in the HierSplit experiments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class ChannelAlign(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.projection = (
            nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1))
            if input_channels > output_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is not None:
            return self.projection(x)
        if self.input_channels < self.output_channels:
            padding = x.new_zeros(
                x.shape[0],
                self.output_channels - self.input_channels,
                x.shape[2],
                x.shape[3],
            )
            return torch.cat((x, padding), dim=1)
        return x


class TemporalGLU(nn.Module):
    def __init__(self, kernel_size: int, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.output_channels = output_channels
        self.align = ChannelAlign(input_channels, output_channels)
        self.convolution = nn.Conv2d(
            input_channels,
            2 * output_channels,
            kernel_size=(kernel_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.align(x)[:, :, self.kernel_size - 1 :, :]
        left, gate = self.convolution(x).split(self.output_channels, dim=1)
        return (left + residual) * torch.sigmoid(gate)


class ChebyshevGraphConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        order: int,
        graph_shift: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("Chebyshev order must be positive")
        self.order = order
        self.register_buffer("graph_shift", torch.as_tensor(graph_shift).float())
        self.weight = nn.Parameter(torch.empty(order, input_channels, output_channels))
        self.bias = nn.Parameter(torch.empty(output_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,N] -> [B,T,N,C]
        x = x.permute(0, 2, 3, 1)
        graph_shift = self.graph_shift.to(dtype=x.dtype)
        polynomials = [x]
        if self.order > 1:
            polynomials.append(torch.einsum("ij,btjc->btic", graph_shift, x))
        for _ in range(2, self.order):
            polynomials.append(
                2 * torch.einsum("ij,btjc->btic", graph_shift, polynomials[-1]) - polynomials[-2]
            )

        stacked = torch.stack(polynomials, dim=2)
        output = torch.einsum("btkni,kio->btno", stacked, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class ResidualGraphConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        order: int,
        graph_shift: torch.Tensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.align = ChannelAlign(input_channels, output_channels)
        self.convolution = ChebyshevGraphConv(
            output_channels,
            output_channels,
            order,
            graph_shift,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.align(x)
        convolved = self.convolution(residual).permute(0, 3, 1, 2)
        return convolved + residual


class STConvBlock(nn.Module):
    def __init__(
        self,
        temporal_kernel: int,
        graph_order: int,
        num_nodes: int,
        input_channels: int,
        channels: Sequence[int],
        graph_shift: torch.Tensor,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("Each ST-Conv block requires three channel widths")
        self.temporal1 = TemporalGLU(temporal_kernel, input_channels, channels[0])
        self.graph = ResidualGraphConv(
            channels[0],
            channels[1],
            graph_order,
            graph_shift,
            bias,
        )
        self.temporal2 = TemporalGLU(temporal_kernel, channels[1], channels[2])
        self.norm = nn.LayerNorm([num_nodes, channels[2]])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal1(x)
        x = torch.relu(self.graph(x))
        x = self.temporal2(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x)


class OutputBlock(nn.Module):
    def __init__(
        self,
        remaining_steps: int,
        input_channels: int,
        hidden_channels: Sequence[int],
        out_steps: int,
        num_nodes: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.temporal = TemporalGLU(remaining_steps, input_channels, hidden_channels[0])
        self.norm = nn.LayerNorm([num_nodes, hidden_channels[0]])
        self.linear1 = nn.Linear(hidden_channels[0], hidden_channels[1], bias=bias)
        self.linear2 = nn.Linear(hidden_channels[1], out_steps, bias=bias)
        del dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = torch.relu(self.linear1(x))
        return self.linear2(x).permute(0, 3, 1, 2)


class STGCN(nn.Module):
    """Two-block STGCN with an explicit encoder/decoder interface."""

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        input_dim: int = 1,
        output_dim: int = 1,
        Ks: int = 3,
        Kt: int = 3,
        blocks: Sequence[Sequence[int]] = (
            (1,),
            (64, 16, 64),
            (64, 16, 64),
            (128, 128),
            (12,),
        ),
        adj_matrix: torch.Tensor | None = None,
        bias: bool = True,
        droprate: float = 0.5,
        **_: object,
    ) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError("The paper STGCN configuration supports output_dim=1")
        if adj_matrix is None:
            raise ValueError("STGCN requires adj_matrix")
        if input_dim != blocks[0][0]:
            raise ValueError("input_dim must match blocks[0][0]")

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_dim = int(blocks[2][-1])

        self.block1 = STConvBlock(
            Kt,
            Ks,
            num_nodes,
            blocks[0][0],
            blocks[1],
            adj_matrix,
            droprate,
            bias,
        )
        self.block2 = STConvBlock(
            Kt,
            Ks,
            num_nodes,
            blocks[1][-1],
            blocks[2],
            adj_matrix,
            droprate,
            bias,
        )

        remaining_steps = in_steps - 4 * (Kt - 1)
        if remaining_steps <= 0:
            raise ValueError("Temporal kernels leave no steps for the output block")
        self.output = OutputBlock(
            remaining_steps,
            blocks[2][-1],
            blocks[-2],
            out_steps,
            num_nodes,
            droprate,
            bias,
        )

    def forward_encoder(self, history_data: torch.Tensor, **_: object) -> torch.Tensor:
        x = history_data.permute(0, 3, 1, 2)
        x = self.block2(self.block1(x))
        return x.permute(0, 2, 3, 1)

    def forward_spatial(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        return encoded

    def forward_decoder(self, encoded: torch.Tensor, **_: object) -> torch.Tensor:
        x = self.output(encoded.permute(0, 3, 1, 2))
        return x.permute(0, 1, 3, 2)

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
        return self.forward_decoder(encoded, **kwargs)


class STGCNSpatialServer(nn.Module):
    """Full-graph spatial operation used by the STGCN SL baseline."""

    def __init__(
        self,
        model_dim: int,
        graph_order: int,
        adj_matrix: torch.Tensor,
        dropout: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.graph = ResidualGraphConv(
            model_dim,
            model_dim,
            graph_order,
            adj_matrix,
            bias,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        x = encoded.permute(0, 3, 1, 2)
        x = self.dropout(torch.relu(self.graph(x)))
        return x.permute(0, 2, 3, 1)
