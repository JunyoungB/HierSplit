"""Model-level implementations of IL, SL, and attention-based HierSplit."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .models.attention import CrossAttentionBlock, SelfAttentionBlock
from .models.stgcn import STGCNSpatialServer
from .partition import subgraph


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _prediction_tensor(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    return output["prediction"] if isinstance(output, dict) else output


def _local_model_parameters(
    base_parameters: dict[str, Any],
    nodes: Sequence[int],
) -> dict[str, Any]:
    parameters = dict(base_parameters)
    parameters["num_nodes"] = len(nodes)
    adjacency = parameters.get("adj_matrix")
    if adjacency is not None:
        parameters["adj_matrix"] = subgraph(torch.as_tensor(adjacency), nodes)
    return parameters


class _PartitionedModel(nn.Module):
    def __init__(self, client_nodes: list[list[int]], total_nodes: int) -> None:
        super().__init__()
        self.client_nodes = [list(map(int, nodes)) for nodes in client_nodes]
        self.total_nodes = total_nodes
        self.communication_bytes = 0
        for client, nodes in enumerate(client_nodes):
            self.register_buffer(
                f"client_nodes_{client}",
                torch.tensor(nodes, dtype=torch.long),
                persistent=False,
            )

    def nodes(self, client: int) -> torch.Tensor:
        return getattr(self, f"client_nodes_{client}")

    @property
    def communication_megabytes(self) -> float:
        return self.communication_bytes / 1024**2

    def _record_communication(self, forward_bytes: int, train: bool) -> dict[str, int]:
        # Backpropagation sends gradients with the same tensor shapes in reverse.
        backward_bytes = forward_bytes
        round_trip_bytes = forward_bytes + backward_bytes
        if train and torch.is_grad_enabled():
            self.communication_bytes += round_trip_bytes
        return {
            "forward_bytes": forward_bytes,
            "backward_bytes": backward_bytes,
            "round_trip_bytes": round_trip_bytes,
        }


class IndependentModel(_PartitionedModel):
    """One complete, independently optimized backbone per client."""

    def __init__(
        self,
        base_model: type[nn.Module],
        base_parameters: dict[str, Any],
        client_nodes: list[list[int]],
        total_nodes: int,
    ) -> None:
        super().__init__(client_nodes, total_nodes)
        self.clients = nn.ModuleList(
            [
                base_model(**_local_model_parameters(base_parameters, nodes))
                for nodes in client_nodes
            ]
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        batch_size = history_data.shape[0]
        first = self.clients[0]
        prediction = history_data.new_zeros(
            batch_size,
            first.out_steps,
            self.total_nodes,
            first.output_dim,
        )
        for client_index, client in enumerate(self.clients):
            nodes = self.nodes(client_index)
            local_history = history_data.index_select(2, nodes)
            local_future = future_data.index_select(2, nodes) if future_data is not None else None
            local_prediction = _prediction_tensor(
                client(
                    history_data=local_history,
                    future_data=local_future,
                    batch_seen=batch_seen,
                    epoch=epoch,
                    train=train,
                    **kwargs,
                )
            )
            prediction = prediction.index_copy(2, nodes, local_prediction)
        return {"prediction": prediction}


class SpatialAttentionServer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    model_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            encoded = layer(encoded, dim=2)
        return encoded


class SplitModel(_PartitionedModel):
    """Clients send node-level activations to a full-graph spatial server."""

    def __init__(
        self,
        backbone: str,
        base_model: type[nn.Module],
        base_parameters: dict[str, Any],
        client_nodes: list[list[int]],
        total_nodes: int,
    ) -> None:
        super().__init__(client_nodes, total_nodes)
        self.clients = nn.ModuleList(
            [
                base_model(**_local_model_parameters(base_parameters, nodes))
                for nodes in client_nodes
            ]
        )

        if backbone == "staeformer":
            self.server = SpatialAttentionServer(
                model_dim=self.clients[0].model_dim,
                feed_forward_dim=base_parameters["feed_forward_dim"],
                num_heads=base_parameters["num_heads"],
                num_layers=base_parameters["num_layers"],
                dropout=base_parameters["dropout"],
            )
        elif backbone == "stgcn":
            self.server = STGCNSpatialServer(
                model_dim=self.clients[0].model_dim,
                graph_order=base_parameters["Ks"],
                adj_matrix=base_parameters["adj_matrix"],
                dropout=base_parameters["droprate"],
                bias=base_parameters["bias"],
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | dict[str, int]]:
        del batch_seen, epoch
        encoded_by_client = []
        for client_index, client in enumerate(self.clients):
            local_history = history_data.index_select(2, self.nodes(client_index))
            encoded_by_client.append(client.forward_encoder(local_history, **kwargs))

        example = encoded_by_client[0]
        global_encoded = example.new_zeros(
            example.shape[0],
            example.shape[1],
            self.total_nodes,
            example.shape[-1],
        )
        for client_index, encoded in enumerate(encoded_by_client):
            global_encoded = global_encoded.index_copy(2, self.nodes(client_index), encoded)

        server_encoded = self.server(global_encoded)
        upload_bytes = sum(_tensor_bytes(encoded) for encoded in encoded_by_client)
        download_bytes = _tensor_bytes(server_encoded)
        communication = self._record_communication(upload_bytes + download_bytes, train)

        first = self.clients[0]
        prediction = history_data.new_zeros(
            history_data.shape[0],
            first.out_steps,
            self.total_nodes,
            first.output_dim,
        )
        for client_index, client in enumerate(self.clients):
            nodes = self.nodes(client_index)
            local_prediction = client.forward_decoder(
                server_encoded.index_select(2, nodes),
                **kwargs,
            )
            prediction = prediction.index_copy(2, nodes, local_prediction)

        return {"prediction": prediction, "communication": communication}


class AttentionSummarizer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_tokens: int,
        feed_forward_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, 1, num_tokens, model_dim))
        self.transformer = CrossAttentionBlock(
            model_dim,
            feed_forward_dim,
            num_heads,
            dropout,
        )

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(nodes.shape[0], nodes.shape[1], -1, -1)
        return self.transformer(queries, nodes, nodes)


class TokenServer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SelfAttentionBlock(
                    model_dim,
                    feed_forward_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tokens = layer(tokens, dim=2)
        return tokens


class HierSplitModel(_PartitionedModel):
    """Attention-based HierSplit (HS-A) with compact client/server tokens."""

    def __init__(
        self,
        base_model: type[nn.Module],
        base_parameters: dict[str, Any],
        client_nodes: list[list[int]],
        total_nodes: int,
        num_tokens: int,
        token_heads: int = 4,
        server_heads: int = 4,
        server_layers: int = 1,
        server_feed_forward_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(client_nodes, total_nodes)
        if num_tokens < 1:
            raise ValueError("num_tokens must be positive")
        self.num_tokens = num_tokens
        self.clients = nn.ModuleList(
            [
                base_model(**_local_model_parameters(base_parameters, nodes))
                for nodes in client_nodes
            ]
        )
        model_dim = self.clients[0].model_dim
        self.summarizers = nn.ModuleList(
            [
                AttentionSummarizer(
                    model_dim,
                    num_tokens,
                    server_feed_forward_dim,
                    token_heads,
                    dropout,
                )
                for _ in client_nodes
            ]
        )
        self.server = TokenServer(
            model_dim,
            server_feed_forward_dim,
            server_heads,
            server_layers,
            dropout,
        )
        self.integrators = nn.ModuleList(
            [
                CrossAttentionBlock(
                    model_dim,
                    server_feed_forward_dim,
                    token_heads,
                    dropout,
                )
                for _ in client_nodes
            ]
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | dict[str, int]]:
        del future_data, batch_seen, epoch
        local_representations = []
        local_tokens = []
        for client_index, client in enumerate(self.clients):
            history = history_data.index_select(2, self.nodes(client_index))
            encoded = client.forward_encoder(history, **kwargs)
            represented = client.forward_spatial(encoded, **kwargs)
            local_representations.append(represented)
            local_tokens.append(self.summarizers[client_index](represented))

        uploaded_tokens = torch.cat(local_tokens, dim=2)
        returned_tokens = self.server(uploaded_tokens)
        communication = self._record_communication(
            sum(_tensor_bytes(tokens) for tokens in local_tokens) + _tensor_bytes(returned_tokens),
            train,
        )
        token_slices = returned_tokens.split(self.num_tokens, dim=2)

        first = self.clients[0]
        prediction = history_data.new_zeros(
            history_data.shape[0],
            first.out_steps,
            self.total_nodes,
            first.output_dim,
        )
        for client_index, client in enumerate(self.clients):
            integrated = self.integrators[client_index](
                local_representations[client_index],
                token_slices[client_index],
                token_slices[client_index],
            )
            local_prediction = client.forward_decoder(integrated, **kwargs)
            prediction = prediction.index_copy(
                2,
                self.nodes(client_index),
                local_prediction,
            )

        return {"prediction": prediction, "communication": communication}
