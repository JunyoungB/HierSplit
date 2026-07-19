"""BasicTS integration for attention-based HierSplit (HS-A)."""

from __future__ import annotations

from typing import Any

from hiersplit.protocols import HierSplitModel

from .base import PartitionedRunner


class HierSplitRunner(PartitionedRunner):
    def define_model(self, cfg: dict[str, Any]) -> HierSplitModel:
        model_type, parameters = self._model_parameters(cfg)
        settings = cfg["HIERSPLIT"]
        return HierSplitModel(
            model_type,
            parameters,
            self.client_nodes,
            self.total_nodes,
            num_tokens=int(settings["NUM_TOKENS"]),
            token_heads=int(settings["TOKEN_HEADS"]),
            server_heads=int(settings["SERVER_HEADS"]),
            server_layers=int(settings["SERVER_LAYERS"]),
            server_feed_forward_dim=int(settings["SERVER_FF_DIM"]),
            dropout=float(settings["DROPOUT"]),
        )
