"""BasicTS integration for the node-activation split-learning baseline."""

from __future__ import annotations

from typing import Any

from hiersplit.protocols import SplitModel

from .base import PartitionedRunner


class SplitRunner(PartitionedRunner):
    def define_model(self, cfg: dict[str, Any]) -> SplitModel:
        model_type, parameters = self._model_parameters(cfg)
        return SplitModel(
            self.backbone,
            model_type,
            parameters,
            self.client_nodes,
            self.total_nodes,
        )
