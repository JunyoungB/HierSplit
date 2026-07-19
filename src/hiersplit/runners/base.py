"""Shared BasicTS runner support for deterministic node partitions."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from basicts.runners.runner_zoo.simple_tsf_runner import SimpleTimeSeriesForecastingRunner
from easytorch.utils import is_master

from hiersplit.partition import make_partition


class PartitionedRunner(SimpleTimeSeriesForecastingRunner):
    """Compute a partition before BasicTS constructs the scaler and model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        partition = cfg["PARTITION"]
        self.client_nodes = make_partition(
            method=partition["METHOD"],
            num_nodes=int(partition["NUM_NODES"]),
            num_clients=int(partition["NUM_CLIENTS"]),
            adjacency=partition.get("ADJ_MATRIX"),
            seed=partition.get("SEED"),
        )
        self.total_nodes = int(partition["NUM_NODES"])
        self.backbone = str(cfg["EXPERIMENT"]["BACKBONE"])
        super().__init__(cfg)

        if is_master():
            path = Path(self.ckpt_save_dir) / "partition.json"
            path.write_text(
                json.dumps(
                    {
                        "method": partition["METHOD"],
                        "seed": partition.get("SEED"),
                        "client_nodes": self.client_nodes,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    def build_scaler(self, cfg: dict[str, Any]):
        if "SCALER" not in cfg:
            return None
        scaler_type = cfg["SCALER"]["TYPE"]
        parameters = dict(cfg["SCALER"]["PARAM"])
        signature = inspect.signature(scaler_type.__init__).parameters
        if "client_nodes" in signature:
            parameters.setdefault("client_nodes", self.client_nodes)
        return scaler_type(**parameters)

    def _model_parameters(self, cfg: dict[str, Any]) -> tuple[type, dict[str, Any]]:
        model_type = cfg["MODEL"].get("CLASS") or cfg["MODEL"]["ARCH"]
        return model_type, dict(cfg["MODEL"]["PARAM"])

    def _unwrapped_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _append_communication_result(self) -> None:
        if not is_master():
            return
        path = Path(self.ckpt_save_dir) / "test_metrics.json"
        if not path.exists():
            return
        metrics = json.loads(path.read_text(encoding="utf-8"))
        model = self._unwrapped_model()
        total_bytes = int(getattr(model, "communication_bytes", 0))
        metrics["communication"] = {
            "training_round_trip_bytes": total_bytes,
            "training_round_trip_megabytes": total_bytes / 1024**2,
        }
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def on_training_end(self, cfg: dict[str, Any], train_epoch: int | None = None) -> None:
        super().on_training_end(cfg, train_epoch)
        self._append_communication_result()
