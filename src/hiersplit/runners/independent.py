"""Independent-learning runner with per-client validation checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from easytorch.utils import is_master, master_only

from hiersplit.protocols import IndependentModel

from .base import PartitionedRunner


class IndependentRunner(PartitionedRunner):
    """Train local models independently and select each client's best epoch."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        count = len(self.client_nodes)
        self.client_best_metric: list[float | None] = [None] * count
        self.client_best_epoch: list[int | None] = [None] * count
        self.client_validation_sum = [0.0] * count
        self.client_validation_weight = [0.0] * count
        self.client_checkpoint_dir = Path(self.ckpt_save_dir) / "client_best"
        self.client_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def define_model(self, cfg: dict[str, Any]) -> IndependentModel:
        model_type, parameters = self._model_parameters(cfg)
        return IndependentModel(
            model_type,
            parameters,
            self.client_nodes,
            self.total_nodes,
        )

    def val_iters(self, iter_index: int, data) -> None:
        if iter_index == 0:
            self.client_validation_sum = [0.0] * len(self.client_nodes)
            self.client_validation_weight = [0.0] * len(self.client_nodes)

        result = self.forward(data=data, epoch=None, iter_num=iter_index, train=False)
        loss = self.metric_forward(self.loss, result)
        weight = self._get_metric_weight(result["target"])
        self.update_epoch_meter("val/loss", loss.item(), weight)
        for name, metric in self.metrics.items():
            value = self.metric_forward(metric, result)
            self.update_epoch_meter(f"val/{name}", value.item(), weight)

        selection_metric = (
            self.loss if self.target_metrics == "loss" else self.metrics[self.target_metrics]
        )
        for client, nodes in enumerate(self.client_nodes):
            index = torch.as_tensor(nodes, dtype=torch.long, device=result["prediction"].device)
            local_result = {
                "prediction": result["prediction"].index_select(2, index),
                "target": result["target"].index_select(2, index),
            }
            value = self.metric_forward(selection_metric, local_result)
            local_weight = self._get_metric_weight(local_result["target"])
            self.client_validation_sum[client] += float(value) * local_weight
            self.client_validation_weight[client] += local_weight

    @master_only
    def on_validating_end(self, train_epoch: int | None) -> None:
        if train_epoch is None:
            return
        greater_is_better = self.metrics_best == "max"
        model = self._unwrapped_model()

        for client, local_model in enumerate(model.clients):
            weight = self.client_validation_weight[client]
            if not weight:
                continue
            metric = self.client_validation_sum[client] / weight
            previous = self.client_best_metric[client]
            improved = previous is None or (
                metric > previous if greater_is_better else metric < previous
            )
            if not improved:
                continue
            self.client_best_metric[client] = metric
            self.client_best_epoch[client] = train_epoch
            torch.save(
                {
                    "client": client,
                    "epoch": train_epoch,
                    "metric": metric,
                    "model_state_dict": {
                        key: value.detach().cpu() for key, value in local_model.state_dict().items()
                    },
                },
                self.client_checkpoint_dir / f"client_{client}.pt",
            )

        # Reload client-wise validation-best models after every epoch without
        # scanning the test set during training.
        self.load_client_best_models()

    def load_client_best_models(self) -> None:
        model = self._unwrapped_model()
        for client, local_model in enumerate(model.clients):
            path = self.client_checkpoint_dir / f"client_{client}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing client checkpoint: {path}")
            checkpoint = torch.load(path, map_location="cpu")
            local_model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    @torch.no_grad()
    @master_only
    def test(
        self,
        train_epoch: int | None = None,
        save_metrics: bool = False,
        save_results: bool = False,
    ):
        self.load_client_best_models()
        return super().test(train_epoch, save_metrics, save_results)

    @master_only
    def on_training_end(self, cfg: dict[str, Any], train_epoch: int | None = None) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        if hasattr(cfg, "TEST"):
            self.logger.info("Evaluating the per-client validation-best checkpoints.")
            self.load_client_best_models()
            self.test_pipeline(
                cfg=cfg,
                train_epoch=train_epoch,
                save_metrics=True,
                save_results=self.save_results,
            )
        if is_master():
            self._append_communication_result()
