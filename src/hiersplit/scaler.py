"""Client-local normalization used by the independent-learning baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from basicts.scaler.base_scaler import BaseScaler


class PartitionedZScoreScaler(BaseScaler):
    """Use one scalar mean and standard deviation per client's training data."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio: float,
        norm_each_channel: bool,
        rescale: bool,
        client_nodes: list[list[int]],
        target_channel: int = 0,
    ) -> None:
        super().__init__(dataset_name, train_ratio, norm_each_channel, rescale)
        self.target_channel = target_channel

        dataset_dir = Path("datasets") / dataset_name
        with (dataset_dir / "desc.json").open(encoding="utf-8") as file:
            shape = tuple(json.load(file)["shape"])
        data = np.memmap(dataset_dir / "data.dat", dtype="float32", mode="r", shape=shape)
        training = data[: int(len(data) * train_ratio), :, target_channel].copy()

        means = np.zeros((1, shape[1]), dtype=np.float32)
        standard_deviations = np.ones((1, shape[1]), dtype=np.float32)
        for nodes in client_nodes:
            local = training[:, nodes]
            if norm_each_channel:
                mean = local.mean(axis=0, keepdims=True)
                standard_deviation = local.std(axis=0, keepdims=True)
                standard_deviation[standard_deviation == 0] = 1.0
            else:
                mean = float(local.mean())
                standard_deviation = float(local.std()) or 1.0
            means[:, nodes] = mean
            standard_deviations[:, nodes] = standard_deviation

        self.mean = torch.from_numpy(means)
        self.std = torch.from_numpy(standard_deviations)

    def transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        standard_deviation = self.std.to(input_data.device)
        input_data[..., self.target_channel] = (
            input_data[..., self.target_channel] - mean
        ) / standard_deviation
        return input_data

    def inverse_transform(self, input_data: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(input_data.device)
        standard_deviation = self.std.to(input_data.device)
        output = input_data.clone()
        output[..., self.target_channel] = (
            output[..., self.target_channel] * standard_deviation + mean
        )
        return output
