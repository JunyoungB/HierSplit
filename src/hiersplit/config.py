"""Build one compact BasicTS configuration for all released experiment cells."""

from __future__ import annotations

from pathlib import Path

import torch
from easydict import EasyDict

from .models import STGCN, STAEformer

DATASETS = {
    "METR-LA": {"nodes": 207, "batch_size": 32, "tokens": 4},
    "PEMS-BAY": {"nodes": 325, "batch_size": 32, "tokens": 6},
    "PEMS04": {"nodes": 307, "batch_size": 32, "tokens": 6},
    "PEMS08": {"nodes": 170, "batch_size": 64, "tokens": 3},
}


def build_config(
    *,
    method: str,
    backbone: str,
    dataset: str,
    partition: str,
    learning_rate: float,
    seed: int,
    partition_seed: int | None,
    num_clients: int,
    num_tokens: int | None,
    epochs: int,
    output_dir: Path,
    use_gpu: bool,
) -> EasyDict:
    """Return the EasyDict consumed directly by ``basicts.launch_training``."""

    from basicts.data import TimeSeriesForecastingDataset
    from basicts.metrics import masked_mae, masked_mape, masked_rmse
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings, load_adj

    from .runners import HierSplitRunner, IndependentRunner, SplitRunner
    from .scaler import PartitionedZScoreScaler

    if method not in {"independent", "split", "hiersplit"}:
        raise ValueError(f"Unsupported method: {method}")
    if backbone not in {"staeformer", "stgcn"}:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataset_info = DATASETS[dataset]
    regular = get_regular_settings(dataset)
    adjacency_path = f"datasets/{dataset}/adj_mx.pkl"
    _, raw_adjacency = load_adj(adjacency_path, "original")
    partition_adjacency = torch.as_tensor(raw_adjacency, dtype=torch.float32)

    if backbone == "staeformer":
        model_type = STAEformer
        model_parameters = {
            "in_steps": regular["INPUT_LEN"],
            "out_steps": regular["OUTPUT_LEN"],
            "steps_per_day": 288,
            "input_dim": 3,
            "output_dim": 1,
            "input_embedding_dim": 24,
            "tod_embedding_dim": 24,
            "dow_embedding_dim": 24,
            "adaptive_embedding_dim": 24,
            "feed_forward_dim": 256,
            "num_heads": 4,
            "num_layers": 1,
            "dropout": 0.1,
        }
        forward_features = [0, 1, 2]
    else:
        model_type = STGCN
        normalized_laplacian, _ = load_adj(adjacency_path, "normlap")
        normalized_laplacian = torch.as_tensor(normalized_laplacian[0], dtype=torch.float32)
        # Use the same graph representation for STGCN partitioning and graph
        # convolution.
        partition_adjacency = normalized_laplacian
        model_parameters = {
            "in_steps": regular["INPUT_LEN"],
            "out_steps": regular["OUTPUT_LEN"],
            "input_dim": 1,
            "output_dim": 1,
            "Ks": 3,
            "Kt": 3,
            "blocks": [[1], [64, 16, 64], [64, 16, 64], [128, 128], [12]],
            "adj_matrix": normalized_laplacian,
            "bias": True,
            "droprate": 0.5,
        }
        forward_features = [0]

    runner_type = {
        "independent": IndependentRunner,
        "split": SplitRunner,
        "hiersplit": HierSplitRunner,
    }[method]
    token_count = int(num_tokens or dataset_info["tokens"])
    run_name = (
        f"{dataset}_{method}_{backbone}_{partition}_c{num_clients}_"
        f"t{token_count}_lr{learning_rate:g}_seed{seed}"
    )

    cfg = EasyDict()
    cfg.DESCRIPTION = "Minimal HierSplit paper reproduction"
    cfg.GPU_NUM = 1 if use_gpu else 0
    cfg.ENV = EasyDict(
        {
            "SEED": seed,
            "TF32": False,
            "DETERMINISTIC": False,
            "CUDNN": EasyDict({"ENABLED": True, "BENCHMARK": True, "DETERMINISTIC": False}),
        }
    )
    cfg.RUNNER = runner_type
    cfg.EXPERIMENT = EasyDict({"METHOD": method, "BACKBONE": backbone})
    cfg.PARTITION = EasyDict(
        {
            "METHOD": partition,
            "NUM_NODES": dataset_info["nodes"],
            "NUM_CLIENTS": num_clients,
            "ADJ_MATRIX": partition_adjacency,
            "SEED": partition_seed,
        }
    )
    cfg.HIERSPLIT = EasyDict(
        {
            "NUM_TOKENS": token_count,
            "TOKEN_HEADS": 4,
            "SERVER_HEADS": 4,
            "SERVER_LAYERS": 1,
            "SERVER_FF_DIM": 256,
            "DROPOUT": 0.1,
        }
    )

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = dataset
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": dataset,
            "train_val_test_ratio": regular["TRAIN_VAL_TEST_RATIO"],
            "input_len": regular["INPUT_LEN"],
            "output_len": regular["OUTPUT_LEN"],
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = PartitionedZScoreScaler if method == "independent" else ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": dataset,
            "train_ratio": regular["TRAIN_VAL_TEST_RATIO"][0],
            "norm_each_channel": regular["NORM_EACH_CHANNEL"],
            "rescale": regular["RESCALE"],
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = run_name
    cfg.MODEL.ARCH = model_type
    cfg.MODEL.CLASS = model_type
    cfg.MODEL.PARAM = EasyDict(model_parameters)
    cfg.MODEL.FORWARD_FEATURES = forward_features
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict({"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape})
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.BEST = "min"
    cfg.METRICS.NULL_VAL = regular["NULL_VAL"]

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = epochs
    cfg.TRAIN.CKPT_SAVE_DIR = str(output_dir / run_name)
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict(
        {"TYPE": "Adam", "PARAM": {"lr": learning_rate, "weight_decay": 3e-4}}
    )
    if backbone == "stgcn":
        scheduler_parameters = {"milestones": [1, 50], "gamma": 0.5}
    else:
        scheduler_parameters = {"milestones": [20, 25], "gamma": 0.1}
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": scheduler_parameters})
    cfg.TRAIN.DATA = EasyDict({"BATCH_SIZE": dataset_info["batch_size"], "SHUFFLE": True})

    cfg.VAL = EasyDict()
    cfg.VAL.INTERVAL = 1
    cfg.VAL.DATA = EasyDict({"BATCH_SIZE": dataset_info["batch_size"]})

    cfg.TEST = EasyDict()
    cfg.TEST.INTERVAL = epochs + 1
    cfg.TEST.DATA = EasyDict({"BATCH_SIZE": dataset_info["batch_size"]})

    cfg.EVAL = EasyDict({"HORIZONS": [3, 6, 12], "USE_GPU": use_gpu, "SAVE_RESULTS": False})
    return cfg
