#!/usr/bin/env python3
"""Launch one released HierSplit experiment through an unmodified BasicTS."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from hiersplit import BASICTS_COMMIT, BASICTS_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("independent", "split", "hiersplit"),
    )
    parser.add_argument(
        "--backbone",
        required=True,
        choices=("staeformer", "stgcn"),
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("METR-LA", "PEMS-BAY", "PEMS04", "PEMS08"),
    )
    parser.add_argument("--partition", required=True, choices=("random", "metis"))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partition-seed", type=int)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-tokens", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--basicts-root",
        type=Path,
        default=Path(os.environ.get("BASICTS_ROOT", REPOSITORY_ROOT / "third_party/BasicTS")),
    )
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument(
        "--allow-unpinned-basicts",
        action="store_true",
        help="Allow a BasicTS checkout other than the documented commit.",
    )
    return parser.parse_args()


def verify_basicts(path: Path, allow_unpinned: bool) -> None:
    if not (path / "basicts/__init__.py").is_file():
        raise FileNotFoundError(
            f"BasicTS was not found at {path}. Follow the README setup instructions or "
            "pass --basicts-root."
        )
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        if allow_unpinned:
            return
        raise RuntimeError("Could not verify the BasicTS git commit") from error
    if commit != BASICTS_COMMIT and not allow_unpinned:
        raise RuntimeError(
            f"Expected BasicTS {BASICTS_COMMIT}, found {commit}. "
            "Checkout the pinned commit or explicitly pass --allow-unpinned-basicts."
        )


def main() -> None:
    args = parse_args()
    basicts_root = args.basicts_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    verify_basicts(basicts_root, args.allow_unpinned_basicts)
    sys.path.insert(0, str(basicts_root))

    import basicts
    import torch

    if basicts.__version__ != BASICTS_VERSION and not args.allow_unpinned_basicts:
        raise RuntimeError(
            f"Expected BasicTS version {BASICTS_VERSION}, found {basicts.__version__}"
        )

    from hiersplit.config import build_config

    os.chdir(basicts_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cfg = build_config(
        method=args.method,
        backbone=args.backbone,
        dataset=args.dataset,
        partition=args.partition,
        learning_rate=args.lr,
        seed=args.seed,
        partition_seed=args.partition_seed,
        num_clients=args.num_clients,
        num_tokens=args.num_tokens,
        epochs=args.epochs,
        output_dir=output_dir,
        use_gpu=args.device == "gpu",
    )
    basicts.launch_training(cfg, args.gpus if args.device == "gpu" else None, node_rank=0)


if __name__ == "__main__":
    main()
