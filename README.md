# HierSplit

Implementation of **Communication-Efficient Federated Traffic Prediction via
Hierarchical Split Learning**.

The repository provides the paper's core experimental setup:

| Backbones | Training paradigms | Partitions | Datasets |
|---|---|---|---|
| STAEformer, STGCN | independent learning (IL), split learning (SL), HierSplit (HS-A) | Random, METIS | METR-LA, PEMS-BAY, PEMS04, PEMS08 |

The HierSplit implementation corresponds to HS-A, with attention summarization,
server token self-attention, and attention-based local integration.

## BasicTS dependency

The experiments use [BasicTS](https://github.com/GestaltCogTeam/BasicTS) as the
data, metric, training, and evaluation framework. The reproducible base is:

- BasicTS internal version: `0.5.8`
- upstream commit: `63fa7192a94c7a59519c0a891f0ad9622f3da37c`
- EasyTorch: `1.3.3`

Project-specific models, partitioning code, scalers, and runners are provided
in this repository and integrate with the pinned BasicTS checkout.

```bash
git clone https://github.com/GestaltCogTeam/BasicTS.git third_party/BasicTS
git -C third_party/BasicTS checkout 63fa7192a94c7a59519c0a891f0ad9622f3da37c

conda env create -f environment.yml
conda activate hiersplit
```

The reference environment and experiment settings are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Data

Download BasicTS's preprocessed datasets and place them under the pinned
checkout:

```text
third_party/BasicTS/datasets/
├── METR-LA/{data.dat,desc.json,adj_mx.pkl,...}
├── PEMS-BAY/{data.dat,desc.json,adj_mx.pkl,...}
├── PEMS04/{data.dat,desc.json,adj_mx.pkl,...}
└── PEMS08/{data.dat,desc.json,adj_mx.pkl,...}
```

The download and file format are documented in BasicTS's
[`tutorial/dataset_design.md`](https://github.com/GestaltCogTeam/BasicTS/blob/63fa7192a94c7a59519c0a891f0ad9622f3da37c/tutorial/dataset_design.md).

## Run one experiment

Run commands from this repository's root. `--basicts-root` may be omitted when
BasicTS is at `third_party/BasicTS`.

```bash
python experiments/train.py \
  --method hiersplit \
  --backbone staeformer \
  --dataset METR-LA \
  --partition metis \
  --lr 0.001 \
  --seed 0 \
  --gpus 0
```

Valid methods are `independent`, `split`, and `hiersplit`; valid backbones are
`staeformer` and `stgcn`. CPU smoke checks can use `--device cpu`.

## Paper protocol

The main table uses 10 clients, 12 input and 12 output steps, 30 epochs, Adam
with weight decay `3e-4`, and batch size 32 except on PEMS08, where it is 64.
For every setting, sweep learning rates `0.001`, `0.002`, and `0.005`, select
the lowest validation-MAE checkpoint, and repeat seeds 0 through 4.

```bash
for lr in 0.001 0.002 0.005; do
  for seed in 0 1 2 3 4; do
    python experiments/train.py \
      --method hiersplit --backbone staeformer \
      --dataset METR-LA --partition random \
      --lr "$lr" --seed "$seed" --gpus 0
  done
done
```

The default token budgets are 4, 6, 6, and 3 per client for METR-LA,
PEMS-BAY, PEMS04, and PEMS08, respectively—approximately 20% of the nodes in a
balanced client partition. Override with `--num-tokens` for sensitivity runs.

## Verification

- `python scripts/smoke_test.py` runs a dependency-light CPU forward/backward check.
- `pytest` runs the model, protocol, and partition test suite.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
