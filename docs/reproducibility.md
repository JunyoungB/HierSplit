# Reproducibility

## Dependencies

- BasicTS commit: `63fa7192a94c7a59519c0a891f0ad9622f3da37c`
- BasicTS internal version: `0.5.8`
- EasyTorch: `1.3.3`
- Python: `3.10`
- PyTorch: `2.3.1`
- CUDA: `12.1`
- NumPy: `1.24.4`

The complete environment is defined in `environment.yml` and
`requirements.txt`.

## Protocol details

- 10 clients; Random or METIS graph partition.
- 12 input steps and 12 prediction steps.
- 30 epochs; Adam; weight decay `3e-4`.
- Learning-rate sweep: `0.001`, `0.002`, `0.005`.
- STAEformer uses MultiStepLR milestones 20 and 25 with gamma `0.1`.
- STGCN uses MultiStepLR milestones 1 and 50 with gamma `0.5`.
- Batch size 32, except PEMS08 uses 64.
- Lowest validation MAE checkpoint; five model seeds.
- IL uses client-local scalar normalization and per-client validation-best
  checkpoints. The best local checkpoint is reloaded after validation. SL and
  HierSplit use the standard global BasicTS scaler.
- Random partitions use the deterministic seed rule from the original runner.
  Model seed and partition seed are configured separately.
- METIS uses `metis.part_graph` with the `metis` Python package and `libmetis`.
- METIS uses raw adjacency for STAEformer and the normalized Laplacian for
  STGCN.
