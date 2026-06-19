# 02 Drug Split HVG2000

This fork keeps the existing `02_drug_split` benchmark contract, but switches the
cell-line expression input to an explicit HVG mode:

- `hvg`: select the top `n_hvg` genes and use them directly
- `pca`: use PCA on the full expression matrix
- `hvg_then_pca`: select the top `n_hvg` genes first, then apply PCA

The default mode for this fork is `hvg` with `n_hvg = 2000`.

Run entrypoints live in this folder only. Shared benchmark code is reused at
runtime through local monkeypatching, so the fork stays self-contained without
editing the parent benchmark tree.

This fork reuses the processed data directory from:

`/mnt/raid5/xujing/KG/benchmarking/01_random_cell_split/01_random_cell_split_GDSC_v2_full/data`

In this folder, `data` should be a symlink to that location.
