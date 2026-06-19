# 01 Random Cell Split HVG2000

This fork keeps the existing `01_random_cell_split` benchmark contract, but adds an
explicit expression-input mode switch:

- `hvg`: select the top `n_hvg` genes and use them directly
- `pca`: use PCA on the full expression matrix
- `hvg_then_pca`: select the top `n_hvg` genes first, then apply PCA

The default mode for this fork is `hvg`, and the HVG count is controlled only by
`n_hvg` in YAML. If `n_hvg` exceeds the available gene count, the fork uses all
available genes as input.

Run entrypoints live in this folder only. Shared benchmark code is reused at runtime
through local monkeypatching, so the fork stays self-contained without editing the
parent benchmark tree.
