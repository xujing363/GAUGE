# strategy_quantile_map_hvg2000_task01

This folder contains a task01-only quantile-mapping alignment experiment.

Strategy:

- use the fixed HVG2000 source run artifacts and model
- for each HVG gene, map TCGA values onto the empirical distribution of the
  GDSC fit-cell population
- keep the downstream frozen `imputer/scaler/pca/model` unchanged
- export task01-style comparison outputs locally
