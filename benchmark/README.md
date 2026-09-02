# GAUGE benchmark

Reference implementation, benchmark data and trained checkpoints for GAUGE —
state-adaptive knowledge-graph gating for cancer drug-response prediction.

Everything here is self-contained: `git clone`, install the requirements, and
you can retrain every model in the paper or score the shipped checkpoints
without any additional download.

## Install

```bash
pip install -r requirements.txt          # torch, numpy, pandas, pyarrow
```

A GPU is recommended (a full 50-epoch run takes ~17 min on one A100 and
0.6 GB of GPU memory); `--device cpu` works but is much slower.

## Reproduce

```bash
# train one split / seed  (writes model.pt, predictions.csv, gdsc_metrics.csv, run_meta.json)
python -m gauge_bench.train --split drug_split --seed 5 --out-dir runs/drug_split_seed5

# all 3 splits x 5 seeds, spread over 4 GPUs
bash scripts/train_all.sh

# score a shipped checkpoint instead of retraining
python -m gauge_bench.evaluate --split drug_split --seed 5 \
    --checkpoint checkpoints/drug_split/seed5 --out-dir runs/eval_drug_seed5
```

## Reproducibility

Training is **not** bit-reproducible on a GPU: the relation-aware graph
attention aggregates messages with `index_add_`, whose atomic float additions
are order-dependent, so two runs of the same command with the same seed differ
slightly and the gap compounds over epochs. Retraining therefore reproduces the
reported numbers statistically, not exactly.

The published numbers correspond to the weights in `checkpoints/`. To reproduce
them exactly, score those checkpoints with `gauge_bench.evaluate` rather than
retraining — the released data pipeline reproduces the shipped predictions to
4.2e-7 (float32 round-trip precision) across all 15 runs.

Retraining lands close but not on top: a fresh 50-epoch `drug_split` seed 5 run
reaches test within-drug PCC 0.358 (shipped checkpoint: 0.351) and overall PCC
0.263 (shipped: 0.296) — within the 0.051 seed-to-seed s.d. of that split.

## Model

Three-stage architecture (`gauge_bench/model.py`):

1. **Encoders** — tumour state `x_s` (2,000 HVG) → `z_s` (128); drug Morgan
   fingerprint (r=2, 2,048 bit) → `z_chem` (128); per-source KG node states
   propagated by two relation-aware graph-attention layers.
2. **State-adaptive KG gating** — a context-conditioned softmax attention over
   the three sources (ChEMBL / DRKG / PrimeKG), conditioned on tumour state,
   drug chemistry, source coverage and node degree, gives `z_kg`; a sigmoid
   injection gate `g` fuses it as `z_a = z_chem + g · z_kg`.
3. **Interaction and heads** — `b = T([z_s, z_a, z_s ⊙ z_a])` feeds an
   absolute-AUC head, a relative-sensitive-value head and a tumour-residual
   head. The residual is fused into the AUC prediction with a weight selected
   on the validation split.

3.73 M parameters. Training uses AdamW (lr 6e-5, weight decay 1e-4), 50 epochs,
batch 512, edge dropout 0.1, chemical-pathway dropout 0.6, and a hybrid sampler
that mixes random pairs with grouped (32 cell lines × 16 drugs) ranking batches.
Checkpoint selection is on validation overall Pearson r.

## Data

`data/` holds the GDSC2 benchmark in a plain, pickle-free format.

| path | contents |
|---|---|
| `data/shared/state_matrix.npz` | 944 cell lines × 2,000 HVG tumour-state matrix |
| `data/shared/kg_nodes.parquet`, `kg_edges.parquet` | 12,541 nodes / 86,376 edges over ChEMBL, DRKG, PrimeKG |
| `data/shared/kg_coverage.parquet` | per-drug source coverage, node id and graph degree |
| `data/shared/drug_bank.npz` | Morgan fingerprints, in KG bank order |
| `data/shared/drugs.csv` | drug ids, names, canonical SMILES, InChIKey |
| `data/<split>/responses/seed<N>.parquet` | 194,058 cell-line × drug AUC rows with the split labels and training targets |

The three held-out-drug partitions differ only in how compounds are assigned:

| split | assignment unit |
|---|---|
| `drug_split` | canonical drug entity (no-leakage name group) |
| `scaffold_split` | Bemis–Murcko scaffold |
| `chemcluster_split` | Butina chemical cluster |

All three use the same 236 structure-canonicalised compounds and the same
165 train / 24 validation / 47 test drug budget, over five split seeds
(5, 7, 43, 91, 98). Structure canonicalisation (RDKit isomeric canonical SMILES
+ InChIKey, exact-duplicate collapse, removal of compounds with no public
structure) is applied *before* the partitions are drawn, so no drug entity
crosses a split.

## Checkpoints and results

`checkpoints/<split>/seed<N>/` — trained weights, per-epoch history, metrics
and the run configuration for each of the 15 runs.

Held-out test performance, mean ± s.d. over the five seeds:

| split | overall PCC | within-drug PCC | within-drug Spearman | within-cell PCC | RMSE |
|---|---|---|---|---|---|
| `drug_split` | 0.328 ± 0.051 | 0.327 ± 0.024 | 0.307 ± 0.028 | 0.241 ± 0.065 | 0.155 ± 0.025 |
| `scaffold_split` | 0.298 ± 0.079 | 0.304 ± 0.032 | 0.280 ± 0.035 | 0.210 ± 0.087 | 0.150 ± 0.016 |
| `chemcluster_split` | 0.248 ± 0.071 | 0.307 ± 0.038 | 0.290 ± 0.049 | 0.163 ± 0.057 | 0.153 ± 0.026 |

Per-seed values are in `results/benchmark_per_seed.csv`; the aggregate table is
`results/benchmark_summary.csv`.

`gdsc_metrics.csv` additionally reports the response decomposition — the
between-drug, between-cell and interaction components of the prediction.

## Layout

```
benchmark/
├── gauge_bench/
│   ├── model.py        architecture, targets and training objectives
│   ├── data.py         dataset loader
│   ├── train.py        training / evaluation entry point
│   ├── evaluate.py     score a trained checkpoint
│   └── metrics.py      regression, value and decomposition metrics
├── scripts/
│   ├── train_all.sh    3 splits x 5 seeds across 4 GPUs
│   ├── summarize.py    rebuild results/ from run directories
│   └── export_dataset.py  regenerate data/ from the internal preparation objects
├── data/               benchmark dataset (see above)
├── checkpoints/        15 trained models
└── results/            per-seed and aggregate metric tables
```

## Data sources

GDSC2 drug-response and cell-line expression (Sanger), ChEMBL, DRKG and
PrimeKG. See `../docs/DATA_SOURCES.md` for versions and licence terms.
