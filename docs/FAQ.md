# Frequently Asked Questions

**Do I need to know Python or machine learning to use this?**
No. Every feature is available through the point-and-click web interface.
Programming is only needed if you want to call `gauge_core` directly from
your own analysis scripts (see `gauge_core/__init__.py` for the public API).

**Is this a clinical diagnostic tool?**
No. GAUGE is a research tool for computational hypothesis generation. See
the limitations and intended-use statements on the "About & Model Card" page.

**Why is the predicted AUC sometimes outside the 0–1 range?**
`auc_hat` is an unconstrained regression output and can exceed the natural
[0, 1] dose-response-curve range, especially after the residual-fusion step
used by the published checkpoints, and on PRISM, where compounds a cell line
outgrows genuinely score AUC > 1. This is expected and does not stop it from
being the right quantity for cross-drug comparison — AUC is on a common
scale for every compound.

**Which score should I use to compare different drugs?**
The **absolute predicted AUC** (`auc_hat`), lower = more sensitive. The
"relative sensitive value" (RTV, `value_hat`) is bounded in [0, 1] but it is
a *within-drug percentile* — `RTV = 1 - rank_d(AUC) / N_d`, computed inside
each drug's own frozen training reference distribution. Because every drug is
normalised against itself, RTV deliberately discards between-drug potency:
a weak drug's best-responding cell line and a potent drug's best-responding
cell line both score RTV ≈ 1. Use RTV to ask "does this sample respond
unusually well to *this* drug", and absolute AUC to ask "which drug should I
pick". Every ranking surface in the app (Drug Ranking, best-drug-per-sample,
Molecular Design, Patient Stratification, Combination Scoring) is scored on
the absolute AUC for this reason.

How far out of range it goes depends on the bundle's residual-fusion weight.
Scoring the whole library for one cell line gives, for the shipped bundles:

| bundle | fusion weight | `auc_hat` range | outside [0, 1.5] |
|---|---|---|---|
| `gdsc_cell_split` | 2.0 | −1.55 … 1.13 | 5.3 % |
| `gdsc_drug_split` | 0.0 | 0.18 … 0.98 | 0 % |
| `prism_cell_split` | 1.0 | −0.85 … 3.44 | 22.8 % |
| `prism_drug_split` | 0.0 | 0.20 … 1.95 | 2.0 % |

The ordering is still the right one to use for a cross-drug question — it is
the only read-out that carries between-drug potency — but on the two
residual-fused bundles (`gdsc_cell_split`, `prism_cell_split`) the absolute
head is visibly uncalibrated at the tails, so treat those cross-drug rankings
as coarse prioritisation rather than as calibrated AUC estimates.

**Can I add my own drug that isn't in the library?**
Yes — switch to "Custom SMILES" wherever a drug picker appears. The
prediction will use chemistry-only reasoning (no knowledge-graph attention),
since GAUGE's knowledge graphs are pre-indexed to a fixed compound set.

**Can I add my own knowledge graph or retrain the model?**
Not from this app — it loads frozen, published checkpoints. Retraining
requires the original training repository and is outside the scope of this
software package.

**My organisation only allows offline software — does this need internet access?**
No. Once installed (Docker image built, or conda environment created), the
app runs fully offline; all model weights and reference data are bundled
locally under `models/`.

**Where do I report a bug or ask for a new feature?**
Open an issue on the project's GitHub repository (see `CITATION.cff`).
