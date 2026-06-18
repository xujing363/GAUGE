# Frequently Asked Questions

**Do I need to know Python or machine learning to use this?**
No. Every feature is available through the point-and-click web interface.
Programming is only needed if you want to call `gauge_core` directly from
your own analysis scripts (see `gauge_core/__init__.py` for the public API).

**Is this a clinical diagnostic tool?**
No. GAUGE is a research tool for computational hypothesis generation. See
the limitations and intended-use statements on the "About & Model Card" page.

**Why is the predicted AUC sometimes outside the 0–1 range?**
`auc_hat` is an unconstrained linear regression output and can slightly
exceed the natural [0, 1] dose-response-curve range, especially after the
residual-fusion step used by the published checkpoints. The bounded
"relative sensitive value" score is the one intended for cross-drug
comparison and is always in [0, 1].

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
