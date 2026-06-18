# GAUGE User Guide

GAUGE is a point-and-click tool — every page in the left sidebar is a
self-contained workflow, and **every page has a one-click demo** so you can
see real output before bringing your own data. This guide explains what
each page does and how to read its output.

## Choosing a model mode (sidebar)

- **GDSC known-compound mode** (recommended default): the checkpoint
  evaluated under a held-out-*cell-line* split in the paper. Use this for
  any drug already in the GAUGE library (283 GDSC compounds) — it gives
  full knowledge-graph explainability.
- **GDSC novel-compound mode**: the checkpoint evaluated under a
  held-out-*drug* split. Use this when you specifically want the model
  variant trained to generalise to chemically unseen compounds.
- **PRISM mode**: trained on the Broad Institute Drug Repurposing Hub
  secondary screen — a much larger library (1,400+ approved/investigational
  compounds) on DepMap cell lines. Also a held-out-drug split. Use this when
  the drug you care about isn't in the GDSC library.

All three modes accept the same sample inputs and produce the same two
outputs described below.

## The two outputs, explained

Every prediction returns:

1. **Relative sensitive value** (`value_hat`, 0–100 on the gauge chart). A
   sigmoid-bounded score, comparable *across different drugs*. This is the
   headline number — higher means GAUGE expects this tumour to respond
   better to this drug relative to the typical cell line tested against it.
2. **Predicted absolute AUC** (`auc_hat`, technical). The model's raw
   dose-response-curve estimate. It can fall slightly outside the natural
   [0, 1] range (and, for PRISM, is on a different real-valued scale
   entirely) because it is an unconstrained regression output — this is
   expected model behaviour, not an error. Use `value_hat` for cross-drug
   and cross-dataset comparison.

## Pages

| Page | What it does | When to use it |
|---|---|---|
| **Single Prediction** | One sample × one drug | Quick check, one-click demo |
| **Batch Prediction** | Many samples × many drugs from an uploaded file | Screening a panel |
| **Drug Ranking** | Rank the whole library (or a subset) for one sample | "What should I try next?" |
| **Combination Scoring** | Pairwise combination heuristics from single-agent predictions, plus a tab that checks them against **real DrugComb** synergy measurements | Combination prioritisation, NOT a synergy assay replacement |
| **KG Explainability** | Source-level attention + gate strength for one prediction | Understanding *why* |
| **Patient Stratification** | Compare two treatment options (vh_diff) using real, de-identified **TCGA** patient demo samples, or your own data | Hypothesis generation only — see in-app caveats |
| **Pharmacogenomic Explorer** | Real GDSC/PRISM response distributions, a gene-expression-vs-drug-response biomarker scatter, and drug-similarity clustering — no GAUGE model involved | Sanity-checking a biomarker, understanding the underlying screen |
| **KG Network Viewer** | Renders the actual ChEMBL/DRKG/PrimeKG graph neighbourhood around a chosen drug | Seeing the literal prior knowledge GAUGE reasons over |
| **Molecular Design Scoring** | Score/rank candidate SMILES against a context sample; includes the paper's real EGFR/ERBB lung-adenocarcinoma design output | Generative-design reward scoring |
| **Expression Data Analysis** | PCA/UMAP, clustering heatmap, QC, two-group volcano plot, and a tumour-vs-normal-tissue (**GTEx**) comparison | General exploratory analysis, not GAUGE-specific |
| **GAUGE Assistant** | Chat with an LLM agent that calls the real GAUGE model for you (predict/rank/combine/search) and explains the result, with every tool call shown transparently | When you'd rather ask in plain language than click through forms |
| **About & Model Card** | Architecture, metrics, intended use, limitations, citation | Reference |

## The GAUGE Assistant

The Assistant page is an LLM (DeepSeek by default) wired up as a **tool-using
agent**, not a free-standing chatbot: it cannot answer "what's the predicted
response of X to Y" from its own knowledge — it must call the same
`predict_drug_response` / `rank_drugs_for_sample` / `score_drug_combination` /
`search_drugs` / `search_cell_lines` tools that the rest of this app uses,
and every tool call (with its exact arguments and result) is shown in an
expander under the answer so you can verify nothing was invented. Needs an
API key — see "Configuring the GAUGE Assistant" in the main `README.md`.
Without a key, every other page is unaffected.

## Providing your own data

Expression files (CSV or TSV) are auto-detected in either orientation:
- **samples as rows** (first column = sample name, remaining columns = genes), or
- **genes as rows** (first column = gene symbol, remaining columns = samples).

See `example_data/` for templates. GAUGE's gene panel is the top 2,000
most-variable genes from its training data; missing genes are imputed with
the training-set mean, and the app warns you if coverage is low (<50%).

To score a drug not in the library, switch to "Custom SMILES" and paste a
valid SMILES string — GAUGE falls back to chemistry-only reasoning (no
knowledge-graph evidence) for compounds it has never indexed.

## Where the demo data comes from

All demo data is real, not synthetic: a curated subset of de-identified TCGA
patient expression profiles (matched to drugs they actually received), real
DrugComb synergy measurements for pairs covered by a GAUGE library, GTEx v11
median tissue expression, and the paper's actual REINVENT4-generated
EGFR/ERBB design candidates. See `docs/DATA_SOURCES.md` for provenance and
`scripts/extract_demo_data.py` for how each file was produced.
