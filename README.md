# GAUGE

**State-adaptive knowledge-graph gating for cancer drug-response prediction — packaged as a point-and-click application.**

GAUGE predicts how a tumour will respond to a drug from a gene-expression
profile, the drug's chemical structure, and three curated pharmacological
knowledge graphs (ChEMBL, DRKG, PrimeKG). This repository packages the
published model into a full-featured web application that students,
clinicians, and biologists can run **without writing any code**.

> Internal development codename: `drugwm`. The model, software, and all
> user-facing material are called **GAUGE**.

## Quickstart

```bash
conda env create -f environment.yml
conda activate gauge
./run_gauge.sh
```
Then open http://localhost:8501. See `docs/INSTALL.md` for Docker and
Windows instructions, and `docs/USER_GUIDE.md` for what each page does.

## What you get

Three checkpoints, selectable from the sidebar: **GDSC known-compound**
(recommended default), **GDSC novel-compound** (held-out-drug split), and
**PRISM** (Broad Repurposing Hub secondary screen, 1,400+ compounds). Every
page below has a one-click demo using real data.

- **Single & batch drug-response prediction** — pick a known cell line or
  upload your own expression profile; pick a library drug or paste a
  custom SMILES.
- **Drug ranking** — best candidates for a sample, library-wide.
- **Combination scoring** — Bliss / activity-product / complementarity
  heuristics from independent single-agent predictions, with a tab that
  checks them against **real DrugComb** synergy measurements.
- **Knowledge-graph explainability** — which of the 3 knowledge graphs
  (and how strongly) drove a prediction.
- **KG network viewer** — renders the actual ChEMBL/DRKG/PrimeKG graph
  neighbourhood around a chosen drug.
- **Patient-style stratification** — the paper's treatment-comparison
  (`vh_diff`) framing, demoed on real de-identified **TCGA** patients
  matched to the drug they actually received.
- **Pharmacogenomic explorer** — real GDSC/PRISM response distributions,
  gene-expression-vs-response biomarker scatter plots, and drug-similarity
  clustering — independent of the GAUGE model.
- **Molecular design** — a full loop: **generate** candidate molecules from a
  seed (offline BRICS fragment-recombination), **score** them with GAUGE for a
  context sample, and inspect **2-D structures, drug-likeness (QED/Lipinski/…),
  and Tanimoto similarity to the nearest library drug**; includes the paper's
  real REINVENT4-generated EGFR/ERBB lung-adenocarcinoma design output.
- **General expression-data analysis toolkit** — PCA/UMAP, hierarchical
  clustering heatmaps, QC, two-group volcano plots, and a tumour-vs-normal
  (**GTEx**) comparison — useful even without running the GAUGE model.
- **Drug Sensitivity Knowledge Base** — a point-and-click page that does **not**
  use the GAUGE model: profile a gene (druggable drugs via DGIdb, disease
  associations via OpenTargets, pathways via Reactome, pan-cancer mutation
  frequency via cBioPortal, protein function via UniProt), profile a drug
  (mechanism / clinical phase / indications via ChEMBL, chemistry via PubChem,
  trials), and search ClinicalTrials.gov / Europe PMC — all from free public APIs.
- **Model card** — architecture, reported metrics, intended use, and
  limitations, in-app.
- **GAUGE Assistant** — a *virtual-biologist* chat agent backed by an LLM
  (DeepSeek by default, or OpenAI) wired up as a **tool-using agent**. It plans,
  then calls the same `gauge_core` functions as the other pages for every GAUGE
  number it gives you (predict/rank/combine/**explain**/search) — shown
  transparently in the chat so it's never a black box. It can also reach **local
  knowledge-graph tools** (drug KG neighbourhood, prediction explainability) and,
  when you enable it, **external biomedical databases** (OpenTargets, PubChem,
  UniProt, ChEMBL, DGIdb, Reactome, cBioPortal, ClinicalTrials.gov, and the
  literature via Europe PMC) for target biology, drug pharmacology, mutation
  frequency, trials and citations — many of which answer drug-sensitivity
  questions without needing a GAUGE prediction. The chat keeps **multiple
  conversations in the left sidebar** (new / switch / rename / delete). Two modes:
  **⚡ Quick Analysis** (conversational) and **📋 Deep Report** (plans → gathers
  evidence across many tool calls → writes a structured, cited, downloadable report
  using a reasoning model); a **"Show process live"** toggle reveals the plan and
  each tool call as it runs. GAUGE response numbers always come from the model;
  external tools add context only, never a prediction. See "Configuring the GAUGE
  Assistant" below.

All five evaluation regimes from the paper (cell-line generalisation,
zero-shot TCGA-style stratification, single-cell vulnerability scoring,
combination prioritisation, and design guidance) are reachable from one
running, frozen model — no retraining, no command line.

## Repository layout

```
GAUGE/
├── app/                 Streamlit application (Home.py + pages/)
├── gauge_core/           Inference engine + Assistant (bundles, predictions, agent)
│   ├── agent.py          Tool-using LLM agent: Quick Analysis + Deep Report modes
│   ├── providers.py      Multi-provider LLM config (DeepSeek/OpenAI) with fallback
│   ├── kg_tools.py       Local KG-neighbourhood + prediction-explainability tools
│   ├── bio_tools.py      Opt-in external biomedical DB tools (OpenTargets/PubChem/UniProt/ChEMBL/DGIdb/Reactome/cBioPortal/ClinicalTrials.gov/Europe PMC)
│   └── vendor/drugwm/    Vendored minimal model code (no training code, no absolute paths)
├── models/                Self-contained exported model bundles (gdsc_cell_split, gdsc_drug_split, prism_secondary)
├── example_data/          Demo data: real TCGA patients, DrugComb pairs, GTEx tissues, design candidates, expression templates
├── kg_types.py            Pickle-compatibility class for the bundled knowledge-graph artifacts
├── scripts/               Maintenance-only: re-export bundles / re-extract demo data from source repositories
├── tests/                 pytest + headless Streamlit (AppTest) test suite
├── docs/                  INSTALL / USER_GUIDE / FAQ / DATA_SOURCES
├── .env.example           Template for the GAUGE Assistant's API key (copy to .env)
├── environment.yml, requirements.txt, pyproject.toml, Dockerfile
└── run_gauge.sh / run_gauge.bat   one-command launchers
```

## Configuring the GAUGE Assistant

The Assistant page needs an OpenAI-API-compatible LLM key (defaults to
DeepSeek):

```bash
cp .env.example .env
# edit .env and set DEEPSEEK_API_KEY=sk-...   (and optionally OPENAI_API_KEY=...)
```

`gauge_core` reads `.env` automatically (never overriding a real environment
variable). `.env` is gitignored and is never baked into the Docker image — pass
it at container run time instead: `docker run --env-file .env ...`. Without a key
configured, every other page still works normally; only the Assistant page is
affected, and it lets a user paste a key into the sidebar for just their session.

**Providers** are defined in `gauge_core/providers.py` (DeepSeek chat, DeepSeek
reasoner, OpenAI) and chosen from the sidebar; the chat falls back to another
configured provider if the primary one errors. Set defaults with
`GAUGE_ASSISTANT_PROVIDER` (chat) and `GAUGE_REPORT_PROVIDER` (the reasoning model
used for Deep Report) in `.env`.

**External biomedical databases** (OpenTargets / PubChem / UniProt / ChEMBL /
DGIdb / Reactome / cBioPortal / ClinicalTrials.gov / Europe PMC) are *opt-in* via
the sidebar toggle and require internet access. They are implemented in
`gauge_core/bio_tools.py` over free, key-less public APIs, and degrade gracefully
(a clear "lookup unavailable" message) when offline — GAUGE predictions are never
affected. When disabled, the app is fully self-contained. The same databases also
power the **Drug Sensitivity KB** page, which works entirely without the GAUGE
model (it does not need an LLM key either).

## For developers: using `gauge_core` programmatically

```python
from gauge_core import load_bundle, predict_one, rank_drugs

bundle = load_bundle("gdsc_cell_split")
result = predict_one(bundle, "SIDM00003", "Camptothecin")
print(result.value_hat, result.auc_hat, result.kg_alpha)
```

`gauge_core` has no dependency on the original training repository or any
absolute path — it is fully self-contained (see `docs/DATA_SOURCES.md` for
provenance and `scripts/export_model_bundle.py` for how bundles were made).

## Testing

```bash
pytest tests/ -q
```
The suite covers the inference engine (`gauge_core`), the LLM agent's tools
(local KG/explainability and external biomedical lookups, the latter mocked so
they run offline), the multi-provider factory and its fallback, plus live
round-trips for both Quick Analysis and Deep Report (skipped automatically if no
API key is configured), and every application page's golden path plus edge cases
(invalid SMILES, unknown cell lines, low gene-panel coverage), using Streamlit's
official headless `AppTest` framework against the real, bundled models.

## Citation, license

See `CITATION.cff` and `LICENSE`.

## Limitations

GAUGE is a research tool for computational prioritisation and hypothesis
generation. It is not a clinical diagnostic and has not been prospectively
validated. Full limitations are documented on the in-app "About & Model
Card" page.
