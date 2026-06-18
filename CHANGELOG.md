# Changelog

## 1.4.0 — 2026-06-18
GAUGE Assistant expansion + a GAUGE-free drug-sensitivity knowledge base.

- **More external databases** (`gauge_core/bio_tools.py`), all free and key-less,
  focused on drug sensitivity: **DGIdb** (drug–gene interactions / druggable
  targets), **ChEMBL** (mechanism of action, max clinical phase, indications),
  **ClinicalTrials.gov** (trials by condition/intervention), **Reactome**
  (pathways for a target), and **cBioPortal** (pan-cancer mutation frequency,
  MSK-IMPACT 2017). All are exposed as Assistant tools and degrade gracefully
  offline. The Assistant can now answer pharmacology / drug-sensitivity questions
  **without** calling GAUGE when no prediction is needed.
- **Multi-conversation chat**: the Assistant now keeps multiple conversations in
  the left sidebar (ChatGPT-style) — New chat, switch, rename, delete — instead of
  a single transcript. Save/Load operates per conversation.
- **Live Deep Report process**: the plan → gather evidence → write → self-review
  pipeline is now shown live via a `progress` callback on `run_report`/`run_turn`
  and an `st.status` panel, with a sidebar **"Show process live"** toggle so users
  can reveal or hide the full research process (and per-tool evidence) as it runs.
- **New page — Drug Sensitivity KB** (`9_💊_Drug_Sensitivity_KB.py`): a
  point-and-click knowledge base that does **not** depend on the GAUGE model —
  profile a gene (druggable drugs, disease associations, pathways, mutation
  frequency, protein function), profile a drug (mechanism, phase, chemistry,
  trials), and search trials/literature directly from public databases.
- **Home** refreshed to describe the external-database integrations, the new
  page, and the upgraded Assistant.
- Test suite grown with network-mocked tests for every new external tool and the
  new page/Assistant wiring.

## 1.3.1 — 2026-06-17
Page improvements and fixes.

- **Batch Prediction**: can now score **known cell lines without uploading**
  (searchable multiselect over the bundle's cell lines) in addition to file
  upload / demo. Results gained richer visualisation — a tabbed view with a
  sample×drug heatmap (with a bar-chart fallback for 1×N selections), a
  best-drug-per-sample summary, and a per-drug cohort comparison (ranking bar +
  distribution box plot).
- **Molecular Design**: now a full design loop, not just scoring — **generate**
  novel candidates from a seed via offline BRICS fragment-recombination
  (`gauge_core/moldesign.py`), then **score + inspect** them with 2-D structure
  depictions (RDKit), drug-likeness/physicochemical descriptors (MW, logP, QED,
  TPSA, Lipinski), and Tanimoto similarity to the nearest GAUGE-library drug.
- Fix: the Molecular Design demo button (and now the generator) populate the
  candidate box via session_state, fixing the keyed-`text_area` `value=` no-op
  that made the demo button appear to do nothing.
- Removed the model-uncertainty value from all user-facing output (Assistant
  tools, Single/Batch prediction, Molecular Design, and the Drug Ranking page,
  which now ranks purely by relative sensitive value `value_hat` as in the
  paper) — the value is not described in the paper.
- "About & Model Card" moved to the end of the page list.
- Test suite grown to 66 cases (new molecular-design and batch/known-cell tests).

## 1.3.0 — 2026-06-17
GAUGE Assistant upgraded toward a "virtual disease biologist" (OriGene-inspired).

- **Deep Report mode**: a new `GaugeAgent.run_report` deep-research loop that
  plans, gathers evidence across many tool calls, synthesises a structured,
  cited Markdown report (Summary / Target & disease biology / GAUGE predictions /
  External evidence / Caveats / References) with a reasoning model, then
  self-critiques and revises. Downloadable from the UI. The existing
  conversational answer is now "⚡ Quick Analysis".
- **External biomedical-database tools** (opt-in, `gauge_core/bio_tools.py`):
  OpenTargets target–disease associations, PubChem compound identity, UniProt
  protein function/disease, and Europe PMC literature for citations. Free, keyless
  public APIs over `httpx`, cached, timeout-guarded, and gracefully degrading when
  offline. Toggled per session in the sidebar; off by default (app stays
  self-contained).
- **Local KG/explainability tools** (`gauge_core/kg_tools.py`): `kg_neighborhood`
  (drug's ChEMBL/DRKG/PrimeKG subgraph) and `explain_prediction` (KG
  source-attention + percentile). The KG Network Viewer page now shares
  `build_drug_subgraph` with the agent (single source of truth).
- **Multi-provider LLM support with fallback** (`gauge_core/providers.py`):
  DeepSeek chat, DeepSeek reasoner, and OpenAI, selectable in the sidebar;
  reasoning models drop unsupported params automatically. New env defaults
  `GAUGE_ASSISTANT_PROVIDER` / `GAUGE_REPORT_PROVIDER`; `OPENAI_API_KEY` added.
- Stronger agent: planning-aware system prompt, larger tool-round budget, optional
  self-reflection pass, and dynamic tool registration (external tools hidden until
  enabled). The original 6 GAUGE tools and `gauge_core.agent` public API are
  unchanged and backward-compatible.
- Test suite grown from 46 to 60 cases (new local-tool, provider-fallback, and
  mocked external-tool tests run offline; a Deep Report live round-trip skips
  without a key).
- UI: the Model / provider / mode controls now render **above** the GAUGE
  checkpoint selector, and conversations can be **saved/loaded** (Markdown + JSON
  download, JSON upload to restore).
- Fix: reasoning models (e.g. `deepseek-reasoner`) are no longer offered as the
  chat model — they cannot do tool calling, which previously caused a "model
  error" when running Deep Report with external databases. They are now used only
  for the report's tool-less synthesis step, with automatic fallback to the chat
  model if the account lacks reasoner access. Added request timeouts and
  persistent, mode-aware error messages.

## 1.2.0 — 2026-06-17
LLM agent integration.

- New **GAUGE Assistant** page: a chat UI backed by an OpenAI-API-compatible
  LLM (defaults to DeepSeek `deepseek-chat`) wired up as a tool-using agent
  (`gauge_core/agent.py`). The LLM never invents a numeric prediction —
  every number comes from a real call into `gauge_core.predict` via 5 tools
  (`predict_drug_response`, `rank_drugs_for_sample`, `score_drug_combination`,
  `search_drugs`, `search_cell_lines`), shown transparently in the UI.
  `temperature=0` for reliable, grounded tool-calling behaviour.
- Added `gauge_core.predict.search_drugs`/`search_cell_lines` (forgiving
  substring search) to support natural-language-driven, fuzzy entity
  resolution, reusable outside the agent too.
- New `.env`-based key configuration (`gauge_core/_env.py`, `.env.example`,
  gitignored `.env`); the Assistant page also accepts a session-only key
  typed into the sidebar so the feature works without `.env` for other
  deployments.
- Test suite grown from 36 to 46 cases; live LLM round-trip tests skip
  automatically when no API key is configured.

## 1.1.0 — 2026-06-17
Multi-dataset, multi-feature expansion.

- Added a third model bundle, `models/prism_secondary` (Broad Repurposing
  Hub PRISM secondary screen, DepMap cell lines, ~1,400 compounds). Renamed
  `cell_split`/`drug_split` to `gdsc_cell_split`/`gdsc_drug_split` for
  consistency.
- Added a `response_table.csv.gz` to every bundle (real per-pair AUC/split)
  powering the new Pharmacogenomic Explorer page.
- New pages: **Pharmacogenomic Explorer** (real response distributions,
  gene-expression-vs-response biomarker scatter, drug-similarity
  clustering), **KG Network Viewer** (renders the actual ChEMBL/DRKG/PrimeKG
  subgraph around a drug), **Molecular Design Scoring** (rank candidate
  SMILES; includes the paper's real EGFR/ERBB design output).
- Real demo data added throughout: de-identified TCGA patients (matched to
  the drug they actually received) on Patient Stratification; real DrugComb
  synergy measurements on Combination Scoring; GTEx v11 tumour-vs-normal
  comparison on Expression Data Analysis. Every page now has an explicit
  one-click demo.
- `kg_types.py` (top-level, decoupled from `gauge_core`'s vendored `drugwm`
  namespace) replaces the original `drugwm.kg_prior.MultiKGGraphArtifacts`
  in pickled bundles, carrying `node_table`/`edge_table` needed for the
  network viewer.
- Global Plotly theme (`plotly_white` + fixed colourway) applied app-wide
  for visual consistency.
- Test suite grown from 24 to 33 cases.

## 1.0.0 — 2026-06-17
Initial release.

- `gauge_core` inference engine wrapping the published GAUGE model
  (single/batch prediction, drug ranking, combination scoring, source-level
  KG explainability), fully decoupled from the original training repository
  (vendored minimal model code, no absolute paths, no sqlite3 dependency).
- Self-contained model bundles exported from the published benchmark
  checkpoints: `models/cell_split` (held-out-cell-line split, recommended
  default) and `models/drug_split` (held-out-drug split).
- 8-page Streamlit application: Single Prediction, Batch Prediction, Drug
  Ranking, Combination Scoring, KG Explainability, Patient Stratification,
  Expression Data Analysis (general-purpose, model-independent), About &
  Model Card.
- Docker, conda, and pip packaging; one-command launchers
  (`run_gauge.sh` / `run_gauge.bat`).
- 24-test suite (`pytest tests/`) covering the inference engine and every
  app page's golden path + edge cases via Streamlit's headless `AppTest`.
