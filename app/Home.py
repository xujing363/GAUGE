import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import APP_NAME, APP_TAGLINE, configure_page, sidebar_mode_selector  # noqa: E402

import streamlit as st  # noqa: E402

configure_page("Home")
bundle = sidebar_mode_selector()

st.title(f"🧬 {APP_NAME}")
st.subheader(APP_TAGLINE)

st.markdown(
    """
**GAUGE** predicts how a tumour will respond to a drug using only three things you can
obtain non-invasively or computationally: a gene-expression profile, the drug's chemical
structure, and three curated pharmacological knowledge graphs (ChEMBL, DRKG, PrimeKG).
You do **not** need any programming or machine-learning background to use this tool —
every analysis below is point-and-click.

GAUGE is the same single, frozen model used in the paper across five evaluation settings:
cell-line drug response, zero-shot TCGA patient stratification, single-cell drug-tolerant-
persister vulnerability scoring, drug-combination prioritisation, and molecular-design
guidance. This software exposes all of those capabilities, plus general pharmacogenomic
and transcriptome analysis tools, through the pages in the left sidebar. **Every page has
a one-click demo** — no data upload required to try anything.

Beyond GAUGE's own predictions, the app also connects to a dozen free public biomedical
databases (DGIdb, ChEMBL, OpenTargets, UniProt, Reactome, cBioPortal, ClinicalTrials.gov,
PubChem, Europe PMC) so you can study druggable targets, drug mechanisms, mutation
frequency, pathways, trials and literature — **with or without a GAUGE prediction**.

Four checkpoints are available from the sidebar: the standard **GDSC** library, a **GDSC
novel-compound** mode for chemically unseen drugs, the standard **PRISM** library
, and , a **PRISM novel-compound** mode for chemically unseen drugs.
"""
)

st.info(
    "**New here? Click a page in the left sidebar.** Start with **Single Prediction** — "
    "it has a one-click demo button that runs an example with no data upload required.",
    icon="👈",
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("### 🔬 Predict")
    st.markdown(
        "- **Single Prediction** — one sample × one drug\n"
        "- **Batch Prediction** — many samples × many drugs from a file\n"
        "- **Drug Ranking** — best drugs for one sample\n"
        "- **Molecular Design Scoring** — rank candidate molecules for a context"
    )
with col2:
    st.markdown("### 🧩 Combine & explain")
    st.markdown(
        "- **Combination Scoring** — two-drug synergy estimate, validated against real "
        "DrugComb measurements\n"
        "- **KG Explainability** — why GAUGE made a prediction\n"
        "- **KG Network Viewer** — the actual ChEMBL/DRKG/PrimeKG subgraph around a drug\n"
        "- **Patient Stratification** — real de-identified TCGA patient demos"
    )
with col3:
    st.markdown("### 📊 Explore data")
    st.markdown(
        "- **Pharmacogenomic Explorer** — real GDSC/PRISM dose-response data, "
        "gene-expression-vs-response biomarker scatter, drug-similarity clustering\n"
        "- **Expression Data Analysis** — PCA/UMAP, clustering, QC, volcano plots, "
        "tumour-vs-normal (GTEx) comparison, on any uploaded matrix\n"
        "- **Drug Sensitivity KB** — druggable targets, drug mechanism & phase, mutation "
        "frequency, pathways, trials & literature — **no GAUGE prediction required**\n"
        "- **About & Model Card** — performance, intended use, limitations, citation"
    )
with col4:
    st.markdown("### 🤖 Just ask")
    st.markdown(
        "- **GAUGE Assistant** — a chat interface (LLM tool-use agent) that calls the real "
        "GAUGE model for you and explains the result in plain language. Every number it "
        "gives you is a genuine model output, shown transparently — not made up by the LLM.\n"
        "- Reaches **external databases** (targets, mechanisms, mutations, trials, literature) "
        "for context and citations, and answers drug-sensitivity questions even without GAUGE.\n"
        "- **Deep Report** mode plans → gathers evidence → writes a cited report, with the "
        "whole process shown live; keep multiple **conversations** in the left sidebar."
    )

st.divider()
test_metrics = next((m for m in bundle.meta.get("reported_metrics_this_seed", []) if m.get("split") == "test"), None)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Known cell lines", bundle.meta.get("n_known_cell_lines"))
m2.metric("Known drugs", bundle.meta.get("n_known_drugs"))
if test_metrics:
    m3.metric("Held-out Pearson r", f"{test_metrics.get('overall_pcc'):.3f}")
    m4.metric("Held-out n pairs", int(test_metrics.get("n")))

st.caption(
    "GAUGE is a research tool intended for hypothesis generation and computational "
    "prioritisation. It does not replace clinical judgement or regulatory-approved "
    "diagnostics. See the **About & Model Card** page for full intended-use and "
    "limitations statements."
)
