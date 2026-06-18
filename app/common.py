"""Shared utilities for every GAUGE Streamlit page: bundle loading/caching,
sidebar mode selector, plotting helpers, and branding constants.

Every page must `import common` (or from this module) before importing
pandas/numpy itself, because `gauge_core` needs to run its libstdc++ preload
before those libraries are loaded (see gauge_core/_drugwm_path.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SOFTWARE_ROOT = APP_DIR.parent
EXAMPLE_DATA_DIR = SOFTWARE_ROOT / "example_data"
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

import gauge_core  # noqa: E402  must precede streamlit's own pandas/pyarrow import where possible

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.bundle import ModelBundle, load_bundle  # noqa: E402

# Consistent look across every page's charts: a clean white template and a
# fixed, colour-blind-friendly qualitative palette used wherever a chart
# doesn't have a more specific reason to choose its own colours.
GAUGE_COLORWAY = ["#2E86AB", "#E8743B", "#19A979", "#945ECF", "#E15F99", "#499894", "#D37295", "#B07AA1"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = GAUGE_COLORWAY

APP_NAME = "GAUGE"
APP_TAGLINE = "State-adaptive knowledge-graph gating for cancer drug-response prediction"
MODE_LABELS = {
    "gdsc_cell_split": "GDSC — known compound library (recommended default)",
    "gdsc_drug_split": "GDSC — novel-compound mode (unseen chemistry)",
    "prism_cell_split": "PRISM — large repurposing library (1,487 compounds, recommended for PRISM)",
    "prism_drug_split": "PRISM — large repurposing library, novel-compound mode",
}
PRISM_MODES = {"prism_cell_split", "prism_drug_split"}
MODE_HELP = (
    "GAUGE ships four checkpoints from the published benchmarks, two per dataset. "
    "**Cell-line-split modes** (GDSC / PRISM recommended defaults) were evaluated under a "
    "held-out-cell-line split: they have seen most of that library's drugs during training "
    "and give the most reliable, knowledge-graph-explained predictions for drugs already in "
    "the library. **Novel-compound modes** were evaluated under a held-out-drug split: use "
    "them when scoring chemistry that is genuinely new to GAUGE (knowledge-graph attention "
    "is calibrated to redistribute across sources for unfamiliar compounds, as shown in the "
    "paper). **GDSC** is the original Genomics of Drug Sensitivity in Cancer screen. "
    "**PRISM** is the Broad Institute Repurposing Hub secondary screen — a much larger "
    "library of approved and investigational compounds, on DepMap cell lines."
)


def configure_page(title: str, icon: str = "🧬") -> None:
    st.set_page_config(page_title=f"{APP_NAME} — {title}", page_icon=icon, layout="wide")


@st.cache_resource(show_spinner="Loading GAUGE model bundle (first launch only takes a few seconds)...")
def _cached_bundle(mode: str) -> ModelBundle:
    return load_bundle(mode)


def sidebar_mode_selector() -> ModelBundle:
    st.sidebar.markdown(f"## {APP_NAME}")
    st.sidebar.caption(APP_TAGLINE)
    mode_key = st.sidebar.radio(
        "Model mode",
        options=list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
        key="gauge_mode",
        help=MODE_HELP,
    )
    bundle = _cached_bundle(mode_key)
    with st.sidebar.expander("About this checkpoint", expanded=False):
        st.write(f"**Split type:** {bundle.meta.get('split_type')}")
        st.write(f"**Known cell lines:** {bundle.meta.get('n_known_cell_lines')}")
        st.write(f"**Known drugs:** {bundle.meta.get('n_known_drugs')}")
        st.write(f"**Knowledge graphs:** {', '.join(bundle.meta.get('kg_sources', []))}")
        test_metrics = next(
            (m for m in bundle.meta.get("reported_metrics_this_seed", []) if m.get("split") == "test"), None
        )
        if test_metrics:
            st.write(f"**Held-out overall Pearson r:** {test_metrics.get('overall_pcc'):.3f}")
    st.sidebar.divider()
    st.sidebar.caption(
        "GAUGE predicts a tumour's response to a drug from gene expression + chemical "
        "structure + 3 pharmacological knowledge graphs. It is a research tool: "
        "predictions are decision support, not a clinical diagnosis."
    )
    return bundle


def known_sample_label(bundle: ModelBundle) -> str:
    """Radio-option label for 'pick one of the bundled known cell lines',
    worded correctly for whichever dataset the current mode is trained on."""
    if bundle.mode in PRISM_MODES:
        return "Known DepMap cell line (PRISM)"
    return "Known GDSC cell line"


def cell_line_picker(bundle: ModelBundle, key_prefix: str) -> str | None:
    """Sidebar-free picker widget for choosing a known cell line (GDSC or,
    in PRISM mode, DepMap)."""
    meta = bundle.cell_metadata
    if meta is None or meta.empty:
        options = bundle.cell_state_matrix.index.tolist()
        return st.selectbox("Known cell line", options, key=f"{key_prefix}_cellpick")
    meta = meta.copy()
    meta["display"] = meta["model_name"].astype(str) + "  —  " + meta["tissue"].astype(str) + " / " + meta["cancer_type"].astype(str)
    meta = meta.sort_values("display")
    choice = st.selectbox("Known cell line", meta["display"].tolist(), key=f"{key_prefix}_cellpick")
    row = meta.loc[meta["display"] == choice]
    if row.empty:
        return None
    return str(row.iloc[0]["SANGER_MODEL_ID"])


def expression_upload_help(key: str) -> None:
    """Downloadable example expression files next to every file_uploader, so
    users can see the exact expected format before preparing their own data."""
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Example file (samples as rows)",
            (EXAMPLE_DATA_DIR / "example_expression_multi_sample.csv").read_bytes(),
            file_name="gauge_example_expression_samples_as_rows.csv",
            mime="text/csv",
            key=f"{key}_dl_rows",
            help="First column = sample name, remaining columns = genes.",
        )
    with c2:
        st.download_button(
            "⬇️ Example file (genes as rows)",
            (EXAMPLE_DATA_DIR / "example_expression_genes_as_rows.csv").read_bytes(),
            file_name="gauge_example_expression_genes_as_rows.csv",
            mime="text/csv",
            key=f"{key}_dl_cols",
            help="First column = gene symbol, remaining columns = samples.",
        )


def drug_picker(bundle: ModelBundle, key_prefix: str) -> int:
    lib = bundle.drug_library.sort_values("DRUG_NAME")
    choice = st.selectbox("Drug (GAUGE library)", lib["DRUG_NAME"].tolist(), key=f"{key_prefix}_drugpick")
    return int(lib.loc[lib["DRUG_NAME"] == choice].iloc[0]["DRUG_ID"])


def kg_alpha_bar(kg_alpha: dict[str, float] | None) -> go.Figure | None:
    if not kg_alpha:
        return None
    fig = px.bar(
        x=list(kg_alpha.keys()),
        y=list(kg_alpha.values()),
        labels={"x": "Knowledge-graph source", "y": "Attention weight (α)"},
        text=[f"{v:.2f}" for v in kg_alpha.values()],
        color=list(kg_alpha.keys()),
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1], height=320, margin=dict(t=20, b=20))
    return fig


def value_hat_gauge(value_hat: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_hat * 100,
            number={"suffix": "%"},
            title={"text": "Predicted relative sensitive value"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2E86AB"},
                "steps": [
                    {"range": [0, 33], "color": "#fde0dd"},
                    {"range": [33, 66], "color": "#fcc5c0"},
                    {"range": [66, 100], "color": "#c6dbef"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def interpretation_sentence(result) -> str:
    band = "high" if result.value_hat >= 0.66 else ("moderate" if result.value_hat >= 0.33 else "low")
    sentence = (
        f"GAUGE predicts a **{band} relative sensitive value** "
        f"({result.value_hat * 100:.0f}/100 on the cross-drug-comparable scale) for "
        f"**{result.drug.name}** on this sample."
    )
    if not result.drug.known:
        sentence += (
            " This compound is not in the bundled training library, so the prediction falls back "
            "to chemistry-only reasoning (no knowledge-graph attention is available for it)."
        )
    elif result.kg_alpha:
        top_source = max(result.kg_alpha, key=result.kg_alpha.get)
        sentence += (
            f" The strongest knowledge-graph contribution came from **{top_source}** "
            f"(α = {result.kg_alpha[top_source]:.2f})."
        )
    if result.percentile_text:
        sentence += " " + result.percentile_text
    return sentence
