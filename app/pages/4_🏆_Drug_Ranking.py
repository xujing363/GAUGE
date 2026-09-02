import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import rank_drugs  # noqa: E402

common.configure_page("Drug Ranking")
bundle = common.sidebar_mode_selector()

st.title("🏆 Drug Ranking")
st.write(
    "Given one sample, rank candidate drugs from most to least promising by their predicted "
    "**relative sensitive value** — the cross-drug-comparable score (`relative_sensitive_value`) GAUGE uses "
    "to order drugs in the paper. Higher means a better predicted response."
)

demo = st.button("▶️ Try a one-click demo (rank the entire library for one sample)", key="dr_demo")

known_label = common.known_sample_label(bundle)
st.subheader("1. Choose a sample")
sample_source = st.radio(
    "Sample source", [known_label, "Upload my own expression file"], horizontal=True, key="dr_sample_source"
)
sample_input = None
if demo:
    sample_input = bundle.cell_state_matrix.index[0]
    st.caption(f"Demo sample: known cell line **{sample_input}**.")
elif sample_source == known_label:
    sample_input = common.cell_line_picker(bundle, "dr")
else:
    uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="dr_upload")
    common.expression_upload_help("dr")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
            sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="dr_sample_in_file")
            sample_input = samples[sample_name]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

st.subheader("2. Choose the candidate drug pool")
pool_choice = st.radio("Candidate pool", ["Entire GAUGE library", "Choose specific drugs"], horizontal=True, key="dr_pool")
lib = bundle.drug_library.sort_values("DRUG_NAME")
if pool_choice == "Entire GAUGE library":
    candidate_ids = lib["DRUG_ID"].astype(int).tolist()
else:
    chosen_names = st.multiselect("Drugs", lib["DRUG_NAME"].tolist(), key="dr_pool_select")
    candidate_ids = lib.loc[lib["DRUG_NAME"].isin(chosen_names), "DRUG_ID"].astype(int).tolist()

top_k = st.slider("Show top N", 5, 50, 15, key="dr_topk")

if demo or st.button("🚀 Rank drugs", type="primary", disabled=sample_input is None or len(candidate_ids) == 0):
    with st.spinner(f"Scoring {len(candidate_ids)} drugs..."):
        # Ranking different drugs against each other is a CROSS-DRUG comparison,
        # so it is scored on the absolute predicted AUC (lower = more sensitive).
        # The relative sensitive value (RTV) is a within-drug percentile and is
        # deliberately not used to order the table -- it is kept as a reference
        # column only.
        ranked = rank_drugs(bundle, sample_input, candidate_drug_ids=candidate_ids, lambda_u=0.0)
    ranked = ranked.sort_values("auc_hat", ascending=True).reset_index(drop=True).rename(
        columns={"value_hat": "relative_sensitive_value_within_drug", "auc_hat": "absolute_auc"}
    )
    ranked.insert(0, "rank", ranked.index + 1)
    st.session_state["dr_ranked"] = ranked

if "dr_ranked" in st.session_state:
    ranked = st.session_state["dr_ranked"]
    top = ranked.head(top_k)
    st.subheader(f"Top {len(top)} drugs")
    fig = px.bar(
        top.sort_values("absolute_auc", ascending=False),
        x="absolute_auc",
        y="DRUG_NAME",
        orientation="h",
        color="absolute_auc",
        color_continuous_scale="Viridis_r",
        labels={"absolute_auc": "Predicted absolute AUC (lower = more sensitive)", "DRUG_NAME": "Drug"},
    )
    fig.update_layout(height=max(320, 28 * len(top)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ranked by **absolute predicted AUC** (lower = more sensitive), which is on a common "
        "scale for every compound. The relative sensitive value is shown for reference only: "
        "it is a within-drug percentile (rank of this sample inside *that drug's* own response "
        "distribution), so it cannot be used to compare one drug against another."
    )
    display_cols = ["rank", "DRUG_NAME", "DRUG_ID", "absolute_auc", "relative_sensitive_value_within_drug"]
    st.dataframe(
        top[[c for c in display_cols if c in top.columns]],
        use_container_width=True,
    )
    st.download_button(
        "⬇️ Download full ranking (CSV)",
        ranked[[c for c in display_cols if c in ranked.columns]].to_csv(index=False).encode(),
        file_name="gauge_drug_ranking.csv",
        mime="text/csv",
    )
