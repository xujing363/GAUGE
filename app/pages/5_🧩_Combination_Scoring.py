import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import score_combination  # noqa: E402

common.configure_page("Combination Scoring")
bundle = common.sidebar_mode_selector()

st.title("🧩 Combination Scoring")
st.warning(
    "GAUGE was never trained on combination-response labels. These scores combine two "
    "independent single-agent predictions (as in the NCI-ALMANAC analysis in the paper) and "
    "are a prioritisation heuristic, not a synergy measurement.",
    icon="⚠️",
)

tab_score, tab_validate = st.tabs(["Score a combination", "Validate against real synergy data (DrugComb)"])

with tab_validate:
    st.write(
        "These rows are real recorded synergy scores from **DrugComb** for pairs where both "
        "drugs are in a GAUGE library. Where the cell line also matches one of GAUGE's "
        "bundled GDSC cell lines, GAUGE's own combination score is computed live alongside "
        "the real measurement — a sanity check, not a guarantee of agreement (GAUGE has "
        "never seen combination labels)."
    )
    dc = pd.read_csv(common.EXAMPLE_DATA_DIR / "example_drugcomb_pairs.csv")
    dc = dc.rename(columns={"Cell line": "cell_line"})
    dc_matched = dc.dropna(subset=["matched_gdsc_cell_id"])
    n_show = st.slider("Number of real pairs to show", 5, 50, 15, key="cs_dc_n")
    sample_pairs = dc_matched.sample(n=min(n_show, len(dc_matched)), random_state=1)
    if st.button("🚀 Compute GAUGE scores for these real pairs", key="cs_dc_compute"):
        gdsc_bundle = bundle if bundle.mode == "gdsc_cell_split" else None
        if gdsc_bundle is None:
            from gauge_core.bundle import load_bundle as _load_bundle

            gdsc_bundle = _load_bundle("gdsc_cell_split")
        rows = []
        scoring_errors = []
        with st.spinner("Scoring..."):
            for r in sample_pairs.itertuples(index=False):
                try:
                    out = score_combination(gdsc_bundle, r.matched_gdsc_cell_id, r.Drug1, r.Drug2, mode="bliss")
                    rows.append(
                        {
                            "drug_a": r.Drug1, "drug_b": r.Drug2, "cell_line": r.cell_line,
                            "gauge_bliss": out["combination_score"], "real_bliss": r.Bliss, "real_zip": r.ZIP, "real_loewe": r.Loewe,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    scoring_errors.append(str(exc))
        if scoring_errors:
            st.caption(f"{len(scoring_errors)} pair(s) could not be scored (e.g. {scoring_errors[0]}).")
        if rows:
            cmp_df = pd.DataFrame(rows)
            st.dataframe(cmp_df, use_container_width=True)
            fig = px.scatter(
                cmp_df, x="real_bliss", y="gauge_bliss", hover_data=["drug_a", "drug_b", "cell_line"],
                title="GAUGE's heuristic combination score vs DrugComb's real recorded Bliss score",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "GAUGE's score is built from two independent single-agent predictions and is not "
                "expected to closely track wet-lab synergy scores point-for-point; this view is for "
                "transparency, not a claimed validation."
            )

with tab_score:
    known_label = common.known_sample_label(bundle)
    st.subheader("1. Choose a sample")
    sample_source = st.radio(
        "Sample source", [known_label, "Upload my own expression file"], horizontal=True, key="cs_sample_source"
    )
    sample_input = None
    if sample_source == known_label:
        sample_input = common.cell_line_picker(bundle, "cs")
    else:
        uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="cs_upload")
        common.expression_upload_help("cs")
        if uploaded is not None:
            try:
                samples = parse_expression_table(uploaded, bundle.artifacts.genes)
                sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="cs_sample_in_file")
                sample_input = samples[sample_name]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse this file: {exc}")

    st.subheader("2. Choose 2 or more drugs")
    demo = st.button("▶️ Try a 3-drug demo", key="cs_demo")
    lib = bundle.drug_library.sort_values("DRUG_NAME")
    demo_defaults = [n for n in ["Erlotinib", "Cisplatin", "Paclitaxel"] if n in lib["DRUG_NAME"].values][:3]
    chosen_names = st.multiselect(
        "Drugs to combine (pairwise)", lib["DRUG_NAME"].tolist(),
        default=demo_defaults if demo else [], key="cs_drug_select",
    )
    mode = st.selectbox(
        "Combination model",
        ["bliss", "activity_product", "complementarity"],
        format_func=lambda m: {
            "bliss": "Bliss independence (A + B − A·B)",
            "activity_product": "Activity product (A × B)",
            "complementarity": "Complementarity-weighted product",
        }[m],
        key="cs_mode",
    )

    if st.button("🚀 Score combinations", type="primary", disabled=sample_input is None or len(chosen_names) < 2):
        pairs = list(combinations(chosen_names, 2))
        rows = []
        with st.spinner(f"Scoring {len(pairs)} pairs..."):
            for name_a, name_b in pairs:
                id_a = int(lib.loc[lib["DRUG_NAME"] == name_a].iloc[0]["DRUG_ID"])
                id_b = int(lib.loc[lib["DRUG_NAME"] == name_b].iloc[0]["DRUG_ID"])
                out = score_combination(bundle, sample_input, id_a, id_b, mode=mode)
                rows.append(
                    {
                        "drug_a": name_a,
                        "drug_b": name_b,
                        "relative_sensitive_value_a": out["value_hat_a"],
                        "relative_sensitive_value_b": out["value_hat_b"],
                        "combination_score": out["combination_score"],
                    }
                )
        st.session_state["cs_results"] = pd.DataFrame(rows)

    if "cs_results" in st.session_state:
        results = st.session_state["cs_results"]
        st.subheader("Results")
        st.dataframe(results, use_container_width=True)
        if len(chosen_names) > 2:
            names = sorted(set(results["drug_a"]) | set(results["drug_b"]))
            matrix = pd.DataFrame(index=names, columns=names, dtype=float)
            for _, r in results.iterrows():
                matrix.loc[r["drug_a"], r["drug_b"]] = r["combination_score"]
                matrix.loc[r["drug_b"], r["drug_a"]] = r["combination_score"]
            fig = px.imshow(matrix, color_continuous_scale="Plasma", labels=dict(color="combination score"))
            st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "⬇️ Download results (CSV)",
            results.to_csv(index=False).encode(),
            file_name="gauge_combination_scores.csv",
            mime="text/csv",
        )
