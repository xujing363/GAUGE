import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402  must precede pandas import below

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import DrugNotFoundError, SampleResolutionError, predict_one  # noqa: E402

common.configure_page("Single Prediction")
bundle = common.sidebar_mode_selector()

st.title("🔬 Single Prediction")
st.write("Predict one sample's response to one drug. Pick options below, then click **Predict**.")

demo_col, _ = st.columns([1, 3])
if demo_col.button("▶️ Try a one-click demo (no data needed)"):
    st.session_state["sp_demo"] = True

known_label = common.known_sample_label(bundle)
st.subheader("1. Choose a sample")
sample_source = st.radio(
    "Sample source",
    [known_label, "Upload my own expression file"],
    horizontal=True,
    key="sp_sample_source",
)

resolved_sample_input = None
gene_coverage_note = None
if st.session_state.get("sp_demo"):
    st.success(f"Demo mode: using a {known_label.lower()} and a known library drug.")
    resolved_sample_input = bundle.cell_state_matrix.index[0]
elif sample_source == known_label:
    resolved_sample_input = common.cell_line_picker(bundle, "sp")
else:
    uploaded = st.file_uploader(
        "Expression file (CSV/TSV; samples-as-rows or genes-as-rows, auto-detected)",
        type=["csv", "tsv", "txt"],
        key="sp_upload",
    )
    common.expression_upload_help("sp")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")
            samples = {}
        if samples:
            sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="sp_sample_in_file")
            resolved_sample_input = samples[sample_name]
            n_present = int(pd.Series(resolved_sample_input.index).isin(bundle.artifacts.genes).sum())
            coverage = n_present / len(bundle.artifacts.genes)
            gene_coverage_note = (coverage, n_present, len(bundle.artifacts.genes))

st.subheader("2. Choose a drug")
drug_source = st.radio("Drug source", ["GAUGE library", "Custom SMILES"], horizontal=True, key="sp_drug_source")
if st.session_state.get("sp_demo"):
    drug_query = int(bundle.drug_library.iloc[0]["DRUG_ID"])
    st.caption(f"Demo drug: **{bundle.drug_library.iloc[0]['DRUG_NAME']}**")
elif drug_source == "GAUGE library":
    drug_query = common.drug_picker(bundle, "sp")
else:
    drug_query = st.text_input("SMILES string", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O (aspirin)", key="sp_smiles")

st.divider()
if st.button("🚀 Predict", type="primary", disabled=resolved_sample_input is None or not drug_query):
    try:
        result = predict_one(bundle, resolved_sample_input, drug_query)
    except (DrugNotFoundError, SampleResolutionError) as exc:
        st.error(str(exc))
    else:
        if gene_coverage_note:
            coverage, n_present, n_total = gene_coverage_note
            if coverage < 0.5:
                st.warning(
                    f"Only {n_present}/{n_total} ({coverage * 100:.0f}%) of GAUGE's gene panel was found in your "
                    "file. Predictions may be less reliable — missing genes are imputed with the training mean."
                )
            else:
                st.caption(f"Gene panel coverage: {n_present}/{n_total} ({coverage * 100:.0f}%).")

        st.subheader("Result")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(common.value_hat_gauge(result.value_hat), use_container_width=True)
        with c2:
            st.metric("Predicted absolute AUC (technical)", f"{result.auc_hat:.3f}")
            st.caption(
                "Absolute AUC is the model's raw dose-response-curve estimate and can fall slightly outside "
                "[0, 1]; the relative sensitive value (gauge above) is the bounded, cross-drug-comparable "
                "score emphasised in the paper — use it to compare across different compounds."
            )

        st.markdown(common.interpretation_sentence(result))

        kg_fig = common.kg_alpha_bar(result.kg_alpha)
        if kg_fig is not None:
            st.subheader("Knowledge-graph source attention")
            st.plotly_chart(kg_fig, use_container_width=True)
            st.caption(
                f"Gate strength (how much KG evidence was injected into the drug representation): "
                f"{result.gate_strength:.2f} (0 = ignored, 1 = fully relied upon)."
            )

        out_row = pd.DataFrame(
            [
                {
                    "sample": result.sample.label,
                    "drug": result.drug.name,
                    "absolute_auc": result.auc_hat,
                    "relative_sensitive_value": result.value_hat,
                    **({f"kg_alpha_{k}": v for k, v in result.kg_alpha.items()} if result.kg_alpha else {}),
                }
            ]
        )
        st.download_button(
            "⬇️ Download this result (CSV)",
            out_row.to_csv(index=False).encode(),
            file_name="gauge_single_prediction.csv",
            mime="text/csv",
        )
