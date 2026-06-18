import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import DrugNotFoundError, predict_one  # noqa: E402

common.configure_page("Knowledge Graph Explainability")
bundle = common.sidebar_mode_selector()

st.title("🧠 Knowledge-Graph Explainability")
st.write(
    "See *why* GAUGE made a prediction: which of the three knowledge graphs "
    "(ChEMBL = mechanism of action, DRKG = gene/protein drug targets, "
    "PrimeKG = protein-disease associations) it relied on, and how strongly."
)

demo = st.button("▶️ Try a one-click demo", key="kg_demo")

known_label = common.known_sample_label(bundle)
st.subheader("1. Choose a sample")
sample_source = st.radio(
    "Sample source", [known_label, "Upload my own expression file"], horizontal=True, key="kg_sample_source"
)
sample_input = None
if demo:
    sample_input = bundle.cell_state_matrix.index[0]
    st.caption(f"Demo sample: known cell line **{sample_input}**.")
elif sample_source == known_label:
    sample_input = common.cell_line_picker(bundle, "kg")
else:
    uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="kg_upload")
    common.expression_upload_help("kg")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
            sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="kg_sample_in_file")
            sample_input = samples[sample_name]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

st.subheader("2. Choose a drug from the GAUGE library")
st.caption("Explainability requires a library drug — novel SMILES fall back to chemistry-only prediction with no KG attention.")
drug_id = int(bundle.drug_library.iloc[0]["DRUG_ID"]) if demo else common.drug_picker(bundle, "kg")
if demo:
    st.caption(f"Demo drug: **{bundle.drug_library.iloc[0]['DRUG_NAME']}**.")

if demo or st.button("🚀 Explain", type="primary", disabled=sample_input is None):
    try:
        result = predict_one(bundle, sample_input, drug_id)
    except DrugNotFoundError as exc:
        st.error(str(exc))
    else:
        if result.kg_alpha is None:
            st.warning("This drug has no knowledge-graph routing in the current bundle; only chemistry-only prediction is available.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.pie(
                    names=list(result.kg_alpha.keys()),
                    values=list(result.kg_alpha.values()),
                    color=list(result.kg_alpha.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="Source attention (α)",
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.metric("Gate strength", f"{result.gate_strength:.2f}", help="0 = KG evidence ignored, 1 = fully relied upon")
                st.metric("Predicted relative sensitive value", f"{result.value_hat * 100:.0f} / 100")
                st.write("**Knowledge-graph coverage for this drug:**")
                for source, has_it in result.drug.kg_coverage.items():
                    st.write(f"{'✅' if has_it else '⬜'} {source}")

            st.markdown(common.interpretation_sentence(result))
            top_source = max(result.kg_alpha, key=result.kg_alpha.get)
            descriptions = {
                "ChEMBL": "curated mechanism-of-action and bioactivity annotations",
                "DRKG": "gene- and protein-level drug-target interaction edges",
                "PrimeKG": "protein-disease and pathway-level association edges",
            }
            st.info(
                f"GAUGE leaned most heavily on **{top_source}** ({descriptions.get(top_source, '')}) "
                "for this tumour–drug pair. In the paper, attention rebalances toward a more uniform "
                "split across all three sources for chemically novel drugs — a sign the model knows "
                "when it is extrapolating."
            )
