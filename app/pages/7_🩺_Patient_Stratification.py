import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import DrugNotFoundError, predict_one  # noqa: E402

common.configure_page("Patient Stratification")
bundle = common.sidebar_mode_selector()

st.title("🩺 Patient-style Stratification & Treatment Comparison")
st.write(
    "GAUGE was trained only on cell-line data but, in the paper, was applied **zero-shot** "
    "(no retraining) to TCGA patient tumour transcriptomes for response discrimination and "
    "for comparing clinically interchangeable drug regimens."
)
st.error(
    "**Research use only.** This page reproduces the *prediction framing* used in the paper's "
    "TCGA analysis. It is not a clinical decision tool, has not been prospectively validated, "
    "and does not account for dosing, comorbidities, or other clinical covariates that a "
    "treating physician must consider.",
    icon="🚫",
)

st.subheader("1. Patient / tumour sample")
sample_source = st.radio(
    "Sample source",
    ["Real de-identified TCGA patient (demo)", common.known_sample_label(bundle), "Upload a tumour expression profile"],
    horizontal=True, key="tcga_sample_source",
)
sample_input = None
actual_drug_received = None
if sample_source.startswith("Real"):
    tcga_expr = pd.read_csv(common.EXAMPLE_DATA_DIR / "example_tcga_patients.csv", index_col=0)
    tcga_meta = pd.read_csv(common.EXAMPLE_DATA_DIR / "example_tcga_patients_meta.csv", index_col=0)
    tcga_meta["display"] = (
        tcga_meta.index.astype(str) + "  —  " + tcga_meta["project_id"].astype(str) + "  —  received: " + tcga_meta["drug"].astype(str)
    )
    choice = st.selectbox("TCGA patient sample", tcga_meta["display"].tolist(), key="tcga_patient_pick")
    sample_id = choice.split("  —  ")[0]
    sample_input = tcga_expr.loc[sample_id]
    actual_drug_received = str(tcga_meta.loc[sample_id, "drug"])
    st.caption(
        f"Cancer type: **{tcga_meta.loc[sample_id, 'project_id']}** "
        f"({tcga_meta.loc[sample_id, 'primary_site']}). Drug actually received: **{actual_drug_received}**. "
        f"Vital status on record: {tcga_meta.loc[sample_id, 'vital_status']}."
    )
elif sample_source.startswith("Known"):
    sample_input = common.cell_line_picker(bundle, "tcga")
else:
    uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="tcga_upload")
    common.expression_upload_help("tcga")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
            sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="tcga_sample_in_file")
            sample_input = samples[sample_name]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

st.subheader("2. Compare two treatment options")
st.caption("e.g. the paper's FOLFIRI (irinotecan) vs FOLFOX (oxaliplatin) comparison in colorectal cancer.")
lib_names = bundle.drug_library.sort_values("DRUG_NAME")["DRUG_NAME"].tolist()
col_a, col_b = st.columns(2)
with col_a:
    default_a = 0
    if actual_drug_received and actual_drug_received.title() in lib_names:
        default_a = lib_names.index(actual_drug_received.title())
    drug_a = st.selectbox(
        "Drug A" + (" (actually received)" if actual_drug_received else ""), lib_names, index=default_a, key="tcga_a_drugpick"
    )
with col_b:
    drug_b = st.selectbox("Drug B (alternative)", lib_names, index=min(1, len(lib_names) - 1), key="tcga_b_drugpick")

if st.button("🚀 Compare", type="primary", disabled=sample_input is None):
    try:
        result_a = predict_one(bundle, sample_input, drug_a)
        result_b = predict_one(bundle, sample_input, drug_b)
    except DrugNotFoundError as exc:
        st.error(str(exc))
    else:
        vh_diff = result_a.value_hat - result_b.value_hat
        preferred = result_a.drug.name if vh_diff > 0 else result_b.drug.name

        c1, c2, c3 = st.columns(3)
        c1.metric(result_a.drug.name, f"{result_a.value_hat * 100:.0f} / 100")
        c2.metric(result_b.drug.name, f"{result_b.value_hat * 100:.0f} / 100")
        c3.metric("Model-preferred option", preferred, delta=f"Δ relative sensitive value = {abs(vh_diff) * 100:.1f}")

        st.markdown(
            f"GAUGE's relative sensitive value favours **{preferred}** for this sample "
            f"(Δ relative sensitive value = {vh_diff:+.3f}, using the same definition as the paper's virtual-RCT "
            "analysis: positive values favour the first-listed drug)."
        )
        if actual_drug_received:
            agreement = "agrees with" if preferred.lower() == actual_drug_received.lower() else "differs from"
            st.info(
                f"This (real, de-identified) patient actually received **{actual_drug_received}**. "
                f"GAUGE's preference here **{agreement}** the treatment on record. This is a single "
                "anecdote, not a validation — the paper's actual stratification result pools many "
                "patients per cancer/drug stratum with IPTW weighting."
            )
        st.caption(
            "The paper's published TCGA AUROC benchmarking and IPTW-weighted survival analysis used "
            "additional clinical covariates (age, stage) and a propensity-score model not reproduced "
            "in this lightweight comparison — treat this output as hypothesis-generating only."
        )
