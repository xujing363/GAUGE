import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402
from gauge_core.predict import predict_one  # noqa: E402

common.configure_page("Batch Prediction")
bundle = common.sidebar_mode_selector()

st.title("📊 Batch Prediction")
st.write(
    "Score many samples against many drugs in one go. Pick known cell lines (no upload needed), "
    "or upload your own expression file, then choose which drugs to test against every sample."
)

known_label = common.known_sample_label(bundle)
st.subheader("1. Provide expression")
input_mode = st.radio(
    "Sample source",
    [f"{known_label}s (no upload needed)", "Upload expression file", "Bundled demo file"],
    horizontal=True,
    key="bp_input_mode",
)

# samples maps a display name -> either a known cell-line ID (str) or an expression Series.
# predict_one accepts both, so downstream code is identical for either source.
samples: dict[str, object] = {}

if input_mode.startswith(known_label):
    meta = bundle.cell_metadata
    if meta is not None and not meta.empty:
        meta = meta.copy()
        meta["display"] = (
            meta["model_name"].astype(str) + "  —  "
            + meta["tissue"].astype(str) + " / " + meta["cancer_type"].astype(str)
        )
        meta = meta.sort_values("display")
        id_by_display = dict(zip(meta["display"], meta["SANGER_MODEL_ID"].astype(str)))
        chosen = st.multiselect(
            f"{known_label}s to score", meta["display"].tolist(), key="bp_known_cells",
            help="Search by cell-line name, tissue, or cancer type.",
        )
        samples = {id_by_display[d]: id_by_display[d] for d in chosen}
    else:
        ids = bundle.cell_state_matrix.index.astype(str).tolist()
        chosen = st.multiselect(f"{known_label}s to score", ids, key="bp_known_cells")
        samples = {c: c for c in chosen}
    if samples:
        st.success(f"Selected {len(samples)} known cell line(s).")
    else:
        st.info("Pick one or more cell lines above to score.")
elif input_mode == "Bundled demo file":
    demo_path = common.EXAMPLE_DATA_DIR / "example_expression_multi_sample.csv"
    samples = parse_expression_table(str(demo_path), bundle.artifacts.genes)
    st.success(f"Loaded demo file with {len(samples)} samples.")
else:
    uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="bp_upload")
    common.expression_upload_help("bp")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
            st.success(f"Parsed {len(samples)} samples from the uploaded file.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

selected_samples = st.multiselect(
    "Samples to include", list(samples.keys()), default=list(samples.keys())[:10], key="bp_sample_select"
) if samples else []

st.subheader("2. Choose drugs")
drug_lib = bundle.drug_library.sort_values("DRUG_NAME")
selected_drug_names = st.multiselect(
    "Drugs to test (from GAUGE library)",
    drug_lib["DRUG_NAME"].tolist(),
    default=drug_lib["DRUG_NAME"].tolist()[:5],
    key="bp_drug_select",
)
selected_drug_ids = drug_lib.loc[drug_lib["DRUG_NAME"].isin(selected_drug_names), "DRUG_ID"].astype(int).tolist()

n_jobs = len(selected_samples) * len(selected_drug_ids)
st.caption(f"This will run {n_jobs} predictions ({len(selected_samples)} samples × {len(selected_drug_ids)} drugs).")
if n_jobs > 4000:
    st.warning("That is a large batch and may take a while in this browser session. Consider narrowing the selection.")

if st.button("🚀 Run batch prediction", type="primary", disabled=n_jobs == 0):
    progress = st.progress(0.0, text="Running predictions...")
    rows = []
    for i, sample_name in enumerate(selected_samples):
        sample_input = samples[sample_name]
        for drug_id in selected_drug_ids:
            result = predict_one(bundle, sample_input, drug_id)
            rows.append(
                {
                    "sample": sample_name,
                    "DRUG_ID": drug_id,
                    "drug": result.drug.name,
                    "absolute_auc": result.auc_hat,
                    "relative_sensitive_value": result.value_hat,
                }
            )
        progress.progress((i + 1) / len(selected_samples), text=f"Running predictions... ({i + 1}/{len(selected_samples)} samples)")
    progress.empty()

    results = pd.DataFrame(rows)
    st.session_state["bp_results"] = results

if "bp_results" in st.session_state:
    import plotly.express as px

    results = st.session_state["bp_results"]
    st.subheader("Results")

    # Headline metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Predictions", len(results))
    m2.metric("Samples", results["sample"].nunique())
    m3.metric("Drugs", results["drug"].nunique())

    best = results.loc[results.groupby("sample")["relative_sensitive_value"].idxmax()]
    pivot = results.pivot_table(index="sample", columns="drug", values="relative_sensitive_value")

    tab_heat, tab_best, tab_drug, tab_table = st.tabs(
        ["🔥 Heatmap", "🏅 Best drug / sample", "💊 Drug comparison", "📋 Table"]
    )

    with tab_heat:
        if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
            fig = px.imshow(
                pivot, color_continuous_scale="RdBu", aspect="auto",
                labels=dict(color="relative sensitive value", x="drug", y="sample"),
                title="Predicted relative sensitive value (sample × drug)",
            )
            fig.update_layout(height=max(320, 26 * pivot.shape[0] + 120))
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Heatmap needs a 2-D grid; with a single sample or drug, show a bar chart instead.
            single = results.sort_values("relative_sensitive_value", ascending=False)
            x, y = ("drug", "sample") if results["sample"].nunique() == 1 else ("sample", "drug")
            fig = px.bar(
                single, x="relative_sensitive_value", y=x, color=y, orientation="h",
                labels={"relative_sensitive_value": "Relative sensitive value", x: ""},
                title="Predicted relative sensitive value",
            )
            fig.update_layout(height=max(320, 30 * len(single)))
            st.plotly_chart(fig, use_container_width=True)

    with tab_best:
        st.caption("The single most promising drug GAUGE predicts for each sample.")
        show_best = best[["sample", "drug", "relative_sensitive_value", "absolute_auc"]].sort_values(
            "relative_sensitive_value", ascending=False
        )
        st.dataframe(show_best, use_container_width=True, hide_index=True)
        fig_best = px.bar(
            show_best, x="relative_sensitive_value", y="sample", color="drug", orientation="h",
            title="Best predicted drug per sample", labels={"sample": ""},
        )
        fig_best.update_layout(height=max(320, 30 * len(show_best)))
        st.plotly_chart(fig_best, use_container_width=True)

    with tab_drug:
        st.caption("How each drug performs across the selected samples (mean ± spread).")
        agg = (
            results.groupby("drug")["relative_sensitive_value"]
            .agg(["mean", "min", "max", "count"]).reset_index()
            .sort_values("mean", ascending=False)
        )
        fig_drug = px.bar(
            agg, x="mean", y="drug", orientation="h", color="mean",
            color_continuous_scale="Viridis", error_x=(agg["max"] - agg["mean"]),
            labels={"mean": "Mean relative sensitive value", "drug": ""},
            title="Drug ranking across the cohort",
        )
        fig_drug.update_layout(height=max(320, 30 * len(agg)))
        st.plotly_chart(fig_drug, use_container_width=True)
        if results["sample"].nunique() >= 2:
            fig_box = px.box(
                results, x="drug", y="relative_sensitive_value", points="all",
                labels={"relative_sensitive_value": "Relative sensitive value", "drug": ""},
                title="Distribution of predicted response per drug",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_table:
        st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download all results (CSV)",
        results.to_csv(index=False).encode(),
        file_name="gauge_batch_predictions.csv",
        mime="text/csv",
    )
