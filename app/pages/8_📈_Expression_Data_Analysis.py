import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from gauge_core.expression_io import parse_expression_table  # noqa: E402

common.configure_page("Expression Data Analysis")
bundle = common.sidebar_mode_selector()

st.title("📈 Expression Data Analysis")
st.write(
    "General-purpose exploratory analysis for any gene-expression matrix you upload — this page "
    "does not use the GAUGE model and works for any multi-sample dataset, not just drug-response data."
)

use_demo = st.checkbox("Use the bundled demo expression file", key="ea_demo")
samples: dict[str, pd.Series] = {}
if use_demo:
    demo_path = common.EXAMPLE_DATA_DIR / "example_expression_multi_sample.csv"
    samples = parse_expression_table(str(demo_path), bundle.artifacts.genes)
else:
    uploaded = st.file_uploader("Expression file (CSV/TSV, multiple samples)", type=["csv", "tsv", "txt"], key="ea_upload")
    common.expression_upload_help("ea")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

if not samples:
    st.info("Upload a file (or check the demo box) to begin.")
    st.stop()

matrix = pd.DataFrame(samples).T  # samples x genes
matrix = matrix.dropna(axis=1, how="all").fillna(0.0)
st.success(f"Loaded {matrix.shape[0]} samples × {matrix.shape[1]} genes.")

tab_qc, tab_dimred, tab_cluster, tab_diff, tab_gtex = st.tabs(
    ["QC summary", "Dimensionality reduction", "Clustering heatmap", "Two-group comparison", "Tumour vs normal tissue (GTEx)"]
)

with tab_qc:
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Samples", matrix.shape[0])
        st.metric("Genes", matrix.shape[1])
    with c2:
        lib_size = matrix.sum(axis=1)
        fig = px.bar(x=lib_size.index, y=lib_size.values, labels={"x": "Sample", "y": "Total expression (library size)"})
        st.plotly_chart(fig, use_container_width=True)
    top_var = matrix.var(axis=0).sort_values(ascending=False).head(20)
    st.write("**Top 20 most variable genes in this dataset:**")
    st.bar_chart(top_var)

with tab_dimred:
    if matrix.shape[0] < 3:
        st.info("Need at least 3 samples for a meaningful projection.")
    else:
        method = st.radio("Method", ["PCA", "UMAP"], horizontal=True, key="ea_dimred_method")
        x = matrix.to_numpy(dtype=np.float32)
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
        if method == "PCA":
            n_comp = min(2, matrix.shape[0] - 1, matrix.shape[1])
            coords = PCA(n_components=n_comp, random_state=0).fit_transform(x)
        else:
            try:
                import umap

                coords = umap.UMAP(n_components=2, random_state=0, n_neighbors=min(15, matrix.shape[0] - 1)).fit_transform(x)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"UMAP unavailable ({exc}); falling back to PCA.")
                coords = PCA(n_components=2, random_state=0).fit_transform(x)
        coords_df = pd.DataFrame(coords[:, :2], columns=["dim1", "dim2"], index=matrix.index)
        fig = px.scatter(coords_df, x="dim1", y="dim2", text=coords_df.index, title=f"{method} projection")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

with tab_cluster:
    n_top = st.slider("Number of top-variance genes to show", 10, min(500, matrix.shape[1]), min(50, matrix.shape[1]), key="ea_cluster_n")
    top_genes = matrix.var(axis=0).sort_values(ascending=False).head(n_top).index
    sub = matrix[top_genes]
    if sub.shape[0] > 2:
        row_order = dendrogram(linkage(sub.to_numpy(), method="average"), no_plot=True, labels=sub.index.tolist())["ivl"]
        sub = sub.loc[row_order]
    fig = px.imshow(
        sub.T,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        labels=dict(color="expression"),
        title="Sample × top-variance-gene heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_diff:
    st.write("Compare two groups of samples (e.g. responders vs non-responders, treated vs control).")
    all_samples = matrix.index.tolist()
    group_a = st.multiselect("Group A", all_samples, key="ea_group_a")
    group_b = st.multiselect("Group B", [s for s in all_samples if s not in group_a], key="ea_group_b")
    if len(group_a) >= 2 and len(group_b) >= 2:
        a = matrix.loc[group_a]
        b = matrix.loc[group_b]
        t_stat, p_val = stats.ttest_ind(a.to_numpy(), b.to_numpy(), axis=0, equal_var=False, nan_policy="omit")
        diff = pd.DataFrame(
            {
                "gene": matrix.columns,
                "mean_diff_A_minus_B": a.mean(axis=0).to_numpy() - b.mean(axis=0).to_numpy(),
                "p_value": p_val,
            }
        )
        diff["neg_log10_p"] = -np.log10(diff["p_value"].clip(lower=1e-300))
        fig = px.scatter(
            diff, x="mean_diff_A_minus_B", y="neg_log10_p", hover_name="gene",
            title="Volcano plot (Group A vs Group B)", labels={"mean_diff_A_minus_B": "Mean difference (A − B)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(diff.sort_values("p_value").head(50), use_container_width=True)
        st.download_button(
            "⬇️ Download full differential table (CSV)",
            diff.sort_values("p_value").to_csv(index=False).encode(),
            file_name="gauge_differential_expression.csv",
            mime="text/csv",
        )
    else:
        st.caption("Select at least 2 samples in each group to run the comparison.")

with tab_gtex:
    st.write(
        "Compare one of your loaded samples against **GTEx v11** median TPM in a chosen healthy "
        "tissue — a quick way to spot genes that look tumour-elevated relative to the matched "
        "normal tissue."
    )
    gtex = pd.read_csv(common.EXAMPLE_DATA_DIR / "example_gtex_median_tpm_by_tissue.csv", index_col=0)
    sample_choice = st.selectbox("Your sample", matrix.index.tolist(), key="ea_gtex_sample")
    tissue_choice = st.selectbox("GTEx reference tissue", sorted(gtex.columns), key="ea_gtex_tissue")
    common_genes = [g for g in matrix.columns if g in gtex.index]
    if len(common_genes) < 10:
        st.warning("Not enough overlapping genes with the GTEx reference panel to compare.")
    else:
        tumor_vals = matrix.loc[sample_choice, common_genes].astype(float)
        normal_vals = gtex.loc[common_genes, tissue_choice].astype(float)
        cmp_df = pd.DataFrame({"gene": common_genes, "sample_value": tumor_vals.to_numpy(), "gtex_median_tpm": normal_vals.to_numpy()})
        cmp_df["log2_ratio"] = np.log2((cmp_df["sample_value"] + 1) / (cmp_df["gtex_median_tpm"] + 1))
        fig = px.scatter(
            cmp_df, x="gtex_median_tpm", y="sample_value", hover_name="gene", log_x=True, log_y=True,
            title=f"{sample_choice} vs GTEx {tissue_choice} (median TPM)",
            labels={"gtex_median_tpm": f"GTEx {tissue_choice} median TPM", "sample_value": f"{sample_choice} value"},
        )
        max_val = max(cmp_df["sample_value"].max(), cmp_df["gtex_median_tpm"].max())
        fig.add_shape(type="line", x0=0.01, y0=0.01, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, use_container_width=True)
        top_up = cmp_df.sort_values("log2_ratio", ascending=False).head(15)
        top_down = cmp_df.sort_values("log2_ratio").head(15)
        c1, c2 = st.columns(2)
        c1.write(f"**Most elevated vs {tissue_choice}**")
        c1.dataframe(top_up[["gene", "log2_ratio"]], use_container_width=True, hide_index=True)
        c2.write(f"**Most reduced vs {tissue_choice}**")
        c2.dataframe(top_down[["gene", "log2_ratio"]], use_container_width=True, hide_index=True)
        st.caption(
            "GTEx and your sample may come from different RNA-seq quantification pipelines; "
            "treat log2 ratios as approximate, not pipeline-harmonised differential expression."
        )
