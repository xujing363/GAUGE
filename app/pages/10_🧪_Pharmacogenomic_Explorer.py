import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
from scipy import stats  # noqa: E402
import streamlit as st  # noqa: E402

common.configure_page("Pharmacogenomic Explorer")
bundle = common.sidebar_mode_selector()

st.title("🧪 Pharmacogenomic Data Explorer")
st.write(
    "Explore the **real experimental** drug-sensitivity data behind this checkpoint directly "
    "— no GAUGE model prediction involved on this page. Useful for sanity-checking a "
    "candidate biomarker or just understanding the screen GAUGE was trained on."
)

resp = bundle.response_table
if resp.empty:
    st.warning("No response table bundled for this mode.")
    st.stop()

tab_dist, tab_biomarker, tab_similarity = st.tabs(
    ["Drug response distribution", "Expression ↔ response biomarker", "Drug similarity clustering"]
)

with tab_dist:
    drug_name = st.selectbox(
        "Drug", sorted(resp["DRUG_NAME"].astype(str).unique()), key="pe_dist_drug"
    )
    sub = resp.loc[resp["DRUG_NAME"] == drug_name]
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(sub, x="AUC", nbins=40, color="split", barmode="overlay", opacity=0.7,
                            title=f"AUC distribution across {len(sub)} cell lines tested with {drug_name}")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Cell lines tested", len(sub))
        st.metric("Median AUC", f"{sub['AUC'].median():.3f}")
        st.metric("IQR", f"{sub['AUC'].quantile(.75) - sub['AUC'].quantile(.25):.3f}")
    most_sensitive = sub.nsmallest(10, "AUC")[["SANGER_MODEL_ID", "AUC", "split"]]
    most_resistant = sub.nlargest(10, "AUC")[["SANGER_MODEL_ID", "AUC", "split"]]
    c3, c4 = st.columns(2)
    c3.write("**Most sensitive (lowest AUC)**")
    c3.dataframe(most_sensitive, use_container_width=True, hide_index=True)
    c4.write("**Most resistant (highest AUC)**")
    c4.dataframe(most_resistant, use_container_width=True, hide_index=True)

with tab_biomarker:
    if not bundle.gene_level_state:
        st.info(
            "This mode's transcriptome encoder uses a PCA rotation rather than per-gene "
            "features, so a single gene's expression can't be read back out for this "
            "checkpoint. Switch to a GDSC mode in the sidebar for this analysis."
        )
    else:
        st.caption("Classic pharmacogenomics view: does this gene's expression correlate with response to this drug?")
        demo = st.button("▶️ Try the EGFR × Erlotinib demo", key="pe_biomarker_demo")
        genes = bundle.artifacts.genes
        default_gene = "EGFR" if "EGFR" in genes else genes[0]
        gene = st.selectbox("Gene", sorted(genes), index=sorted(genes).index(default_gene) if demo else 0, key="pe_gene")
        drug_options = sorted(resp["DRUG_NAME"].astype(str).unique())
        default_drug = "Erlotinib" if "Erlotinib" in drug_options else drug_options[0]
        drug_name = st.selectbox(
            "Drug", drug_options, index=drug_options.index(default_drug) if demo else 0, key="pe_biomarker_drug"
        )
        expr_proxy = bundle.gene_proxy_series(gene)
        sub = resp.loc[resp["DRUG_NAME"] == drug_name].copy()
        sub["expression_z"] = sub["SANGER_MODEL_ID"].map(expr_proxy)
        sub = sub.dropna(subset=["expression_z", "AUC"])
        if len(sub) < 5:
            st.warning("Not enough overlapping cell lines to plot.")
        else:
            r, p = stats.pearsonr(sub["expression_z"], sub["AUC"])
            fig = px.scatter(
                sub, x="expression_z", y="AUC", trendline="ols", hover_data=["SANGER_MODEL_ID"],
                labels={"expression_z": f"{gene} expression (standardized)", "AUC": f"{drug_name} AUC"},
                title=f"{gene} expression vs {drug_name} response (n={len(sub)}, r={r:.2f}, p={p:.1e})",
            )
            st.plotly_chart(fig, use_container_width=True)
            direction = "lower AUC (more sensitive)" if r < 0 else "higher AUC (more resistant)"
            st.markdown(
                f"Pearson r = **{r:.2f}** (p = {p:.1e}). Higher {gene} expression is associated with {direction} "
                f"in this dataset. This is a raw correlation in the real screening data, independent of GAUGE."
            )

with tab_similarity:
    top_n = st.slider("Number of most-tested drugs to cluster", 10, 60, 25, key="pe_sim_n")
    top_drugs = resp["DRUG_NAME"].value_counts().head(top_n).index.tolist()
    sub = resp.loc[resp["DRUG_NAME"].isin(top_drugs)]
    pivot = sub.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_NAME", values="AUC")
    corr = pivot.corr(min_periods=20)
    if corr.shape[0] >= 2:
        from scipy.cluster.hierarchy import dendrogram, linkage

        order = dendrogram(linkage(corr.fillna(0).to_numpy(), method="average"), no_plot=True, labels=corr.index.tolist())["ivl"]
        corr = corr.loc[order, order]
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Cross-drug response-profile correlation (similar drugs cluster together)",
            labels=dict(color="Pearson r"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Two drugs with strongly correlated AUC across the same cell lines often share "
            "a mechanism of action or target pathway — a model-free sanity check for "
            "GAUGE's knowledge-graph-derived mechanism groupings."
        )
