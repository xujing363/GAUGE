import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core.predict import DrugNotFoundError, predict_one  # noqa: E402
from gauge_core import moldesign  # noqa: E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

common.configure_page("Molecular Design Scoring")
bundle = common.sidebar_mode_selector()

st.title("🧬 Molecular Design Scoring")
st.write(
    "A full design loop: **generate** candidate molecules from a seed (or paste your own), "
    "**score** them with GAUGE's relative sensitive value against a chosen patient/cell "
    "context, and inspect their **structures, drug-likeness, and similarity to known drugs** — "
    "the same role GAUGE played scoring REINVENT4-generated, EGFR-directed compounds for "
    "lung adenocarcinoma in the paper."
)

# ── 1. Context sample ─────────────────────────────────────────────────────────
known_label = common.known_sample_label(bundle)
st.subheader("1. Choose a context sample")
sample_source = st.radio(
    "Sample source", [known_label, "Upload my own expression file"], horizontal=True, key="md_sample_source"
)
sample_input = None
if sample_source == known_label:
    sample_input = common.cell_line_picker(bundle, "md")
else:
    from gauge_core.expression_io import parse_expression_table

    uploaded = st.file_uploader("Expression file (CSV/TSV)", type=["csv", "tsv", "txt"], key="md_upload")
    common.expression_upload_help("md")
    if uploaded is not None:
        try:
            samples = parse_expression_table(uploaded, bundle.artifacts.genes)
            sample_name = st.selectbox("Sample in this file", list(samples.keys()), key="md_sample_in_file")
            sample_input = samples[sample_name]
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not parse this file: {exc}")

# ── 2. Candidate molecules (generate / paste / demo) ──────────────────────────
st.subheader("2. Candidate molecules")
source = st.radio(
    "Where do the candidates come from?",
    ["🧪 Generate from a seed (de novo)", "✍️ Paste SMILES", "📄 Published REINVENT4 demo"],
    key="md_candidate_source",
)

if source.startswith("🧪"):
    st.caption(
        "Fragment one or more seed molecules (BRICS) and recombine them into new, valid, "
        "drug-like candidates — an offline stand-in for the paper's generative design step."
    )
    seed_mode = st.radio(
        "Seed source", ["Pick GAUGE-library drug(s)", "Paste seed SMILES"], horizontal=True, key="md_seed_mode"
    )
    seed_smiles: list[str] = []
    if seed_mode == "Pick GAUGE-library drug(s)":
        lib = bundle.drug_library.sort_values("DRUG_NAME")
        default_seed = ["Erlotinib"] if "Erlotinib" in lib["DRUG_NAME"].values else []
        seed_names = st.multiselect("Seed drug(s)", lib["DRUG_NAME"].tolist(), default=default_seed, key="md_seed_drugs")
        seed_smiles = [
            str(lib.loc[lib["DRUG_NAME"] == n].iloc[0].get("canonical_smiles") or lib.loc[lib["DRUG_NAME"] == n].iloc[0].get("smiles"))
            for n in seed_names
        ]
    else:
        seed_text = st.text_area("Seed SMILES (one per line)", height=80, key="md_seed_smiles_text")
        seed_smiles = [s.strip() for s in seed_text.splitlines() if s.strip()]

    n_gen = st.slider("How many candidates to generate", 5, 40, 15, key="md_n_gen")
    if st.button("✨ Generate candidates", disabled=not seed_smiles):
        with st.spinner("Generating analogs (BRICS recombination)…"):
            generated = moldesign.generate_analogs(seed_smiles, n=n_gen)
        if generated:
            st.session_state["md_smiles_text"] = "\n".join(generated)
            st.success(f"Generated {len(generated)} novel candidate(s). Review/edit them below, then score.")
            st.rerun()
        else:
            st.warning(
                "Could not generate analogs from this seed (it may not be BRICS-fragmentable). "
                "Try a larger or more complex seed molecule, or paste SMILES directly."
            )

elif source.startswith("📄"):
    if st.button("▶️ Load the published EGFR/ERBB design demo candidates", key="md_demo"):
        demo_path = common.EXAMPLE_DATA_DIR / "example_design_ranked_candidates.csv"
        demo_df = pd.read_csv(demo_path)
        st.session_state["md_smiles_text"] = "\n".join(demo_df["canonical_smiles"].tolist())
        st.rerun()

smiles_text = st.text_area(
    "Candidate SMILES (one per line) — generated/loaded candidates appear here and can be edited",
    height=150,
    placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nCOCCOc1cc2ncnc(Nc3cccc(c3)C#C)c2cc1OCCOC\n...",
    key="md_smiles_text",
)
candidate_smiles = [s.strip() for s in smiles_text.splitlines() if s.strip()]
st.caption(f"{len(candidate_smiles)} candidate(s) entered.")

# ── 3. Score & rank ───────────────────────────────────────────────────────────
st.subheader("3. Score & rank")
if st.button("🚀 Score candidates", type="primary", disabled=sample_input is None or not candidate_smiles):
    rows = []
    errors = []
    with st.spinner(f"Scoring {len(candidate_smiles)} candidates with GAUGE…"):
        for smi in candidate_smiles:
            try:
                result = predict_one(bundle, sample_input, smi)
            except DrugNotFoundError as exc:
                errors.append(f"{smi}: {exc}")
                continue
            props = moldesign.molecular_properties(smi)
            near = moldesign.nearest_library_drug(bundle, smi) or {}
            rows.append(
                {
                    "smiles": moldesign.canonical(smi) or smi,
                    "relative_sensitive_value": result.value_hat,
                    "absolute_auc": result.auc_hat,
                    "nearest_drug": near.get("nearest_drug"),
                    "tanimoto": near.get("tanimoto"),
                    **{k: v for k, v in props.items() if k != "valid"},
                }
            )
    if errors:
        st.warning(f"{len(errors)} candidate(s) could not be parsed:\n" + "\n".join(errors[:5]))
    if rows:
        ranked = pd.DataFrame(rows).sort_values("relative_sensitive_value", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        st.session_state["md_ranked"] = ranked

if "md_ranked" in st.session_state:
    ranked = st.session_state["md_ranked"]
    st.subheader("Ranked candidates")

    tab_rank, tab_struct, tab_props = st.tabs(["📊 Ranking", "🧪 Structures", "🔬 Properties"])

    with tab_rank:
        fig = px.bar(
            ranked.head(20).sort_values("relative_sensitive_value"), x="relative_sensitive_value", y="smiles", orientation="h",
            color="relative_sensitive_value", color_continuous_scale="Viridis",
            hover_data=["nearest_drug", "tanimoto", "qed"],
            labels={"relative_sensitive_value": "Predicted relative sensitive value", "smiles": ""},
        )
        fig.update_layout(height=max(320, 26 * min(len(ranked), 20)), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher = better predicted response. Hover for the nearest known drug (Tanimoto) and QED drug-likeness."
        )

    with tab_struct:
        st.caption("2-D structures of the top candidates (highest predicted relative sensitive value first).")
        top = ranked.head(12)
        mols, legends = [], []
        for r in top.itertuples(index=False):
            m = Chem.MolFromSmiles(r.smiles)
            if m is not None:
                mols.append(m)
                legends.append(f"#{r.rank}  value={r.relative_sensitive_value:.3f}  QED={r.qed:.2f}")
        if mols:
            grid = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(240, 200), legends=legends)
            st.image(grid, use_container_width=False)
        else:
            st.info("No depictable structures among the top candidates.")

    with tab_props:
        st.caption("Drug-likeness and physicochemical profile of each candidate.")
        prop_cols = ["rank", "smiles", "relative_sensitive_value", "absolute_auc", "nearest_drug", "tanimoto",
                     "mol_weight", "logp", "qed", "tpsa", "h_donors", "h_acceptors", "rotatable_bonds", "rings", "lipinski_ok"]
        show = ranked[[c for c in prop_cols if c in ranked.columns]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        n_lipinski = int(ranked["lipinski_ok"].sum()) if "lipinski_ok" in ranked.columns else 0
        st.caption(f"{n_lipinski} / {len(ranked)} candidates satisfy Lipinski's rule of five.")

    st.download_button(
        "⬇️ Download ranked candidates (CSV)",
        ranked.to_csv(index=False).encode(),
        file_name="gauge_design_ranking.csv", mime="text/csv",
    )

with st.expander("Reference: the published EGFR/ERBB lung-adenocarcinoma design result"):
    st.write(
        "These are real REINVENT4-generated candidates from the paper's design task, already "
        "scored and ranked by GAUGE against a TCGA lung-adenocarcinoma patient context. "
        "`final_rank_score` blends GAUGE's relative_sensitive_value with a knowledge-graph anchor score."
    )
    demo_path = common.EXAMPLE_DATA_DIR / "example_design_ranked_candidates.csv"
    if demo_path.exists():
        ref_df = pd.read_csv(demo_path)
        ref_df = ref_df[[c for c in ref_df.columns if "uncertainty" not in c.lower()]]
        st.dataframe(ref_df, use_container_width=True)
