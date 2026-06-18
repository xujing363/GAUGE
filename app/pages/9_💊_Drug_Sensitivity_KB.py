"""Drug-sensitivity knowledge base — pharmacology and cancer-genomics context
that does **not** depend on the GAUGE model.

Everything here is point-and-click and answered directly from free public
biomedical databases (DGIdb, ChEMBL, OpenTargets, UniProt, Reactome, cBioPortal,
ClinicalTrials.gov, PubChem, Europe PMC). It complements GAUGE's predictions with
the surrounding biology: which drugs hit a target, how a drug works, how often a
gene is mutated, what trials exist, and what the literature says.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from gauge_core import bio_tools  # noqa: E402

common.configure_page("Drug Sensitivity Knowledge Base", icon="💊")

# This page is deliberately GAUGE-free, so it skips the model-bundle loader and
# instead shows a light sidebar note explaining what it does and does not use.
st.sidebar.markdown("## 💊 Drug Sensitivity KB")
st.sidebar.caption(
    "Pharmacology & cancer-genomics context from public databases. "
    "**No GAUGE model required** — these analyses are independent of GAUGE predictions."
)
st.sidebar.info(
    "Needs internet access. All sources are free and key-less:\n\n"
    "- **DGIdb** — drug–gene interactions\n"
    "- **ChEMBL** — mechanism, phase, indications\n"
    "- **OpenTargets** — target–disease evidence\n"
    "- **UniProt** — protein function\n"
    "- **Reactome** — pathways\n"
    "- **cBioPortal** — mutation frequency\n"
    "- **ClinicalTrials.gov** — trials\n"
    "- **PubChem / Europe PMC** — chemistry & literature",
    icon="🌐",
)

st.title("💊 Drug Sensitivity Knowledge Base")
st.caption(
    "Common drug-sensitivity analyses that do **not** rely on GAUGE model outputs — "
    "druggable targets, drug mechanisms, mutation frequency, pathways, trials and literature, "
    "pulled live from public biomedical databases."
)


def _err(d: dict) -> str | None:
    """Return a human note if a tool result is an error/empty, else None."""
    if not isinstance(d, dict):
        return "Unexpected response."
    if d.get("error"):
        return f"{d.get('hint', 'Lookup failed.')}\n\n`{d['error']}`"
    return None


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


tab_target, tab_drug, tab_trials, tab_lit = st.tabs(
    ["🎯 Target / gene", "💊 Drug profile", "🧪 Clinical trials", "📚 Literature"]
)

# ── Target / gene ─────────────────────────────────────────────────────────────
with tab_target:
    st.subheader("Profile a gene / target")
    st.write(
        "Enter an HGNC gene symbol to see which drugs act on it, its disease associations, "
        "molecular function, pathways, and how often it is mutated across tumours."
    )
    c1, c2 = st.columns([3, 1])
    gene = c1.text_input("Gene symbol", value="EGFR", key="kb_gene").strip()
    go_gene = c2.button("🔎 Look up", key="kb_gene_go", use_container_width=True)
    st.caption("Examples: EGFR · TP53 · ERBB2 · KRAS · BRAF · ALK · BRCA1")

    if go_gene and gene:
        with st.spinner(f"Querying public databases for {gene}…"):
            druggable = bio_tools.lookup_drug_gene_interactions(gene)
            disease = bio_tools.lookup_target_disease_associations(gene)
            protein = bio_tools.lookup_protein(gene)
            pathways = bio_tools.lookup_pathways(gene)
            muts = bio_tools.lookup_cancer_mutations(gene)

        # Mutation frequency headline
        e = _err(muts)
        if e:
            st.warning(f"Mutation frequency unavailable. {e}")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Pan-cancer mutation freq.", f"{muts.get('mutation_frequency_pct')}%")
            m2.metric("Mutated tumours", f"{muts.get('n_mutated_samples'):,}")
            m3.metric("Cohort size", f"{muts.get('n_samples'):,}")
            st.caption(f"Source: cBioPortal — {muts.get('cohort')}")

        st.markdown("#### 💊 Drugs that act on this target (DGIdb)")
        e = _err(druggable)
        if e:
            st.warning(e)
        elif druggable.get("interactions"):
            df = _df(druggable["interactions"])
            df["interaction_types"] = df["interaction_types"].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{druggable.get('n_interactions')} interactions found · showing top {len(df)} · source: DGIdb")
        else:
            st.info("No drug–gene interactions found in DGIdb for this gene.")

        cda, cdb = st.columns(2)
        with cda:
            st.markdown("#### 🧬 Disease associations (OpenTargets)")
            e = _err(disease)
            if e:
                st.warning(e)
            elif disease.get("top_associations"):
                st.dataframe(_df(disease["top_associations"]), use_container_width=True, hide_index=True)
                st.caption(f"{disease.get('approved_name') or ''} · source: OpenTargets")
            else:
                st.info("No OpenTargets associations found.")
        with cdb:
            st.markdown("#### 🛤️ Pathways (Reactome)")
            e = _err(pathways)
            if e:
                st.warning(e)
            elif pathways.get("pathways"):
                st.dataframe(_df(pathways["pathways"]), use_container_width=True, hide_index=True)
                st.caption(f"{pathways.get('n_pathways')} pathways · source: Reactome")
            else:
                st.info("No Reactome pathways mapped.")

        st.markdown("#### 🔬 Protein function & disease involvement (UniProt)")
        e = _err(protein)
        if e:
            st.warning(e)
        else:
            st.write(f"**{protein.get('protein_name') or gene}** ({protein.get('accession')})")
            for fn in protein.get("function", []):
                st.write(f"- {fn}")
            if protein.get("disease_involvement"):
                st.write("**Disease involvement:** " + "; ".join(protein["disease_involvement"]))

# ── Drug profile ──────────────────────────────────────────────────────────────
with tab_drug:
    st.subheader("Profile a drug")
    st.write("Mechanism of action, clinical phase, approved indications, chemistry and active trials.")
    c1, c2 = st.columns([3, 1])
    drug = c1.text_input("Drug name", value="Erlotinib", key="kb_drug").strip()
    go_drug = c2.button("🔎 Look up", key="kb_drug_go", use_container_width=True)
    st.caption("Examples: Erlotinib · Imatinib · Osimertinib · Trametinib · Olaparib")

    if go_drug and drug:
        with st.spinner(f"Querying public databases for {drug}…"):
            mech = bio_tools.lookup_drug_mechanism(drug)
            chem = bio_tools.lookup_compound(drug)
            trials = bio_tools.search_clinical_trials(intervention=drug, limit=6)

        e = _err(mech)
        if e:
            st.warning(e)
        else:
            m1, m2 = st.columns(2)
            m1.metric("Max clinical phase", str(mech.get("max_clinical_phase") or "—"))
            m2.metric("ChEMBL ID", str(mech.get("chembl_id") or "—"))
            if mech.get("mechanisms_of_action"):
                st.markdown("**Mechanism(s) of action**")
                st.dataframe(_df(mech["mechanisms_of_action"]), use_container_width=True, hide_index=True)
            if mech.get("indications"):
                st.markdown("**Indications:** " + ", ".join(mech["indications"]))
            st.caption("Source: ChEMBL")

        st.markdown("#### ⚗️ Chemistry (PubChem)")
        e = _err(chem)
        if e:
            st.warning(e)
        elif chem.get("cid"):
            cc = st.columns(4)
            cc[0].metric("PubChem CID", str(chem.get("cid")))
            cc[1].metric("Formula", str(chem.get("molecular_formula") or "—"))
            cc[2].metric("Mol. weight", str(chem.get("molecular_weight") or "—"))
            cc[3].metric("XLogP", str(chem.get("xlogp") if chem.get("xlogp") is not None else "—"))
            st.caption(f"IUPAC: {chem.get('iupac_name') or '—'}")
        else:
            st.info("No PubChem record found.")

        st.markdown("#### 🧪 Recent trials for this drug")
        e = _err(trials)
        if e:
            st.warning(e)
        elif trials.get("trials"):
            tdf = _df(trials["trials"])
            for col in ("phases", "conditions"):
                if col in tdf:
                    tdf[col] = tdf[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            st.caption("Source: ClinicalTrials.gov")
        else:
            st.info("No trials found.")

# ── Clinical trials ───────────────────────────────────────────────────────────
with tab_trials:
    st.subheader("Search clinical trials")
    st.write("Find interventional trials by condition, drug, or both (ClinicalTrials.gov).")
    c1, c2, c3 = st.columns([2, 2, 1])
    cond = c1.text_input("Condition", value="lung adenocarcinoma", key="kb_tr_cond").strip()
    intr = c2.text_input("Intervention / drug", value="Erlotinib", key="kb_tr_intr").strip()
    go_tr = c3.button("🔎 Search", key="kb_tr_go", use_container_width=True)
    n_tr = st.slider("How many trials", 3, 20, 8, key="kb_tr_n")

    if go_tr and (cond or intr):
        with st.spinner("Searching ClinicalTrials.gov…"):
            res = bio_tools.search_clinical_trials(condition=cond, intervention=intr, limit=n_tr)
        e = _err(res)
        if e:
            st.warning(e)
        elif res.get("trials"):
            tdf = _df(res["trials"])
            for col in ("phases", "conditions"):
                if col in tdf:
                    tdf[col] = tdf[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
            tdf["link"] = tdf["nct_id"].apply(lambda n: f"https://clinicaltrials.gov/study/{n}" if n else "")
            st.dataframe(
                tdf,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )
            st.caption(f"{res.get('count')} trials · source: ClinicalTrials.gov")
        else:
            st.info("No trials matched.")

# ── Literature ────────────────────────────────────────────────────────────────
with tab_lit:
    st.subheader("Search the literature")
    st.write("Find citable primary literature (Europe PMC), ranked by citation count.")
    c1, c2 = st.columns([4, 1])
    q = c1.text_input("Query", value="EGFR inhibitor resistance lung adenocarcinoma", key="kb_lit_q").strip()
    go_lit = c2.button("🔎 Search", key="kb_lit_go", use_container_width=True)
    n_lit = st.slider("How many papers", 3, 20, 8, key="kb_lit_n")

    if go_lit and q:
        with st.spinner("Searching Europe PMC…"):
            res = bio_tools.search_literature(q, limit=n_lit)
        e = _err(res)
        if e:
            st.warning(e)
        elif res.get("papers"):
            ldf = _df(res["papers"])
            ldf["link"] = ldf.apply(
                lambda r: (f"https://doi.org/{r['doi']}" if r.get("doi") else
                           (f"https://europepmc.org/article/MED/{r['pmid']}" if r.get("pmid") else "")),
                axis=1,
            )
            st.dataframe(
                ldf,
                use_container_width=True,
                hide_index=True,
                column_config={"link": st.column_config.LinkColumn("link")},
            )
            st.caption(f"{res.get('count')} papers · source: Europe PMC")
        else:
            st.info("No papers found.")

st.divider()
st.caption(
    "These analyses are independent of GAUGE and are intended for research context only — "
    "not clinical advice. Database contents are owned by their respective providers."
)
