"""Opt-in external biomedical-database tools for the GAUGE Assistant.

These give the Assistant the kind of *biological context* a "virtual disease
biologist" needs -- target-disease evidence, compound chemistry, protein
function, and primary literature -- by querying free, key-less public APIs.

Design rules (these matter):
- **Opt-in.** Nothing here runs unless the caller passes ``enable_external``.
  GAUGE stays a fully self-contained, offline-capable app by default.
- **Never a drug-response number.** These tools return biological context
  only. Every GAUGE prediction still comes from ``gauge_core.predict``; the
  LLM is told (in the agent system prompt) not to conflate the two.
- **Graceful degradation.** Every call is wrapped: a timeout, an offline
  machine, or an upstream outage returns ``{"error": ..., "hint": ...}``
  rather than raising, so the chat never crashes when the network is down.

Transport is ``httpx`` (already an ``openai`` dependency -- no new package).
Responses are LRU-cached in-process so repeated lookups in one report are cheap.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import httpx

_TIMEOUT = httpx.Timeout(12.0, connect=6.0)
_HEADERS = {"User-Agent": "GAUGE-Assistant/1.0 (research tool; biomedical lookups)"}

# Public endpoints (all free, no API key required).
_OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
_EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_DGIDB_GQL = "https://dgidb.org/api/graphql"
_CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
_CLINICALTRIALS = "https://clinicaltrials.gov/api/v2/studies"
_REACTOME = "https://reactome.org/ContentService"
_CBIOPORTAL = "https://www.cbioportal.org/api"
# A large, curated pan-cancer cohort used as a single, reproducible reference for
# mutation frequency (MSK-IMPACT 2017, ~10.9k tumours sequenced clinically).
_CBIO_STUDY = "msk_impact_2017"


def _degraded(exc: Exception) -> dict[str, Any]:
    return {
        "error": f"{type(exc).__name__}: {exc}",
        "hint": "External biomedical lookup unavailable (offline or upstream error). "
        "GAUGE model predictions are unaffected.",
    }


def _get_json(url: str, *, params: dict | None = None) -> Any:
    resp = httpx.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post_json(url: str, *, json: dict) -> Any:
    resp = httpx.post(url, json=json, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# ── OpenTargets: target ↔ disease association evidence ────────────────────────
_OT_QUERY = """
query TargetDiseases($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits { id name }
  }
}
"""
_OT_ASSOC_QUERY = """
query Assoc($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id approvedSymbol approvedName
    associatedDiseases(page: {index: 0, size: 10}) {
      count
      rows { score disease { id name } }
    }
  }
}
"""


@lru_cache(maxsize=256)
def lookup_target_disease_associations(gene_symbol: str) -> dict[str, Any]:
    """Top disease associations for a gene/target from OpenTargets (0-1 scores)."""
    try:
        hit = _post_json(_OPENTARGETS_GQL, json={"query": _OT_QUERY, "variables": {"q": gene_symbol}})
        hits = (hit.get("data", {}).get("search", {}) or {}).get("hits", [])
        if not hits:
            return {"gene": gene_symbol, "associations": [], "note": "No OpenTargets target found."}
        ensembl_id = hits[0]["id"]
        data = _post_json(_OPENTARGETS_GQL, json={"query": _OT_ASSOC_QUERY, "variables": {"ensemblId": ensembl_id}})
        target = (data.get("data", {}) or {}).get("target") or {}
        assoc = target.get("associatedDiseases", {}) or {}
        rows = [
            {"disease": r["disease"]["name"], "efo_id": r["disease"]["id"], "association_score": round(r["score"], 3)}
            for r in assoc.get("rows", [])
        ]
        return {
            "gene": gene_symbol,
            "ensembl_id": ensembl_id,
            "approved_symbol": target.get("approvedSymbol"),
            "approved_name": target.get("approvedName"),
            "total_associated_diseases": assoc.get("count"),
            "top_associations": rows,
            "source": "OpenTargets Platform",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── PubChem: compound identity & properties ──────────────────────────────────
@lru_cache(maxsize=256)
def lookup_compound(name_or_smiles: str) -> dict[str, Any]:
    """Resolve a drug name or SMILES to PubChem CID + key chemical properties."""
    looks_like_smiles = any(c in name_or_smiles for c in "()=#[]") and " " not in name_or_smiles.strip()
    namespace = "smiles" if looks_like_smiles else "name"
    props = "MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName,XLogP"
    try:
        # Pass the identifier as a query parameter so SMILES special characters
        # ('(', '=', '#', '[', ']') are encoded by httpx rather than breaking the path.
        url = f"{_PUBCHEM}/compound/{namespace}/property/{props}/JSON"
        data = _get_json(url, params={namespace: name_or_smiles})
        rows = data.get("PropertyTable", {}).get("Properties", [])
        if not rows:
            return {"query": name_or_smiles, "note": "No PubChem compound found."}
        row = rows[0]
        return {
            "query": name_or_smiles,
            "cid": row.get("CID"),
            "iupac_name": row.get("IUPACName"),
            "molecular_formula": row.get("MolecularFormula"),
            "molecular_weight": row.get("MolecularWeight"),
            "canonical_smiles": row.get("CanonicalSMILES"),
            "xlogp": row.get("XLogP"),
            "source": "PubChem (NCBI)",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── UniProt: protein function & disease involvement ──────────────────────────
_ACCESSION_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")


@lru_cache(maxsize=256)
def lookup_protein(gene_or_uniprot: str) -> dict[str, Any]:
    """Protein function + disease involvement for a gene symbol or UniProt accession.

    Prefers the reviewed (Swiss-Prot) human entry. A UniProt accession is queried
    directly; anything else is treated as a gene symbol (``accession:<symbol>`` is
    rejected by the API with a 400, so the two must be distinguished).
    """
    token = gene_or_uniprot.strip()
    if _ACCESSION_RE.match(token.upper()):
        query = f"accession:{token}"
    else:
        query = f"gene_exact:{token} AND organism_id:9606 AND reviewed:true"
    fields = "accession,id,protein_name,gene_names,cc_function,cc_disease"
    try:
        data = _get_json(_UNIPROT, params={"query": query, "fields": fields, "format": "json", "size": 1})
        results = data.get("results", [])
        if not results:
            # Fall back to a looser gene search (unreviewed entries, e.g. niche symbols).
            data = _get_json(
                _UNIPROT,
                params={"query": f"gene:{token} AND organism_id:9606", "fields": fields, "format": "json", "size": 1},
            )
            results = data.get("results", [])
        if not results:
            return {"query": gene_or_uniprot, "note": "No human UniProt entry found."}
        entry = results[0]

        def _texts(comment_type: str) -> list[str]:
            out = []
            for c in entry.get("comments", []):
                if c.get("commentType") == comment_type:
                    for t in c.get("texts", []):
                        if t.get("value"):
                            out.append(t["value"])
                    dz = c.get("disease", {})
                    if dz.get("diseaseId"):
                        out.append(dz["diseaseId"])
            return out

        desc = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value")
        return {
            "query": gene_or_uniprot,
            "accession": entry.get("primaryAccession"),
            "protein_name": desc,
            "gene_names": [g.get("geneName", {}).get("value") for g in entry.get("genes", []) if g.get("geneName")],
            "function": _texts("FUNCTION")[:2],
            "disease_involvement": _texts("DISEASE")[:5],
            "source": "UniProtKB",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── Europe PMC: primary literature (citations for the report) ────────────────
@lru_cache(maxsize=256)
def search_literature(query: str, limit: int = 5) -> dict[str, Any]:
    """Search Europe PMC for recent literature; returns citable title/PMID/DOI rows."""
    try:
        data = _get_json(
            _EUROPEPMC,
            params={"query": query, "format": "json", "pageSize": int(limit), "resultType": "lite", "sort": "CITED desc"},
        )
        results = data.get("resultList", {}).get("result", [])
        papers = [
            {
                "title": r.get("title"),
                "authors": r.get("authorString"),
                "journal": r.get("journalTitle"),
                "year": r.get("pubYear"),
                "pmid": r.get("pmid"),
                "doi": r.get("doi"),
                "cited_by": r.get("citedByCount"),
            }
            for r in results
        ]
        return {"query": query, "count": len(papers), "papers": papers, "source": "Europe PMC"}
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── DGIdb: drug ↔ gene interactions (druggable targets) ──────────────────────
_DGIDB_GENE_QUERY = """
query Interactions($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      interactions {
        drug { name conceptId approved }
        interactionScore
        interactionTypes { type directionality }
        sources { sourceDbName }
      }
    }
  }
}
"""


@lru_cache(maxsize=256)
def lookup_drug_gene_interactions(gene_symbol: str, limit: int = 15) -> dict[str, Any]:
    """Which drugs are known to act on a gene/target, from DGIdb (aggregated
    drug-gene interaction databases). Core for 'what could hit GENE?' reasoning.

    Returns biological/pharmacological context (known interactions and an
    interaction score), NOT a GAUGE drug-response prediction.
    """
    try:
        data = _post_json(
            _DGIDB_GQL,
            json={"query": _DGIDB_GENE_QUERY, "variables": {"names": [gene_symbol.upper()]}},
        )
        nodes = (data.get("data", {}).get("genes", {}) or {}).get("nodes", [])
        if not nodes:
            return {"gene": gene_symbol, "interactions": [], "note": "No DGIdb gene record found.", "source": "DGIdb"}
        interactions = nodes[0].get("interactions", []) or []
        rows = []
        for it in interactions:
            drug = it.get("drug") or {}
            rows.append(
                {
                    "drug": drug.get("name"),
                    "approved": drug.get("approved"),
                    "interaction_types": [t.get("type") for t in (it.get("interactionTypes") or []) if t.get("type")],
                    "interaction_score": round(it["interactionScore"], 3) if it.get("interactionScore") is not None else None,
                    "n_sources": len({s.get("sourceDbName") for s in (it.get("sources") or [])}),
                }
            )
        # Strongest, best-evidenced interactions first.
        rows.sort(key=lambda r: (r["interaction_score"] or 0, r["n_sources"]), reverse=True)
        return {
            "gene": gene_symbol,
            "n_interactions": len(rows),
            "interactions": rows[: int(limit)],
            "source": "DGIdb (drug-gene interaction database)",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── ChEMBL: mechanism of action, max clinical phase, indications ──────────────
def _chembl_molecule_id(drug_name: str) -> dict[str, Any] | None:
    """Resolve a drug name to its ChEMBL molecule record (preferring an exact
    preferred-name match, falling back to the free-text search index)."""
    exact = _get_json(
        f"{_CHEMBL}/molecule.json",
        params={"pref_name__iexact": drug_name, "limit": 1},
    )
    mols = exact.get("molecules", [])
    if not mols:
        srch = _get_json(f"{_CHEMBL}/molecule/search.json", params={"q": drug_name, "limit": 1})
        mols = srch.get("molecules", [])
    return mols[0] if mols else None


@lru_cache(maxsize=256)
def lookup_drug_mechanism(drug_name: str) -> dict[str, Any]:
    """Mechanism of action, molecular target, max clinical phase and approved
    indications for a drug, from ChEMBL. Biological/pharmacological context,
    NOT a GAUGE prediction."""
    try:
        mol = _chembl_molecule_id(drug_name)
        if not mol:
            return {"drug": drug_name, "note": "No ChEMBL molecule found.", "source": "ChEMBL"}
        chembl_id = mol.get("molecule_chembl_id")
        max_phase = mol.get("max_phase")
        # Mechanisms are attached to the parent compound, so a salt/child molecule
        # (e.g. erlotinib hydrochloride) returns nothing under molecule_chembl_id;
        # query parent_molecule_chembl_id and fall back to the molecule id.
        mech = _get_json(f"{_CHEMBL}/mechanism.json", params={"parent_molecule_chembl_id": chembl_id, "limit": 5})
        if not mech.get("mechanisms"):
            mech = _get_json(f"{_CHEMBL}/mechanism.json", params={"molecule_chembl_id": chembl_id, "limit": 5})
        mechanisms = [
            {
                "mechanism_of_action": m.get("mechanism_of_action"),
                "action_type": m.get("action_type"),
                "target_chembl_id": m.get("target_chembl_id"),
            }
            for m in mech.get("mechanisms", [])
        ]
        ind = _get_json(
            f"{_CHEMBL}/drug_indication.json",
            params={"molecule_chembl_id": chembl_id, "limit": 8},
        )
        indications = sorted(
            {i.get("efo_term") or i.get("mesh_heading") for i in ind.get("drug_indications", []) if (i.get("efo_term") or i.get("mesh_heading"))}
        )
        return {
            "drug": drug_name,
            "chembl_id": chembl_id,
            "max_clinical_phase": max_phase,
            "mechanisms_of_action": mechanisms,
            "indications": indications[:10],
            "source": "ChEMBL",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── ClinicalTrials.gov: relevant interventional trials ───────────────────────
@lru_cache(maxsize=256)
def search_clinical_trials(condition: str = "", intervention: str = "", limit: int = 6) -> dict[str, Any]:
    """Search ClinicalTrials.gov (API v2) for trials matching a condition and/or
    an intervention (drug). Returns citable NCT IDs, status and phase. Provide at
    least one of `condition` or `intervention`."""
    if not (condition or intervention):
        return {"error": "Provide at least one of `condition` or `intervention`.", "trials": []}
    params: dict[str, Any] = {
        "pageSize": int(limit),
        "format": "json",
        "sort": "LastUpdatePostDate:desc",
    }
    if condition:
        params["query.cond"] = condition
    if intervention:
        params["query.intr"] = intervention
    try:
        data = _get_json(_CLINICALTRIALS, params=params)
        trials = []
        for study in data.get("studies", []):
            ps = study.get("protocolSection", {})
            ident = ps.get("identificationModule", {})
            status = ps.get("statusModule", {})
            design = ps.get("designModule", {})
            cond = ps.get("conditionsModule", {})
            trials.append(
                {
                    "nct_id": ident.get("nctId"),
                    "title": ident.get("briefTitle"),
                    "status": status.get("overallStatus"),
                    "phases": design.get("phases"),
                    "conditions": (cond.get("conditions") or [])[:4],
                }
            )
        return {
            "condition": condition or None,
            "intervention": intervention or None,
            "count": len(trials),
            "trials": trials,
            "source": "ClinicalTrials.gov (API v2)",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── Reactome: pathway membership for a target ────────────────────────────────
@lru_cache(maxsize=256)
def lookup_pathways(gene_symbol: str, limit: int = 12) -> dict[str, Any]:
    """Reactome pathways a gene/protein participates in (mechanistic context for
    why a target matters). Resolves the gene to a UniProt accession first."""
    try:
        # 1. gene symbol -> reviewed human UniProt accession
        prot = _get_json(
            _UNIPROT,
            params={
                "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
                "fields": "accession",
                "format": "json",
                "size": 1,
            },
        )
        results = prot.get("results", [])
        if not results:
            return {"gene": gene_symbol, "pathways": [], "note": "No reviewed human UniProt entry found.", "source": "Reactome"}
        accession = results[0].get("primaryAccession")
        # 2. UniProt accession -> Reactome pathways (lowest-level pathways)
        resp = httpx.get(
            f"{_REACTOME}/data/mapping/UniProt/{accession}/pathways",
            params={"species": "9606"},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        if resp.status_code == 404:
            return {"gene": gene_symbol, "accession": accession, "pathways": [], "note": "No Reactome pathways mapped.", "source": "Reactome"}
        resp.raise_for_status()
        rows = resp.json()
        pathways = [
            {"stable_id": p.get("stId"), "name": p.get("displayName")}
            for p in rows
            if p.get("displayName")
        ]
        return {
            "gene": gene_symbol,
            "accession": accession,
            "n_pathways": len(pathways),
            "pathways": pathways[: int(limit)],
            "source": "Reactome",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# ── cBioPortal: pan-cancer mutation frequency ────────────────────────────────
@lru_cache(maxsize=256)
def lookup_cancer_mutations(gene_symbol: str) -> dict[str, Any]:
    """Pan-cancer somatic-mutation frequency for a gene across a large clinical
    cohort (MSK-IMPACT 2017, cBioPortal). Cancer-genomics context, NOT a GAUGE
    prediction."""
    try:
        gene = _get_json(f"{_CBIOPORTAL}/genes/{gene_symbol.upper()}")
        entrez = gene.get("entrezGeneId")
        if not entrez:
            return {"gene": gene_symbol, "note": "Gene not found in cBioPortal.", "source": "cBioPortal"}
        sample_list = _get_json(f"{_CBIOPORTAL}/sample-lists/{_CBIO_STUDY}_all")
        n_total = len(sample_list.get("sampleIds", [])) or sample_list.get("sampleCount")
        muts = _post_json(
            f"{_CBIOPORTAL}/molecular-profiles/{_CBIO_STUDY}_mutations/mutations/fetch?projection=SUMMARY",
            json={"entrezGeneIds": [entrez], "sampleListId": f"{_CBIO_STUDY}_all"},
        )
        mutated_samples = {m.get("sampleId") for m in muts if m.get("sampleId")}
        n_mut = len(mutated_samples)
        freq = round(100.0 * n_mut / n_total, 2) if n_total else None
        return {
            "gene": gene_symbol,
            "entrez_gene_id": entrez,
            "cohort": "MSK-IMPACT 2017 (pan-cancer clinical sequencing)",
            "n_samples": n_total,
            "n_mutated_samples": n_mut,
            "mutation_frequency_pct": freq,
            "source": "cBioPortal",
        }
    except Exception as exc:  # noqa: BLE001
        return _degraded(exc)


# Map tool name -> callable, mirroring the agent's TOOL_IMPLS convention.
EXTERNAL_TOOL_IMPLS: dict[str, Any] = {
    "lookup_target_disease_associations": lookup_target_disease_associations,
    "lookup_compound": lookup_compound,
    "lookup_protein": lookup_protein,
    "search_literature": search_literature,
    "lookup_drug_gene_interactions": lookup_drug_gene_interactions,
    "lookup_drug_mechanism": lookup_drug_mechanism,
    "search_clinical_trials": search_clinical_trials,
    "lookup_pathways": lookup_pathways,
    "lookup_cancer_mutations": lookup_cancer_mutations,
}

EXTERNAL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_target_disease_associations",
            "description": (
                "Look up which diseases a gene/protein target is associated with, and how strongly "
                "(OpenTargets evidence scores, 0-1). Use for 'is GENE a good target in DISEASE?' "
                "questions. Returns biological evidence, NOT a GAUGE drug-response prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {"gene_symbol": {"type": "string", "description": "HGNC gene symbol, e.g. EGFR"}},
                "required": ["gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_compound",
            "description": (
                "Resolve a drug name or SMILES to PubChem identity and chemical properties "
                "(formula, weight, canonical SMILES, XLogP). Use to identify/characterise a compound."
            ),
            "parameters": {
                "type": "object",
                "properties": {"name_or_smiles": {"type": "string", "description": "Drug name or a SMILES string"}},
                "required": ["name_or_smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_protein",
            "description": (
                "Get a human protein's molecular function and known disease involvement from UniProt, "
                "by gene symbol or UniProt accession. Use to explain target biology/mechanism."
            ),
            "parameters": {
                "type": "object",
                "properties": {"gene_or_uniprot": {"type": "string", "description": "Gene symbol (e.g. ERBB2) or UniProt accession"}},
                "required": ["gene_or_uniprot"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "Search the primary biomedical literature (Europe PMC) and return citable papers "
                "(title, authors, journal, year, PMID, DOI). Use to ground claims with references, "
                "especially in Deep Report mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search, e.g. 'EGFR inhibitor lung adenocarcinoma resistance'"},
                    "limit": {"type": "integer", "description": "How many papers to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_gene_interactions",
            "description": (
                "List drugs known to act on a gene/target (DGIdb aggregated drug-gene interactions, "
                "with interaction type and an evidence score). Use for 'what drugs could target GENE?' "
                "or to find druggable nodes behind a GAUGE prediction. Biological context, NOT a prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string", "description": "HGNC gene symbol, e.g. EGFR"},
                    "limit": {"type": "integer", "description": "How many interacting drugs to return (default 15)"},
                },
                "required": ["gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_mechanism",
            "description": (
                "Mechanism of action, molecular target, maximum clinical phase and approved indications "
                "for a drug, from ChEMBL. Use to characterise how a drug works and how far it has advanced "
                "clinically. Pharmacological context, NOT a GAUGE prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {"drug_name": {"type": "string", "description": "Drug name, e.g. Erlotinib"}},
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": (
                "Search ClinicalTrials.gov for interventional trials by condition and/or intervention (drug). "
                "Returns citable NCT IDs, recruitment status and trial phase. Use to ground drug-in-disease "
                "claims in real clinical activity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {"type": "string", "description": "Disease/condition, e.g. 'lung adenocarcinoma'"},
                    "intervention": {"type": "string", "description": "Drug/intervention, e.g. 'Erlotinib'"},
                    "limit": {"type": "integer", "description": "How many trials to return (default 6)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_pathways",
            "description": (
                "Reactome biological pathways a gene/protein participates in. Use to explain the mechanistic "
                "context of a target (signalling, metabolism, repair). Biological context, NOT a prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string", "description": "HGNC gene symbol, e.g. EGFR"},
                    "limit": {"type": "integer", "description": "How many pathways to return (default 12)"},
                },
                "required": ["gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_cancer_mutations",
            "description": (
                "Pan-cancer somatic-mutation frequency for a gene across a large clinical cohort "
                "(MSK-IMPACT 2017, via cBioPortal). Use to gauge how often a target is altered in tumours. "
                "Cancer-genomics context, NOT a GAUGE prediction."
            ),
            "parameters": {
                "type": "object",
                "properties": {"gene_symbol": {"type": "string", "description": "HGNC gene symbol, e.g. TP53"}},
                "required": ["gene_symbol"],
            },
        },
    },
]
