"""Tests for the optional external biomedical tools and the provider factory.

All tests here are network-free: httpx is monkeypatched to return canned JSON
(or to raise, to prove graceful degradation). This keeps the suite runnable
offline and without cost, mirroring the philosophy in test_agent.py.
"""
import httpx
import pytest

from gauge_core import bio_tools, providers


# ── external bio tools (network mocked) ──────────────────────────────────────
@pytest.fixture(autouse=True)
def _clear_caches():
    # bio_tools results are lru_cached; clear between tests so monkeypatches apply.
    for fn in bio_tools.EXTERNAL_TOOL_IMPLS.values():
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    yield


def test_lookup_compound_parses_pubchem(monkeypatch):
    payload = {"PropertyTable": {"Properties": [{
        "CID": 176870, "IUPACName": "erlotinib", "MolecularFormula": "C22H23N3O4",
        "MolecularWeight": "393.4", "CanonicalSMILES": "C#Cc1cccc(...)c1", "XLogP": 2.7,
    }]}}
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: payload)
    out = bio_tools.lookup_compound("Erlotinib")
    assert out["cid"] == 176870
    assert out["molecular_formula"] == "C22H23N3O4"
    assert out["source"].startswith("PubChem")


def test_search_literature_parses_europepmc(monkeypatch):
    payload = {"resultList": {"result": [
        {"title": "EGFR in NSCLC", "authorString": "Doe J", "journalTitle": "Cell",
         "pubYear": "2020", "pmid": "12345", "doi": "10.1/x", "citedByCount": 99},
    ]}}
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: payload)
    out = bio_tools.search_literature("EGFR lung", limit=3)
    assert out["count"] == 1
    assert out["papers"][0]["pmid"] == "12345"


def test_lookup_target_disease_associations_parses_opentargets(monkeypatch):
    search_resp = {"data": {"search": {"hits": [{"id": "ENSG00000146648", "name": "EGFR"}]}}}
    assoc_resp = {"data": {"target": {
        "approvedSymbol": "EGFR", "approvedName": "epidermal growth factor receptor",
        "associatedDiseases": {"count": 200, "rows": [
            {"score": 0.91, "disease": {"id": "EFO_0000305", "name": "carcinoma"}},
        ]},
    }}}
    calls = iter([search_resp, assoc_resp])
    monkeypatch.setattr(bio_tools, "_post_json", lambda url, json: next(calls))
    out = bio_tools.lookup_target_disease_associations("EGFR")
    assert out["approved_symbol"] == "EGFR"
    assert out["top_associations"][0]["association_score"] == 0.91


def test_lookup_drug_gene_interactions_parses_dgidb(monkeypatch):
    payload = {"data": {"genes": {"nodes": [{
        "name": "EGFR",
        "interactions": [
            {"drug": {"name": "AFATINIB", "conceptId": "x", "approved": True},
             "interactionScore": 1.2, "interactionTypes": [{"type": "inhibitor", "directionality": None}],
             "sources": [{"sourceDbName": "A"}, {"sourceDbName": "B"}]},
            {"drug": {"name": "FOO", "conceptId": "y", "approved": False},
             "interactionScore": 0.1, "interactionTypes": [], "sources": [{"sourceDbName": "A"}]},
        ],
    }]}}}
    monkeypatch.setattr(bio_tools, "_post_json", lambda url, json: payload)
    out = bio_tools.lookup_drug_gene_interactions("EGFR")
    assert out["n_interactions"] == 2
    # Highest interaction_score first.
    assert out["interactions"][0]["drug"] == "AFATINIB"
    assert out["interactions"][0]["n_sources"] == 2


def test_lookup_drug_mechanism_parses_chembl(monkeypatch):
    mol_resp = {"molecules": [{"molecule_chembl_id": "CHEMBL553", "max_phase": "4.0"}]}
    mech_resp = {"mechanisms": [
        {"mechanism_of_action": "EGFR inhibitor", "action_type": "INHIBITOR", "target_chembl_id": "CHEMBL203"},
    ]}
    ind_resp = {"drug_indications": [{"efo_term": "non-small cell lung carcinoma"}, {"mesh_heading": "Pancreatic Neoplasms"}]}
    calls = iter([mol_resp, mech_resp, ind_resp])
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: next(calls))
    out = bio_tools.lookup_drug_mechanism("Erlotinib")
    assert out["chembl_id"] == "CHEMBL553"
    assert out["mechanisms_of_action"][0]["action_type"] == "INHIBITOR"
    assert "non-small cell lung carcinoma" in out["indications"]


def test_search_clinical_trials_parses_v2(monkeypatch):
    payload = {"studies": [{"protocolSection": {
        "identificationModule": {"nctId": "NCT001", "briefTitle": "Erlotinib in NSCLC"},
        "statusModule": {"overallStatus": "COMPLETED"},
        "designModule": {"phases": ["PHASE3"]},
        "conditionsModule": {"conditions": ["Lung Adenocarcinoma"]},
    }}]}
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: payload)
    out = bio_tools.search_clinical_trials(condition="lung", intervention="Erlotinib")
    assert out["count"] == 1
    assert out["trials"][0]["nct_id"] == "NCT001"
    assert out["trials"][0]["phases"] == ["PHASE3"]


def test_search_clinical_trials_requires_a_query():
    out = bio_tools.search_clinical_trials()
    assert "error" in out and out["trials"] == []


def test_lookup_pathways_resolves_uniprot_then_reactome(monkeypatch):
    uniprot_resp = {"results": [{"primaryAccession": "P00533"}]}
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: uniprot_resp)

    class _Resp:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return [{"stId": "R-HSA-177929", "displayName": "Signaling by EGFR"}]

    monkeypatch.setattr(bio_tools.httpx, "get", lambda *a, **k: _Resp())
    out = bio_tools.lookup_pathways("EGFR")
    assert out["accession"] == "P00533"
    assert out["pathways"][0]["name"] == "Signaling by EGFR"


def test_lookup_cancer_mutations_computes_frequency(monkeypatch):
    gene_resp = {"entrezGeneId": 7157, "hugoGeneSymbol": "TP53"}
    samples_resp = {"sampleIds": [f"S{i}" for i in range(10)]}
    get_calls = iter([gene_resp, samples_resp])
    monkeypatch.setattr(bio_tools, "_get_json", lambda url, params=None: next(get_calls))
    muts = [{"sampleId": "S1"}, {"sampleId": "S2"}, {"sampleId": "S2"}]  # 2 unique mutated
    monkeypatch.setattr(bio_tools, "_post_json", lambda url, json: muts)
    out = bio_tools.lookup_cancer_mutations("TP53")
    assert out["entrez_gene_id"] == 7157
    assert out["n_samples"] == 10
    assert out["n_mutated_samples"] == 2
    assert out["mutation_frequency_pct"] == 20.0


def test_external_tool_degrades_gracefully_offline(monkeypatch):
    def _boom(*a, **k):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(bio_tools, "_get_json", _boom)
    out = bio_tools.lookup_compound("Erlotinib")
    assert "error" in out and "unavailable" in out["hint"]


def test_external_tool_schemas_and_impls_are_consistent():
    schema_names = {s["function"]["name"] for s in bio_tools.EXTERNAL_TOOL_SCHEMAS}
    impl_names = set(bio_tools.EXTERNAL_TOOL_IMPLS)
    assert schema_names == impl_names, schema_names ^ impl_names


# ── provider factory + fallback ──────────────────────────────────────────────
def test_get_provider_and_base_url():
    p = providers.get_provider("deepseek-chat")
    assert providers.provider_base_url(p) == "https://api.deepseek.com"
    assert providers.get_provider("deepseek-reasoner").is_reasoning is True


def test_build_client_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LookupError):
        providers.build_client(providers.get_provider("openai"), api_key=None)


def test_chat_completion_falls_back_to_second_provider(monkeypatch):
    primary = providers.get_provider("openai")
    fallback = providers.get_provider("deepseek-chat")

    class _BoomClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    raise RuntimeError("primary down")

    class _OKClient:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    return "OK-from-fallback"

    def _fake_build(provider, api_key=None):
        return (_BoomClient(), "m") if provider.name == "openai" else (_OKClient(), "m")

    monkeypatch.setattr(providers, "build_client", _fake_build)
    out = providers.chat_completion(primary, fallbacks=[fallback], messages=[], temperature=0)
    assert out == "OK-from-fallback"
