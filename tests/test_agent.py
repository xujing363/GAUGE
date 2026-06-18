"""Tests for the GAUGE Assistant (LLM tool-using agent).

Tool-implementation tests run network-free (they call straight into
gauge_core, exactly like the agent does internally). The live round-trip
tests actually call the configured LLM API and are skipped automatically
if no key is configured, so the suite stays runnable without network/cost
by default while still being exercised whenever a key is present (as it is
in this repository's `.env`).
"""
import os

import pytest

from gauge_core import load_bundle
from gauge_core.agent import (
    AgentNotConfiguredError,
    GaugeAgent,
    _tool_combo,
    _tool_predict,
    _tool_rank,
    _tool_search_cells,
    _tool_search_drugs,
)
from gauge_core.kg_tools import explain_prediction, kg_neighborhood
from gauge_core._env import load_dotenv

load_dotenv()

HAS_KEY = bool(os.environ.get("DEEPSEEK_API_KEY"))


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("gdsc_cell_split")


def test_tool_predict_known_pair(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    drug_id = int(bundle.drug_library.iloc[0]["DRUG_ID"])
    out = _tool_predict(bundle, {}, cell_id, drug_id)
    assert "error" not in out
    assert 0.0 <= out["relative_sensitive_value"] <= 1.0
    assert out["kg_source_attention"] is not None


def test_tool_predict_unknown_cell_line_returns_error_not_raise(bundle):
    out = _tool_predict(bundle, {}, "NOT_A_REAL_ID", "Cisplatin")
    assert "error" in out


def test_tool_predict_with_uploaded_sample(bundle):
    import pandas as pd
    cell_id = bundle.cell_state_matrix.index[0]
    state_row = bundle.cell_state_matrix.loc[cell_id]
    genes = bundle.artifacts.genes
    gene_int_cols = [c for c in bundle.cell_state_matrix.columns if str(c).isdigit()][:2000]
    expr = pd.Series({genes[int(c)]: float(state_row[c]) for c in gene_int_cols}, dtype="float32")
    uploaded = {"my_sample": expr}
    out = _tool_predict(bundle, uploaded, "my_sample", bundle.drug_library.iloc[0]["DRUG_NAME"])
    assert "error" not in out
    assert out["sample_is_uploaded"] is True


def test_tool_rank(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    out = _tool_rank(bundle, {}, cell_id, top_k=5)
    assert len(out["ranked_drugs"]) == 5


def test_tool_combo(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    ids = bundle.drug_library["DRUG_NAME"].tolist()[:2]
    out = _tool_combo(bundle, {}, cell_id, ids[0], ids[1], mode="bliss")
    assert "error" not in out
    assert isinstance(out["combination_score"], float)


def test_tool_search_drugs(bundle):
    out = _tool_search_drugs(bundle, {}, "plat")
    assert isinstance(out["matches"], list)
    assert any("lat" in m["drug_name"].lower() for m in out["matches"])


def test_tool_search_cells(bundle):
    out = _tool_search_cells(bundle, {}, "lung")
    assert isinstance(out["matches"], list)


def test_agent_not_configured_without_key(bundle, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(AgentNotConfiguredError):
        GaugeAgent(bundle, api_key="")


def test_reasoning_model_rejected_as_chat_provider(bundle):
    # Reasoning models can't do tool calling, so they must not be usable as the
    # chat/primary provider (this is the bug behind the Deep Report + external error).
    with pytest.raises(AgentNotConfiguredError):
        GaugeAgent(bundle, provider="deepseek-reasoner", api_key="sk-dummy")


# ── new local tools (network-free) ───────────────────────────────────────────
def test_explain_prediction_local_tool(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    drug_name = bundle.drug_library.iloc[0]["DRUG_NAME"]
    out = explain_prediction(bundle, {}, cell_id, drug_name)
    assert "error" not in out
    assert 0.0 <= out["relative_sensitive_value"] <= 1.0
    # dominant KG is one of the three (or None when the drug lacks KG coverage)
    assert out["dominant_knowledge_graph"] in {"ChEMBL", "DRKG", "PrimeKG", None}


def test_kg_neighborhood_local_tool(bundle):
    drug_name = bundle.drug_library.iloc[0]["DRUG_NAME"]
    out = kg_neighborhood(bundle, {}, drug_name)
    assert "error" not in out
    assert "kg_coverage" in out
    assert isinstance(out["edges"], list)


# ── tool-registry gating on enable_external ──────────────────────────────────
@pytest.mark.skipif(not HAS_KEY, reason="DEEPSEEK_API_KEY not configured")
def test_external_tools_hidden_unless_enabled(bundle):
    base = GaugeAgent(bundle)
    names = {s["function"]["name"] for s in base._tool_schemas()}
    assert "predict_drug_response" in names
    assert "explain_prediction" in names
    assert "lookup_target_disease_associations" not in names  # gated off by default

    ext = GaugeAgent(bundle, enable_external=True)
    ext_names = {s["function"]["name"] for s in ext._tool_schemas()}
    assert "lookup_target_disease_associations" in ext_names
    assert "search_literature" in ext_names
    # New drug-sensitivity external tools are registered when external is enabled.
    for name in (
        "lookup_drug_gene_interactions",
        "lookup_drug_mechanism",
        "search_clinical_trials",
        "lookup_pathways",
        "lookup_cancer_mutations",
    ):
        assert name in ext_names


@pytest.mark.skipif(not HAS_KEY, reason="DEEPSEEK_API_KEY not configured")
def test_agent_live_round_trip(bundle):
    agent = GaugeAgent(bundle)
    cell_id = bundle.cell_state_matrix.index[0]
    drug_name = bundle.drug_library.iloc[0]["DRUG_NAME"]
    result = agent.run_turn([{"role": "user", "content": f"What does GAUGE predict for {cell_id} with {drug_name}?"}])
    assert result.reply
    assert any(tc.name == "predict_drug_response" for tc in result.tool_calls)


@pytest.mark.skipif(not HAS_KEY, reason="DEEPSEEK_API_KEY not configured")
def test_agent_report_mode_round_trip(bundle):
    # Deep report mode: gather (chat model) -> synthesise (reasoner). Local tools only.
    agent = GaugeAgent(bundle)
    cell_id = bundle.cell_state_matrix.index[0]
    drug_name = bundle.drug_library.iloc[0]["DRUG_NAME"]
    result = agent.run_report(
        [{"role": "user", "content": f"Write a short report on {drug_name} for {cell_id}."}]
    )
    assert result.reply and len(result.reply) > 100
    assert any(tc.name in {"predict_drug_response", "explain_prediction"} for tc in result.tool_calls)
