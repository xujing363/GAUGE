import numpy as np
import pytest

from gauge_core import load_bundle, predict_one, rank_drugs, resolve_drug, resolve_sample, score_combination
from gauge_core.predict import DrugNotFoundError, SampleResolutionError


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("gdsc_cell_split")


def test_bundle_loads(bundle):
    assert bundle.state_dim == 2003
    assert bundle.cell_state_matrix.shape[0] > 500
    assert bundle.drug_library.shape[0] > 100


def test_known_drug_resolution(bundle):
    drug_id = int(bundle.drug_library.iloc[0]["DRUG_ID"])
    drug = resolve_drug(bundle, drug_id)
    assert drug.known
    assert drug.drug_id == drug_id


def test_novel_smiles_resolution(bundle):
    drug = resolve_drug(bundle, "CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
    assert not drug.known
    assert drug.drug_id is None
    assert drug.fingerprint.shape == (2048,)


def test_invalid_drug_raises(bundle):
    with pytest.raises(DrugNotFoundError):
        resolve_drug(bundle, "definitely_not_a_real_drug_or_smiles_!!!")


def test_known_cell_line_resolution(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    sample = resolve_sample(bundle, cell_id)
    assert sample.known_cell_line
    assert sample.state_vector.shape == (bundle.state_dim,)


def test_unknown_cell_line_string_raises(bundle):
    with pytest.raises(SampleResolutionError):
        resolve_sample(bundle, "NOT_A_REAL_SANGER_ID")


def test_custom_expression_resolution(bundle):
    genes = bundle.artifacts.genes[:100]
    expr = {g: float(np.random.rand() * 10) for g in genes}
    sample = resolve_sample(bundle, expr)
    assert not sample.known_cell_line
    assert sample.state_vector.shape == (bundle.state_dim,)
    assert sample.n_genes_used == 100


def test_predict_one_known_known(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    drug_id = int(bundle.drug_library.iloc[0]["DRUG_ID"])
    result = predict_one(bundle, cell_id, drug_id)
    assert 0.0 <= result.value_hat <= 1.0
    assert result.kg_alpha is not None
    assert pytest.approx(sum(result.kg_alpha.values()), abs=1e-3) == 1.0


def test_predict_one_novel_drug_has_no_kg_attention(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    result = predict_one(bundle, cell_id, "CC(=O)OC1=CC=CC=C1C(=O)O")
    assert result.kg_alpha is None


def test_rank_drugs_subset(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    candidate_ids = bundle.drug_library["DRUG_ID"].astype(int).tolist()[:6]
    ranked = rank_drugs(bundle, cell_id, candidate_drug_ids=candidate_ids)
    assert len(ranked) == 6
    assert list(ranked["planner_rank"]) == sorted(ranked["planner_rank"])


def test_score_combination_modes(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    ids = bundle.drug_library["DRUG_ID"].astype(int).tolist()[:2]
    for mode in ("activity_product", "bliss", "complementarity"):
        out = score_combination(bundle, cell_id, ids[0], ids[1], mode=mode)
        assert np.isfinite(out["combination_score"])


# ── RTV must never drive a cross-drug comparison ──────────────────────────────

def test_absolute_activity_is_monotone_in_auc_and_unclipped():
    from gauge_core import absolute_activity

    # strictly decreasing in AUC, and NOT clipped: PRISM reports AUC > 1, and
    # clipping would tie every such drug at one value and destroy the ranking.
    assert float(absolute_activity(0.2)) > float(absolute_activity(0.9))
    assert float(absolute_activity(1.2)) < 0.0
    assert float(absolute_activity(2.5)) < float(absolute_activity(1.2))


def test_rank_drugs_orders_by_absolute_auc_not_rtv(bundle):
    cell_id = bundle.cell_state_matrix.index[0]
    ranked = rank_drugs(bundle, cell_id, lambda_u=0.0)
    # With no uncertainty penalty the planner order is exactly absolute AUC
    # ascending (lower AUC = more sensitive).
    assert ranked["auc_hat"].is_monotonic_increasing
    assert "absolute_activity" in ranked.columns
    # ...and it is genuinely a different order from ranking by RTV, i.e. this
    # is not a no-op rename.
    assert not ranked["value_hat"].is_monotonic_decreasing


def test_rank_candidates_refuses_to_score_without_absolute_auc():
    import pandas as pd
    import pytest as _pytest
    from drugwm.planner import rank_candidates

    frame = pd.DataFrame({"entity_id": ["a"], "value_hat": [0.9], "uncertainty": [0.1]})
    with _pytest.raises(KeyError, match="auc_hat"):
        rank_candidates(frame)


def test_combination_uses_absolute_activity_not_rtv(bundle):
    drug_ids = bundle.drug_library["DRUG_ID"].astype(int).tolist()[:2]
    cell_id = bundle.cell_state_matrix.index[0]
    out = score_combination(bundle, cell_id, drug_ids[0], drug_ids[1])
    assert "absolute_activity_a" in out and "absolute_activity_b" in out
    a = min(max(1.0 - out["auc_hat_a"], 0.0), 1.0)
    b = min(max(1.0 - out["auc_hat_b"], 0.0), 1.0)
    assert out["combination_score"] == pytest.approx(a + b - a * b)
