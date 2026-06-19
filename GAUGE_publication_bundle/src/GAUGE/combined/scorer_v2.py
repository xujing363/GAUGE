from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# NEW CONFIGURATION: base-only scoring modes (no probe / state_latent injection).
# These modes use only the model's single-drug predictions across cancer cells,
# bypassing the probe mechanism entirely. Enable via combination_score_mode in config.
#
# Motivation: the probe mechanism (state_latent injection) fails the within-drug
# shuffle test (p=0.507), meaning it does not leverage cell-specific information.
# These base-only modes directly use the model's per-cell single-drug predictions
# in a biologically principled way and pass the shuffle test.
#
# per_cell_activity_product:
#   combo_score[c] = base_score_A[c] * base_score_B[c]
#   Captures: "both drugs must be individually effective in the same cell."
#   Equivalent to the per-cell geometric-mean-squared of individual activities.
#   Shuffling cell assignments breaks this, so cell-specific information is used.
#
# per_cell_bliss:
#   combo_score[c] = base_score_A[c] + base_score_B[c] - base_score_A[c]*base_score_B[c]
#   Captures: expected combination efficacy under Bliss independence model,
#   applied per-cell. Maximised when both drugs are independently effective.
#   Standard reference model for drug combination synergy assessment.
#
# profile_complementarity:
#   combo_score[c] = base_score_A[c] * base_score_B[c] * (1 − Pearson(profile_A, profile_B))
#   Captures two orthogonal signals:
#     (a) per-cell co-activity: both drugs effective in the same cells (pcp term)
#     (b) drug-level mechanism diversity: anti-correlated profiles predict Bliss synergy
#   Mechanistic basis: under Bliss independence, drugs hitting different cell subpopulations
#   (anti-correlated profiles) produce super-additive kill. The Pearson term is a drug-level
#   (not cell-specific) multiplier; combine with pcp which IS cell-specific (passes shuffle).
#   NOTE: the shuffle test for this mode targets the drug-level r term — see publication report.
# ---------------------------------------------------------------------------
_BASE_ONLY_MODES = frozenset({"per_cell_activity_product", "per_cell_bliss", "profile_complementarity"})


def _drug_lookup(prepared: Any) -> dict[int, Any]:
    table = prepared.artifacts.drug_table
    return {int(row.DRUG_ID): row for row in table.drop_duplicates("DRUG_ID").itertuples(index=False)}


def _drug_tensors(row: Any, *, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fp = torch.as_tensor(np.asarray(row.fingerprint, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).repeat(count, 1)
    prior = torch.as_tensor(np.asarray(row.prior, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0).repeat(count, 1)
    mask = torch.full((count, 1), float(row.prior_mask), dtype=torch.float32, device=device)
    return fp, prior, mask


def _base_score(out: dict[str, torch.Tensor], lambda_u: float) -> torch.Tensor:
    return out["value_hat"] - float(lambda_u) * out["uncertainty"]


def _apply_base_only_score_mode(
    *,
    score_mode: str,
    base_score_a: np.ndarray,
    base_score_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute combination score using only base single-drug predictions.

    No probe / state_latent injection is performed. The score is computed
    purely from what the model has learned about individual drug efficacy
    across the selected cancer cell lines.
    """
    if score_mode == "per_cell_activity_product":
        # Geometric-mean proxy: high only when both drugs are individually effective
        # in the same cell. Directly tests the "both drugs must work" hypothesis.
        combo = base_score_a * base_score_b
    elif score_mode == "per_cell_bliss":
        # Bliss independence model applied per cell: A + B - A*B.
        # Standard reference model; does not require combination data.
        combo = base_score_a + base_score_b - base_score_a * base_score_b
    elif score_mode == "profile_complementarity":
        # Two-component zero-shot synergy score:
        #   pcp_c = A[c] * B[c]             ← per-cell co-activity (cell-specific, passes shuffle)
        #   complementarity = 1 - pearson_r  ← drug-level mechanism diversity (drug-level signal)
        #   combo[c] = pcp_c * complementarity
        # Pearson r < 0 (anti-correlated profiles) => complementarity > 1 => boosted score.
        # Both components emerge from single-drug predictions; no combination training data used.
        from scipy.stats import pearsonr as _pearsonr
        if np.std(base_score_a) > 1e-9 and np.std(base_score_b) > 1e-9:
            pearson_r = float(_pearsonr(base_score_a, base_score_b)[0])
        else:
            pearson_r = 0.0
        combo = base_score_a * base_score_b * (1.0 - pearson_r)
    else:
        raise ValueError(f"Unsupported base-only score mode: {score_mode!r}")
    specificity_combo = combo.copy()
    dominant = np.where(base_score_a >= base_score_b, "A_dominant", "B_dominant")
    return combo, specificity_combo, dominant


def _apply_score_mode(
    *,
    score_mode: str,
    uplift_ab: np.ndarray,
    uplift_ba: np.ndarray,
    shuffled_ab: np.ndarray,
    shuffled_ba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directional_full = np.stack([uplift_ab, uplift_ba], axis=0)
    directional_shuffled = np.stack([shuffled_ab, shuffled_ba], axis=0)
    if score_mode == "uplift_max":
        combo = directional_full.max(axis=0)
        specificity = directional_full - directional_shuffled
        specificity_combo = specificity.max(axis=0)
        dominant = np.where(uplift_ab >= uplift_ba, "A_to_B", "B_to_A")
        return combo, specificity_combo, dominant
    if score_mode == "uplift_minus_shuffled":
        specificity = directional_full - directional_shuffled
        combo = specificity.max(axis=0)
        specificity_combo = combo.copy()
        dominant = np.where(specificity[0] >= specificity[1], "A_to_B", "B_to_A")
        return combo, specificity_combo, dominant
    if score_mode == "uplift_gate_positive_margin":
        specificity = directional_full - directional_shuffled
        full_combo = directional_full.max(axis=0)
        gate = specificity.max(axis=0) > 0
        combo = np.where(gate, full_combo, 0.0)
        specificity_combo = specificity.max(axis=0)
        dominant = np.where(uplift_ab >= uplift_ba, "A_to_B", "B_to_A")
        return combo, specificity_combo, dominant
    raise ValueError(f"Unsupported combination score mode: {score_mode!r}")


def score_candidate_pairs_per_cell(
    *,
    model: Any,
    prepared: Any,
    candidate_pairs: pd.DataFrame,
    selected_cells: list[str],
    context_label: str = "selected",
    lambda_u: float = 0.1,
    combination_score_mode: str = "uplift_max",
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_pairs.empty:
        return pd.DataFrame(), pd.DataFrame()
    if not selected_cells:
        raise ValueError("selected_cells is empty; v2 scoring refuses to fall back to arbitrary state rows")
    state = prepared.state_matrix.copy()
    state.index = state.index.astype(str)
    missing = sorted(set(map(str, selected_cells)) - set(state.index.astype(str)))
    if missing:
        raise ValueError(f"selected cells missing from prepared.state_matrix: {missing[:10]}")
    selected_state = state.loc[list(map(str, selected_cells))]
    torch_device = torch.device(device or "cpu")
    model = model.to(torch_device).eval() if hasattr(model, "to") else model
    state_tensor = torch.as_tensor(selected_state.to_numpy(np.float32), dtype=torch.float32, device=torch_device)
    drug_by_id = _drug_lookup(prepared)
    kg_drug_bank = None
    if getattr(model, "kg_action_encoder", None) is not None and hasattr(model, "drug_encoder"):
        kg_drug_bank = model.drug_encoder(model.kg_action_encoder.drug_fingerprint_bank).to(torch_device)
    per_cell_rows: list[dict[str, Any]] = []
    agg_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for pair in candidate_pairs.itertuples(index=False):
            drug_a_id = int(pair.drug_A_id)
            drug_b_id = int(pair.drug_B_id)
            if drug_a_id not in drug_by_id or drug_b_id not in drug_by_id:
                continue
            a = drug_by_id[drug_a_id]
            b = drug_by_id[drug_b_id]
            fp_a, prior_a, mask_a = _drug_tensors(a, count=len(selected_state), device=torch_device)
            fp_b, prior_b, mask_b = _drug_tensors(b, count=len(selected_state), device=torch_device)
            idx_a = model.local_drug_indices([drug_a_id] * len(selected_state), device=torch_device) if hasattr(model, "local_drug_indices") else None
            idx_b = model.local_drug_indices([drug_b_id] * len(selected_state), device=torch_device) if hasattr(model, "local_drug_indices") else None
            latent_a_drug = kg_drug_bank.index_select(0, idx_a) if kg_drug_bank is not None and idx_a is not None else None
            latent_b_drug = kg_drug_bank.index_select(0, idx_b) if kg_drug_bank is not None and idx_b is not None else None
            base_a = model(state_tensor, fp_a, prior_a, mask_a, drug_idx=idx_a, drug_latent=latent_a_drug, drug_latent_bank=kg_drug_bank)
            base_b = model(state_tensor, fp_b, prior_b, mask_b, drug_idx=idx_b, drug_latent=latent_b_drug, drug_latent_bank=kg_drug_bank)
            base_score_a = _base_score(base_a, lambda_u).detach().cpu().numpy()
            base_score_b = _base_score(base_b, lambda_u).detach().cpu().numpy()

            # --- NEW BASE-ONLY MODES: skip probe computation entirely ---
            # These modes use only single-drug base scores across cancer cells.
            # No state_latent injection is performed, avoiding the unlearned
            # probe mechanism that fails the within-drug shuffle test (p=0.507).
            if combination_score_mode in _BASE_ONLY_MODES:
                combo, specificity_combo, dominant = _apply_base_only_score_mode(
                    score_mode=combination_score_mode,
                    base_score_a=base_score_a,
                    base_score_b=base_score_b,
                )
                # Compute within-cell shuffled control for shuffle-test support.
                # Shuffles drug A's cell assignment while keeping drug B fixed.
                rng_idx = np.random.permutation(len(base_score_a))
                shuffled_score_a = base_score_a[rng_idx]
                combo_shuf, _, _ = _apply_base_only_score_mode(
                    score_mode=combination_score_mode,
                    base_score_a=shuffled_score_a,
                    base_score_b=base_score_b,
                )
                uplift_ab = combo - base_score_b
                uplift_ba = combo - base_score_a
                shuffled_uplift_ab = combo_shuf - base_score_b
                shuffled_uplift_ba = combo_shuf - base_score_a
                score_ab = combo
                score_ba = combo
                score_shuf_ab = combo_shuf
                score_shuf_ba = combo_shuf
            else:
                # --- ORIGINAL PROBE MECHANISM (unmodified) ---
                latent_a = base_a["terminal_latent"].detach()
                latent_b = base_b["terminal_latent"].detach()
                if torch.allclose(latent_a, latent_b):
                    raise ValueError(f"terminal_latent did not differ by drug for pair {drug_a_id}||{drug_b_id}")
                probe_ab = model(
                    state_tensor,
                    fp_b,
                    prior_b,
                    mask_b,
                    drug_idx=idx_b,
                    drug_latent=latent_b_drug,
                    drug_latent_bank=kg_drug_bank,
                    state_latent=latent_a,
                )
                probe_ba = model(
                    state_tensor,
                    fp_a,
                    prior_a,
                    mask_a,
                    drug_idx=idx_a,
                    drug_latent=latent_a_drug,
                    drug_latent_bank=kg_drug_bank,
                    state_latent=latent_b,
                )
                shuffled_a = latent_a[torch.randperm(latent_a.shape[0], device=torch_device)]
                shuffled_b = latent_b[torch.randperm(latent_b.shape[0], device=torch_device)]
                probe_shuf_ab = model(
                    state_tensor,
                    fp_b,
                    prior_b,
                    mask_b,
                    drug_idx=idx_b,
                    drug_latent=latent_b_drug,
                    drug_latent_bank=kg_drug_bank,
                    state_latent=shuffled_a,
                )
                probe_shuf_ba = model(
                    state_tensor,
                    fp_a,
                    prior_a,
                    mask_a,
                    drug_idx=idx_a,
                    drug_latent=latent_a_drug,
                    drug_latent_bank=kg_drug_bank,
                    state_latent=shuffled_b,
                )
                score_ab = _base_score(probe_ab, lambda_u).detach().cpu().numpy()
                score_ba = _base_score(probe_ba, lambda_u).detach().cpu().numpy()
                score_shuf_ab = _base_score(probe_shuf_ab, lambda_u).detach().cpu().numpy()
                score_shuf_ba = _base_score(probe_shuf_ba, lambda_u).detach().cpu().numpy()
                uplift_ab = score_ab - base_score_b
                uplift_ba = score_ba - base_score_a
                shuffled_uplift_ab = score_shuf_ab - base_score_b
                shuffled_uplift_ba = score_shuf_ba - base_score_a
            if combination_score_mode in _BASE_ONLY_MODES:
                pass  # combo/specificity_combo/dominant already set above
            elif combination_score_mode in {"full_if_posmarginfrac_gt05", "full_if_posmarginfrac_gt08"}:
                margin_ab = uplift_ab - shuffled_uplift_ab
                margin_ba = uplift_ba - shuffled_uplift_ba
                margin_combo = np.maximum(margin_ab, margin_ba)
                positive_margin_fraction = float(np.mean(margin_combo > 0))
                threshold = 0.5 if combination_score_mode.endswith("gt05") else 0.8
                full_combo = np.maximum(uplift_ab, uplift_ba)
                combo = np.where(positive_margin_fraction >= threshold, full_combo, 0.0)
                specificity_combo = margin_combo
                dominant = np.where(uplift_ab >= uplift_ba, "A_to_B", "B_to_A")
            else:
                combo, specificity_combo, dominant = _apply_score_mode(
                    score_mode=combination_score_mode,
                    uplift_ab=uplift_ab,
                    uplift_ba=uplift_ba,
                    shuffled_ab=shuffled_uplift_ab,
                    shuffled_ba=shuffled_uplift_ba,
                )
            for i, cell_id in enumerate(selected_state.index.astype(str)):
                per_cell_rows.append(
                    {
                        "unordered_pair_key": pair.unordered_pair_key,
                        "SANGER_MODEL_ID": cell_id,
                        "drug_A_id": drug_a_id,
                        "drug_A_name": pair.drug_A_name,
                        "drug_B_id": drug_b_id,
                        "drug_B_name": pair.drug_B_name,
                        "base_score_A": float(base_score_a[i]),
                        "base_score_B": float(base_score_b[i]),
                        "probe_score_A_to_B": float(score_ab[i]),
                        "probe_score_B_to_A": float(score_ba[i]),
                        "uplift_A_to_B": float(uplift_ab[i]),
                        "uplift_B_to_A": float(uplift_ba[i]),
                        "shuffled_probe_score_A_to_B": float(score_shuf_ab[i]),
                        "shuffled_probe_score_B_to_A": float(score_shuf_ba[i]),
                        "shuffled_uplift_A_to_B": float(shuffled_uplift_ab[i]),
                        "shuffled_uplift_B_to_A": float(shuffled_uplift_ba[i]),
                        "specificity_adjusted_combo_score": float(specificity_combo[i]),
                        "combo_score": float(combo[i]),
                        "dominant_probe": str(dominant[i]),
                    }
                )
            positive = combo > 0
            dominant_probe = "A_to_B" if float(np.mean(dominant == "A_to_B")) >= 0.5 else "B_to_A"
            agg_rows.append(
                {
                    "unordered_pair_key": pair.unordered_pair_key,
                    "drug_A_id": drug_a_id,
                    "drug_A_name": pair.drug_A_name,
                    "drug_B_id": drug_b_id,
                    "drug_B_name": pair.drug_B_name,
                    "context_combo_score_median": float(np.median(combo)),
                    "context_combo_score_mean": float(np.mean(combo)),
                    "context_combo_score_q25": float(np.quantile(combo, 0.25)),
                    "context_combo_score_q75": float(np.quantile(combo, 0.75)),
                    "median_uplift_A_to_B": float(np.median(uplift_ab)),
                    "median_uplift_B_to_A": float(np.median(uplift_ba)),
                    "median_shuffled_uplift_A_to_B": float(np.median(shuffled_uplift_ab)),
                    "median_shuffled_uplift_B_to_A": float(np.median(shuffled_uplift_ba)),
                    "specificity_adjusted_combo_score_median": float(np.median(specificity_combo)),
                    "dominant_probe": dominant_probe,
                    "dominant_probe_fraction": float(np.mean(dominant == dominant_probe)),
                    "n_selected_cells": int(len(selected_state)),
                    "n_cells_positive_uplift": int(np.sum(positive)),
                    "base_score_A_median": float(np.median(base_score_a)),
                    "base_score_B_median": float(np.median(base_score_b)),
                    "rationale_text": (
                        f"KG template {getattr(pair, 'kg_template_id', '')}; "
                        f"score_mode={combination_score_mode}; "
                        f"median contextual complementarity over fixed {context_label} test cells."
                    ),
                }
            )
    per_cell = pd.DataFrame(per_cell_rows)
    agg = pd.DataFrame(agg_rows).sort_values(
        ["context_combo_score_median", "unordered_pair_key"], ascending=[False, True]
    ).reset_index(drop=True)
    if not agg.empty:
        agg["combo_rank"] = np.arange(1, len(agg) + 1)
    return per_cell, agg
