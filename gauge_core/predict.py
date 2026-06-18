from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import _drugwm_path  # noqa: F401  must precede numpy/pandas/torch (see module docstring)

import numpy as np
import pandas as pd
import torch

from drugwm.features import morgan_fp, project_expression  # noqa: E402
from drugwm.planner import rank_candidates  # noqa: E402

from .bundle import ModelBundle

KG_BRANCH_NAMES = ("ChEMBL", "DRKG", "PrimeKG")


class DrugNotFoundError(ValueError):
    pass


class SampleResolutionError(ValueError):
    pass


@dataclass
class ResolvedDrug:
    drug_id: int | None
    name: str
    smiles: str
    fingerprint: np.ndarray
    known: bool
    has_kg_routing: bool
    kg_coverage: dict[str, bool] = field(default_factory=dict)


@dataclass
class ResolvedSample:
    label: str
    state_vector: np.ndarray
    known_cell_line: bool
    sanger_id: str | None = None
    gene_coverage: float | None = None
    n_genes_used: int | None = None
    n_genes_total: int | None = None


@dataclass
class PredictionResult:
    auc_hat: float
    value_hat: float
    uncertainty: float
    raw_auc_base: float
    cell_residual_hat: float
    kg_alpha: dict[str, float] | None
    gate_strength: float | None
    drug: ResolvedDrug
    sample: ResolvedSample
    percentile_text: str | None = None


def resolve_drug(bundle: ModelBundle, query: str | int) -> ResolvedDrug:
    """Look up a drug by ID or name in the bundle's drug library (GDSC or
    PRISM, depending on the loaded bundle), falling back to treating the
    query as a raw SMILES string for an arbitrary compound."""
    lib = bundle.drug_library
    query_str = str(query).strip()

    row = None
    if query_str.lstrip("-").isdigit():
        match = lib.loc[lib["DRUG_ID"].astype(int) == int(query_str)]
        if not match.empty:
            row = match.iloc[0]
    if row is None:
        name_match = lib.loc[lib["DRUG_NAME"].astype(str).str.lower() == query_str.lower()]
        if not name_match.empty:
            row = name_match.iloc[0]
    if row is None and "drug_key" in lib.columns:
        key_match = lib.loc[lib["drug_key"].astype(str).str.lower() == query_str.lower()]
        if not key_match.empty:
            row = key_match.iloc[0]

    if row is not None:
        drug_id = int(row["DRUG_ID"])
        smiles = str(row.get("canonical_smiles") or row.get("smiles"))
        fp = morgan_fp(smiles)
        if fp is None:
            raise DrugNotFoundError(f"Library entry for {query!r} has an unparseable SMILES string.")
        coverage = {src: bool(row.get(f"has_{src}", False)) for src in KG_BRANCH_NAMES if f"has_{src}" in row.index}
        return ResolvedDrug(
            drug_id=drug_id,
            name=str(row["DRUG_NAME"]),
            smiles=smiles,
            fingerprint=fp,
            known=True,
            has_kg_routing=bundle.has_kg_routing(drug_id),
            kg_coverage=coverage,
        )

    # Not in the library: treat the query as a raw SMILES string for a novel compound.
    fp = morgan_fp(query_str)
    if fp is None:
        raise DrugNotFoundError(
            f"{query!r} is not a recognised GAUGE library drug name/ID and is not a valid SMILES string."
        )
    return ResolvedDrug(
        drug_id=None,
        name=query_str,
        smiles=query_str,
        fingerprint=fp,
        known=False,
        has_kg_routing=False,
        kg_coverage={},
    )


def resolve_sample(bundle: ModelBundle, sample: str | pd.Series | dict[str, float]) -> ResolvedSample:
    """Resolve a sample into a model-ready state vector.

    `sample` is either:
      - a known SANGER_MODEL_ID string (exact replay of the bundled, fitted state), or
      - a mapping of gene symbol -> expression value for a brand-new sample
        (a patient biopsy, a new cell line, etc.), which is projected through
        the bundle's already-fitted gene selection/imputer/scaler.
    """
    if isinstance(sample, str):
        if sample in bundle.cell_state_matrix.index:
            vector = bundle.cell_state_matrix.loc[sample].to_numpy(dtype=np.float32)
            return ResolvedSample(label=sample, state_vector=vector, known_cell_line=True, sanger_id=sample)
        raise SampleResolutionError(
            f"{sample!r} is not a known cell line in this model bundle ({bundle.mode}). "
            "Pass a gene-expression mapping instead to score a new sample."
        )

    expr = pd.Series(sample, dtype=np.float32)
    expr.index = expr.index.astype(str)
    genes = bundle.artifacts.genes
    n_present = int(expr.index.isin(genes).sum())
    coverage = n_present / max(len(genes), 1)
    expr_df = pd.DataFrame([expr])
    projected = project_expression(expr_df, genes, bundle.artifacts.imputer, bundle.artifacts.scaler, bundle.artifacts.pca)
    extra_stats = np.array(
        [
            bundle.global_auc_stats["global_auc_train_mean"],
            bundle.global_auc_stats["global_auc_train_median"],
            0.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate([projected[0].astype(np.float32), extra_stats])
    return ResolvedSample(
        label="custom sample",
        state_vector=vector,
        known_cell_line=False,
        gene_coverage=coverage,
        n_genes_used=n_present,
        n_genes_total=len(genes),
    )


def _forward(bundle: ModelBundle, state_vector: np.ndarray, drug: ResolvedDrug) -> dict[str, torch.Tensor]:
    model = bundle.model
    state_t = torch.tensor(state_vector, dtype=torch.float32, device=bundle.device).unsqueeze(0)
    fp_t = torch.tensor(drug.fingerprint, dtype=torch.float32, device=bundle.device).unsqueeze(0)
    use_kg = drug.has_kg_routing and bundle.precomputed_kg_payload is not None
    with torch.no_grad():
        if use_kg:
            drug_idx_t = model.local_drug_indices([drug.drug_id], device=bundle.device)
            out = model(
                state_t,
                fp_t,
                drug_idx=drug_idx_t,
                use_prior=True,
                use_terminal=True,
                precomputed_kg_payload=bundle.precomputed_kg_payload,
                fusion_weight=bundle.fusion_weight,
                return_explanations=True,
                explanation_level="source",
            )
        else:
            out = model(
                state_t,
                fp_t,
                drug_idx=None,
                use_prior=False,
                use_terminal=True,
                fusion_weight=bundle.fusion_weight,
            )
    return out


def _percentile_text(bundle: ModelBundle, drug: ResolvedDrug, auc_hat: float) -> str | None:
    if drug.drug_id is None:
        return None
    values = bundle.artifacts.drug_auc_train_values.get(int(drug.drug_id))
    if values is None or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    rank = float(np.searchsorted(arr, auc_hat, side="right") / len(arr))
    more_sensitive_than_pct = (1.0 - rank) * 100.0
    return (
        f"Predicted AUC ranks more sensitive than {more_sensitive_than_pct:.0f}% of the "
        f"{len(arr)} training cell lines profiled against {drug.name} (lower AUC = more sensitive)."
    )


def predict_one(bundle: ModelBundle, sample: str | pd.Series | dict[str, float], drug: str | int) -> PredictionResult:
    resolved_sample = resolve_sample(bundle, sample)
    resolved_drug = resolve_drug(bundle, drug)
    out = _forward(bundle, resolved_sample.state_vector, resolved_drug)

    kg_alpha = None
    gate_strength = None
    if "kg_alpha" in out:
        alpha = out["kg_alpha"].squeeze(0).cpu().numpy()
        kg_alpha = {name: float(w) for name, w in zip(KG_BRANCH_NAMES, alpha)}
    if "gate" in out:
        gate_strength = float(out["gate"].squeeze(0).abs().mean().item())

    auc_hat = float(out["auc_hat"].item())
    return PredictionResult(
        auc_hat=auc_hat,
        value_hat=float(out["value_hat"].item()),
        uncertainty=float(out["uncertainty"].item()),
        raw_auc_base=float(out["raw_auc_base"].item()),
        cell_residual_hat=float(out["cell_residual_hat"].item()),
        kg_alpha=kg_alpha,
        gate_strength=gate_strength,
        drug=resolved_drug,
        sample=resolved_sample,
        percentile_text=_percentile_text(bundle, resolved_drug, auc_hat),
    )


def rank_drugs(
    bundle: ModelBundle,
    sample: str | pd.Series | dict[str, float],
    candidate_drug_ids: list[int] | None = None,
    lambda_u: float = 0.1,
) -> pd.DataFrame:
    """Rank all (or a chosen subset of) bundled library drugs for one sample."""
    resolved_sample = resolve_sample(bundle, sample)
    lib = bundle.drug_library
    if candidate_drug_ids is not None:
        lib = lib.loc[lib["DRUG_ID"].astype(int).isin([int(x) for x in candidate_drug_ids])]
    rows = []
    for _, lib_row in lib.iterrows():
        drug = resolve_drug(bundle, int(lib_row["DRUG_ID"]))
        out = _forward(bundle, resolved_sample.state_vector, drug)
        rows.append(
            {
                "entity_id": resolved_sample.label,
                "DRUG_ID": drug.drug_id,
                "DRUG_NAME": drug.name,
                "auc_hat": float(out["auc_hat"].item()),
                "value_hat": float(out["value_hat"].item()),
                "uncertainty": float(out["uncertainty"].item()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return rank_candidates(frame, lambda_u=lambda_u).reset_index(drop=True)


def score_combination(
    bundle: ModelBundle,
    sample: str | pd.Series | dict[str, float],
    drug_a: str | int,
    drug_b: str | int,
    mode: str = "bliss",
) -> dict[str, Any]:
    """Score a two-drug combination from two independently-trained single-agent
    predictions (GAUGE was never trained on combination labels). Modes mirror
    the combination scoring used for the NCI-ALMANAC analysis in the paper:
      - "activity_product": A_eff * B_eff
      - "bliss": A_eff + B_eff - A_eff * B_eff  (Bliss independence)
      - "complementarity": A_eff * B_eff * (1 - corr-free proxy)
    Effective single-agent "activity" is the predicted relative sensitive
    value (value_hat), already on a common [0, 1] cross-drug scale.
    """
    pred_a = predict_one(bundle, sample, drug_a)
    pred_b = predict_one(bundle, sample, drug_b)
    a, b = pred_a.value_hat, pred_b.value_hat
    if mode == "activity_product":
        combo = a * b
    elif mode == "bliss":
        combo = a + b - a * b
    elif mode == "complementarity":
        combo = a * b * (1.0 + abs(a - b))
    else:
        raise ValueError(f"Unsupported combination mode: {mode!r}")
    return {
        "drug_a": pred_a.drug.name,
        "drug_b": pred_b.drug.name,
        "value_hat_a": a,
        "value_hat_b": b,
        "combination_score": float(combo),
        "mode": mode,
        "prediction_a": pred_a,
        "prediction_b": pred_b,
    }


def search_drugs(bundle: ModelBundle, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Forgiving substring search over the bundled drug library, for callers
    (e.g. the LLM agent) that don't know the exact drug name/ID."""
    lib = bundle.drug_library
    q = str(query).strip().lower()
    mask = lib["DRUG_NAME"].astype(str).str.lower().str.contains(q, regex=False)
    matches = lib.loc[mask].head(limit)
    return [{"drug_id": int(r["DRUG_ID"]), "drug_name": str(r["DRUG_NAME"])} for _, r in matches.iterrows()]


def search_cell_lines(bundle: ModelBundle, query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Forgiving substring search over bundled cell-line metadata (name,
    tissue, or cancer type), for callers that don't know the exact ID."""
    meta = bundle.cell_metadata
    if meta is None or meta.empty:
        return []
    q = str(query).strip().lower()
    text_cols = [c for c in ("model_name", "tissue", "cancer_type") if c in meta.columns]
    mask = pd.Series(False, index=meta.index)
    for col in text_cols:
        mask = mask | meta[col].astype(str).str.lower().str.contains(q, regex=False)
    matches = meta.loc[mask].head(limit)
    return [
        {
            "sanger_model_id": str(r["SANGER_MODEL_ID"]),
            "model_name": str(r.get("model_name", "")),
            "tissue": str(r.get("tissue", "")),
            "cancer_type": str(r.get("cancer_type", "")),
        }
        for _, r in matches.iterrows()
    ]
