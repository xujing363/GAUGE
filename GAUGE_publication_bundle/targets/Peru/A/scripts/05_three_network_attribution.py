"""
Script 05: Three Prior Network Attribution
===========================================
The model integrates three knowledge graphs as prior networks:
  ChEMBL  — pharmacological MoA, drug-target action types, mechanism data
  DRKG    — broad multi-type biological network (gene-disease-pathway)
  PrimeKG — comprehensive protein interaction and disease-gene network

Key findings from Script 02:
  - DRKG alpha = 0 for ALL drugs → model learned DRKG adds no incremental signal
  - 56.9% of drugs (156/274) activate NO KG → purely fingerprint-based
  - 43.1% of drugs (118/274) activate ChEMBL and/or PrimeKG selectively

This script quantifies the contribution of each network to:
  1. Baseline prediction accuracy (Spearman r vs true AUC)
  2. Gene perturbation effect magnitude (how much the network is needed to
     "know" which gene affects which drug)

Ablation approach: kg_mask string passed to predict_frame
  "ChEMBL_off"  → remove ChEMBL edges from KG graph
  "DRKG_off"    → remove DRKG edges
  "PrimeKG_off" → remove PrimeKG edges
  "all_off"     → remove all KG (fingerprint-only baseline)

Outputs (saved to results/05_network_attribution/):
  network_baseline_accuracy.csv        - Spearman r by drug under each ablation
  network_perturbation_effect.csv      - ΔAU retention under each ablation
  kg_active_vs_silent_accuracy.csv     - accuracy comparison: KG-active vs KG-silent drugs
  network_attribution_summary.csv      - publication-ready summary table

Usage:
    python scripts/05_three_network_attribution.py
"""
from __future__ import annotations

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, os.environ.get("KGPUB_PY_ROOT", "/mnt/raid5/xujing/KG"))

from config import (
    BATCH_SIZE,
    CONFIG_YAML,
    DEVICE,
    DRUG_FAMILIES,
    GENE_PANEL,
    KNOWN_DRUG_TARGETS,
    MIN_SAMPLES_PER_DRUG,
    PREPARED_PKL,
    RESULTS_DIR,
    RESULT_DIR,
)
from utils import (
    build_inference_context,
    load_experiment,
    perturb_state_at_gene,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from GAUGE.train import (
        TrainExecutionContext,
        _build_prediction_session,
        predict_frame,
    )
    from GAUGE.drug_level import select_best_fusion_weight, with_fused_auc
    from GAUGE.repro import RUNTIME_PROFILE_STABLE

OUT_DIR = RESULTS_DIR / "05_network_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# KG mask conditions to test
CONDITIONS = {
    "full":         None,
    "ChEMBL_off":   "ChEMBL_off",
    "DRKG_off":     "DRKG_off",
    "PrimeKG_off":  "PrimeKG_off",
    "all_off":      "all_off",
}

# Focus drug-gene pairs (using genes available in HVG2000)
FOCUS_PAIRS = [
    ("Erlotinib",   "EGFR"),
    ("Gefitinib",   "EGFR"),
    ("Lapatinib",   "ERBB2"),
    ("Venetoclax",  "BCL2"),
    ("Navitoclax",  "BCL2"),
    ("Palbociclib", "CDKN1A"),
    ("Trametinib",  "MYC"),
    ("Olaparib",    "CDKN2A"),
    ("Dasatinib",   "KIT"),
]


def run_predict_frame_with_mask(model, frame, prepared, config, device, batch_size, kg_mask=None):
    """Run predict_frame with optional kg_mask and return auc_hat array."""
    context = TrainExecutionContext()
    session = _build_prediction_session(
        model, prepared, device=device,
        benchmark=config, controls=config.controls,
        context=context, runtime_profile=RUNTIME_PROFILE_STABLE,
        eval_compile=False, explain_dir=None,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        outputs = predict_frame(
            model, frame, prepared,
            batch_size=batch_size, device=device,
            benchmark=config, controls=config.controls,
            context=context, runtime_profile=RUNTIME_PROFILE_STABLE,
            eval_compile=False, kg_mask=kg_mask, session=session,
        )
    pred = outputs.core.copy()
    return pred


def compute_manual_perturbation_delta(
    model, infra, local_idx, gene_idx,
    pca_components, scaler_scale, imputer_stats, device, batch_size
):
    """Compute delta_AUC for gene silencing using manual forward passes (no predict_frame)."""
    tensors      = infra["tensors"]
    tensor_banks = infra["tensor_banks"]
    kg_idx_bank  = infra["kg_drug_idx_bank"]
    kg_payload   = infra["precomputed_kg_payload"]

    base_list = []
    pert_list = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(local_idx), batch_size):
            end = min(start + batch_size, len(local_idx))
            bidx = local_idx[start:end]
            si = tensors.state_idx[bidx]
            di = tensors.drug_idx[bidx]
            s  = tensor_banks.state_bank.index_select(0, si)
            fp = tensor_banks.fp_bank.index_select(0, di)
            ki = kg_idx_bank.index_select(0, di) if kg_idx_bank is not None else None
            out_b = model(state=s, drug_fp=fp, drug_idx=ki,
                          use_prior=True, precomputed_kg_payload=kg_payload)
            s_p = perturb_state_at_gene(s, gene_idx, pca_components, scaler_scale, imputer_stats)
            out_p = model(state=s_p, drug_fp=fp, drug_idx=ki,
                          use_prior=True, precomputed_kg_payload=kg_payload)
            base_list.append(out_b["auc_hat"].cpu().numpy())
            pert_list.append(out_p["auc_hat"].cpu().numpy())
    return np.concatenate(base_list), np.concatenate(pert_list)


def main():
    print("=" * 70)
    print("Script 05: Three Prior Network Attribution")
    print("ChEMBL × DRKG × PrimeKG — KG Source Ablation")
    print("=" * 70)

    model, prepared, config = load_experiment(PREPARED_PKL, RESULT_DIR, CONFIG_YAML, DEVICE)

    genes = prepared.artifacts.genes
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    test_frame = prepared.responses[prepared.responses["split"] == "test"].copy()
    test_frame = test_frame.reset_index(drop=True)
    print(f"  Test rows: {len(test_frame)}, Test cells: {test_frame['SANGER_MODEL_ID'].nunique()}")

    infra = build_inference_context(model, prepared, config, test_frame, DEVICE)

    pca_components = prepared.artifacts.pca.components_
    scaler_scale   = prepared.artifacts.scaler.scale_
    imputer_stats  = prepared.artifacts.imputer.statistics_

    # ── Load KG alpha profile from script 02 results ───────────────────────
    alpha_csv = RESULTS_DIR / "02_gate" / "drug_kg_activation_profile.csv"
    if alpha_csv.exists():
        alpha_df = pd.read_csv(alpha_csv)
        kg_active_drugs = set(alpha_df[alpha_df["alpha_total"] > 0.01]["DRUG_NAME"])
        kg_silent_drugs = set(alpha_df[alpha_df["alpha_total"] <= 0.01]["DRUG_NAME"])
        print(f"  KG-active drugs: {len(kg_active_drugs)}, KG-silent: {len(kg_silent_drugs)}")
    else:
        kg_active_drugs = set()
        kg_silent_drugs = set()

    # ── Part 1: Baseline accuracy under each KG ablation ───────────────────
    print("\n--- Part 1: Baseline Prediction Accuracy by KG Source ---")

    # Sample representative drugs: all KG-active + 20 KG-silent for comparison
    study_drugs_active = [d for d in kg_active_drugs if d in set(test_frame["DRUG_NAME"])]
    study_drugs_silent = [d for d in list(kg_silent_drugs)[:20] if d in set(test_frame["DRUG_NAME"])]
    study_drugs = study_drugs_active + study_drugs_silent
    print(f"  Study drugs: {len(study_drugs_active)} KG-active + {len(study_drugs_silent)} KG-silent")

    accuracy_rows = []
    for cond_name, kg_mask in CONDITIONS.items():
        print(f"  Condition: {cond_name} ...")
        study_frame = test_frame[test_frame["DRUG_NAME"].isin(study_drugs)].copy()
        pred_df = run_predict_frame_with_mask(
            model, study_frame, prepared, config, DEVICE, BATCH_SIZE, kg_mask=kg_mask
        )

        # Compute per-drug Spearman r
        for drug_name in study_drugs:
            drug_pred = pred_df[pred_df["DRUG_NAME"] == drug_name] if "DRUG_NAME" in pred_df.columns else \
                        pred_df[pred_df["DRUG_ID"] == study_frame[study_frame["DRUG_NAME"] == drug_name]["DRUG_ID"].iloc[0]]
            if len(drug_pred) < 5:
                continue
            if "AUC" not in drug_pred.columns or "auc_hat" not in drug_pred.columns:
                continue
            true_auc = drug_pred["AUC"].values
            pred_auc = drug_pred["auc_hat"].values
            if np.isnan(true_auc).any():
                continue
            r, p = spearmanr(true_auc, pred_auc)
            accuracy_rows.append({
                "condition":     cond_name,
                "DRUG_NAME":     drug_name,
                "spearman_r":    float(r),
                "p_value":       float(p),
                "n_cells":       len(drug_pred),
                "kg_active":     drug_name in kg_active_drugs,
            })

    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_df.to_csv(OUT_DIR / "network_baseline_accuracy.csv", index=False)

    # Summary
    if not accuracy_df.empty:
        summary_acc = (
            accuracy_df.groupby(["condition", "kg_active"])["spearman_r"]
            .agg(["mean", "std", "count"]).reset_index()
        )
        print("\n  Accuracy by condition × KG activation status:")
        print(summary_acc.to_string(index=False))

    # ── Part 2: Gene perturbation effect under KG ablation ─────────────────
    print("\n--- Part 2: Gene Perturbation Effect Retention under KG Ablation ---")
    print("  (Using direct forward passes for perturbation; predict_frame for baseline)")

    pert_rows = []
    for drug_name, gene_name in FOCUS_PAIRS:
        if gene_name not in gene_to_idx:
            continue
        gene_idx = gene_to_idx[gene_name]
        drug_frame = test_frame[test_frame["DRUG_NAME"] == drug_name]
        if len(drug_frame) < 5:
            continue
        local_idx = drug_frame.index.to_numpy()

        print(f"\n  {drug_name} ← {gene_name} (n={len(drug_frame)})")

        # Full-KG baseline perturbation delta
        base_full, pert_full = compute_manual_perturbation_delta(
            model, infra, local_idx, gene_idx,
            pca_components, scaler_scale, imputer_stats, DEVICE, BATCH_SIZE
        )
        delta_full = np.abs(base_full - pert_full).mean()

        pert_rows.append({
            "DRUG_NAME":      drug_name,
            "gene_name":      gene_name,
            "condition":      "full",
            "abs_delta_auc":  float(delta_full),
            "retention":      1.0,
            "is_known_target": gene_name in KNOWN_DRUG_TARGETS.get(drug_name, []),
        })
        print(f"    full:          |ΔAU| = {delta_full:.5f}")

        # For KG ablation: use predict_frame with kg_mask on the drug subset
        for cond_name, kg_mask in CONDITIONS.items():
            if cond_name == "full":
                continue
            # Run predict_frame with mask for this drug's rows
            drug_sub = test_frame.iloc[local_idx].copy()
            pred_base = run_predict_frame_with_mask(
                model, drug_sub, prepared, config, DEVICE, BATCH_SIZE, kg_mask=kg_mask
            )
            # For perturbed state, we need direct model calls (predict_frame doesn't support state perturbation)
            # Instead, compare full vs ablated delta as proxy for network importance
            if "auc_hat" in pred_base.columns:
                delta_ablated = np.abs(
                    base_full - pred_base["auc_hat"].values[:len(local_idx)]
                ).mean() if len(pred_base) >= len(local_idx) else np.nan
            else:
                delta_ablated = np.nan

            retention = float(delta_ablated / (delta_full + 1e-8)) if not np.isnan(delta_ablated) else np.nan

            pert_rows.append({
                "DRUG_NAME":      drug_name,
                "gene_name":      gene_name,
                "condition":      cond_name,
                "abs_delta_auc":  float(delta_ablated) if not np.isnan(delta_ablated) else np.nan,
                "retention":      retention,
                "is_known_target": gene_name in KNOWN_DRUG_TARGETS.get(drug_name, []),
            })
            print(f"    {cond_name:15s}: |Δfull_vs_ablated| = {delta_ablated:.5f}")

    pert_df = pd.DataFrame(pert_rows)
    pert_df.to_csv(OUT_DIR / "network_perturbation_effect.csv", index=False)

    # ── Part 3: KG-active vs KG-silent accuracy comparison ─────────────────
    print("\n--- Part 3: KG-Active vs KG-Silent Drug Accuracy ---")
    if not accuracy_df.empty:
        active_full = accuracy_df[
            (accuracy_df["condition"] == "full") & (accuracy_df["kg_active"] == True)
        ]["spearman_r"]
        silent_full = accuracy_df[
            (accuracy_df["condition"] == "full") & (accuracy_df["kg_active"] == False)
        ]["spearman_r"]

        if len(active_full) > 0 and len(silent_full) > 0:
            print(f"  KG-active drugs (full KG):  r = {active_full.mean():.4f} ± {active_full.std():.4f} (n={len(active_full)})")
            print(f"  KG-silent drugs (full KG):  r = {silent_full.mean():.4f} ± {silent_full.std():.4f} (n={len(silent_full)})")
            if len(active_full) > 1 and len(silent_full) > 1:
                _, p = mannwhitneyu(active_full, silent_full, alternative="greater")
                print(f"  Mann-Whitney p (active > silent): {p:.3e}")

        # Delta Spearman when each source removed (for KG-active drugs only)
        active_acc = accuracy_df[accuracy_df["kg_active"] == True]
        if not active_acc.empty:
            print("\n  ΔSpearman (full − ablated) for KG-active drugs:")
            full_r = active_acc[active_acc["condition"] == "full"].set_index("DRUG_NAME")["spearman_r"]
            for cond in ["ChEMBL_off", "DRKG_off", "PrimeKG_off", "all_off"]:
                abl_r = active_acc[active_acc["condition"] == cond].set_index("DRUG_NAME")["spearman_r"]
                common = full_r.index.intersection(abl_r.index)
                if len(common) > 0:
                    delta = full_r[common] - abl_r[common]
                    print(f"    {cond:15s}: ΔSpearman = {delta.mean():+.4f} ± {delta.std():.4f}")

        kg_comparison = pd.DataFrame({
            "category": ["KG-active", "KG-silent"],
            "n_drugs":  [len(active_full), len(silent_full)],
            "mean_spearman": [active_full.mean() if len(active_full) > 0 else np.nan,
                              silent_full.mean() if len(silent_full) > 0 else np.nan],
            "std_spearman": [active_full.std() if len(active_full) > 0 else np.nan,
                             silent_full.std() if len(silent_full) > 0 else np.nan],
        })
        kg_comparison.to_csv(OUT_DIR / "kg_active_vs_silent_accuracy.csv", index=False)

    # ── Summary table ──────────────────────────────────────────────────────
    if not accuracy_df.empty:
        summary_rows = []
        for cond in CONDITIONS:
            sub = accuracy_df[accuracy_df["condition"] == cond]
            sub_active = sub[sub["kg_active"] == True]
            summary_rows.append({
                "condition":              cond,
                "n_drugs_active":         len(sub_active),
                "mean_spearman_active":   float(sub_active["spearman_r"].mean()) if len(sub_active) > 0 else np.nan,
                "n_drugs_silent":         len(sub[sub["kg_active"] == False]),
                "mean_spearman_silent":   float(sub[sub["kg_active"] == False]["spearman_r"].mean()) if len(sub) > 0 else np.nan,
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUT_DIR / "network_attribution_summary.csv", index=False)
        print("\n--- Network Attribution Summary ---")
        print(summary_df.to_string(index=False))

    print(f"\nAll outputs → {OUT_DIR}")
    print("Script 05 complete.")


if __name__ == "__main__":
    main()
