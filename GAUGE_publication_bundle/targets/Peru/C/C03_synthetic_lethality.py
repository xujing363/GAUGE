"""
Scenario C — Step 3: Synthetic Lethality Prediction and SynLeth Validation
===========================================================================

Combines outputs from C01 (drug-sensitive genes) and C02 (cell-sensitive targets)
to construct synthetic lethality (SL) candidate pairs and validate them against
the SynLeth database.

Conceptual framework
---------------------
  Drug D inhibits target T (from ChEMBL prior in KG).
  Gene G sensitises cells to drug D (from Step 1).
  → (T, G) is a drug-context-specific SL candidate:
      - Inhibiting T (via drug) + knocking out G → enhanced lethality
      - This is a form of synthetic dosage lethality (SDL)

  Cell C has sensitive target S (from Step 2).
  Gene G sensitises cell C to drug D (from Step 1, filtered to cell C).
  → (S, G) is a cell-context-specific SL candidate.

World-model contribution
------------------------
  Static PrimeKG already contains SL edges, but they are context-independent.
  Our model's dynamic α-gates (α_ChEMBL, α_DRKG, α_PrimeKG) weight each prior
  per (cell, drug) pair. The gene perturbation experiments reveal which
  gene–gene functional relationships are ACTIVE in each drug context — this is
  the "static → dynamic" transformation.

Validation strategy
-------------------
  1. SynLeth-SL enrichment: are predicted SL candidates enriched in SynLeth SL pairs?
  2. SynLeth-NONSL specificity: predicted candidates should NOT overlap with known NON-SL pairs.
  3. Fisher's exact test + hypergeometric p-value per cancer type.
  4. Novel candidates = high-confidence predictions NOT in SynLeth (new biology).

Outputs (results/C03_sl_prediction/)
--------------------------------------
  sl_candidates_full.csv       — all (drug_target, gene) SL predictions
  sl_candidates_confident.csv  — filtered by delta_auc > threshold + frac_sensitising > 0.5
  synleth_validation.csv       — SynLeth overlap statistics
  novel_sl_predictions.csv     — high-confidence pairs absent from SynLeth
  cancer_specific_sl.csv       — cancer-type focused SL candidates
"""
from __future__ import annotations

import sys
from pathlib import Path
from scipy.stats import fisher_exact, hypergeom

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DELTA_AUC_THRESHOLD,
    ENRICHMENT_FDR_CUTOFF,
    GDSC_MODEL_LIST,
    KNOWN_DRUG_TARGETS,
    RESULTS_DIR,
    SENSITISING_PERCENTILE,
    SYNLETH_DIR,
    CANCER_TYPE_FOCUS,
)
from utils import load_cell_metadata, load_synleth_database

OUT_DIR = RESULTS_DIR / "C03_sl_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds for "confident" SL predictions
# Using percentile-based ranking: top 5% = sensitising, top 2% = confident
CONFIDENT_PERCENTILE = 2.0      # top 2% genes per drug = "confident sensitising"


def build_drug_target_map() -> dict[str, list[str]]:
    """Return {drug_name: [target_genes]}."""
    return {k: [t for t in v if t != "DNA"] for k, v in KNOWN_DRUG_TARGETS.items()}


def load_c01_results() -> pd.DataFrame:
    """Load drug-gene perturbation results from Step 1."""
    path = RESULTS_DIR / "C01_drug_gene" / "drug_gene_delta_auc.csv"
    if not path.exists():
        raise FileNotFoundError(f"C01 results not found: {path}\nRun C01_drug_gene_perturbation.py first.")
    return pd.read_csv(path)


def load_c02_results() -> pd.DataFrame:
    """Load cell-target perturbation results from Step 2."""
    path = RESULTS_DIR / "C02_cell_target" / "cell_target_delta_auc.csv"
    if not path.exists():
        raise FileNotFoundError(f"C02 results not found: {path}\nRun C02_cell_target_perturbation.py first.")
    return pd.read_csv(path)


def hypergeometric_enrichment(
    k: int, N: int, K: int, n: int
) -> float:
    """
    Hypergeometric p-value: drawing k successes in n draws from a population
    of N items with K successes.
    """
    return hypergeom.sf(k - 1, N, K, n)


def main() -> None:
    print("=" * 70)
    print("Scenario C — Step 3: Synthetic Lethality Prediction + Validation")
    print("=" * 70)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("\nLoading Step 1 & 2 results …")
    drug_gene_df  = load_c01_results()
    cell_target_df = load_c02_results()
    # Compute within-cell percentile rank (not persisted in C02 CSV)
    cell_target_df["within_cell_percentile"] = (
        cell_target_df.groupby("SANGER_MODEL_ID")["mean_delta_auc"].rank(pct=True) * 100
    )
    synleth_db     = load_synleth_database(SYNLETH_DIR)
    cell_meta      = load_cell_metadata(GDSC_MODEL_LIST)
    drug_target_map = build_drug_target_map()

    print(f"  Drug-gene pairs (C01): {len(drug_gene_df):,}")
    print(f"  Cell-target pairs (C02): {len(cell_target_df):,}")
    print(f"  SynLeth SL pairs: {len(synleth_db.get('SL', set())):,}")
    print(f"  SynLeth NONSL pairs: {len(synleth_db.get('NONSL', set())):,}")

    # ── Compute per-drug percentile thresholds ────────────────────────────────
    # The double-disjoint model produces small absolute delta values (~1e-4 scale)
    # Use percentile ranking within each drug for robust gene selection.
    drug_thresholds: dict[str, float] = {}
    for drug_name, grp in drug_gene_df.groupby("DRUG_NAME"):
        threshold = np.percentile(grp["mean_delta_auc"], 100 - SENSITISING_PERCENTILE)
        drug_thresholds[drug_name] = float(threshold)
    conf_thresholds: dict[str, float] = {}
    for drug_name, grp in drug_gene_df.groupby("DRUG_NAME"):
        threshold = np.percentile(grp["mean_delta_auc"], 100 - CONFIDENT_PERCENTILE)
        conf_thresholds[drug_name] = float(threshold)

    print(f"  Sensitising threshold (top {SENSITISING_PERCENTILE}%) range: "
          f"{min(drug_thresholds.values()):.5f} – {max(drug_thresholds.values()):.5f}")

    # ── APPROACH 1: Drug-centric SL candidates (drug_target × sensitising_gene) ─
    print("\n--- Approach 1: Drug-centric SL candidates ---")
    sl_candidates = []

    for drug_name, drug_targets in drug_target_map.items():
        drug_subset = drug_gene_df[drug_gene_df["DRUG_NAME"] == drug_name]
        if drug_subset.empty:
            continue

        # Sensitising genes: top 5% by delta_AUC for this drug
        threshold = drug_thresholds.get(drug_name, 0)
        sens_genes = drug_subset[drug_subset["mean_delta_auc"] >= threshold].copy()
        sens_genes = sens_genes.sort_values("mean_delta_auc", ascending=False)

        for target in drug_targets:
            for _, row in sens_genes.iterrows():
                cand_gene = row["gene_name"]
                if cand_gene == target:
                    continue

                # Check SynLeth annotations
                pair_fwd = (target, cand_gene)
                pair_rev = (cand_gene, target)
                in_sl    = pair_fwd in synleth_db.get("SL", set()) or pair_rev in synleth_db.get("SL", set())
                in_sdl   = pair_fwd in synleth_db.get("SDL", set()) or pair_rev in synleth_db.get("SDL", set())
                in_nonsl = pair_fwd in synleth_db.get("NONSL", set()) or pair_rev in synleth_db.get("NONSL", set())

                sl_candidates.append({
                    "drug_name":          drug_name,
                    "drug_target":        target,
                    "sensitising_gene":   cand_gene,
                    "mean_delta_auc":     row["mean_delta_auc"],
                    "std_delta_auc":      row["std_delta_auc"],
                    "frac_sensitising":   row["frac_sensitising"],
                    "n_cells":            row["n_cells"],
                    "in_synleth_SL":      in_sl,
                    "in_synleth_SDL":     in_sdl,
                    "in_synleth_NONSL":   in_nonsl,
                    "synleth_validated":  in_sl or in_sdl,
                    "novel":              not (in_sl or in_sdl or in_nonsl),
                })

    sl_df = pd.DataFrame(sl_candidates)
    sl_df.to_csv(OUT_DIR / "sl_candidates_full.csv", index=False)
    print(f"  Total SL candidates: {len(sl_df):,}")
    print(f"  SynLeth-validated:   {sl_df['synleth_validated'].sum():,} ({100*sl_df['synleth_validated'].mean():.1f}%)")
    print(f"  Novel (not in SynLeth): {sl_df['novel'].sum():,} ({100*sl_df['novel'].mean():.1f}%)")

    # ── Confident candidates (top 2% per drug, not in NONSL) ─────────────────
    def is_confident(row):
        threshold = conf_thresholds.get(row["drug_name"], 0)
        return row["mean_delta_auc"] >= threshold and not row["in_synleth_NONSL"]

    confident_df = sl_df[sl_df.apply(is_confident, axis=1)].copy().sort_values(
        "mean_delta_auc", ascending=False
    )
    confident_df.to_csv(OUT_DIR / "sl_candidates_confident.csv", index=False)
    print(f"\n  Confident SL candidates: {len(confident_df):,}")
    print(f"  Confident & validated: {confident_df['synleth_validated'].sum():,}")
    print(f"  Confident & novel:     {confident_df['novel'].sum():,}")

    # ── APPROACH 2: Cell-context SL (top-5% sensitive targets per cell) ─────────
    print("\n--- Approach 2: Cell-context SL candidates ---")
    # Pre-compute SL lookup set once (sorted tuples for O(1) lookup)
    sl_sorted_set: set[tuple[str, str]] = {
        (min(a, b), max(a, b)) for a, b in synleth_db.get("SL", set())
    }
    # Filter to top 5% sensitive targets per cell (by within-cell percentile)
    cell_target_sens_top = cell_target_df[
        cell_target_df["within_cell_percentile"] >= (100 - SENSITISING_PERCENTILE)
    ].copy() if "within_cell_percentile" in cell_target_df.columns else cell_target_df[
        cell_target_df["mean_delta_auc"] > DELTA_AUC_THRESHOLD
    ].copy()

    cell_sl = []
    for cell_id, c2_grp in cell_target_sens_top.groupby("SANGER_MODEL_ID"):
        sens_targets = c2_grp["target_gene"].tolist()
        # All pairs of sensitive targets within the cell
        for i, target in enumerate(sens_targets):
            for cand_gene in sens_targets[i + 1:]:
                pair_key = (min(target, cand_gene), max(target, cand_gene))
                in_sl = pair_key in sl_sorted_set
                cell_sl.append({
                    "SANGER_MODEL_ID": cell_id,
                    "gene_A": target,
                    "gene_B": cand_gene,
                    "in_synleth_SL": in_sl,
                })

    if cell_sl:
        cell_sl_df = pd.DataFrame(cell_sl).drop_duplicates(["SANGER_MODEL_ID", "gene_A", "gene_B"])
        cell_sl_df.to_csv(OUT_DIR / "cell_context_sl_candidates.csv", index=False)
        print(f"  Cell-context SL candidates: {len(cell_sl_df):,}")
    else:
        print("  No cell-context SL candidates generated")

    # ── SynLeth enrichment analysis ───────────────────────────────────────────
    print("\n--- SynLeth enrichment analysis ---")
    sl_pairs   = synleth_db.get("SL", set())
    total_gene_pairs = len(sl_df)
    n_sl_universe    = len(sl_pairs)
    # Estimate gene universe size (genes tested in perturbation × drug targets)
    n_candidate_genes = drug_gene_df["gene_name"].nunique()
    n_drug_targets    = len([t for ts in drug_target_map.values() for t in ts])
    N_universe = n_candidate_genes * n_drug_targets

    k_in_sl   = sl_df["in_synleth_SL"].sum()
    k_in_sdl  = sl_df["in_synleth_SDL"].sum()
    k_conf_sl = confident_df["in_synleth_SL"].sum() if len(confident_df) > 0 else 0

    enrichment_rows = []
    for subset_name, subset_df in [("all_candidates", sl_df), ("confident", confident_df)]:
        n_subset = len(subset_df)
        k_sl = subset_df["in_synleth_SL"].sum()
        if n_subset > 0 and N_universe > 0:
            expected = n_subset * len(sl_pairs) / N_universe
            pval = hypergeometric_enrichment(k_sl, N_universe, len(sl_pairs), n_subset)
            enrichment_rows.append({
                "subset":     subset_name,
                "n_pairs":    n_subset,
                "k_in_SL":    k_sl,
                "expected":   round(expected, 2),
                "enrichment": round(k_sl / expected, 3) if expected > 0 else float("nan"),
                "pval":       pval,
            })

    enrich_df = pd.DataFrame(enrichment_rows)
    enrich_df.to_csv(OUT_DIR / "synleth_validation.csv", index=False)
    print(enrich_df.to_string(index=False))

    # ── Novel high-confidence predictions ─────────────────────────────────────
    print("\n--- Novel SL predictions (not in SynLeth) ---")
    novel_df = confident_df[confident_df["novel"]].copy()
    novel_df.to_csv(OUT_DIR / "novel_sl_predictions.csv", index=False)
    print(f"  Novel confident candidates: {len(novel_df):,}")
    print("\nTop 20 novel SL predictions:")
    print(novel_df[["drug_name", "drug_target", "sensitising_gene", "mean_delta_auc", "frac_sensitising"]]
          .head(20).to_string(index=False))

    # ── Cancer-type specific SL ───────────────────────────────────────────────
    print("\n--- Cancer-type specific SL analysis ---")
    # Use C02 cell-target data + cancer metadata
    ct_merge = cell_target_df.merge(cell_meta, on="SANGER_MODEL_ID", how="left")
    ct_merge = ct_merge.dropna(subset=["cancer_type"])

    cancer_sl_rows = []
    for cancer_type in CANCER_TYPE_FOCUS:
        ct_subset = ct_merge[ct_merge["cancer_type"] == cancer_type]
        if len(ct_subset) == 0:
            continue
        n_cells = ct_subset["SANGER_MODEL_ID"].nunique()
        # Find targets consistently sensitising in this cancer type
        target_agg = (
            ct_subset.groupby("target_gene")
            .agg(
                mean_delta   = ("mean_delta_auc", "mean"),
                n_cells_sens = ("frac_sensitising", lambda x: (x > 0.3).sum()),
            )
            .reset_index()
            .sort_values("mean_delta", ascending=False)
        )
        for _, r in target_agg.head(5).iterrows():
            cancer_sl_rows.append({
                "cancer_type":   cancer_type,
                "target_gene":   r.target_gene,
                "n_cells":       n_cells,
                "mean_delta_auc": r.mean_delta,
                "n_cells_sensitising": r.n_cells_sens,
                "frac_cells_sens": round(r.n_cells_sens / n_cells, 3) if n_cells > 0 else 0,
            })

    cancer_sl_df = pd.DataFrame(cancer_sl_rows)
    cancer_sl_df.to_csv(OUT_DIR / "cancer_specific_sl.csv", index=False)
    print(cancer_sl_df.to_string(index=False))

    print(f"\nAll Step 3 results saved to {OUT_DIR}")
    print("\nSummary:")
    print(f"  Drug-centric SL candidates: {len(sl_df):,}")
    print(f"  Confident candidates: {len(confident_df):,}")
    print(f"  SynLeth-validated: {confident_df['synleth_validated'].sum():,}")
    print(f"  Novel predictions:  {len(novel_df):,}")


if __name__ == "__main__":
    main()
