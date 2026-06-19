#!/usr/bin/env python3
"""
Layer 4 (part 3): Transcriptomic reversal score computation.

For each analogue's proxy signature, computes how well it reverses
the cancer disease signature (LUAD/SKCM tumor vs GTEx normal).

Reversal score = mean(proxy_sig[disease_down_genes]) - mean(proxy_sig[disease_up_genes])
Higher reversal score = analogue better reverses cancer state.

Compares: top improved analogues vs parent drug vs all analogues vs random controls.

Run with: conda run -n kg_GAUGE python v04_transcriptome_reversal.py
Output: results/layer4_transcriptome/
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import fdrcorrection

ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "layer4_transcriptome"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LINCS_DIR  = ROOT / "data" / "lincs"
DISEASE_DIR = ROOT / "data" / "disease_sig"

TOP_N_DISEASE_GENES = 200  # top N up/down genes from disease signature to use


def load_gene_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [g.strip() for g in path.read_text().splitlines() if g.strip()]


def reversal_score(sig_row: pd.Series, up_genes: list[str], down_genes: list[str],
                   gene_cols: list[str]) -> float:
    """
    Reversal score: drug should down-regulate disease-up genes
    and up-regulate disease-down genes.
    score = mean(drug_sig[disease_down]) - mean(drug_sig[disease_up])
    """
    # Intersect with both gene_cols and the row's actual index
    row_genes = set(sig_row.index)
    available_up   = [g for g in up_genes   if g in gene_cols and g in row_genes]
    available_down = [g for g in down_genes if g in gene_cols and g in row_genes]
    if not available_up or not available_down:
        return float("nan")
    val_up   = sig_row[available_up].values.astype(float)
    val_down = sig_row[available_down].values.astype(float)
    return float(np.mean(val_down) - np.mean(val_up))


def main():
    print("=" * 60)
    print("Layer 4 (Part 3): Transcriptomic Reversal Score")
    print("=" * 60)

    # ── Load proxy signatures ─────────────────────────────────────────────────
    proxy_path = LINCS_DIR / "proxy_signatures.csv"
    if not proxy_path.exists():
        print("  ERROR: proxy_signatures.csv not found. Run v03_lincs_proxy.py first.")
        return
    proxy = pd.read_csv(proxy_path)
    meta_cols = ["DRUG_NAME", "DRUG_ID", "smiles", "seed_drug", "cancer_type",
                 "delta_improvement", "mean_value_hat", "tanimoto", "best_lincs_sim",
                 "best_lincs_name"]
    gene_cols = [c for c in proxy.columns if c not in meta_cols]
    print(f"  Loaded {len(proxy)} proxy signatures with {len(gene_cols)} gene cols")

    all_results = []

    for cancer_tag, cancer_type, seed_drug in [
        ("LUAD", "TCGA-LUAD", "erlotinib"),
        ("SKCM", "TCGA-SKCM", "trametinib"),
    ]:
        print(f"\n  [{cancer_tag}] {cancer_type} / {seed_drug}")

        # Load disease signature
        up_genes   = load_gene_list(DISEASE_DIR / f"{cancer_tag}_up_genes_lincs.txt")
        down_genes = load_gene_list(DISEASE_DIR / f"{cancer_tag}_down_genes_lincs.txt")
        if not up_genes or not down_genes:
            # Fallback to non-LINCS
            up_genes   = load_gene_list(DISEASE_DIR / f"{cancer_tag}_up_genes.txt")
            down_genes = load_gene_list(DISEASE_DIR / f"{cancer_tag}_down_genes.txt")
        if not up_genes or not down_genes:
            print(f"  SKIP: no disease signature for {cancer_tag}")
            continue

        up_genes   = up_genes[:TOP_N_DISEASE_GENES]
        down_genes = down_genes[:TOP_N_DISEASE_GENES]
        avail_up   = [g for g in up_genes   if g in gene_cols]
        avail_down = [g for g in down_genes if g in gene_cols]
        print(f"  Disease up genes: {len(up_genes)} (in signature: {len(avail_up)})")
        print(f"  Disease down genes: {len(down_genes)} (in signature: {len(avail_down)})")

        # Load parent signature
        parent_path = LINCS_DIR / f"parent_signature_{seed_drug}.csv"
        parent_rev = None
        if parent_path.exists():
            parent_df = pd.read_csv(parent_path)
            if len(parent_df) > 0:
                parent_rev = reversal_score(parent_df.iloc[0], up_genes, down_genes, gene_cols)
                print(f"  Parent ({seed_drug}) reversal score: {parent_rev:.4f}")

        # Compute reversal for each analogue in this cancer type
        sub = proxy[proxy["cancer_type"] == cancer_type].copy()
        print(f"  Analogues with proxy signatures: {len(sub)}")

        sub["reversal_score"] = sub.apply(
            lambda r: reversal_score(r, up_genes, down_genes, gene_cols), axis=1
        )
        if parent_rev is not None:
            sub["delta_reversal_vs_parent"] = sub["reversal_score"] - parent_rev
        else:
            sub["delta_reversal_vs_parent"] = float("nan")

        sub["is_improved"] = sub["delta_improvement"] > 0
        sub["cancer_tag"]  = cancer_tag
        all_results.append(sub)

        # Summary stats
        improved = sub[sub["is_improved"]]
        non_imp  = sub[~sub["is_improved"]]
        print(f"\n  Reversal score distribution:")
        print(f"    All analogues: mean={sub['reversal_score'].mean():.4f}, "
              f"std={sub['reversal_score'].std():.4f}")
        if parent_rev is not None:
            print(f"    Parent drug:   {parent_rev:.4f}")
        if len(improved) > 0:
            print(f"    Improved (Δ>0): mean={improved['reversal_score'].mean():.4f}, "
                  f"n={len(improved)}")
            print(f"    delta_reversal: mean={improved['delta_reversal_vs_parent'].mean():.4f}")

        # Statistical test: do improved analogues have higher reversal than non-improved?
        if len(improved) >= 2 and len(non_imp) >= 2:
            try:
                stat, p = mannwhitneyu(
                    improved["reversal_score"].dropna(),
                    non_imp["reversal_score"].dropna(),
                    alternative="greater"
                )
                print(f"  MWU (improved > non-improved reversal): stat={stat:.1f}, p={p:.3e}")
            except Exception:
                pass

        # Empirical p-value: fraction of all analogues with reversal >= best improved
        if len(improved) > 0:
            top_rev = improved["reversal_score"].max()
            n_controls = len(sub)
            n_beat = (sub["reversal_score"] >= top_rev).sum()
            emp_p = (1 + n_beat) / (1 + n_controls)
            print(f"  Empirical p-value (top improved vs all): {emp_p:.4f}")

        # Print improved analogues
        if len(improved) > 0:
            cols_show = ["DRUG_NAME", "delta_improvement", "reversal_score",
                         "delta_reversal_vs_parent", "best_lincs_sim", "best_lincs_name"]
            print(f"\n  Improved analogues reversal scores:")
            print(improved[cols_show].sort_values("reversal_score", ascending=False).to_string(index=False))

    if not all_results:
        print("  No results to save")
        return

    # ── Save combined results ─────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    save_cols = meta_cols + ["reversal_score", "delta_reversal_vs_parent", "is_improved", "cancer_tag"]
    combined[save_cols].to_csv(OUT_DIR / "reversal_scores.csv", index=False)
    print(f"\n  Saved → {OUT_DIR}/reversal_scores.csv ({len(combined)} rows)")

    # Build summary JSON
    summary = {}
    for cancer_tag in ["LUAD", "SKCM"]:
        seed = {"LUAD": "erlotinib", "SKCM": "trametinib"}[cancer_tag]
        sub = combined[combined["cancer_tag"] == cancer_tag]
        imp = sub[sub["is_improved"]]
        if len(sub) == 0:
            continue
        parent_path = LINCS_DIR / f"parent_signature_{seed}.csv"
        parent_rev = None
        if parent_path.exists():
            pdf = pd.read_csv(parent_path)
            up_g   = load_gene_list(DISEASE_DIR / f"{cancer_tag}_up_genes_lincs.txt") or \
                     load_gene_list(DISEASE_DIR / f"{cancer_tag}_up_genes.txt")
            down_g = load_gene_list(DISEASE_DIR / f"{cancer_tag}_down_genes_lincs.txt") or \
                     load_gene_list(DISEASE_DIR / f"{cancer_tag}_down_genes.txt")
            g_cols = [c for c in pdf.columns if c not in ["DRUG_NAME", "seed_drug"]]
            if len(pdf) > 0 and up_g and down_g:
                parent_rev = reversal_score(pdf.iloc[0], up_g, down_g, g_cols)

        summary[cancer_tag] = {
            "n_analogues":    int(len(sub)),
            "n_improved":     int(len(imp)),
            "parent_reversal_score": round(float(parent_rev), 4) if parent_rev else None,
            "mean_reversal_all":     round(float(sub["reversal_score"].mean()), 4),
            "mean_reversal_improved": round(float(imp["reversal_score"].mean()), 4) if len(imp) else None,
            "mean_delta_reversal_improved": round(float(imp["delta_reversal_vs_parent"].mean()), 4)
                                             if len(imp) else None,
            "n_improved_with_positive_delta_reversal": int((imp["delta_reversal_vs_parent"] > 0).sum())
                                                        if len(imp) else 0,
            "improved_details": [
                {
                    "DRUG_NAME":    r["DRUG_NAME"],
                    "delta_improvement": round(float(r["delta_improvement"]), 6),
                    "reversal_score":    round(float(r["reversal_score"]), 4),
                    "delta_reversal":    round(float(r["delta_reversal_vs_parent"]), 4),
                    "best_lincs_sim":    round(float(r["best_lincs_sim"]), 3),
                    "best_lincs_name":   r["best_lincs_name"],
                }
                for _, r in imp.sort_values("delta_improvement", ascending=False).iterrows()
            ]
        }

    with open(OUT_DIR / "reversal_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
