#!/usr/bin/env python3
"""
CTRdb Dual-Drug Patient Inference and Combination Validation
=============================================================
Validates GAUGE drug combination recommendations on real patients from
the Cancer Treatment Response database (CTRdb) who received dual-drug
(combination) therapy and have known clinical outcomes.

DATA SOURCES:
  - CTR_RNAseq.h5ad  (3109 patients, RNA-seq, raw counts)
  - CTR_Microarray.h5ad (7507 patients, microarray, log2 expression)

NORMALIZATION STRATEGY:
  - RNAseq: log1p(counts) → gene-wise quantile mapping to GDSC distribution
  - Microarray: log2 values directly → gene-wise quantile mapping to GDSC
  Both strategies use the same quantile mapping used for TCGA inference.
  This is appropriate because quantile mapping aligns the per-gene rank
  distribution regardless of absolute scale, making it robust to different
  normalization methods.

PIPELINE:
  1. Filter for dual-drug patients with both drugs in GDSC model
  2. Normalize expression → extract HVG2000 genes → quantile map → impute → scale
  3. Apply model (checkpoint: full_20260524_140841) → predict all drug sensitivities
  4. Compute complementarity scores for actual drug pairs
  5. Validate against clinical response (Response vs Non_response labels)

OUTPUTS:
  results/
    ctrdb_predictions_rnaseq.csv      - Per-patient drug predictions (RNAseq)
    ctrdb_predictions_microarray.csv  - Per-patient drug predictions (Microarray)
    ctrdb_dual_combo_scores.csv       - Combo scores for dual-drug patients
    table_response_validation.csv     - Response AUC and statistics
    table_response_by_cancer.csv      - Per-cancer-type breakdown
    figure_data_response.csv          - Combined data for figures
"""
from __future__ import annotations

import pickle
import os
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.metrics import roc_auc_score

KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))
DATA_ROOT = Path(os.environ.get("KGPUB_DATA_ROOT", "/mnt/raid5/xujing/Agent/Datasets"))
sys.path.insert(0, os.environ.get("KGPUB_PY_ROOT", str(KG_ROOT)))

# ── Import pipeline utilities from existing TCGA scripts ─────────────────────
from GAUGE.external import (
    _align_projected_states,
    _drug_lookup,
    _predict_many_pairs,
)
from GAUGE.train import load_model

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = KG_ROOT / "benchmarking/01_random_cell_split/HVG2000/results/full_20260524_140841"
GDSC_EXPR_PATH = KG_ROOT / "KG_GAUGE_PublicData/GDSC/rnaseq_merged_rsem_tpm_20260323.csv"
GDSC_GENEID_PATH = KG_ROOT / "KG_GAUGE_PublicData/GDSC/gene_identifiers_20241212.csv"
CTRDB_RNASEQ = DATA_ROOT / "ctrdbv2/CTR_RNAseq.h5ad"
CTRDB_MICRO = DATA_ROOT / "ctrdbv2/CTR_Microarray.h5ad"

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_U = 0.1
DEVICE = "cpu"
SEED = 42
rng = np.random.RandomState(SEED)

# ── TCGA cross-patient Pearson reference (v5 complementarity formula) ─────────
TCGA_PREDS = KG_ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000/strategy_quantile_map_hvg2000_task01/results/predictions.csv"

# CTRdb cancer_type (Cancer_type_level1) → TCGA project_id
_CANCER_MAP: dict[str, str] = {
    "breast cancer":                                          "TCGA-BRCA",
    "breast ductal carcinoma":                                "TCGA-BRCA",
    "breast lobular carcinoma":                               "TCGA-BRCA",
    "invasive breast carcinoma":                              "TCGA-BRCA",
    "breast carcinoma":                                       "TCGA-BRCA",
    "urinary bladder cancer":                                 "TCGA-BLCA",
    "bladder urothelial carcinoma":                           "TCGA-BLCA",
    "lung cancer":                                            "TCGA-LUAD",
    "lung non-small cell carcinoma":                          "TCGA-LUAD",
    "lung adenocarcinoma":                                    "TCGA-LUAD",
    "lung squamous cell carcinoma":                           "TCGA-LUSC",
    "melanoma":                                               "TCGA-SKCM",
    "skin cancer":                                            "TCGA-SKCM",
    "soft tissue tumours":                                    "TCGA-SARC",
    "leiomyosarcoma":                                         "TCGA-SARC",
    "dedifferentiated liposarcoma":                           "TCGA-SARC",
    "myxofibrosarcoma":                                       "TCGA-SARC",
    "ovarian cancer":                                         "TCGA-OV",
    "malignant ovarian surface epithelial-stromal neoplasm":  "TCGA-OV",
    "ovarian carcinoma":                                      "TCGA-OV",
    "uterine cancer":                                         "TCGA-UCEC",
    "bile duct cancer":                                       "TCGA-CHOL",
    "bile duct adenocarcinoma":                               "TCGA-CHOL",
    "mesothelioma":                                           "TCGA-MESO",
    "malignant biphasic mesothelioma":                        "TCGA-MESO",
    "stomach cancer":                                         "TCGA-STAD",
    "gastric adenocarcinoma":                                 "TCGA-STAD",
    "liver cancer":                                           "TCGA-LIHC",
    "esophageal cancer":                                      "TCGA-ESCA",
    "esophagus squamous cell carcinoma":                      "TCGA-ESCA",
    "pancreatic cancer":                                      "TCGA-PAAD",
    "pancreatic adenocarcinoma":                              "TCGA-PAAD",
    "cervix carcinoma":                                       "TCGA-CESC",
}

# pair_key (sorted, "|"-separated) → TCGA fallback when cancer_type is missing
_PAIR_FALLBACK: dict[str, str] = {
    "Dabrafenib|Trametinib": "TCGA-SKCM",
    "MK-2206|Paclitaxel":    "TCGA-BRCA",
}

print("Loading TCGA predictions for cross-patient Pearson computation...")
_tcga_raw = pd.read_csv(TCGA_PREDS)
_vh_tcga: pd.DataFrame = (
    _tcga_raw.groupby(["entity_id", "DRUG_NAME"], observed=True)["value_hat"]
    .mean().unstack("DRUG_NAME")
)
_tcga_cancer: pd.Series = (
    _tcga_raw.drop_duplicates("entity_id").set_index("entity_id")["project_id"]
)
print(f"  TCGA pivot: {_vh_tcga.shape}")
_pearson_cache: dict[tuple, float] = {}


def _map_cancer_to_tcga(cancer_type: str, pair_key: str) -> str | None:
    ct = str(cancer_type).strip().lower()
    if ct and ct != "nan":
        if ct in _CANCER_MAP:
            return _CANCER_MAP[ct]
        for key, proj in _CANCER_MAP.items():
            if key.split()[0] in ct:
                return proj
    return _PAIR_FALLBACK.get(pair_key, None)


def _get_cross_patient_pearson(drug_a: str, drug_b: str,
                                tcga_project: str | None) -> float:
    cache_key = (drug_a, drug_b, tcga_project or "pan")
    if cache_key in _pearson_cache:
        return _pearson_cache[cache_key]
    if tcga_project and tcga_project != "pan":
        pids = _tcga_cancer[_tcga_cancer == tcga_project].index
    else:
        pids = _vh_tcga.index
    if drug_a not in _vh_tcga.columns or drug_b not in _vh_tcga.columns:
        _pearson_cache[cache_key] = 0.0
        return 0.0
    sub_a = _vh_tcga.loc[_vh_tcga.index.intersection(pids), drug_a].dropna()
    sub_b = _vh_tcga.loc[_vh_tcga.index.intersection(pids), drug_b].dropna()
    common = sub_a.index.intersection(sub_b.index)
    va, vb = sub_a.loc[common].values.astype(np.float64), sub_b.loc[common].values.astype(np.float64)
    if len(va) < 5 or np.std(va) < 1e-9 or np.std(vb) < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(va, vb)[0, 1])
        if np.isnan(r):
            r = 0.0
    _pearson_cache[cache_key] = r
    return r


def norm_drug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# ── Load model and artifacts ──────────────────────────────────────────────────
print("Loading GAUGE model...")
with open(CHECKPOINT_DIR / "artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)
model = load_model(CHECKPOINT_DIR, artifacts, strict=True)
model.eval()

genes: list[str] = artifacts.genes
print(f"  HVG genes: {len(genes)}")
print(f"  State dim: {artifacts.state_dim}")

# Drug lookup
drug_lookup = _drug_lookup(artifacts)
# drug_lookup: dict[str_key → dict with DRUG_ID, DRUG_NAME, fingerprint, etc.]
gdsc_drugs_norm = {norm_drug(d["DRUG_NAME"]): d for d in drug_lookup.values()}
print(f"  Model drugs: {len(drug_lookup)}")

# ── Load GDSC expression (source distribution for quantile mapping) ───────────
print("Loading GDSC expression for quantile mapping reference...")
from GAUGE.data import load_gdsc_expression
gdsc_expr = load_gdsc_expression(GDSC_EXPR_PATH, GDSC_GENEID_PATH)
fit_cells = [c for c, role in artifacts.split_by_cell.items()
             if role == "fit" and c in gdsc_expr.index]
source_hvg = (
    gdsc_expr.loc[fit_cells, genes]
    .reindex(columns=genes, fill_value=0.0)
    .astype(np.float32)
    .to_numpy()
)
print(f"  GDSC fit cells used: {len(fit_cells)}")
print(f"  Source HVG shape: {source_hvg.shape}")

# ── Quantile mapping functions (from TCGA script) ─────────────────────────────
def _quantile_map_gene(source_col: np.ndarray, target_col: np.ndarray) -> np.ndarray:
    src = np.asarray(source_col, dtype=np.float32)
    tgt = np.asarray(target_col, dtype=np.float32)
    src_sorted = np.sort(src)
    ranks = np.argsort(np.argsort(tgt, kind="mergesort"), kind="mergesort").astype(np.float32)
    quantiles = (ranks + 0.5) / max(len(tgt), 1)
    grid = np.linspace(0.0, 1.0, num=len(src_sorted), endpoint=False, dtype=np.float32) + (
        0.5 / max(len(src_sorted), 1)
    )
    mapped = np.interp(quantiles, grid, src_sorted, left=src_sorted[0], right=src_sorted[-1])
    return mapped.astype(np.float32, copy=False)

def _quantile_map_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = np.empty_like(target, dtype=np.float32)
    for idx in range(target.shape[1]):
        out[:, idx] = _quantile_map_gene(source[:, idx], target[:, idx])
    return out

def _append_zero_state_padding(states_2d: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = int(states_2d.shape[1])
    if current_dim == target_dim:
        return states_2d.astype(np.float32, copy=False)
    if current_dim > target_dim:
        return states_2d[:, :target_dim].astype(np.float32, copy=False)
    pad = np.zeros((states_2d.shape[0], target_dim - current_dim), dtype=np.float32)
    return np.concatenate([states_2d.astype(np.float32, copy=False), pad], axis=1)


def process_expression_to_states(
    expr_matrix: np.ndarray,
    gene_names: list[str],
    data_type: str,   # "rnaseq" or "microarray"
) -> np.ndarray:
    """
    Transform patient expression matrix to model state vectors.

    Steps:
      1. Normalize expression based on data type:
         - RNAseq (raw counts): log1p(counts) → log scale ~0-11
         - Microarray (log2):   use as-is
      2. Extract HVG2000 genes → align to model gene order
      3. Quantile map gene-by-gene to GDSC train distribution
      4. Impute (mean) → StandardScaler → PCA/identity
      5. Align states (add kg prior if needed)
    """
    n_samples = expr_matrix.shape[0]
    n_target = len(genes)

    # ── Step 1: Normalize ────────────────────────────────────────────────────
    if data_type == "rnaseq":
        # Raw counts → log1p
        expr_normed = np.log1p(expr_matrix.astype(np.float32))
    else:
        # Microarray already log2 normalized, convert nan to 0
        expr_normed = np.where(np.isnan(expr_matrix), 0.0, expr_matrix).astype(np.float32)

    # ── Step 2: Extract HVG genes ─────────────────────────────────────────────
    gene_to_idx = {g: i for i, g in enumerate(gene_names) if g}
    hvg_matrix = np.zeros((n_samples, n_target), dtype=np.float32)
    n_found = 0
    for target_pos, gene in enumerate(genes):
        if gene in gene_to_idx:
            hvg_matrix[:, target_pos] = expr_normed[:, gene_to_idx[gene]]
            n_found += 1
    print(f"    HVG genes found: {n_found}/{n_target} ({n_found/n_target:.1%})")

    # ── Step 3: Quantile mapping ──────────────────────────────────────────────
    print("    Applying quantile mapping...")
    hvg_aligned = _quantile_map_matrix(source_hvg, hvg_matrix)

    # ── Step 4: Impute → Scale ────────────────────────────────────────────────
    imputed = artifacts.imputer.transform(hvg_aligned)
    scaled = artifacts.scaler.transform(imputed)

    # ── Step 5: PCA projection ────────────────────────────────────────────────
    states_core = artifacts.pca.transform(scaled).astype(np.float32, copy=False)
    state_dim = int(artifacts.state_dim or states_core.shape[1])
    states = _append_zero_state_padding(states_core, state_dim)
    states = _align_projected_states(states, artifacts)
    return states


def parse_dual_drugs(drug_list_str: str) -> tuple[str | None, str | None]:
    """Parse 'DrugA+DrugB' → (DrugA, DrugB) if both in model. None otherwise."""
    parts = str(drug_list_str).split("+")
    if len(parts) != 2:
        return None, None
    a, b = parts[0].strip(), parts[1].strip()
    a_mapped = gdsc_drugs_norm.get(norm_drug(a))
    b_mapped = gdsc_drugs_norm.get(norm_drug(b))
    if a_mapped is not None and b_mapped is not None:
        return a, b
    return None, None


def get_drug_attr(drug_raw: str, attr: str):
    d = gdsc_drugs_norm.get(norm_drug(drug_raw))
    return d[attr] if d is not None else None


def run_inference_and_validate(
    h5ad_path: Path,
    data_type: str,
    output_prefix: str,
) -> pd.DataFrame:
    """
    Full pipeline: load CTRdb data → infer → compute combo scores → validate.
    Returns combo score dataframe for dual-drug patients.
    """
    print(f"\n{'='*60}")
    print(f"Processing {data_type.upper()}: {h5ad_path.name}")
    print(f"{'='*60}")

    adata = ad.read_h5ad(h5ad_path)
    obs = adata.obs.copy()
    X = adata.X
    if sps.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    print(f"  Total patients: {len(obs)}")
    print(f"  Genes: {X.shape[1]}")

    # Gene names
    if "gene_name" in adata.var.columns:
        gene_names = adata.var["gene_name"].astype(str).tolist()
    else:
        gene_names = [str(g).split(".")[0] for g in adata.var_names]

    # ── Filter to dual-drug patients with both drugs in model ─────────────────
    obs["_n_plus"] = obs["Drug_list"].astype(str).str.count(r"\+")
    dual_mask = obs["_n_plus"] == 1
    dual_obs = obs[dual_mask].copy()
    dual_obs[["drug_A_raw", "drug_B_raw"]] = pd.DataFrame(
        [parse_dual_drugs(d) for d in dual_obs["Drug_list"]],
        index=dual_obs.index,
        columns=["drug_A_raw", "drug_B_raw"],
    )
    dual_obs = dual_obs[dual_obs["drug_A_raw"].notna() & dual_obs["drug_B_raw"].notna()].copy()

    # Map to GDSC drug objects
    dual_obs["drug_A_name"] = dual_obs["drug_A_raw"].apply(
        lambda d: gdsc_drugs_norm[norm_drug(d)]["DRUG_NAME"]
    )
    dual_obs["drug_B_name"] = dual_obs["drug_B_raw"].apply(
        lambda d: gdsc_drugs_norm[norm_drug(d)]["DRUG_NAME"]
    )
    dual_obs["drug_A_id"] = dual_obs["drug_A_raw"].apply(
        lambda d: gdsc_drugs_norm[norm_drug(d)]["DRUG_ID"]
    )
    dual_obs["drug_B_id"] = dual_obs["drug_B_raw"].apply(
        lambda d: gdsc_drugs_norm[norm_drug(d)]["DRUG_ID"]
    )

    print(f"  Dual-drug patients with both drugs in model: {len(dual_obs)}")

    if len(dual_obs) < 10:
        print("  SKIP: too few patients")
        return pd.DataFrame()

    # Get expression matrix for dual-drug patients
    dual_idx = [obs.index.get_loc(i) for i in dual_obs.index]
    X_dual = X[dual_idx, :]
    print(f"  Expression shape: {X_dual.shape}")

    # ── Process expression → states ────────────────────────────────────────────
    states = process_expression_to_states(X_dual, gene_names, data_type)
    print(f"  State vectors shape: {states.shape}")

    # ── Generate drug predictions for all drugs ───────────────────────────────
    print("  Generating model predictions...")
    entity_ids = dual_obs.index.astype(str).tolist()
    all_drugs = list(drug_lookup.values())

    # Batch prediction for all drugs × all patients
    pred_rows = []
    batch = 256
    for start in range(0, len(entity_ids), batch):
        end = min(start + batch, len(entity_ids))
        batch_states = states[start:end]
        batch_ids = entity_ids[start:end]
        drug_chunks = [all_drugs[i:i+128] for i in range(0, len(all_drugs), 128)]
        for chunk in drug_chunks:
            pr = _predict_many_pairs(
                model, batch_states, batch_ids,
                [chunk for _ in range(len(batch_ids))],
                device=DEVICE, batch_size=2048
            )
            pred_rows.append(pr)
        if (start // batch) % 5 == 0:
            print(f"    Processed {end}/{len(entity_ids)} patients...")

    predictions = pd.concat(pred_rows, ignore_index=True)
    predictions["base_score"] = predictions["value_hat"] - LAMBDA_U * predictions["uncertainty"]
    predictions.to_csv(RESULTS_DIR / f"ctrdb_predictions_{output_prefix}.csv", index=False)
    print(f"  Saved predictions: {len(predictions)} rows")

    # ── Compute personalized combo scores ─────────────────────────────────────
    pred_by_patient = {eid: grp for eid, grp in predictions.groupby("entity_id")}
    combo_rows = []

    for i, (patient_id, row) in enumerate(dual_obs.iterrows()):
        eid = str(patient_id)
        if eid not in pred_by_patient:
            continue
        p = pred_by_patient[eid].set_index("DRUG_NAME")["base_score"]

        bA = float(p.get(row["drug_A_name"], np.nan))
        bB = float(p.get(row["drug_B_name"], np.nan))
        if np.isnan(bA) or np.isnan(bB):
            continue

        pcp = bA * bB
        additive = bA + bB
        max_single = max(bA, bB)

        # Complementarity v5: pcp × (1 − cross_patient_Pearson)
        pk = "|".join(sorted([row["drug_A_name"], row["drug_B_name"]]))
        tcga_proj = _map_cancer_to_tcga(str(row.get("Cancer_type_level1", "")), pk)
        cross_r = _get_cross_patient_pearson(row["drug_A_name"], row["drug_B_name"], tcga_proj)
        complementarity = pcp * (1.0 - cross_r)

        # Response binary
        resp_str = str(row.get("Response", "")).strip()
        if resp_str == "Response":
            y = 1
        elif resp_str == "Non_response":
            y = 0
        else:
            y = None

        combo_rows.append({
            "patient_id": patient_id,
            "data_type": data_type,
            "cancer_type": row.get("Cancer_type_level1", ""),
            "drug_A_name": row["drug_A_name"],
            "drug_A_id": row["drug_A_id"],
            "drug_B_name": row["drug_B_name"],
            "drug_B_id": row["drug_B_id"],
            "pair_key": pk,
            "base_score_A": bA,
            "base_score_B": bB,
            "pcp": pcp,
            "complementarity": complementarity,
            "cross_patient_r": cross_r,
            "additive": additive,
            "max_single": max_single,
            "response_str": resp_str,
            "response_binary": y,
        })

    combo_df = pd.DataFrame(combo_rows)
    print(f"  Combo scores computed: {len(combo_df)} patients")
    print(f"  With response: {combo_df['response_binary'].notna().sum()}")
    if combo_df["response_binary"].notna().sum() > 0:
        pos = int(combo_df["response_binary"].eq(1).sum())
        neg = int(combo_df["response_binary"].eq(0).sum())
        print(f"  Response: {pos} responders, {neg} non-responders")

    return combo_df


# ── Run both data types ────────────────────────────────────────────────────────
combo_rna = run_inference_and_validate(CTRDB_RNASEQ, "rnaseq", "rnaseq")
combo_micro = run_inference_and_validate(CTRDB_MICRO, "microarray", "microarray")

# ── Combine and validate ───────────────────────────────────────────────────────
all_combo = pd.concat([combo_rna, combo_micro], ignore_index=True)
all_combo.to_csv(RESULTS_DIR / "ctrdb_dual_combo_scores.csv", index=False)
print(f"\nCombined CTRdb dual-drug dataset: {len(all_combo)} patients")

resp_valid = all_combo[all_combo["response_binary"].notna()].copy()
resp_valid["response_binary"] = resp_valid["response_binary"].astype(int)
print(f"With valid response labels: {len(resp_valid)}")

# ── Response AUC validation ────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESPONSE VALIDATION (CTRdb dual-drug patients)")
print("="*60)

def compute_auc_stats(y_true, y_score, label, n_perm=3000):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) < 10 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"label": label, "n": len(y_true), "auc": np.nan, "perm_p": np.nan, "mw_p": np.nan}
    auc = float(roc_auc_score(y_true, y_score))
    null = [roc_auc_score(y_true, rng.permutation(y_score)) for _ in range(n_perm)]
    p_perm = max(1/n_perm, float((np.array(null) >= auc).mean()))
    _, mw_p = mannwhitneyu(y_score[y_true==1], y_score[y_true==0], alternative="greater")
    return {
        "label": label, "n": len(y_true),
        "n_pos": int(y_true.sum()), "n_neg": int((y_true==0).sum()),
        "auc": round(auc, 4), "perm_p": round(p_perm, 5),
        "mw_p": round(float(mw_p), 5),
    }

auc_rows = []
for score_col, label in [
    ("complementarity", "Complementarity v5 (pcp×(1-r))"),
    ("pcp", "Personalized co-activity (pcp)"),
    ("additive", "Additive (A+B)"),
    ("max_single", "Max single agent"),
    ("base_score_A", "Drug A alone"),
    ("base_score_B", "Drug B alone"),
]:
    if score_col not in resp_valid.columns:
        continue
    for dt_label, dt_sub in [("Pooled", resp_valid), ("RNAseq", resp_valid[resp_valid["data_type"]=="rnaseq"]), ("Microarray", resp_valid[resp_valid["data_type"]=="microarray"])]:
        if len(dt_sub) < 10:
            continue
        m = compute_auc_stats(dt_sub["response_binary"], dt_sub[score_col], f"{dt_label}|{label}")
        m["data_type"] = dt_label
        m["score"] = score_col
        auc_rows.append(m)
        auc_val = m.get("auc", np.nan)
        p_val = m.get("perm_p", np.nan)
        print(f"  [{dt_label}] {label}: AUC={auc_val:.4f}, n={m.get('n')}, "
              f"pos={m.get('n_pos')}, p={p_val:.4f}")

pd.DataFrame(auc_rows).to_csv(RESULTS_DIR / "table_response_validation.csv", index=False)

# ── Per-cancer breakdown ───────────────────────────────────────────────────────
print("\nPer-cancer-type response AUC (complementarity score, pooled):")
cancer_rows = []
for cancer, grp in resp_valid.groupby("cancer_type"):
    if len(grp) < 10:
        continue
    m = compute_auc_stats(grp["response_binary"], grp["complementarity"], cancer)
    m["cancer_type"] = cancer
    m["n_responders"] = int(grp["response_binary"].eq(1).sum())
    m["n_non_responders"] = int(grp["response_binary"].eq(0).sum())
    cancer_rows.append(m)
    print(f"  {cancer}: AUC={m.get('auc', 'nan')}, n={m.get('n')}, "
          f"pos={m.get('n_pos')}, p={m.get('perm_p', 'nan')}")

pd.DataFrame(cancer_rows).to_csv(RESULTS_DIR / "table_response_by_cancer.csv", index=False)

# ── Per-drug-combo breakdown ────────────────────────────────────────────────────
print("\nPer-drug-combo response statistics (n>=5):")
combo_stats = resp_valid.groupby("pair_key").agg(
    n_total=("patient_id", "count"),
    n_responders=("response_binary", "sum"),
    mean_complementarity=("complementarity", "mean"),
    std_complementarity=("complementarity", "std"),
    mean_pcp=("pcp", "mean"),
    cross_r=("cross_patient_r", "first"),
    resp_rate=("response_binary", "mean"),
).reset_index()
combo_stats = combo_stats[combo_stats["n_total"] >= 5].sort_values("mean_complementarity", ascending=False)
combo_stats.to_csv(RESULTS_DIR / "table_combo_stats.csv", index=False)
print(combo_stats.to_string(index=False))

# Save figure data
resp_valid.to_csv(RESULTS_DIR / "figure_data_response.csv", index=False)

print(f"\n[DONE] All outputs saved to {RESULTS_DIR}")
