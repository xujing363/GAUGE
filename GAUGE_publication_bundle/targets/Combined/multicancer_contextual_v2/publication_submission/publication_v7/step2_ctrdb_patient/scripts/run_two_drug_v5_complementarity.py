"""
双药患者验证 v5 — 统一使用 profile_complementarity 评分

评分规则与 NCI-ALMANAC 细胞系验证完全一致：

  complementarity(patient) = pcp(patient) × (1 − cross_patient_Pearson(A, B))

其中：
  pcp(patient)              = value_hat_A(patient) × value_hat_B(patient)
                              患者特异，来自 GAUGE 对 CTR 患者的推断
  cross_patient_Pearson(A,B)= Pearson({value_hat_A(p)} , {value_hat_B(p)})
                              药物对特异，用 TCGA 大样本患者估计机制多样性
                              （类比细胞系验证中的 Pearson 跨细胞相关）

癌种映射（CTR 癌症名称 → TCGA project_id，用于查找参考 Pearson）：
  Bladder urothelial carcinoma  → TCGA-BLCA
  Lung non-small cell carcinoma → TCGA-LUAD
  Breast * carcinoma            → TCGA-BRCA
  Malignant ovarian *           → TCGA-OV
  Leiomyosarcoma / SARC         → TCGA-SARC
  Cervix carcinoma              → TCGA-CESC
  Bile duct adenocarcinoma      → TCGA-CHOL
  Dabrafenib+Trametinib (NaN)   → TCGA-SKCM (BRAF-mutant reference)
  其他 / 无法映射               → pan-cancer (全 11370 人)

输出：results_v5/
旧版 results_final/ 不修改。
"""
from __future__ import annotations

import os
import warnings, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))
TCGA_PREDS = KG_ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000/strategy_quantile_map_hvg2000_task01/results/predictions.csv"
CTR_SCORES = KG_ROOT / "Combined/multicancer_contextual_v2/publication_submission/two_patient/results_final/ctr_combined_scores.csv"
OUT_DIR = Path(__file__).resolve().parent / "results_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CTR cancer_type → TCGA project_id
CANCER_MAP = {
    "bladder urothelial carcinoma":                          "TCGA-BLCA",
    "lung non-small cell carcinoma":                         "TCGA-LUAD",
    "lung adenocarcinoma":                                   "TCGA-LUAD",
    "lung squamous cell carcinoma":                          "TCGA-LUSC",
    "breast ductal carcinoma":                               "TCGA-BRCA",
    "breast lobular carcinoma":                              "TCGA-BRCA",
    "invasive breast carcinoma":                             "TCGA-BRCA",
    "breast carcinoma":                                      "TCGA-BRCA",
    "malignant ovarian surface epithelial-stromal neoplasm": "TCGA-OV",
    "ovarian carcinoma":                                     "TCGA-OV",
    "leiomyosarcoma":                                        "TCGA-SARC",
    "dedifferentiated liposarcoma":                          "TCGA-SARC",
    "myxofibrosarcoma":                                      "TCGA-SARC",
    "cervix carcinoma":                                      "TCGA-CESC",
    "bile duct adenocarcinoma":                              "TCGA-CHOL",
    "gastric adenocarcinoma":                                "TCGA-STAD",
    "esophagus squamous cell carcinoma":                     "TCGA-ESCA",
    "pancreatic adenocarcinoma":                             "TCGA-PAAD",
    "malignant biphasic mesothelioma":                       "TCGA-MESO",
}

# Dabrafenib+Trametinib 无 cancer_type → 用 SKCM（BRAF V600E 参考）
PAIR_FALLBACK_CANCER = {
    "Dabrafenib+Trametinib": "TCGA-SKCM",
    "MK-2206+Paclitaxel":    "TCGA-BRCA",   # HER2+ breast 常用方案
}

# ── 加载 TCGA 预测 → 全量 pivot ───────────────────────────────────────────────
print("=" * 68)
print("Loading TCGA predictions pivot matrix...")
tcga = pd.read_csv(TCGA_PREDS)
vh_tcga = (
    tcga.groupby(["entity_id", "DRUG_NAME"], observed=True)["value_hat"]
    .mean().unstack("DRUG_NAME")
)
tcga_cancer = (
    tcga.drop_duplicates("entity_id").set_index("entity_id")["project_id"]
)
print(f"  TCGA pivot: {vh_tcga.shape}")

# 预先计算每个 (药物A, 药物B, TCGA_project) 的 cross_patient Pearson
# 使用懒加载缓存（只计算 CTR 实际用到的药对）
_pearson_cache: dict[tuple, float] = {}

def get_cross_patient_pearson(drug_a: str, drug_b: str,
                               tcga_project: str | None) -> float:
    """
    用 TCGA 同癌种患者估计药物 A/B 的跨患者 Pearson 相关。
    找不到→用 pan-cancer（全 TCGA）。
    结果缓存，避免重复计算。
    """
    cache_key = (drug_a, drug_b, tcga_project or "pan")
    if cache_key in _pearson_cache:
        return _pearson_cache[cache_key]

    # 选 TCGA 患者子集
    if tcga_project and tcga_project != "pan":
        pids = tcga_cancer[tcga_cancer == tcga_project].index
    else:
        pids = vh_tcga.index  # pan-cancer

    # 两个药物在该子集的 value_hat 向量
    if drug_a not in vh_tcga.columns or drug_b not in vh_tcga.columns:
        _pearson_cache[cache_key] = 0.0
        return 0.0

    va = vh_tcga.loc[pids, drug_a].dropna().values.astype(np.float64)
    vb = vh_tcga.loc[pids, drug_b].dropna().values.astype(np.float64)

    # 只对两者都有值的患者计算
    mask = ~(np.isnan(va) | np.isnan(vb))
    # 其实 fillna 已经做了，这里作为防护
    va_m, vb_m = va[mask], vb[mask]

    if len(va_m) < 5 or np.std(va_m) < 1e-9 or np.std(vb_m) < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(va_m, vb_m)[0, 1])
        if np.isnan(r):
            r = 0.0

    _pearson_cache[cache_key] = r
    return r

# ── 加载 CTR 已有评分（pcp 已计算） ──────────────────────────────────────────
print("\nLoading CTR combined scores (results_final)...")
df = pd.read_csv(CTR_SCORES)
print(f"  {len(df)} rows, pairs: {df['drug_pair'].nunique()}")

# ── 计算 complementarity ──────────────────────────────────────────────────────
print("\nComputing complementarity = pcp × (1 − cross_patient_Pearson)...")

def map_cancer_to_tcga(cancer_type: str, drug_pair: str) -> str | None:
    """CTR cancer_type 字符串 → TCGA project_id"""
    if pd.isna(cancer_type) or str(cancer_type).strip().lower() == "nan":
        return PAIR_FALLBACK_CANCER.get(drug_pair, None)
    ct_lower = str(cancer_type).strip().lower()
    # 精确匹配
    if ct_lower in CANCER_MAP:
        return CANCER_MAP[ct_lower]
    # 模糊匹配（含 breast / lung / ovarian 等）
    for key, proj in CANCER_MAP.items():
        if key.split()[0] in ct_lower:
            return proj
    return None  # 无法映射 → pan-cancer

pearson_r_vals   = []
complementarity_vals = []

for _, row in df.iterrows():
    tcga_proj = map_cancer_to_tcga(row.get("cancer_type", ""), row["drug_pair"])
    dA = str(row["drug_A"])
    dB = str(row["drug_B"])
    r  = get_cross_patient_pearson(dA, dB, tcga_proj)
    pcp = row["pcp_score"]
    compl = float(pcp) * (1.0 - r) if not (np.isnan(pcp)) else np.nan
    pearson_r_vals.append(round(r, 4))
    complementarity_vals.append(round(compl, 6) if not np.isnan(compl) else np.nan)

df["tcga_ref_cancer"]  = [
    map_cancer_to_tcga(row.get("cancer_type",""), row["drug_pair"])
    for _, row in df.iterrows()
]
df["cross_patient_pearson_r"] = pearson_r_vals
df["complementarity"]         = complementarity_vals

# 打印每个药对的参考 Pearson
print("\n  Drug pair → TCGA reference → cross-patient Pearson r:")
for pair, grp in df.dropna(subset=["complementarity"]).groupby("drug_pair", observed=True):
    r_vals = grp["cross_patient_pearson_r"].unique()
    refs   = grp["tcga_ref_cancer"].unique()
    print(f"  {pair:35s} → ref={refs[0] or 'pan-cancer':15s}  r={r_vals[0]:+.4f}")

# 保存
df.to_csv(OUT_DIR / "ctr_combined_scores_v5.csv", index=False)

# 按 dataset 分别保存
df[df["dataset"] == "RNAseq"].to_csv(OUT_DIR / "ctr_rnaseq_scores_v5.csv",       index=False)
df[df["dataset"] == "Microarray"].to_csv(OUT_DIR / "ctr_microarray_scores_v5.csv", index=False)

print(f"\n  Saved {len(df)} rows with complementarity scores.")
print(f"  Pearson r range: [{min(pearson_r_vals):.4f}, {max(pearson_r_vals):.4f}]")
print(f"  Complementarity range (non-nan): "
      f"[{min(x for x in complementarity_vals if not np.isnan(x)):.4f}, "
      f"{max(x for x in complementarity_vals if not np.isnan(x)):.4f}]")

# ── 验证函数 ──────────────────────────────────────────────────────────────────
def validate(grp: pd.DataFrame, score_col: str, label: str) -> dict:
    g = grp.dropna(subset=[score_col, "response"])
    y = (g["response"] == "Response").astype(int)
    n_r, n_nr = int(y.sum()), int((1-y).sum())
    if n_r < 3 or n_nr < 3:
        return dict(label=label, score=score_col, n=len(g),
                    n_R=n_r, n_NR=n_nr, auc=np.nan, mwu_p=np.nan,
                    delta_median=np.nan, dir="—")
    try:
        auc = roc_auc_score(y, g[score_col])
    except Exception:
        auc = np.nan
    _, p = mannwhitneyu(g.loc[y==1, score_col], g.loc[y==0, score_col],
                        alternative="greater")
    med_r  = float(g.loc[y==1, score_col].median())
    med_nr = float(g.loc[y==0, score_col].median())
    dir_ok = "✓" if med_r > med_nr else "✗"
    return dict(label=label, score=score_col, n=len(g), n_R=n_r, n_NR=n_nr,
                auc=round(auc, 4), mwu_p=round(p, 5),
                median_R=round(med_r, 4), median_NR=round(med_nr, 4),
                delta_median=round(med_r - med_nr, 4), dir=dir_ok)

# ── 整体验证 ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Validation: pcp vs complementarity vs max_single")

val_rows = []
SCORES = ["complementarity", "pcp_score", "max_single_drug"]

# 整体
for sc in SCORES:
    val_rows.append(validate(df, sc, "CTR_combined_all"))

# 按 dataset
for ds in ["RNAseq", "Microarray"]:
    sub = df[df["dataset"] == ds]
    for sc in SCORES:
        val_rows.append(validate(sub, sc, f"CTR_{ds}"))

print(f"\n{'Label':20s} {'Score':22s} {'n':>5} {'R':>4} {'NR':>4} "
      f"{'AUC':>7} {'MWU_p':>8} {'Δmedian':>8} {'Dir'}")
print("─" * 90)
for r in val_rows:
    if np.isnan(r.get('auc', np.nan)):
        continue
    print(f"{r['label']:20s} {r['score']:22s} {r['n']:5d} {r['n_R']:4d} {r['n_NR']:4d} "
          f"{r['auc']:7.4f} {r['mwu_p']:8.5f} {r['delta_median']:8.4f} {r['dir']}")

# ── 逐药对验证 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("Per drug-pair: complementarity vs pcp (n>=10, R>=3, NR>=3)")

pair_rows = []
for (pair, ds), grp in df.groupby(["drug_pair", "dataset"], observed=True):
    g = grp.dropna(subset=["complementarity", "pcp_score"])
    if len(g) < 5: continue
    n_r = (g.response == "Response").sum()
    n_nr = (g.response == "Non_response").sum()
    if n_r < 3 or n_nr < 3: continue
    r_compl  = validate(g, "complementarity",  f"{pair}|{ds}")
    r_pcp    = validate(g, "pcp_score",        f"{pair}|{ds}")
    r_max    = validate(g, "max_single_drug",  f"{pair}|{ds}")
    # Pearson r for this pair
    pearson  = g["cross_patient_pearson_r"].iloc[0]
    ref_c    = g["tcga_ref_cancer"].iloc[0] or "pan"
    pair_rows.append(dict(
        drug_pair=pair, dataset=ds,
        n=len(g), R=int(n_r), NR=int(n_nr),
        tcga_ref=ref_c, pearson_r=round(pearson, 4),
        compl_AUC=r_compl["auc"], compl_p=r_compl["mwu_p"],
        pcp_AUC=r_pcp["auc"],     pcp_p=r_pcp["mwu_p"],
        max_AUC=r_max["auc"],
        compl_wins=1 if (not np.isnan(r_compl["auc"]) and not np.isnan(r_pcp["auc"])
                        and r_compl["auc"] > r_pcp["auc"]) else 0,
    ))

pair_df = pd.DataFrame(pair_rows).sort_values("compl_AUC", ascending=False, na_position="last")
pair_df.to_csv(OUT_DIR / "validation_by_pair_v5.csv", index=False)

print(f"\n{'Drug Pair':35s} {'DS':12s} {'n':>4} {'R':>3} {'NR':>3} "
      f"{'ref':15s} {'r':>6} │ "
      f"{'compl_AUC':>10} {'p':>8} │ {'pcp_AUC':>8} {'p':>8} │ "
      f"{'max_AUC':>8} │ {'compl>pcp'}")
print("─" * 130)
for _, r in pair_df.iterrows():
    win = "✓" if r.compl_wins else ""
    print(f"{r.drug_pair:35s} {r.dataset:12s} {r.n:4.0f} {r.R:3.0f} {r.NR:3.0f} "
          f"{str(r.tcga_ref):15s} {r.pearson_r:6.3f} │ "
          f"{r.compl_AUC:10.4f} {r.compl_p:8.5f} │ "
          f"{r.pcp_AUC:8.4f} {r.pcp_p:8.5f} │ "
          f"{r.max_AUC:8.4f} │ {win}")

# 汇总
total = len(pair_df.dropna(subset=["compl_AUC","pcp_AUC"]))
wins  = pair_df["compl_wins"].sum()
print(f"\n  complementarity > pcp: {wins}/{total} drug-pair × dataset combinations")
print(f"  Overall Pearson r range: [{pair_df['pearson_r'].min():.4f}, "
      f"{pair_df['pearson_r'].max():.4f}]")

# ── 保存汇总 ──────────────────────────────────────────────────────────────────
pd.DataFrame(val_rows).to_csv(OUT_DIR / "validation_summary_v5.csv", index=False)
print(f"\nAll results saved to {OUT_DIR}")
EOF
