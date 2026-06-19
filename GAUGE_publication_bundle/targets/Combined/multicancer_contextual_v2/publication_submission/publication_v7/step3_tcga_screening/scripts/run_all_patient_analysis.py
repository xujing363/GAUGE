"""
全量 TCGA 患者药物组合分析（11,370 人 × 33 癌种）

数据来源：strategy_quantile_map_hvg2000_task01/results/predictions.csv
          （已预计算，11370 患者 × 274 药物，quantile-map 校正）

与旧版 strategy_quantile_map_hvg2000_task01/results 的区别：
  旧版：combo_enrichment 仅覆盖 6 个目标癌种（3867 人）
  本版：覆盖全部 33 个癌种（11370 人）

分析内容：
  A. 全队列 PCA（11370 人）
  B. 全癌种药物组合富集（33 × C(274,2) = 1,234,233 对，向量化计算）
  C. 全癌种生存分析（combo_potential vs OS）
  D. 全癌种患者内部多样性统计
  E. 全癌种 Top-20 组合推荐汇总

输出：strategy_quantile_map_hvg2000_task01_all_patient/results/
"""
from __future__ import annotations

import warnings
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))
SRC_PREDS = KG_ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000/strategy_quantile_map_hvg2000_task01/results/predictions.csv"
SRC_ACT = KG_ROOT / "benchmarking/07_tcga_actual_treatment/HVG2000/strategy_quantile_map_hvg2000_task01/results/tcga_actual_treatment_scores.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 加载预测 ──────────────────────────────────────────────────────────────────
print("=" * 68)
print("Loading predictions (11,370 patients × 274 drugs)...")
preds = pd.read_csv(SRC_PREDS)
n_patients = preds["entity_id"].nunique()
n_drugs    = preds["DRUG_NAME"].nunique()
n_cancers  = preds["project_id"].nunique()
print(f"  {n_patients:,} patients × {n_drugs} drugs × {n_cancers} cancer types")

# 构建透视矩阵（全量）
print("Building pivot matrices...")
vh_pivot = (
    preds.groupby(["entity_id", "DRUG_NAME"], observed=True)["value_hat"]
    .mean().unstack("DRUG_NAME")
)
ah_pivot = (
    preds.groupby(["entity_id", "DRUG_NAME"], observed=True)["auc_hat"]
    .mean().unstack("DRUG_NAME")
)
drug_names = vh_pivot.columns.tolist()
n_d = len(drug_names)
patient_cancer = (
    preds.drop_duplicates("entity_id")
    .set_index("entity_id")["project_id"]
)
all_pids    = vh_pivot.index.tolist()
all_cancers = patient_cancer.reindex(all_pids).values
print(f"  Pivot: {vh_pivot.shape}  NaN rate: {vh_pivot.isna().mean().mean():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis A：全队列 PCA（11,370 人）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Analysis A: Full-cohort PCA (11,370 patients × 274 drugs)")

from sklearn.decomposition import PCA as skPCA

mat_full = vh_pivot.fillna(vh_pivot.mean()).values.astype(np.float32)
pca_full = skPCA(n_components=10, random_state=42)
pc_full  = pca_full.fit_transform(mat_full)
evr      = pca_full.explained_variance_ratio_

pca_df = pd.DataFrame({
    "patient_id":  all_pids,
    "cancer_type": all_cancers,
    "PC1": pc_full[:, 0], "PC2": pc_full[:, 1],
    "PC3": pc_full[:, 2], "PC4": pc_full[:, 3],
    "PC5": pc_full[:, 4],
})
pca_df.to_csv(OUT_DIR / "patient_profile_pca_all.csv", index=False)

print(f"  EVR: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, PC3={evr[2]:.3f}")
print(f"  Cumulative top-5: {evr[:5].sum():.3f}")

# 每癌种质心
centroid_rows = []
for ct in sorted(set(all_cancers)):
    mask = all_cancers == ct
    n    = mask.sum()
    c    = pc_full[mask, :5].mean(axis=0)
    centroid_rows.append({
        "cancer_type": ct, "n": int(n),
        "PC1": round(float(c[0]),4), "PC2": round(float(c[1]),4),
        "PC3": round(float(c[2]),4), "PC4": round(float(c[3]),4),
        "PC5": round(float(c[4]),4),
    })
centroid_df = pd.DataFrame(centroid_rows).sort_values("PC1")
centroid_df.to_csv(OUT_DIR / "cancer_type_centroids.csv", index=False)
print("\n  Cancer-type centroids (PC1, PC2):")
for _, r in centroid_df.sort_values("PC1").iterrows():
    print(f"    {r.cancer_type:15s}: ({r.PC1:+.3f}, {r.PC2:+.3f})  n={r.n}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis B：全癌种药物组合富集（向量化，33 癌种 × 37,401 对）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Analysis B: Drug Combination Enrichment — ALL 33 cancer types")

# 上三角索引（预先计算一次）
i_idx, j_idx = np.triu_indices(n_d, k=1)  # 37,401 对
drug_a_arr = np.array(drug_names)[i_idx]
drug_b_arr = np.array(drug_names)[j_idx]

all_combo_rows = []
diversity_rows = []

cancer_list = sorted(patient_cancer.unique())
print(f"  Processing {len(cancer_list)} cancer types...")

for ct in cancer_list:
    pids = patient_cancer[patient_cancer == ct].index.tolist()
    n_p  = len(pids)
    mat  = vh_pivot.reindex(pids).fillna(0).values.astype(np.float32)  # (n_p, 274)

    # ── mean pcp：向量化外积均值 ──────────────────────────────────────────
    mean_pcp_mat = (mat.T @ mat) / n_p   # (274, 274)
    pcp_vals = mean_pcp_mat[i_idx, j_idx]

    # ── cross-patient Pearson 相关矩阵（向量化）──────────────────────────
    mu   = mat.mean(axis=0, keepdims=True)
    std_ = mat.std(axis=0) + 1e-9
    mat_z = (mat - mu) / std_
    pearson_mat = (mat_z.T @ mat_z) / n_p   # (274, 274)
    r_vals    = pearson_mat[i_idx, j_idx]
    compl_vals = pcp_vals * (1.0 - r_vals)

    # 追加到列表（只保存 top-50 和完整对数的 parquet）
    chunk = pd.DataFrame({
        "cancer":                  ct,
        "drug_a":                  drug_a_arr,
        "drug_b":                  drug_b_arr,
        "mean_pcp":                pcp_vals.astype(np.float32),
        "cross_patient_pearson_r": r_vals.astype(np.float32),
        "inv_pearson":             (1.0 - r_vals).astype(np.float32),
        "complementarity":         compl_vals.astype(np.float32),
    })
    all_combo_rows.append(chunk)

    # 多样性统计
    std_per_drug = mat.std(axis=0)
    top3 = pd.Series(std_per_drug, index=drug_names).nlargest(3)
    diversity_rows.append({
        "cancer_type":            ct,
        "n_patients":             n_p,
        "median_per_drug_std":    round(float(np.median(std_per_drug)), 4),
        "mean_per_drug_std":      round(float(np.mean(std_per_drug)),   4),
        "top_variable_drug_1":    top3.index[0],
        "top_variable_drug_2":    top3.index[1],
        "top_variable_drug_3":    top3.index[2],
    })
    print(f"    {ct:15s}: {n_p:5d} patients, "
          f"top compl={compl_vals.max():.4f}")

print("  Concatenating all combo rows...")
combo_df = pd.concat(all_combo_rows, ignore_index=True)
combo_df.to_parquet(OUT_DIR / "combo_enrichment_all_cancers.parquet", index=False)
print(f"  Saved: {len(combo_df):,} rows → combo_enrichment_all_cancers.parquet")

# Top-20 per cancer (complementarity)
top_compl = (
    combo_df.sort_values("complementarity", ascending=False)
    .groupby("cancer", observed=True).head(20)
    .reset_index(drop=True)
)
top_compl.to_csv(OUT_DIR / "top20_combo_by_complementarity.csv", index=False)

# Top-20 per cancer (pcp)
top_pcp = (
    combo_df.sort_values("mean_pcp", ascending=False)
    .groupby("cancer", observed=True).head(20)
    .reset_index(drop=True)
)
top_pcp.to_csv(OUT_DIR / "top20_combo_by_pcp.csv", index=False)

# Diversity
div_df = pd.DataFrame(diversity_rows)
div_df.to_csv(OUT_DIR / "patient_profile_diversity_all.csv", index=False)
print(f"  Diversity saved: {len(div_df)} cancer types")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis C：全癌种生存分析
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Analysis C: Survival Analysis — all cancer types with OS data")

# 构建 combo_potential（全 11370 人）
ah_arr  = ah_pivot.reindex(all_pids).fillna(0).values.astype(np.float32)
sorted_auc = np.sort(ah_arr, axis=1)          # ascending：最敏感（最低 auc）优先
combo_pot  = pd.Series(
    -sorted_auc[:, :10].mean(axis=1),         # top-10 最敏感药物均值取负
    index=all_pids, name="combo_potential"
)
mean_vh = pd.Series(
    vh_pivot.reindex(all_pids).fillna(0).values.mean(axis=1),
    index=all_pids, name="mean_value_hat"
)

# 全量 combo_potential 保存
combo_pot_df = pd.DataFrame({
    "patient_id":       all_pids,
    "cancer_type":      all_cancers,
    "combo_potential":  combo_pot.values,
    "mean_value_hat":   mean_vh.values,
    "min_auc_hat":      ah_arr.min(axis=1),
    "mean_auc_hat":     ah_arr.mean(axis=1),
})
combo_pot_df.to_csv(OUT_DIR / "patient_combo_potential_all.csv", index=False)
print(f"  combo_potential saved: {len(combo_pot_df):,} patients")

# 加载 OS 数据
actual = pd.read_csv(SRC_ACT)
os_frame = (
    actual[["entity_id", "event", "time", "age_at_diagnosis", "project_id"]]
    .drop_duplicates("entity_id").set_index("entity_id")
    .dropna(subset=["time"])
)
os_frame = os_frame[os_frame["time"] >= 0]
print(f"  OS frame: {len(os_frame):,} patients, {int(os_frame['event'].sum())} events")

def km_logrank(t, e, high):
    try:
        from lifelines.statistics import logrank_test
        r = logrank_test(t[high], t[~high], e[high], e[~high])
        return float(r.test_statistic), float(r.p_value)
    except Exception:
        return np.nan, np.nan

def cox_ph(t, e, sc):
    try:
        from lifelines import CoxPHFitter
        df = pd.DataFrame({"time": t, "event": e, "score": sc}).dropna()
        if len(df) < 10 or df["event"].sum() < 3:
            return np.nan, np.nan, np.nan
        cph = CoxPHFitter()
        cph.fit(df, "time", "event")
        coef = float(cph.params_["score"])
        return coef, float(np.exp(coef)), float(cph.summary.loc["score", "p"])
    except Exception:
        return np.nan, np.nan, np.nan

surv_rows = []
for ct in cancer_list:
    os_sub = os_frame[os_frame["project_id"] == ct].copy()
    if len(os_sub) < 15 or os_sub["event"].sum() < 5:
        continue
    os_sub = os_sub.join(combo_pot,  how="inner")
    os_sub = os_sub.join(mean_vh,    how="inner")
    os_sub = os_sub.dropna(subset=["combo_potential", "time"])
    if len(os_sub) < 15:
        continue

    t = os_sub["time"].values
    e = os_sub["event"].values.astype(int)

    for score_col in ["combo_potential", "mean_value_hat"]:
        sc   = os_sub[score_col].values
        coef, hr, p_cox = cox_ph(t, e, sc)
        high = sc >= np.median(sc)
        stat, p_km = km_logrank(t, e, high)

        med_h = med_l = np.nan
        try:
            from lifelines import KaplanMeierFitter
            kmf_h, kmf_l = KaplanMeierFitter(), KaplanMeierFitter()
            kmf_h.fit(t[high], e[high]); kmf_l.fit(t[~high], e[~high])
            med_h = float(kmf_h.median_survival_time_)
            med_l = float(kmf_l.median_survival_time_)
        except Exception:
            pass

        surv_rows.append({
            "cancer":          ct,
            "score":           score_col,
            "n":               len(os_sub),
            "events":          int(e.sum()),
            "cox_hr":          round(hr,    4) if not np.isnan(hr)    else hr,
            "cox_p":           round(p_cox, 5) if not np.isnan(p_cox) else p_cox,
            "km_stat":         round(stat,  4) if not np.isnan(stat)  else stat,
            "km_p":            round(p_km,  5) if not np.isnan(p_km)  else p_km,
            "median_os_high":  round(med_h, 1) if not np.isnan(med_h) else med_h,
            "median_os_low":   round(med_l, 1) if not np.isnan(med_l) else med_l,
        })

surv_df = pd.DataFrame(surv_rows)
surv_df.to_csv(OUT_DIR / "survival_analysis_all_cancers.csv", index=False)
print(f"  Survival results: {len(surv_df)} rows ({len(surv_df)//2} cancer types with OS data)")

# 显示显著结果
sig_surv = surv_df[surv_df["km_p"] < 0.05].sort_values("km_p")
print(f"\n  Significant KM results (p<0.05): {len(sig_surv)}")
print(f"  {'Cancer':15s} {'Score':20s} {'n':>6} {'Events':>7} "
      f"{'Cox HR':>8} {'Cox p':>8} {'KM p':>8}")
for _, r in sig_surv.iterrows():
    print(f"  {r.cancer:15s} {r.score:20s} {r.n:6d} {r.events:7d} "
          f"{r.cox_hr:8.4f} {r.cox_p:8.5f} {r.km_p:8.5f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis E：Top-5 组合推荐汇总表（全癌种）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Analysis E: Top-5 combination recommendations per cancer type")

summary_rows = []
for ct in cancer_list:
    sub = top_compl[top_compl["cancer"] == ct].head(5)
    for rank, (_, row) in enumerate(sub.iterrows(), 1):
        summary_rows.append({
            "cancer":          ct,
            "rank":            rank,
            "drug_a":          row.drug_a,
            "drug_b":          row.drug_b,
            "mean_pcp":        round(float(row.mean_pcp), 4),
            "pearson_r":       round(float(row.cross_patient_pearson_r), 3),
            "complementarity": round(float(row.complementarity), 4),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "top5_recommendations_all_cancers.csv", index=False)
print(f"  Saved top-5 per cancer → {len(summary_df)} rows")

# 打印前3名
print("\n  Sample (top-3 per cancer, first 8 cancers):")
for ct in cancer_list[:8]:
    sub = summary_df[summary_df["cancer"] == ct].head(3)
    print(f"\n  [{ct}]")
    for _, r in sub.iterrows():
        print(f"    #{r.rank} {r.drug_a} + {r.drug_b}: "
              f"pcp={r.mean_pcp:.4f}, r={r.pearson_r:.3f}, "
              f"compl={r.complementarity:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 写报告
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 68)
print("Writing report...")

sig_km = surv_df[
    (surv_df["score"] == "combo_potential") & (surv_df["km_p"] < 0.05)
].sort_values("km_p")
sig_cox = surv_df[
    (surv_df["score"] == "combo_potential") & (surv_df["cox_p"] < 0.05)
].sort_values("cox_p")

lines = [
    "# GAUGE 全量 TCGA 患者药物组合推荐（11,370 人 × 33 癌种）",
    "",
    f"**数据来源**：`strategy_quantile_map_hvg2000_task01/results/predictions.csv`  ",
    f"**患者总数**：11,370（33 个 TCGA 癌种项目）  ",
    f"**药物数量**：{n_d}（GAUGE 模型药物库）  ",
    f"**药物对总数**：{n_d*(n_d-1)//2:,} × 33 癌种 = {n_d*(n_d-1)//2*33:,} 对  ",
    "**日期**：2026-05-29",
    "",
    "---",
    "",
    "## Analysis A：全队列 PCA",
    "",
    f"- 11,370 名患者 × {n_d} 药物 value_hat 向量做 PCA",
    f"- PC1={evr[0]*100:.1f}%，PC2={evr[1]*100:.1f}%，"
    f"PC3={evr[2]*100:.1f}%，累计前5={evr[:5].sum()*100:.1f}%",
    "",
    "各癌种质心（PC1 升序）：",
    "",
    "| 癌症 | n | PC1 | PC2 | PC3 |",
    "|------|---|-----|-----|-----|",
]
for _, r in centroid_df.iterrows():
    lines.append(f"| {r.cancer_type} | {r.n} | {r.PC1:+.3f} | {r.PC2:+.3f} | {r.PC3:+.3f} |")

lines += [
    "",
    "---",
    "",
    "## Analysis B：全癌种药物组合富集",
    "",
    f"- 共 {len(combo_df):,} 行（{n_d*(n_d-1)//2:,} 对 × {len(cancer_list)} 癌种）",
    "- 评分：mean_pcp（跨患者均值乘积）、cross_patient_pearson_r、complementarity = pcp × (1-r)",
    "",
    "**患者内部多样性（每药物跨患者 std 中位数）**：",
    "",
    "| 癌症 | 患者数 | 中位 per-drug std | Top1 变异药物 |",
    "|------|--------|-------------------|--------------|",
]
for _, r in div_df.sort_values("median_per_drug_std", ascending=False).iterrows():
    lines.append(
        f"| {r.cancer_type} | {r.n_patients} | {r.median_per_drug_std} | "
        f"{r.top_variable_drug_1} |"
    )

lines += [
    "",
    "---",
    "",
    "## Analysis C：生存分析（combo_potential vs OS）",
    "",
    f"combo_potential = −mean(auc_hat 最低 Top-10 药物)，按中位数分高低组。",
    f"共 {len(surv_df)//2} 个癌种有足够生存数据（n≥15，事件≥5）。",
    "",
    f"**KM 显著结果（p<0.05，{len(sig_km)} 个）**：",
    "",
    "| 癌症 | n | 事件 | Cox HR | Cox p | KM p | 中位OS高组 | 中位OS低组 |",
    "|------|---|------|--------|-------|------|-----------|-----------|",
]
for _, r in sig_km.iterrows():
    lines.append(
        f"| {r.cancer} | {r.n} | {r.events} | {r.cox_hr} | {r.cox_p} | "
        f"{r.km_p} | {r.median_os_high}d | {r.median_os_low}d |"
    )

if sig_cox.empty:
    lines.append("\n*（Cox 显著结果见 survival_analysis_all_cancers.csv）*")
else:
    lines += [
        "",
        f"**Cox 显著结果（p<0.05，{len(sig_cox)} 个）**：",
        "",
        "| 癌症 | n | 事件 | Cox HR | Cox p |",
        "|------|---|------|--------|-------|",
    ]
    for _, r in sig_cox.iterrows():
        lines.append(
            f"| {r.cancer} | {r.n} | {r.events} | {r.cox_hr} | {r.cox_p} |"
        )

lines += [
    "",
    "---",
    "",
    "## Analysis E：各癌种 Top-5 组合推荐（complementarity 排序）",
    "",
    "| 癌症 | Rank | 药物A | 药物B | pcp | r | complementarity |",
    "|------|------|-------|-------|-----|---|----------------|",
]
for _, r in summary_df[summary_df["rank"] <= 3].iterrows():
    lines.append(
        f"| {r.cancer} | {r.rank} | {r.drug_a} | {r.drug_b} | "
        f"{r.mean_pcp} | {r.pearson_r} | {r.complementarity} |"
    )

lines += [
    "",
    "---",
    "",
    "## 输出文件",
    "",
    "| 文件 | 内容 |",
    "|------|------|",
    "| `patient_profile_pca_all.csv` | 11,370 名患者 PC1-PC5 坐标 |",
    "| `cancer_type_centroids.csv` | 33 个癌种的 PCA 质心 |",
    "| `combo_enrichment_all_cancers.parquet` | 全量组合富集（1.2M 行）|",
    "| `top20_combo_by_complementarity.csv` | 每癌种 Top-20（按 complementarity）|",
    "| `top20_combo_by_pcp.csv` | 每癌种 Top-20（按 pcp）|",
    "| `patient_combo_potential_all.csv` | 全 11,370 名患者 combo_potential |",
    "| `patient_profile_diversity_all.csv` | 全 33 癌种多样性统计 |",
    "| `survival_analysis_all_cancers.csv` | 全癌种生存分析汇总 |",
    "| `top5_recommendations_all_cancers.csv` | 全癌种前5推荐汇总 |",
    "",
    "## 复现",
    "",
    "```bash",
    "conda activate kg_GAUGE",
    "cd /mnt/raid5/xujing/KG/Combined/multicancer_contextual_v2",
    "python publication_submission/strategy_quantile_map_hvg2000_task01_all_patient/"
    "run_all_patient_analysis.py",
    "```",
    "",
    "运行时间：约 5-10 分钟（CPU，无需 GPU，无模型推断，直接使用预计算预测）。",
]

report_path = Path(__file__).resolve().parent / "analysis_report.md"
report_path.write_text("\n".join(lines))
print(f"  Report → {report_path.name}")

# JSON 摘要
summary = {
    "n_patients": int(n_patients),
    "n_drugs": int(n_d),
    "n_cancer_types": int(n_cancers),
    "n_combo_pairs_total": int(len(combo_df)),
    "pca_evr_pc1": round(float(evr[0]), 4),
    "pca_evr_pc2": round(float(evr[1]), 4),
    "km_significant_cancers": sig_km["cancer"].tolist(),
    "cox_significant_cancers": sig_cox["cancer"].tolist(),
}
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

print(f"\n{'='*68}")
print(f"DONE. All results in {OUT_DIR}")
for fp in sorted(OUT_DIR.iterdir()):
    print(f"  {fp.name}  ({fp.stat().st_size // 1024} KB)")
