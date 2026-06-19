# Step 3：TCGA 患者药物组合筛选（探索性）

## 重要前提：这一步没有金标准

TCGA 患者**没有组合用药后的疗效记录**，因此第三层分析是**探索性的**，不能作为主要证据。

使用间接替代指标（总生存期 OS）做间接验证，逻辑是：
> 若模型正确预测了患者的药物敏感性特征，  
> 则"适合接受高强度药物治疗"的患者（combo_potential 高）  
> 在实际临床中应有更好的生存结局（OS 更长）。

这个逻辑链有**至少两个弱点**：
1. TCGA 患者的实际用药并不由 GAUGE 推荐，因此无法直接因果推断
2. combo_potential 是工程性指标（最敏感 10 种药的均值），与实际组合用药的关系未严格论证

**这一层的价值在于**：提供个体化推荐的全景视图，以及初步的生物学合理性验证。

---

## 数据规模

| 指标 | 数值 |
|------|------|
| TCGA 患者总数 | 11,370 人 |
| 覆盖癌种 | 33 种 |
| 评分药物数 | 274 种 |
| 评分药物对数（per 癌种） | 37,401 对（C(274,2)） |
| 总评分条目 | 33 × 37,401 = 1,234,233 对 |

预测使用 `strategy_quantile_map_hvg2000_task01` 流程（分位映射 + GAUGE）的预计算预测文件。

---

## Analysis A：PCA 聚类（质量控制）

以 GAUGE 预测的药物敏感性谱对 11,370 名患者做 PCA：
- PC1 解释 **28.0%** 方差，PC2 解释 **22.9%**（分位映射版本）
- 不同癌种在 PC 空间中形成可识别的聚类，说明模型从表达谱中提取到了癌种特异的药物敏感性信息

PC1 梯度（从高敏感到低敏感）：

| PC1 极端 | 代表癌种 | 生物学解读 |
|---------|---------|-----------|
| 最负端（最敏感） | TCGA-KICH、TCGA-LIHC、TCGA-LGG | 与已知较高药物反应性一致 |
| 最正端（最不敏感） | TCGA-DLBC、TCGA-THYM、TCGA-TGCT | 淋巴瘤/胸腺瘤/睾丸癌用药体系不同 |

PCA 结果支持模型捕捉到了有生物学意义的癌种差异，而非随机噪声。

---

## Analysis B：药物组合推荐

对每个癌种计算所有 37,401 对药物的互补性得分，排序取 Top-20。

**评分公式（癌种均值）**：
```
complementarity(A, B, cancer) = mean_pcp(A,B,cancer) × (1 − cross_patient_Pearson(A,B,cancer))
mean_pcp = 癌种内所有患者的 A×B 均值
```

**代表性推荐（各癌种 Top-1）**：

| 癌种 | 药物 A | 药物 B | 互补性 | Pearson r | 生物学依据 |
|------|--------|--------|--------|---------|-----------|
| TCGA-BRCA | Acetalax | Olaparib | 0.592 | −0.487 | PARP 抑制 + 代谢调节 |
| TCGA-SKCM | PD0325901 | Sepantronium bromide | 0.716 | −0.164 | MEK 抑制 + BCL2 途径 |
| TCGA-LUAD | N-acetyl cysteine | SB505124 | 0.656 | −0.481 | 氧化应激 + TGF-β |
| TCGA-OV | SB505124 | SN-38 | 0.638 | −0.371 | TGF-β + 拓扑异构酶 |
| TCGA-LGG | Daporinad | Tretinoin | 0.750 | −0.505 | NAMPT + 视黄酸（IDH 突变相关） |
| TCGA-CESC | Docetaxel | SGC0946 | 0.578 | −0.429 | 微管 + 表观遗传 |

所有高互补性对均满足 r < 0（机制互补），体现公式的选择性。

---

## Analysis C：生存分析

**定义**：`combo_potential` = 该患者对"最敏感前 10 种药物"的平均 `value_hat`（越低越敏感）

将 11,370 名患者按 combo_potential 中位数分为高/低两组，做 Kaplan-Meier 生存对比。

**有统计意义的癌种（KM p < 0.05）**：

| 癌种 | n | 事件 | Cox HR | Cox p | **KM p** | 方向 |
|------|---|------|--------|-------|---------|------|
| **TCGA-CESC（宫颈癌）** | 118 | 27 | 0.018 | **0.004** | **0.034** | 高敏感 → 更好 OS ✓ |

**趋势（KM p 0.05–0.15）**：

| 癌种 | n | 事件 | Cox p | KM p | 方向 |
|------|---|------|-------|------|------|
| TCGA-LGG（低级别胶质瘤） | 201 | 55 | 0.056 | 0.109 | 低敏感 → 更好（反向）|
| TCGA-UCEC（子宫内膜癌） | 148 | 24 | 0.210 | 0.102 | 高敏感 → 更好 |
| TCGA-KIRC（肾透明细胞癌） | 33 | 29 | 0.081 | 0.252 | 高敏感 → 更好 |

**多数癌种（>15 种）不显著**。

**CESC（宫颈癌）Cox p=0.004**：宫颈癌对铂类和紫杉醇一线化疗有较明确的反应差异，combo_potential 可能捕捉到了真实的化疗敏感性分层。但需注意 n=118 较小，且 Cox HR 极小（0.018）提示存在数值不稳定。

**LGG 反向**：低 combo_potential（不太敏感）预后反而更好。可能的解释：LGG 患者预后主要由 IDH 突变状态决定，IDH 突变型对化疗相对不敏感但预后更好，导致方向性混淆。这说明 combo_potential 在某些癌种中不是简单的单向指标。

---

## 结论的边界

**可成立的主张**：
- ✅ GAUGE 从 11,370 名患者的表达谱中提取出癌种特异的药物敏感性结构（PCA 有意义聚类）
- ✅ 宫颈癌（CESC）中高 combo_potential 与更好 OS 显著相关（KM p=0.034，Cox p=0.004）
- ✅ 推荐的高互补性组合均满足 r < 0（机制互补），生物学合理性有据可查
- ✅ 覆盖 33 种癌症的全景推荐，为个体化治疗提供参考

**不可成立的主张**：
- ✗ 多数癌种生存分析不显著（>15 种），不能说"泛癌种生存验证"
- ✗ LGG 方向相反，说明 combo_potential 的临床解读是癌种依赖的
- ✗ 无因果关系——患者实际用药不由 GAUGE 推荐，OS 关联不能直接解读为"推荐用药使患者活更长"
- ✗ 推荐列表不能作为直接临床指导，需要独立实验验证
- ✗ 第三层不能单独支撑论文的主要结论，只能作为探索性分析放补充材料或讨论部分

---

## 输出文件说明

| 文件 | 内容 |
|------|------|
| `survival_analysis_all_cancers.csv` | 18 个癌种 Cox + KM 生存分析结果 |
| `top5_recommendations_all_cancers.csv` | 各癌种 Top-5 组合推荐（rank/药对/complementarity/r） |
| `top20_combo_by_complementarity.csv` | 各癌种 Top-20 组合（按互补性排序） |
| `patient_profile_pca_all.csv` | 11,370 名患者的 PCA 坐标（PC1-5） |
| `cancer_type_centroids.csv` | 各癌种质心（33 癌种） |
| `patient_profile_diversity_all.csv` | 各癌种药物敏感性异质性统计 |
| `patient_combo_potential_all.csv` | 每名患者的 combo_potential 得分 |
| `summary.json` | 关键统计摘要 |

---

## 复现

```bash
conda activate kg_GAUGE
cd /mnt/raid5/xujing/KG/Combined/multicancer_contextual_v2

# 约 10 分钟（依赖已有 TCGA 预测文件）
python publication_submission/_archive_cell_split/strategy_quantile_map_hvg2000_task01_all_patient/run_all_patient_analysis.py
```
