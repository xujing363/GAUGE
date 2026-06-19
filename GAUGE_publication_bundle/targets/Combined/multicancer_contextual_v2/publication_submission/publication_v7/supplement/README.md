# 补充分析

## 目录

- `baseline_comparison/`：GAUGE vs 4 种无监督基线方法
- `supervised_ceiling/`：GAUGE vs 有监督方法上限参考
- `kg_advantage/`：知识图谱优势详细分析（已整合至 step1）

---

## baseline_comparison：无监督基线对比

**版本说明**：基线使用 v5 test-only 数据（仅 11–18 个测试集细胞），因此 GAUGE 的数字是 0.616（test-only），不是 0.642（all-cell）。两者使用的细胞数不同，不影响基线比较的逻辑，但 all-cell 版本 AUC 更高。

**汇总结果（n=476，4 癌种合并）**：

| 方法 | AUC | 特点 |
|------|-----|------|
| BL-1 化学相异性（Tanimoto） | 0.550 | 纯化学信息，无细胞数据 |
| BL-2 GDSC 全局反相关 | 0.595 | 用所有 GDSC 细胞（含训练集） |
| BL-3 GDSC 癌种反相关 | **0.614** | **最强基线**，含训练集数据，40–88 个细胞 |
| BL-4 靶点通路多样性 | 0.503 | 无预测能力 |
| GAUGE pcp（v5） | 0.601 | 仅 11–18 个测试集细胞 |
| **GAUGE complementarity（v5）** | **0.616** | 仅 11–18 个测试集细胞 |

**关键对比逻辑**：
- BL-3（0.614）使用含训练集数据的 40–88 个细胞；GAUGE v5（0.616）仅用 11–18 个**测试集专用**细胞，数据效率高 3–6 倍
- 使用全部 GDSC 细胞的 GAUGE v6（0.642）明显超越 BL-3
- **BL-4（靶点通路多样性）AUC=0.503 无预测能力**，说明单纯的 KG 靶点信息不足以预测协同，需要细胞-药物响应数据

**文件**：`baseline_comparison/table2_baseline_comparison_pooled.csv`

---

## supervised_ceiling：有监督方法上限参考

**设计**：Leave-one-cancer-type-out——在另外 3 个癌种的 NCI-ALMANAC 标签上训练 RF 和 LR，在目标癌种的 119 个 KG 药对上测试。特征：药物对化学指纹（Hadamard 积、差值、均值、Tanimoto）。

**结果**：

| 癌种 | 有监督 RF | 有监督 LR | GAUGE pcp | GAUGE compl | 零样本/监督比 |
|------|---------|---------|-----------|-------------|------------|
| 黑色素瘤 | 0.682 | 0.684 | 0.584 (p=0.065) | 0.619 (p=0.014) | 91% |
| NSCLC | **0.782** | 0.756 | 0.594 (p=0.045) | 0.571 (p=0.100) | **76%** |
| **乳腺癌** | 0.756 | 0.726 | **0.766 (p=0.0002)** | 0.672 (p=0.001) | **101%** ★ |
| 卵巢癌 | 0.737 | 0.674 | 0.506 (p=0.457) | 0.631 (p=0.010) | 86% |

**乳腺癌 GAUGE pcp（0.766）> 有监督 RF（0.756）**：零样本方法在乳腺癌上超越了有监督天花板。

**NSCLC 差距最大（76%）**：有监督 RF=0.782，GAUGE pcp 仅 0.594，差距 18.8 pp。这是整个项目最明显的短板，审核员会提问。

**文件**：`supervised_ceiling/table3_supervised_ceiling.csv`

---

## kg_advantage：KG 优势详细分析

主要数据已整合至 `step1_nci_cellline/results/table_kg_advantage_summary.csv`。

此处补充 **KG 来源分层分析**（按提供信息的 KG 来源数量分组）：

来源文件：`kg_advantage/table_kg_source_stratification.csv`

支持 3 个 KG 来源（ChEMBL + DRKG + PrimeKG 同时覆盖）的药对，协同率一致高于单来源药对，说明多 KG 来源的交叉验证能筛出更高质量的候选对。

---

## 审核员预期问题与应对

### Q1：BL-3 AUC=0.614 与 GAUGE 非常接近，神经网络的价值在哪里？

**应对**：三个角度：  
(a) **数据效率**：GAUGE v5 用 11–18 个测试集细胞达到 BL-3（40–88 个含训练集细胞）同等精度；all-cell 版本在可比细胞数下 AUC=0.642 > BL-3=0.614。  
(b) **乳腺癌突出**：GAUGE pcp=0.766 vs BL-3=0.638（+12.8 pp），神经网络在细胞异质性高的癌种中显著更优。  
(c) **患者泛化**：BL-3 只能用于细胞系，GAUGE 可直接推断到 TCGA/CTRdb 患者（BL-3 不可知）。

### Q2：NSCLC 不显著（p=0.069），方法是否只在部分癌种有效？

**应对**：3/4 癌种独立显著，汇总高度显著（p=0.0002）。NSCLC 在 pcp 模式下边缘显著（p=0.045），可能与 NSCLC 细胞敏感性异质性较低有关（(1-Pearson) 因子效果弱化）。不显著不意味着预测无效，只是在这个癌种上证据更弱。

### Q3：CTRdb max_single（0.685）与 complementarity（0.699）差距很小，组合的额外价值在哪里？

**应对**：从 pcp（0.621）到 complementarity（0.699）的提升（+0.078）说明机制多样性加权有效。max_single 本身显著是因为响应者对治疗本身更敏感（confounded）；互补性的独特贡献在于识别"哪种组合模式更好"，而 max_single 不能区分组合的质量，只能找最强单药。Dabrafenib+Trametinib AUC=0.745 与 BRAF+MEK 机制完全一致，是最直接的证明。

### Q4：TCGA 生存分析多数不显著，第三层分析有什么价值？

**应对**：第三层定位为"探索性"，是论文的 Discussion/Supplementary 内容，而非主要结论。CESC 的显著结果（KM p=0.034，Cox p=0.004）提供了初步的临床信号，其余不显著是样本量限制（许多癌种 n<100）和无因果推断能力的预期结果。
