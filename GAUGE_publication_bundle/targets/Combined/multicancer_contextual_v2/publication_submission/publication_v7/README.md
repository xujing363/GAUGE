# GAUGE 药物组合预测 — 发表版数据包 v7

**模型**：GAUGE，仅在 GDSC 单药数据上训练  
**核心主张**：零样本药物组合预测——无需任何组合训练数据  
**整理日期**：2026-05-29  

---

## 整体逻辑

药物组合预测的证明需要三层递进的独立证据：

```
第一层（最强）：细胞系实验室验证
  NCI-ALMANAC 是独立于 GDSC 的组合用药金标准
  → 模型在细胞系上的预测 vs 真实协同实验结果

第二层（临床直接验证）：真实患者 + 已知疗效
  CTRdb：接受双药联合治疗的患者，有 CR/PR/SD/PD 记录
  → 预测得分是否区分响应者与非响应者

第三层（探索性）：患者筛选 + 间接验证
  TCGA：无组合用药金标准，以生存期作为间接替代
  → 高预测潜力的患者是否有更好的生存结局
```

---

## 各层核心结果

### 第一层：NCI-ALMANAC 细胞系验证

**来源**：`step1_nci_cellline/`  
**模型版本**：使用全部该癌种 GDSC 细胞系（v6，all-cell 版本）

| 癌种 | 细胞数 | 药对数 | 协同率 | AUC | 95% CI | p 值 | 是否显著 |
|------|--------|--------|--------|-----|--------|------|---------|
| 黑色素瘤 | 54 | 119 | 34.5% | 0.655 | 0.538–0.767 | 0.003 | ✓ |
| NSCLC | 84 | 119 | 38.7% | 0.580 | 0.462–0.693 | 0.069 | ✗ |
| **乳腺癌** | 51 | 119 | 32.8% | **0.705** | 0.604–0.802 | **0.0004** | **✓✓** |
| 卵巢癌 | 40 | 119 | 34.5% | 0.656 | 0.546–0.762 | 0.003 | ✓ |
| **汇总** | — | **476** | 35.1% | **0.642** | 0.587–0.697 | **0.0002** | **✓✓✓** |

**KG 富集**（知识图谱引导候选对的协同率 vs NCI 背景）：  
汇总富集比 = **1.62×**，二项检验 p = **1.45×10⁻¹¹**

### 第二层：CTRdb 真实患者验证

**来源**：`step2_ctrdb_patient/`  
**数据**：CTRdb v2，RNA-seq 队列 n=185（130 响应，55 非响应）

| 评分方式 | AUC | p 值 | 说明 |
|---------|-----|------|------|
| **互补性（complementarity）** | **0.699** | **0.00033** | 最优 ✓ |
| 最佳单药（max_single） | 0.685 | 0.00033 | 次之 |
| 加和（Drug A + B） | 0.639 | 0.001 | — |
| pcp（协同活性） | 0.621 | 0.005 | — |

**关键药物对**（v5 逐对分析）：

| 药物对 | n | R/NR | AUC | p | 生物学意义 |
|--------|---|------|-----|---|-----------|
| **Dabrafenib + Trametinib** | 28 | 14/14 | **0.745** | **0.015** | BRAF+MEK 靶向，最强生物学依据 |
| Cisplatin + Gemcitabine | 59 | 40/19 | 0.576 | 0.175 | DNA 损伤（机制相似，被降权） |

### 第三层：TCGA 患者筛选（探索性）

**来源**：`step3_tcga_screening/`  
**数据**：11,370 TCGA 患者 × 33 癌种 × 274 药物

生存分层有统计意义的癌种（combo_potential vs OS）：

| 癌种 | n | 事件 | Cox p | KM p | 结论 |
|------|---|------|-------|------|------|
| **TCGA-CESC（宫颈癌）** | 118 | 27 | **0.004** | **0.034** | 高敏感 → 更好 OS ✓ |
| TCGA-LGG（低级别胶质瘤） | 201 | 55 | 0.056 | 0.109 | 趋势 |
| TCGA-UCEC（子宫内膜癌） | 148 | 24 | 0.210 | 0.102 | 趋势 |

---

## 目录结构

```
publication_v7/
├── README.md                        ← 本文件（总览）
├── step1_nci_cellline/              ← 第一层：NCI-ALMANAC 细胞系验证
│   ├── README.md
│   ├── results/                     ← 主要结果表格（10 个文件）
│   └── scripts/                     ← 可复现脚本（3 个）
├── step2_ctrdb_patient/             ← 第二层：CTRdb 真实患者验证
│   ├── README.md
│   ├── results/                     ← 主要结果表格（7 个文件）
│   └── scripts/                     ← 可复现脚本（2 个）
├── step3_tcga_screening/            ← 第三层：TCGA 患者筛选（探索性）
│   ├── README.md
│   ├── results/                     ← 主要结果表格（8 个文件）
│   └── scripts/                     ← 可复现脚本（1 个）
└── supplement/                      ← 补充分析
    ├── baseline_comparison/         ← vs 4 种无监督基线
    ├── supervised_ceiling/          ← vs 有监督方法上限
    └── kg_advantage/                ← KG 富集详细分析
```

**注意**：所有数据来源于 `_archive_cell_split/`，原始文件未做任何修改。

---

## 评分公式

```
complementarity(A, B) = median_c[A(c) × B(c)] × (1 − Pearson(A, B))
                         ─────────────────────   ──────────────────────
                              pcp（协同活性）        机制多样性因子
```

- `A(c)`, `B(c)` = `value_hat − 0.1 × uncertainty`（保守基础评分）  
- `Pearson` = 跨细胞（或跨患者）A/B 敏感性谱的 Pearson 相关系数  
- 公式理论依据：Bliss 独立性模型——机制不同且各自有效的药物组合产生协同

---

## 复现方法

```bash
conda activate kg_GAUGE
cd /mnt/raid5/xujing/KG/Combined/multicancer_contextual_v2

# 第一层 NCI 验证（全 3 步，约 16 分钟，无 GPU）
python publication_submission/_archive_cell_split/all_cell_line/scripts/01_allcell_score_computation.py
python publication_submission/_archive_cell_split/all_cell_line/scripts/02_nci_validation_allcell.py
python publication_submission/_archive_cell_split/all_cell_line/scripts/03_kg_advantage_analysis.py

# 第二层 CTRdb 验证（约 40 分钟，无 GPU）
python publication_submission/_archive_cell_split/all_cell_line/two_patient/scripts/01_ctrdb_inference_and_combo_validation.py

# 第三层 TCGA 筛选（约 10 分钟，无 GPU，依赖已有预测文件）
python publication_submission/_archive_cell_split/strategy_quantile_map_hvg2000_task01_all_patient/run_all_patient_analysis.py
```

---

## 必须披露的局限

1. **NSCLC 不显著**：NCI 验证中 NSCLC AUC=0.580，p=0.069，未达 0.05 阈值
2. **CTRdb Microarray 不显著**：AUC=0.429，p=0.941；合并分析（n=361）不显著（p=0.73）
3. **BL-3 原始数据接近**：无监督基线 BL-3（GDSC 癌种反相关）AUC=0.614，与 GAUGE 接近（v5 test-only 0.616）；all-cell 版本 0.642 有改进
4. **TCGA 无组合金标准**：第三层为探索性，生存分析只有宫颈癌 Cox p<0.05，多数不显著
5. **乳腺癌单药接近天花板**：Breast pcp=0.766 虽超有监督 RF=0.756，但 max_single 不作为主要对比
