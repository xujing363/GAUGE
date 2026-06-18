# Data sources behind GAUGE

## Model bundles (`models/`)

| Bundle | Training data | Cell-line ID system | Notes |
|---|---|---|---|
| `gdsc_cell_split` | GDSC1/2 response + RNA-seq (top-2000-variance genes) | Sanger `SIDM...` | Recommended default; held-out-cell-line split |
| `gdsc_drug_split` | Same as above | Sanger `SIDM...` | Held-out-drug split |
| `prism_secondary` | Broad Repurposing Hub PRISM secondary screen response + DepMap `OmicsExpressionTPMLogp1HumanProteinCodingGenes` | DepMap `ACH-...` | Held-out-drug split; ~1,400 compounds |

All three use the same three knowledge graphs:
- **ChEMBL** — curated mechanism-of-action / bioactivity knowledge graph.
- **DRKG** (Drug Repurposing Knowledge Graph) — gene/protein drug-target interaction edges.
- **PrimeKG** — protein-disease and pathway-association edges.

No patient data, mutation calls, copy-number variants, tumour stage, age, or
cancer-type labels are used as model inputs (see the model card's "inputs
explicitly not used" list). Exported bundle provenance (exact source
checkpoints, per-seed metrics) is recorded in each `models/<mode>/bundle_meta.json`.

## Demo data (`example_data/`)

Produced by `scripts/extract_demo_data.py` from real research data, never synthesized:

| File | Source | Used by |
|---|---|---|
| `example_tcga_patients*.csv` | TCGA pan-cancer RNA-seq + therapy records (`Agent/Datasets/TCGA/h5ad_outputs/tcga_gene_expression_tpm_therapies_split.h5ad`), filtered to single-agent treatments with a drug in a GAUGE library | Patient Stratification |
| `example_drugcomb_pairs.csv` | DrugComb `drugcombs_scored.csv`, filtered to pairs where both drugs are in the GDSC GAUGE library, with a fuzzy-matched bundled GDSC cell line where possible | Combination Scoring |
| `example_gtex_median_tpm_by_tissue.csv` | GTEx v11 median-TPM-by-tissue GCT, restricted to GAUGE's gene panel | Expression Data Analysis (tumour-vs-normal tab) |
| `example_design_*.csv` | The paper's actual REINVENT4-generated EGFR/ERBB lung-adenocarcinoma design run (`DrugDesign/05_tcga_egfr_erbb_design`) | Molecular Design Scoring |
| `example_expression_*.csv` | A small slice of the GDSC expression matrix, restricted to GAUGE's gene panel | Single/Batch Prediction, Drug Ranking, etc. templates |

## Datasets considered but intentionally not included

Per the published paper's scope, this software only covers GDSC, PRISM,
TCGA, DrugComb/NCI-ALMANAC-style combination data, DepMap, and GTEx. Other
datasets present in the original research repository (CTRP, BeatAML2, PDX)
are not represented in the paper and were deliberately left out of this
software.

## Re-exporting from a newly trained checkpoint

`scripts/export_model_bundle.py` is a one-time maintenance script (not part
of the deployed app) that converts a raw training-run directory into a slim,
portable bundle. It requires the full original training repository
(`GAUGE_DRUGWM_REPO`, defaults to a hardcoded path used by the authors'
internal training environment) and is not needed to *use* the app.
`scripts/extract_demo_data.py` similarly requires access to the underlying
research data directories and is not needed to use the app once
`example_data/` is populated.
