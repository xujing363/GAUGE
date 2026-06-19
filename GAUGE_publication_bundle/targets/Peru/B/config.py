"""
Scenario B: Drug Repurposing via Drug Network Perturbation
Configuration for all analysis scripts.

Scientific context:
  Drug split model: trained on ~198 drugs, tested on 56 never-seen drugs.
  Three prior knowledge graphs (三大先验网络):
    ChEMBL  - drug mechanism of action, direct target annotation
    DRKG    - broad biological context (pathways, diseases, side effects)
    PrimeKG - protein interaction network (indirect targets, pathway context)

  The model learns to dynamically allocate attention (alpha_ChEMBL, alpha_DRKG,
  alpha_PrimeKG) per (drug, cell) pair, transforming static KG into a dynamic,
  context-specific drug representation — the "world model" aspect.
"""
import os
from pathlib import Path

KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))

# ── Drug split canonical result dir (user-specified) ─────────────────────────
DRUG_SPLIT_RESULT_DIR     = KG_ROOT / "benchmarking/02_drug_split/HVG2000/results/true"
DRUG_SPLIT_PREPARED_PKL   = KG_ROOT / "benchmarking/02_drug_split/HVG2000/data/processed/default/prepared.pkl"
DRUG_SPLIT_CONFIG_YAML    = DRUG_SPLIT_RESULT_DIR / "default.yaml"

# ── GDSC cell-line metadata ───────────────────────────────────────────────────
GDSC_MODEL_LIST = KG_ROOT / "KG_GAUGE_PublicData/GDSC/model_list_20260420.csv"

# ── Pre-computed result files ─────────────────────────────────────────────────
PREDICTIONS_CSV          = DRUG_SPLIT_RESULT_DIR / "predictions.csv"
KG_ATTENTION_CSV         = DRUG_SPLIT_RESULT_DIR / "kg_attention_by_prediction.csv"
KG_COVERAGE_CSV          = DRUG_SPLIT_RESULT_DIR / "kg_coverage_by_drug.csv"
CHEMBL_EDGES_CSV         = DRUG_SPLIT_RESULT_DIR / "chembl_moa_edges.csv"
DRKG_EDGES_CSV           = DRUG_SPLIT_RESULT_DIR / "drkg_filtered_edges.csv"
PRIMEKG_EDGES_CSV        = DRUG_SPLIT_RESULT_DIR / "primekg_filtered_edges.csv"
GDSC_DRUG_PCC_CSV        = DRUG_SPLIT_RESULT_DIR / "gdsc_drug_pcc.csv"

# ── Output dirs ───────────────────────────────────────────────────────────────
B_DIR     = Path(__file__).parent
RESULTS   = B_DIR / "results"
FIGURES   = B_DIR / "figures"

# ── Runtime ───────────────────────────────────────────────────────────────────
DEVICE     = "cuda:0"
BATCH_SIZE = 8192
RANDOM_SEED = 42

# ── Focus drug-target pairs for case studies (all in test split) ─────────────
# These are drugs with known mechanisms and KG coverage
FOCAL_TEST_DRUGS = {
    "Erlotinib":    {"drug_id": 1168, "target": "EGFR",          "family": "EGFR_inhibitor",   "cancer": "Lung"},
    "Gefitinib":    {"drug_id": 1010, "target": "EGFR",          "family": "EGFR_inhibitor",   "cancer": "Lung"},
    "Osimertinib":  {"drug_id": 1919, "target": "EGFR",          "family": "EGFR_inhibitor",   "cancer": "Lung"},
    "Venetoclax":   {"drug_id": 1909, "target": "BCL2",          "family": "BCL2_inhibitor",   "cancer": "Haematological"},
    "Trametinib":   {"drug_id": 1372, "target": "MAP2K1/MAP2K2", "family": "MEK_inhibitor",    "cancer": "Skin"},
    "Crizotinib":   {"drug_id": 1083, "target": "ALK/MET",       "family": "ALK_inhibitor",    "cancer": "Lung"},
    "Talazoparib":  {"drug_id": 1259, "target": "PARP1/PARP2",   "family": "PARP_inhibitor",   "cancer": "Breast"},
    "Ruxolitinib":  {"drug_id": 1507, "target": "JAK1/JAK2",     "family": "JAK_inhibitor",    "cancer": "Haematological"},
    "Cisplatin":    {"drug_id": 1005, "target": "DNA",            "family": "DNA_damaging",     "cancer": "Bladder"},
    "Cytarabine":   {"drug_id": 1006, "target": "DNA_polymerase", "family": "antimetabolite",  "cancer": "Haematological"},
}

# Cancer type groupings for specificity analysis
CANCER_GROUPS = {
    "Lung":            ["Lung Non-Small Cell Carcinoma", "Lung Carcinoma", "Lung Adenocarcinoma",
                        "Non-Small Cell Lung Carcinoma", "Small Cell Lung Carcinoma"],
    "Haematological":  ["Diffuse Large B-Cell Lymphoma", "Acute Lymphoblastic Leukemia",
                        "Acute Myeloid Leukemia", "Multiple Myeloma", "Chronic Myeloid Leukemia",
                        "B-Cell Lymphoma", "T-Cell Lymphoma", "Lymphoma", "Leukemia"],
    "Skin":            ["Melanoma", "Cutaneous Melanoma"],
    "Breast":          ["Breast Carcinoma", "Breast Cancer", "Invasive Ductal Carcinoma"],
    "Colorectal":      ["Colorectal Carcinoma", "Colon Carcinoma", "Rectal Carcinoma", "Colorectal Adenocarcinoma"],
    "Pancreatic":      ["Pancreatic Carcinoma", "Pancreatic Ductal Adenocarcinoma"],
}
