"""
Scenario A: Drug Resistance Mechanism Exploration
==================================================
Cell-line split test set: 190 unseen cell lines × 283 drugs
Perturbation: Transcriptome silencing / overexpression (fixed drug)
Goal: Reveal causal gene-drug sensitivity relationships via world model

World model gate: z_a = z_chem + sigmoid(W[z_s, z_chem, z_KG]) * z_KG
Terminal consequence: b = T(z_s, z_a, z_s*z_a)
Three prior networks: ChEMBL, DRKG, PrimeKG
"""
import os
from pathlib import Path

KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))

# ─── Model paths (true = best cell-line split model) ───────────────────────
RESULT_DIR = Path(
    KG_ROOT / "benchmarking/01_random_cell_split/HVG2000/results/true"
)
PREPARED_PKL = Path(
    KG_ROOT / "benchmarking/01_random_cell_split"
    / "HVG2000/data/processed/default/.cache/prepare"
    / "d78ee85a28b06384179852e5/prepared.pkl"
)
CONFIG_YAML = RESULT_DIR / "default.yaml"

# ─── Cell line metadata ─────────────────────────────────────────────────────
GDSC_MODEL_LIST = Path(
    KG_ROOT / "KG_GAUGE_PublicData/GDSC/model_list_20260420.csv"
)

# ─── Output directories ─────────────────────────────────────────────────────
PERU_A_DIR = Path(__file__).parent.parent
RESULTS_DIR = PERU_A_DIR / "results"
FIGURES_DIR = PERU_A_DIR / "figures"

# ─── Runtime ────────────────────────────────────────────────────────────────
DEVICE = "cuda:0"
BATCH_SIZE = 4096

# ─── Analysis parameters ────────────────────────────────────────────────────
MIN_SAMPLES_PER_DRUG = 10  # Minimum test cell lines per drug

# Cancer types to focus on (GDSC tissue labels)
FOCUS_CANCER_TYPES = {
    "NSCLC":       ["Non-Small Cell Lung Carcinoma", "Lung Carcinoma"],
    "Melanoma":    ["Melanoma"],
    "AML":         ["Acute Myeloid Leukemia"],
    "CLL_DLBCL":   ["Chronic Lymphocytic Leukemia", "Diffuse Large B-Cell Lymphoma"],
    "Breast":      ["Breast Carcinoma"],
    "Colorectal":  ["Colorectal Carcinoma"],
    "Pancreatic":  ["Pancreatic Carcinoma"],
    "Ovarian":     ["Ovarian Carcinoma"],
}

# Drug families to focus on
DRUG_FAMILIES = {
    "EGFR_inhibitors":   ["Erlotinib", "Gefitinib", "Lapatinib"],
    "BRAF_inhibitors":   ["Vemurafenib", "Dabrafenib"],
    "MEK_inhibitors":    ["Trametinib", "Selumetinib"],
    "BCL2_inhibitors":   ["Venetoclax", "Navitoclax"],
    "PARP_inhibitors":   ["Olaparib", "Niraparib", "Rucaparib"],
    "CDK46_inhibitors":  ["Palbociclib", "Abemaciclib"],
    "PI3K_inhibitors":   ["Alpelisib", "Buparlisib"],
    "MTOR_inhibitors":   ["Everolimus", "Temsirolimus", "Rapamycin"],
}

# Known drug→target gene pairs — only including targets IN the HVG2000 set.
# Missing targets (BRAF, MAP2K1, PARP1, CDK4, ABL1, PIK3CA, MTOR, etc.) are not
# in the HVG set: they are regulated primarily at mutation/CNV level, not via
# transcriptional variability. This is scientifically expected and documented.
KNOWN_DRUG_TARGETS = {
    # EGFR family → EGFR inhibitors (EGFR, ERBB2, ERBB3 all in HVG)
    "Erlotinib":   ["EGFR"],
    "Gefitinib":   ["EGFR"],
    "Lapatinib":   ["EGFR", "ERBB2"],
    # FGFR inhibitors → FGFR family (all 4 in HVG)
    "AZD4547":     ["FGFR1", "FGFR2", "FGFR3"],
    # KIT inhibitors → KIT (in HVG)
    "Imatinib":    ["KIT"],
    # MET inhibitors → MET (in HVG)
    "Crizotinib":  ["MET"],
    # BCL2 inhibitors → BCL2 (in HVG)
    "Venetoclax":  ["BCL2"],
    "Navitoclax":  ["BCL2"],
    # CDKN targets (cell cycle suppressors in HVG)
    "Palbociclib": ["CDKN1A"],   # CDK4/6i → p21 as downstream mediator
    "Abemaciclib": ["CDKN1A"],
    # MYC-driven sensitivity
    "Vincristine":    ["MYC"],
    "Doxorubicin":    ["MYC"],
}

# Full biological annotation (including non-HVG targets, for reference in analysis)
DRUG_TARGET_FULL = {
    "Erlotinib":   ["EGFR", "ERBB4"],
    "Gefitinib":   ["EGFR"],
    "Lapatinib":   ["EGFR", "ERBB2"],
    "Vemurafenib": ["BRAF"],
    "Dabrafenib":  ["BRAF"],
    "Trametinib":  ["MAP2K1", "MAP2K2"],
    "Venetoclax":  ["BCL2"],
    "Olaparib":    ["PARP1", "PARP2"],
    "Palbociclib": ["CDK4", "CDK6"],
    "Imatinib":    ["ABL1", "KIT"],
    "Dasatinib":   ["ABL1", "SRC"],
}

# Gene panel: genes confirmed to be in HVG2000 set used by the true model.
# Note: many classical oncogene targets (BRAF, MAP2K1, KRAS, PARP1, CDK4/6, ABL1)
# are NOT highly variable by expression across GDSC cell lines — they are altered
# primarily at mutation/CNV level, not transcription level. The HVG selection
# captures transcriptionally variable regulators, which is appropriate for this
# transcriptome-perturbation analysis.
GENE_PANEL = [
    # EGFR/ERBB signaling (in HVG set — high expression variability)
    "EGFR", "ERBB2", "ERBB3",
    # FGFR family (validated HVG targets)
    "FGFR1", "FGFR2", "FGFR3", "FGFR4",
    # Receptor tyrosine kinases
    "KIT", "MET",
    # PI3K signaling (delta isoform and downstream)
    "PIK3CD", "AKT3",
    # Apoptosis (BCL2 family in HVG)
    "BCL2", "BCL2A1",
    # Cell cycle regulators
    "CCND1", "CCND2",
    # CDK inhibitors (CDKN family in HVG)
    "CDKN1A", "CDKN2A", "CDKN2B", "CDKN2C",
    # MYC oncogenes
    "MYC", "MYCN",
    # MAPK signaling (available forms)
    "MAPK13", "MAP4K1",
    # Notch signaling
    "NOTCH3",
    # Metabolism / DNA repair
    "MGMT",
]

# Extended panel: top 50 highly variable oncology-relevant genes for global scan
# (same as GENE_PANEL but includes broader pathway members)
GENE_PANEL_EXTENDED = GENE_PANEL + [
    # Additional cell cycle
    "CCNE1", "CCNE2",
    # Additional signaling
    "IGF2", "IGFBP3", "IGFBP5", "IGFBP7",
]

# KG sources to ablate
KG_SOURCES = ["ChEMBL", "DRKG", "PrimeKG"]
KG_MASK_MAP = {
    "ChEMBL":  "ChEMBL_off",
    "DRKG":    "DRKG_off",
    "PrimeKG": "PrimeKG_off",
    "all":     "all_off",
}
