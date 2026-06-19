"""
Scenario C — Synthetic Lethality Prediction via Simultaneous Gene + Drug-Target Perturbation.

Uses the double-disjoint split model (both cell AND drug unseen during training),
the most stringent test of the world model's generalisation capability.

Three prior networks:
  ChEMBL  → drug–target pharmacology
  DRKG    → disease–gene biology
  PrimeKG → protein–protein and gene–gene interactions

Static priors → dynamic via per-prediction α-attention gates learned by the model.
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
KG_ROOT = Path(os.environ.get("KGPUB_KG_ROOT", "/mnt/raid5/xujing/KG"))

# Double-disjoint model: neither cell nor drug seen during training
DOUBLE_DISJOINT_RESULT_DIR = (
    KG_ROOT / "benchmarking/05_double_disjoint_split/results/full_20260529_152945"
)
DOUBLE_DISJOINT_PREPARED_PKL = (
    KG_ROOT / "benchmarking/05_double_disjoint_split/data/processed/default/prepared.pkl"
)
DOUBLE_DISJOINT_CONFIG_YAML = DOUBLE_DISJOINT_RESULT_DIR / "default.yaml"

# External validation databases
SYNLETH_DIR = KG_ROOT / "KG_GAUGE_PublicData/SynLeth"
SYNLETH_SL   = SYNLETH_DIR / "gene_sl_gene.tsv"     # Synthetic lethality (37 943 pairs)
SYNLETH_SDL  = SYNLETH_DIR / "gene_sdl_gene.tsv"    # Synthetic dosage lethality
SYNLETH_SDR  = SYNLETH_DIR / "gene_sdr_gene.tsv"    # Synthetic dosage rescue
SYNLETH_NONSL = SYNLETH_DIR / "gene_nonsl_gene.tsv" # Validated non-SL pairs (negative control)

# Cell-line cancer-type metadata
GDSC_MODEL_LIST = KG_ROOT / "KG_GAUGE_PublicData/GDSC/model_list_20260420.csv"

# Output
SCENARIO_C_DIR = Path(__file__).parent
RESULTS_DIR  = SCENARIO_C_DIR / "results"
FIGURES_DIR  = SCENARIO_C_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Analysis parameters ──────────────────────────────────────────────────────
DEVICE      = "cuda:0"
BATCH_SIZE  = 4096
RANDOM_SEED = 42

# Number of HVG genes to perturb (top N by expression variance)
N_HVG_PERTURB = 2000

# Minimum test samples per drug to include drug in analysis
MIN_SAMPLES_PER_DRUG = 5

# Significance thresholds
# The double-disjoint model shows small absolute delta values (~1e-4 scale)
# because silencing a single gene in a 512-dim PCA space has a diluted effect.
# We use RELATIVE PERCENTILE ranking within each drug rather than absolute thresholds.
DELTA_AUC_THRESHOLD   = 0.0001  # Minimum absolute delta to retain a pair
SENSITISING_PERCENTILE = 5.0    # Top N% genes per drug = "sensitising" (rank-based)
ENRICHMENT_FDR_CUTOFF = 0.05    # FDR cutoff for SynLeth enrichment tests

# Drug–target relationships from KG (used for SL candidate definition)
KNOWN_DRUG_TARGETS = {
    "Erlotinib":    ["EGFR"],
    "Gefitinib":    ["EGFR"],
    "Osimertinib":  ["EGFR"],
    "Crizotinib":   ["ALK", "MET"],
    "Savolitinib":  ["MET"],
    "Trametinib":   ["MAP2K1", "MAP2K2"],
    "Venetoclax":   ["BCL2"],
    "TW 37":        ["BCL2", "BCL2L1"],
    "MIM1":         ["MCL1"],
    "Talazoparib":  ["PARP1", "PARP2"],
    "Nutlin-3a (-)":["MDM2"],
    "Cisplatin":    ["DNA"],
    "Oxaliplatin":  ["DNA"],
    "Cyclophosphamide": ["DNA"],
    "Cytarabine":   ["DNA"],
    "AZD8055":      ["MTOR"],
    "Buparlisib":   ["PIK3CA", "PIK3CB"],
    "Alpelisib":    ["PIK3CA"],
    "Entinostat":   ["HDAC1", "HDAC3"],
    "JQ1":          ["BRD4", "BRD3", "BRD2"],
    "Ruxolitinib":  ["JAK1", "JAK2"],
    "Ulixertinib":  ["MAPK3", "MAPK1"],
    "VX-11e":       ["MAPK3"],
    "MK-1775":      ["WEE1"],
    "AZD5438":      ["CDK1", "CDK2"],
    "AZD1208":      ["PIM1", "PIM2", "PIM3"],
    "GSK2578215A":  ["DYRK1B"],
    "GSK269962A":   ["ROCK1", "ROCK2"],
}

# Cancer type groupings for focused analysis
CANCER_TYPE_FOCUS = [
    "Lung Adenocarcinoma",
    "Acute Myeloid Leukemia",
    "Colorectal Carcinoma",
    "Breast Carcinoma",
    "Melanoma",
    "Pancreatic Carcinoma",
    "Ovarian Carcinoma",
]
