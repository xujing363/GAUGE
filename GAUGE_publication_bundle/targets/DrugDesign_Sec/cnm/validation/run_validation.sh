#!/bin/bash
# Run the full 7-layer validation pipeline for GAUGE analogue validation.
# All scripts are idempotent (safe to re-run; cached results are reused where possible).
#
# Prerequisites:
#   - conda env kg_GAUGE with: rdkit, scipy, h5py, pandas, statsmodels, anndata
#   - GDSC data at /mnt/raid5/xujing/KG/KG_GAUGE_PublicData/GDSC/
#   - GTEx V11 at /mnt/raid5/xujing/KG/KG_GAUGE_PublicData/GTEx_V11/
#   - TCGA h5ad at /mnt/raid5/xujing/Agent/Datasets/TCGA/h5ad_outputs/
#   - LINCS GSE70138/GSE92742 at /mnt/raid5/xujing/KG/KG_GAUGE_PublicData/LINCS/
#     NOTE: LINCS gctx files require a >20GB full download. v03_gdsc_proxy.py
#           provides a GDSC-based alternative that does not need the gctx.
#   - GNINA binary at /mnt/raid5/xujing/KG/DrugDesign_Sec/docking/gnina/gnina.1.3.2
#     NOTE: chmod +x the binary; set LD_LIBRARY_PATH as below.
#
# LD_LIBRARY_PATH fix for GNINA (needs libcudnn.so.9):
export LD_LIBRARY_PATH="/mnt/raid5/xujing/miniconda3/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

set -e
cd "$(dirname "$0")"
ENV="conda run -n kg_GAUGE"

echo "============================================================"
echo "GAUGE Analogue Validation Pipeline"
echo "============================================================"

echo ""
echo "[v01] Layer 6: Chemistry / ADMET validation..."
$ENV python scripts/v01_chemistry_admet.py

echo ""
echo "[v02] Layer 4a: Disease signature (TCGA vs GTEx)..."
$ENV python scripts/v02_disease_signature.py

echo ""
echo "[v03] Layer 4b: GDSC transcriptome proxy signatures..."
$ENV python scripts/v03_gdsc_proxy.py

echo ""
echo "[v04] Layer 4c: Transcriptome reversal scores..."
$ENV python scripts/v04_transcriptome_reversal.py

echo ""
echo "[v05] Layer 5: Molecular docking (GNINA)..."
$ENV python scripts/v05_docking.py

echo ""
echo "[v06] Layer 7: Statistical controls..."
$ENV python scripts/v06_statistical_controls.py

echo ""
echo "[v07] Final evidence matrix integration..."
$ENV python scripts/v07_final_evidence.py

echo ""
echo "============================================================"
echo "All done. Results in results/"
echo "============================================================"
ls -lh results/layer6_chemistry/chemistry_admet.csv \
        results/layer4_transcriptome/reversal_scores.csv \
        results/layer5_docking/docking_scores.csv \
        results/layer7_final/final_evidence_matrix.csv 2>/dev/null
