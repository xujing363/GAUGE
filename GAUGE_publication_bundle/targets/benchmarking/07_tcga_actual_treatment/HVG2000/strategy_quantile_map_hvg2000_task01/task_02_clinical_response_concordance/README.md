# Task 02 Clinical Response Concordance

Reproduce and extend TCGA responder-versus-non-responder analyses with documented thresholds and cancer-stratified outputs.

## Scope
- Source run: `/mnt/raid5/xujing/KG/benchmarking/01_random_cell_split/HVG2000/results/full_20260524_140841`
- Analysis scope: pan-cancer plus within-cancer by default
- Output contract: local inputs, outputs, figures, logs, and `outputs/manifest.json`

## Local Workflow
1. Review `config.yaml`.
2. Materialize task-local inputs under `inputs/`.
3. Run `python scripts/run_task.py`.
4. Inspect `outputs/manifest.json` and task-local output tables.
