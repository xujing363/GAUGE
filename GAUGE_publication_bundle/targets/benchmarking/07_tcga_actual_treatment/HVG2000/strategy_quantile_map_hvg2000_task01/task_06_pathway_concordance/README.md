# Strategy Quantile Map Task 06 Pathway Concordance

Score pre-registered canonical pathway programs against strategy-local biomarker truth and predicted sensitivity outputs.

## Scope
- Inputs are restricted to this strategy's `task_08` biomarker truth table and `task_10` KG-ranked biomarker table.
- Pathway programs are declared in `config.yaml` up front.
- Outputs stay local to this task folder, including `outputs/manifest.json`.

## Local Workflow
1. Review `config.yaml`.
2. Run `python scripts/run_task.py`.
3. Inspect `outputs/summary.json`, `outputs/pan_cancer_pathway_concordance.csv`, and `outputs/top_pathway_hits.csv`.
