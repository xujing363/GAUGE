from __future__ import annotations

from pathlib import Path

import pandas as pd

from .benchmarking import PerturbationConfig
from .perturbation import plan_perturbation_mechanisms


def plan_counterfactual_interventions_stub(
    *,
    prediction_frame: pd.DataFrame,
    edge_rows: pd.DataFrame,
    out_dir: Path,
    top_k_actions: int = 10,
) -> pd.DataFrame:
    return plan_perturbation_mechanisms(
        prediction_frame=prediction_frame,
        edge_rows=edge_rows,
        edge_ablation_rows=pd.DataFrame(),
        node_ablation_rows=pd.DataFrame(),
        out_dir=out_dir,
        config=PerturbationConfig(
            enabled=True,
            output_dir="task5",
            action_top_k=int(top_k_actions),
        ),
        kg_graph=None,
    )
