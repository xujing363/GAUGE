from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from GAUGE.config import Paths
from GAUGE.data import attach_smiles
from GAUGE.features import build_drug_table
from GAUGE.kg_prior import build_multikg_graph_artifacts, write_kg_reports


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild KG report files for the HVG2000 benchmark from an existing prepared.pkl.")
    parser.add_argument(
        "--prepared-pkl",
        type=Path,
        default=ROOT / "data" / "processed" / "default" / "prepared.pkl",
        help="Path to prepared.pkl. Default: HVG2000/data/processed/default/prepared.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "rebuild_kg_reports",
        help="Output directory for KG report CSVs.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "default" / ".cache",
        help="Cache directory used by KG artifact builder.",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuilding the KG artifact cache instead of reusing cached artifacts.")
    parser.add_argument("--use-cache", action="store_true", default=True, help="Reuse existing KG artifact cache when available.")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable reuse of existing KG artifact cache.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    paths = Paths()
    if not args.prepared_pkl.exists():
        raise FileNotFoundError(f"prepared.pkl not found: {args.prepared_pkl}")
    prepared = pickle.loads(args.prepared_pkl.read_bytes())
    responses = prepared.responses.copy()
    if "smiles" not in responses.columns:
        raise ValueError("prepared.responses must contain smiles for PRISM rebuilds.")
    mapped = responses.loc[responses["smiles"].notna()].copy()
    prior_matrix = pd.DataFrame(index=sorted(mapped["drug_key"].astype(str).unique()), dtype="float32")
    drug_table, _ = build_drug_table(mapped, prior_matrix)
    artifacts = build_multikg_graph_artifacts(
        paths=paths,
        drug_table=drug_table,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_kg_reports(args.out_dir, artifacts)
    prior_audit = getattr(prepared, "prior_audit", pd.DataFrame(columns=["drug_key", "drug_name", "issue"]))
    prior_audit.to_csv(args.out_dir / "prior_mapping_audit.csv", index=False)
    print(f"wrote KG reports to {args.out_dir}")
    print(f"edge_rows={len(artifacts.edge_table)}")
    print(artifacts.edge_audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
