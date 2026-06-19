from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import canonical_pair_key


def _read_csv_tolerant(path: Path, **kwargs: Any) -> pd.DataFrame:
    for encoding in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception:
            pass
    return pd.DataFrame()


def _load_drugcomb(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    frame = _read_csv_tolerant(Path(path))
    if frame.empty:
        return frame
    cols = {str(c).strip().lower(): c for c in frame.columns}
    d1 = cols.get("drug1") or cols.get("drug_a") or cols.get("drug_a_name")
    d2 = cols.get("drug2") or cols.get("drug_b") or cols.get("drug_b_name")
    zip_col = cols.get("zip")
    bliss = cols.get("bliss")
    hsa = cols.get("hsa")
    if not all([d1, d2]):
        return pd.DataFrame()
    out = pd.DataFrame({"drug_A_name": frame[d1].astype(str), "drug_B_name": frame[d2].astype(str)})
    out["zip"] = pd.to_numeric(frame[zip_col], errors="coerce") if zip_col else np.nan
    out["bliss"] = pd.to_numeric(frame[bliss], errors="coerce") if bliss else np.nan
    out["hsa"] = pd.to_numeric(frame[hsa], errors="coerce") if hsa else np.nan
    out["pair_name_key"] = out.apply(lambda r: canonical_pair_key(r["drug_A_name"], r["drug_B_name"]), axis=1)
    return out


def _load_nci(path: Path) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    path = Path(path)
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    return pd.DataFrame()
                with zf.open(names[0]) as fh:
                    raw = pd.read_csv(fh)
        except Exception:
            return pd.DataFrame()
    else:
        raw = _read_csv_tolerant(path)
    if raw.empty:
        return raw
    cols = {str(c).strip().lower(): c for c in raw.columns}
    d1 = cols.get("drug1") or cols.get("sample1")
    d2 = cols.get("drug2") or cols.get("sample2")
    score = cols.get("score")
    if not all([d1, d2, score]):
        return pd.DataFrame()
    out = pd.DataFrame({"drug_A_name": raw[d1].astype(str), "drug_B_name": raw[d2].astype(str), "nci_score": pd.to_numeric(raw[score], errors="coerce")})
    out["pair_name_key"] = out.apply(lambda r: canonical_pair_key(r["drug_A_name"], r["drug_B_name"]), axis=1)
    return out


def write_external_metrics(result_dir: Path, *, drugcomb_path: Path | None = None, nci_almanac_path: Path | None = None) -> dict[str, dict[str, Any]]:
    result_dir = Path(result_dir)
    pred = pd.read_csv(result_dir / "contextual_combination_predictions.csv")
    pred = pred.copy()
    pred["pair_name_key"] = pred.apply(lambda r: canonical_pair_key(r["drug_A_name"], r["drug_B_name"]), axis=1)
    results: dict[str, dict[str, Any]] = {}
    if drugcomb_path is None or not Path(drugcomb_path).exists():
        drugcomb_metrics = pd.DataFrame([{"analysis": "drugcomb", "status": "unavailable"}])
    else:
        labels = _load_drugcomb(Path(drugcomb_path))
        merged = pred.merge(labels, on="pair_name_key", how="inner", suffixes=("", "_external")) if not labels.empty else pd.DataFrame()
        if merged.empty:
            drugcomb_metrics = pd.DataFrame([{"analysis": "drugcomb", "status": "no_mapped_pairs", "mapped_pairs": 0}])
        else:
            merged = merged.sort_values("context_combo_score_median", ascending=False).reset_index(drop=True)
            y = (merged["zip"].astype(float) >= 10.0).astype(int).to_numpy()
            top = max(1, int(np.ceil(len(merged) * 0.1)))
            base_rate = float(y.mean()) if len(y) else 0.0
            drugcomb_metrics = pd.DataFrame(
                [
                    {
                        "analysis": "drugcomb",
                        "status": "completed",
                        "mapped_pairs": int(len(merged)),
                        "high_synergy_rate": base_rate,
                        "top_decile_enrichment": float(y[:top].mean() / base_rate) if base_rate > 0 else float("nan"),
                        "spearman_context_score_zip": float(
                            merged["context_combo_score_median"].astype(float).corr(merged["zip"].astype(float), method="spearman")
                        ),
                    }
                ]
            )
    drugcomb_metrics.to_csv(result_dir / "drugcomb_external_metrics.csv", index=False)
    results["DrugComb"] = drugcomb_metrics.to_dict("records")[0]
    if nci_almanac_path is None or not Path(nci_almanac_path).exists():
        nci_metrics = pd.DataFrame([{"analysis": "nci", "status": "unavailable"}])
    else:
        labels = _load_nci(Path(nci_almanac_path))
        merged = pred.merge(labels, on="pair_name_key", how="inner", suffixes=("", "_external")) if not labels.empty else pd.DataFrame()
        if merged.empty:
            nci_metrics = pd.DataFrame([{"analysis": "nci", "status": "no_mapped_pairs", "mapped_pairs": 0}])
        else:
            nci_metrics = pd.DataFrame(
                [
                    {
                        "analysis": "nci",
                        "status": "completed",
                        "mapped_pairs": int(len(merged)),
                        "spearman_context_score_nci": float(
                            merged["context_combo_score_median"].astype(float).corr(merged["nci_score"].astype(float), method="spearman")
                        ),
                    }
                ]
            )
    nci_metrics.to_csv(result_dir / "nci_external_metrics.csv", index=False)
    results["NCI_ALMANAC"] = nci_metrics.to_dict("records")[0]
    return results
