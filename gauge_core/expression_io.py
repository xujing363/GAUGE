"""Helpers for turning a user-uploaded expression file into per-sample
gene -> value mappings that `gauge_core.predict.resolve_sample` understands.

Accepts two common layouts and auto-detects which one was given:
  - samples-as-rows: first column = sample name, remaining columns = genes
  - genes-as-rows:    first column = gene symbol, remaining columns = samples
"""
from __future__ import annotations

import io

import pandas as pd

_KNOWN_GENE_SET_HINT_COLS = {"gene", "gene_symbol", "hgnc_symbol", "symbol"}


def parse_expression_table(file_obj: io.BytesIO | str, known_genes: list[str]) -> dict[str, pd.Series]:
    """Parse an uploaded CSV/TSV expression table into {sample_name: gene->value Series}.

    `known_genes` is the model's gene panel; used only to disambiguate
    orientation when it is not obvious from column names.
    """
    raw = pd.read_csv(file_obj, sep=None, engine="python")
    if raw.shape[1] < 2:
        raise ValueError("Expression file needs at least 2 columns (a label column plus one or more value columns).")

    first_col = raw.columns[0]
    raw[first_col] = raw[first_col].astype(str)
    known_gene_set = set(known_genes)

    overlap_as_gene_rows = raw[first_col].isin(known_gene_set).mean()
    overlap_as_sample_rows = raw.columns[1:].astype(str).isin(known_gene_set).mean()

    genes_are_rows = overlap_as_gene_rows >= overlap_as_sample_rows
    if str(first_col).strip().lower() in _KNOWN_GENE_SET_HINT_COLS:
        genes_are_rows = True

    samples: dict[str, pd.Series] = {}
    if genes_are_rows:
        indexed = raw.set_index(first_col)
        for sample_name in indexed.columns:
            samples[str(sample_name)] = pd.to_numeric(indexed[sample_name], errors="coerce").dropna()
    else:
        indexed = raw.set_index(first_col)
        for sample_name, row in indexed.iterrows():
            samples[str(sample_name)] = pd.to_numeric(row, errors="coerce").dropna()

    if not samples:
        raise ValueError("Could not find any numeric expression values in the uploaded file.")
    return samples


def gene_coverage_report(sample: pd.Series, known_genes: list[str]) -> dict[str, float | int]:
    known_gene_set = set(known_genes)
    present = sum(1 for g in sample.index if g in known_gene_set)
    return {
        "n_genes_used": present,
        "n_genes_total": len(known_genes),
        "coverage_fraction": present / max(len(known_genes), 1),
    }
