from __future__ import annotations

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("a:si", NS):
        values.append("".join(t.text or "" for t in si.findall(".//a:t", NS)))
    return values


def _cell_value(cell: ET.Element, shared: list[str]) -> str | float | int | None:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", NS))
    value = cell.find("a:v", NS)
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        return shared[int(value.text)]
    text = value.text
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def read_xlsx_first_sheet(path: Path, usecols: Iterable[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Read a simple XLSX first sheet using only the Python standard library.

    The GDSC fitted files are regular single-sheet XLSX files, but the target
    conda env may not include openpyxl. This reader handles inline strings,
    shared strings, numeric cells, sparse rows, and optional column projection.
    """
    with zipfile.ZipFile(path) as zf:
        shared = _shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = root.findall(".//a:sheetData/a:row", NS)
        if not rows:
            return pd.DataFrame()
        header_cells = rows[0].findall("a:c", NS)
        header_by_idx = {_column_index(c.attrib["r"]): _cell_value(c, shared) for c in header_cells}
        max_idx = max(header_by_idx)
        header = [str(header_by_idx.get(i, "")) for i in range(max_idx + 1)]
        wanted = set(usecols) if usecols is not None else None
        indices = [i for i, name in enumerate(header) if wanted is None or name in wanted]
        records = []
        for row in rows[1:]:
            values = {}
            for cell in row.findall("a:c", NS):
                idx = _column_index(cell.attrib["r"])
                if idx in indices:
                    values[idx] = _cell_value(cell, shared)
            records.append({header[i]: values.get(i) for i in indices})
            if nrows is not None and len(records) >= nrows:
                break
    return pd.DataFrame.from_records(records, columns=[header[i] for i in indices])
