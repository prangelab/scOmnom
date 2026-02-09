from __future__ import annotations

from typing import Iterable

import pandas as pd


_RAW_TYPE_COLS = (
    "gene_type",
    "gene_biotype",
    "biotype",
    "gene_type_ensembl",
)


def _normalize_type(val: object) -> str:
    s = str(val).strip().lower()
    s = s.replace(" ", "_")
    return s


def _classify_gene_type(raw_type: str | None, gene_name: str, has_raw: bool) -> str:
    name = str(gene_name)
    upper = name.upper()
    raw = _normalize_type(raw_type) if raw_type is not None else ""

    if upper.startswith("MT-"):
        return "mt"

    if raw in ("protein_coding", "protein-coding"):
        return "protein_coding"

    if raw.startswith("mt_") or "mitochond" in raw:
        return "mt"

    if "mirna" in raw or "micro" in raw or raw == "mir":
        return "mir"

    if "linc" in raw:
        return "linc"

    if raw in ("lncrna", "lnc_rna"):
        return "other_noncoding"

    if "pseudogene" in raw:
        return "other_noncoding"

    if raw in (
        "snorna",
        "snrna",
        "scarna",
        "rrna",
        "trna",
        "misc_rna",
        "vault_rna",
        "ribozyme",
        "ribozymes",
        "antisense",
        "sense_intronic",
        "sense_overlapping",
        "processed_transcript",
        "retained_intron",
        "non_coding",
        "noncoding",
    ):
        return "other_noncoding"

    if upper.startswith("MIR"):
        return "mir"

    if upper.startswith("LINC"):
        return "linc"

    if upper.startswith(("SNORD", "SNORA", "SCARNA", "RNU", "RN7SK", "RMRP", "RPPH1")):
        return "other_noncoding"

    if upper in ("MALAT1", "NEAT1"):
        return "other_noncoding"

    if has_raw:
        return "other"
    return "protein_coding"


def gene_type_map(adata) -> dict[str, str]:
    if adata is None:
        return {}

    raw_series = None
    for col in _RAW_TYPE_COLS:
        if col in adata.var.columns:
            raw_series = adata.var[col]
            break

    has_raw = raw_series is not None
    if raw_series is None:
        raw_series = pd.Series(index=adata.var_names, data=[None] * adata.var_names.size)

    out: dict[str, str] = {}
    for gene, raw in raw_series.items():
        out[str(gene)] = _classify_gene_type(raw, str(gene), has_raw=has_raw)
    return out


def add_gene_type_column(
    adata,
    df: pd.DataFrame,
    *,
    gene_col: str = "gene",
    gene_type_col: str = "gene_type",
    inplace: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or gene_col not in df.columns:
        return df

    out = df if inplace else df.copy()
    if gene_type_col in out.columns:
        return out

    gmap = gene_type_map(adata)
    out[gene_type_col] = out[gene_col].astype(str).map(gmap).fillna("other")

    if gene_col in out.columns:
        cols = list(out.columns)
        cols.remove(gene_type_col)
        idx = cols.index(gene_col) + 1 if gene_col in cols else 0
        cols.insert(idx, gene_type_col)
        out = out.loc[:, cols]
    return out
