from __future__ import annotations

import pandas as pd


def apply_gene_type_map(
    df: pd.DataFrame,
    gene_map: dict[str, str],
    *,
    gene_col: str = "gene",
    gene_type_col: str = "gene_type",
    gene_chrom_col: str = "gene_chrom",
    gene_id_col: str = "gene_id",
    source_col: str = "gene_type_source",
    source_label: str | None = None,
    chrom_map: dict[str, str] | None = None,
    gene_id_map: dict[str, str] | None = None,
    chrom_source_col: str = "gene_chrom_source",
    chrom_source_label: str | None = None,
    add_source_cols: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty or gene_col not in df.columns:
        return df

    out = df if inplace else df.copy()
    if gene_type_col not in out.columns:
        out[gene_type_col] = out[gene_col].astype(str).map(gene_map).fillna("unknown")

    if add_source_cols and source_col not in out.columns:
        if source_label is None:
            source_label = "unknown"
        out[source_col] = str(source_label)

    if chrom_map is not None and gene_chrom_col not in out.columns:
        out[gene_chrom_col] = out[gene_col].astype(str).map(chrom_map)
        if add_source_cols and chrom_source_col not in out.columns:
            if chrom_source_label is None:
                chrom_source_label = source_label or "unknown"
            out[chrom_source_col] = str(chrom_source_label)

    if gene_id_map is not None and gene_id_col not in out.columns:
        out[gene_id_col] = out[gene_col].astype(str).map(gene_id_map)

    if gene_col in out.columns:
        cols = list(out.columns)
        if gene_type_col in cols:
            cols.remove(gene_type_col)
            idx = cols.index(gene_col) + 1 if gene_col in cols else 0
            cols.insert(idx, gene_type_col)
        if gene_id_col in cols:
            cols.remove(gene_id_col)
            idx = cols.index(gene_type_col) + 1 if gene_type_col in cols else 0
            cols.insert(idx, gene_id_col)
        if add_source_cols and source_col in cols:
            cols.remove(source_col)
            idx = cols.index(gene_type_col) + 1 if gene_type_col in cols else 0
            cols.insert(idx, source_col)
        if gene_chrom_col in cols:
            cols.remove(gene_chrom_col)
            if add_source_cols and source_col in cols:
                anchor = source_col
            elif gene_id_col in cols:
                anchor = gene_id_col
            else:
                anchor = gene_type_col
            idx = cols.index(anchor) + 1 if anchor in cols else 0
            cols.insert(idx, gene_chrom_col)
        if add_source_cols and chrom_source_col in cols:
            cols.remove(chrom_source_col)
            idx = cols.index(gene_chrom_col) + 1 if gene_chrom_col in cols else 0
            cols.insert(idx, chrom_source_col)
        out = out.loc[:, cols]
    return out
