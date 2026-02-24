from __future__ import annotations

from pathlib import Path
from typing import Dict

import anndata as ad
import pandas as pd

from .clustering_utils import create_manual_rename_round


def load_rename_mapping(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] != 2:
        raise ValueError("rename mapping file must have exactly 2 columns (no header).")
    mapping = {str(k).strip(): str(v).strip() for k, v in df.itertuples(index=False)}
    if not mapping:
        raise ValueError("rename mapping file is empty.")
    return mapping


def rename_idents(
    adata: ad.AnnData,
    mapping: Dict[str, str],
    *,
    parent_round_id: str | None = None,
    round_name: str = "manual_rename",
    set_active: bool = True,
    notes: str | None = "Manual rename of pretty labels.",
) -> str:
    return create_manual_rename_round(
        adata,
        mapping=mapping,
        parent_round_id=parent_round_id,
        round_name=round_name,
        set_active=set_active,
        notes=notes,
    )
