from __future__ import annotations

from typing import Mapping

import anndata as ad
import pandas as pd

from .adata_ops import subset_adata_by_cluster_mapping as _subset_adata_by_cluster_mapping
from .rename_utils import rename_idents


def subset_adata_by_cluster_mapping(
    adata: ad.AnnData,
    subset_mapping: Mapping[str, str] | Mapping[str, list[str]],
    round_id: str | None = None,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """
    Public API wrapper for subset creation.

    Accepted mapping forms:
    1) Cnn -> subset (preferred, mirrors subset TSV):
       {"C00": "Hepatocyte", "C01": "Endothelial"}
    2) subset -> [Cnn, ...]:
       {"Hepatocyte": ["C00", "C02"], "Immune": ["C04", "C06"]}
    """
    if not isinstance(subset_mapping, Mapping) or not subset_mapping:
        raise ValueError("subset_mapping must be a non-empty mapping.")

    values = list(subset_mapping.values())
    first = values[0] if values else None

    if isinstance(first, str):
        # Cnn -> subset
        subset_to_clusters: dict[str, list[str]] = {}
        for cnn, subset in subset_mapping.items():
            c = str(cnn).strip()
            s = str(subset).strip()
            if not c or not s:
                raise ValueError("subset_mapping contains empty keys or values.")
            subset_to_clusters.setdefault(s, []).append(c)
    else:
        # assume subset -> [Cnn, ...]
        subset_to_clusters = {}
        for subset, clusters in subset_mapping.items():
            s = str(subset).strip()
            if not s:
                raise ValueError("subset_mapping contains an empty subset name.")
            if not isinstance(clusters, (list, tuple, set)):
                raise ValueError("subset -> clusters mapping values must be list/tuple/set of Cnn strings.")
            subset_to_clusters[s] = [str(c).strip() for c in clusters if str(c).strip()]
            if not subset_to_clusters[s]:
                raise ValueError(f"subset_mapping has no clusters for subset {s!r}.")

    return _subset_adata_by_cluster_mapping(
        adata=adata,
        subset_to_clusters=subset_to_clusters,
        round_id=round_id,
    )

__all__ = [
    "rename_idents",
    "subset_adata_by_cluster_mapping",
]
