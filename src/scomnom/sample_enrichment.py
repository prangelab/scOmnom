"""Internal preparation of replicate-population inputs for sample enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .de_utils import pseudobulk_aggregate


@dataclass(frozen=True)
class _PreparedPseudobulks:
    counts: pd.DataFrame
    expression: pd.DataFrame
    qc: pd.DataFrame
    provenance: dict[str, Any]


def _resolve_count_assay(
    adata: ad.AnnData, requested: str = "auto",
) -> tuple[str | None, dict[str, Any]]:
    """Resolve and validate counts without replacing or rounding the source."""
    priority = ["counts_cb", "counts_raw", "X"]
    if requested not in ["auto", *priority]:
        raise ValueError(f"Unknown counts source {requested!r}; choose auto, counts_cb, counts_raw, or X.")
    resolved = next((key for key in priority[:-1] if key in adata.layers), "X") if requested == "auto" else requested
    if resolved != "X" and resolved not in adata.layers:
        raise ValueError(f"Requested counts source {resolved!r} is unavailable.")
    matrix = adata.X if resolved == "X" else adata.layers[resolved]
    if matrix is None:
        raise ValueError(f"Selected counts source {resolved!r} is unavailable.")
    is_sparse = sparse.issparse(matrix)
    values = matrix.tocsr().data if is_sparse else np.asarray(matrix).reshape(-1)
    if values.dtype.kind not in "biuf":
        raise ValueError(f"Selected counts source {resolved!r} must contain numeric counts.")
    for start in range(0, values.size, 1_000_000):
        block = values[start:start + 1_000_000]
        if not np.isfinite(block).all() or np.any(block < 0):
            raise ValueError(f"Selected counts source {resolved!r} must contain finite, nonnegative counts.")
        if not np.allclose(block, np.rint(block), rtol=0, atol=1e-6):
            raise ValueError(f"Selected counts source {resolved!r} must contain integer-like counts.")
    return (None if resolved == "X" else resolved), {
        "requested": requested,
        "source": "X" if resolved == "X" else f"layers/{resolved}",
        "shape": list(matrix.shape),
        "sparse": is_sparse,
        "dtype": str(matrix.dtype),
        "integer_like": True,
        "integer_atol": 1e-6,
        "source_priority": priority,
        "priority_rank": priority.index(resolved) + 1,
    }


def _identifier_values(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name!r} contains missing identifiers.")
    if values.drop_duplicates().astype(str).duplicated().any():
        raise ValueError(f"{name!r} contains identifiers that collide when converted to strings.")
    return values.astype(str)


def _resolve_replicate_metadata(
    obs: pd.DataFrame, *, replicate_key: str, condition_key: str | None = None,
    covariates: Sequence[str] = (),
) -> pd.DataFrame:
    """Require one metadata value per library; preserve incomplete covariates."""
    columns = list(dict.fromkeys([*([condition_key] if condition_key else []), *covariates]))
    missing = [key for key in [replicate_key, *columns] if key not in obs]
    if missing:
        raise ValueError(f"Replicate metadata columns missing from obs: {missing}")
    replicate_ids = _identifier_values(obs[replicate_key], name=replicate_key)
    if condition_key:
        _identifier_values(obs[condition_key], name=condition_key)
    metadata = obs[columns].copy()
    grouped = metadata.groupby(replicate_ids.to_numpy(), sort=True, observed=True)
    sizes = grouped.nunique(dropna=True)
    ambiguous = sizes > 1
    if ambiguous.to_numpy().any():
        bad = [(str(idx), str(col)) for idx, row in ambiguous.iterrows() for col in columns if row[col]]
        raise ValueError(f"Replicate metadata is ambiguous (replicate, column): {bad}")
    keep = ~replicate_ids.duplicated()
    result = metadata.loc[keep].copy()
    result.index = pd.Index(replicate_ids.loc[keep], name="replicate_id")
    result = result.sort_index()
    # Any missing cell annotation makes that library's covariate incomplete.
    for col in columns:
        missing_by_replicate = metadata[col].isna().groupby(replicate_ids.to_numpy()).any()
        result.loc[missing_by_replicate.reindex(result.index, fill_value=False), col] = np.nan
    return result


def _prepare_pseudobulks(
    adata: ad.AnnData, *, replicate_key: str, population_key: str,
    condition_key: str | None = None, covariates: Sequence[str] = (),
    counts_layer: str = "auto", min_cells_per_replicate_group: int = 20,
) -> _PreparedPseudobulks:
    """Aggregate all observed libraries, retain QC, and normalize eligible rows."""
    threshold = min_cells_per_replicate_group
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, np.integer)) or threshold < 1:
        raise ValueError("min_cells_per_replicate_group must be a positive integer.")
    if population_key not in adata.obs:
        raise ValueError(f"Population column {population_key!r} is missing from obs.")
    if not adata.var_names.is_unique:
        raise ValueError("Gene identifiers in var_names must be unique.")
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Sample enrichment requires cells and genes.")
    metadata = _resolve_replicate_metadata(
        adata.obs, replicate_key=replicate_key, condition_key=condition_key, covariates=covariates,
    )
    populations = _identifier_values(adata.obs[population_key], name=population_key)
    layer, assay = _resolve_count_assay(adata, counts_layer)
    matrix = adata.X if layer is None else adata.layers[layer]
    # Promote small integer counts before the shared sparse summation engine.
    matrix = matrix.astype(np.float64 if matrix.dtype.kind == "f" else np.int64, copy=False)
    rep_codes, rep_values = pd.factorize(adata.obs[replicate_key].astype(str), sort=True)
    pop_codes, pop_values = pd.factorize(populations, sort=True)
    # Encode keys to avoid delimiter collisions inside pseudobulk_aggregate.
    working = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame({"replicate": rep_codes.astype(str), "population": pop_codes.astype(str)}, index=adata.obs_names),
        var=pd.DataFrame(index=adata.var_names),
    )
    counts, keys = pseudobulk_aggregate(
        working, sample_key="replicate", group_key="population", counts_layer=None,
        min_cells_per_sample_group=1,
    )
    pb_ids = pd.Index([f"pb{i:06d}" for i in range(len(keys))], name="pb_id")
    counts.index = pb_ids
    replicates = rep_values.to_numpy()[keys["replicate"].astype(int).to_numpy()]
    groups = pop_values.to_numpy()[keys["population"].astype(int).to_numpy()]
    pb_matrix = counts.sparse.to_coo().tocsr()
    pb_matrix.sum_duplicates()
    pb_matrix.eliminate_zeros()
    library_sizes = np.asarray(pb_matrix.sum(axis=1)).ravel()
    if not np.isfinite(library_sizes).all() or np.any(library_sizes < 0):
        raise ValueError("Aggregated counts produced invalid library sizes.")
    qc = pd.DataFrame({
        "replicate_id": replicates,
        "population_id": groups,
        "n_cells": keys["n_cells"].to_numpy(),
        "library_size": library_sizes,
        "detected_genes": np.diff(pb_matrix.indptr),
    }, index=pb_ids)
    for key in metadata:
        if key in qc.columns:
            raise ValueError(f"Metadata column {key!r} conflicts with a sample-enrichment QC column.")
        qc[key] = pd.Series(metadata[key].reindex(replicates).array, index=pb_ids)
    reasons = [
        ";".join(reason for applies, reason in (
            (cells < threshold, "insufficient_cells"), (size == 0, "zero_library"),
        ) if applies)
        for cells, size in zip(qc["n_cells"], library_sizes)
    ]
    qc["exclusion_reason"] = reasons
    qc["included"] = qc["exclusion_reason"].eq("")
    eligible = qc["included"].to_numpy()
    retained = np.asarray(pb_matrix[eligible].sum(axis=0)).ravel() > 0
    normalized = pb_matrix[eligible][:, retained].astype(np.float64)
    normalized = normalized.multiply((1_000_000 / library_sizes[eligible])[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    expression = pd.DataFrame.sparse.from_spmatrix(
        normalized, index=pb_ids[eligible], columns=adata.var_names[retained],
    )
    provenance = {
        "count_assay": assay,
        "replicate_key": replicate_key,
        "population_key": population_key,
        "condition_key": condition_key,
        "covariates": list(covariates),
        "min_cells_per_replicate_group": int(threshold),
        "normalization": "log1p(counts / library_size * 1000000)",
        "library_size_before_gene_filter": True,
        "gene_filter": {
            "input_genes": adata.n_vars,
            "retained_genes": int(retained.sum()),
            "rule": "total counts > 0 across eligible replicate-population libraries",
            "gene_identifiers": "var_names (unchanged)",
        },
    }
    return _PreparedPseudobulks(counts=counts, expression=expression, qc=qc, provenance=provenance)
