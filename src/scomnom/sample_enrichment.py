"""Internal preparation of replicate-population inputs for sample enrichment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .de_utils import pseudobulk_aggregate
from . import annotation_utils as au
from .composition_utils import _resolve_active_cluster_key
from .config import SampleEnrichmentConfig


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


@dataclass(frozen=True)
class _PopulationSelection:
    round_id: str
    population_key: str
    round_cell_mask: np.ndarray
    mapping: pd.DataFrame
    selected_ids: tuple[str, ...]


def _resolve_populations(
    adata: ad.AnnData, *, round_id: str | None = None, target_groups: Sequence[str] = (),
) -> _PopulationSelection:
    rid = round_id if round_id is not None else adata.uns.get("active_cluster_round")
    rounds = adata.uns.get("cluster_rounds", {})
    if not rid or not isinstance(rounds, dict) or rid not in rounds or not isinstance(rounds[rid], dict):
        raise ValueError(f"Sample enrichment requires a valid clustering round; got {rid!r}.")
    info = rounds[rid]
    native = info.get("labels_obs_key")
    if native and native not in adata.obs:
        raise ValueError(f"Round {rid!r} has no stored labels at {native!r}.")
    if not native and rid != adata.uns.get("active_cluster_round"):
        raise ValueError(f"Round {rid!r} has no round-native labels; refusing active-round fallback.")
    try:
        key = _resolve_active_cluster_key(adata, round_id=rid)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    mask = adata.obs[key].notna().to_numpy()
    if not mask.any():
        raise ValueError(f"Round {rid!r} contains no eligible populations.")
    raw = _identifier_values(adata.obs.loc[mask, key], name=key)
    pretty_key = info.get("annotation", {}).get("pretty_cluster_key")
    if not pretty_key and f"cluster_label__{rid}" in adata.obs:
        pretty_key = f"cluster_label__{rid}"
    if pretty_key and pretty_key not in adata.obs:
        raise ValueError(f"Round {rid!r} has no display labels at {pretty_key!r}.")
    labels = _identifier_values(adata.obs.loc[mask, pretty_key], name=pretty_key) if pretty_key else raw
    pairs = pd.DataFrame({"population_id": raw.to_numpy(), "population_label": labels.to_numpy()}).drop_duplicates()
    if pairs["population_id"].duplicated().any():
        raise ValueError(f"Round {rid!r} maps an internal population to multiple display labels.")
    mapping = pairs.sort_values("population_id", kind="stable").reset_index(drop=True)
    selected = []
    for selector in target_groups:
        matches = mapping.loc[
            mapping["population_id"].eq(selector) | mapping["population_label"].eq(selector)
            | mapping["population_label"].str.split(":", n=1).str[0].eq(selector), "population_id"
        ]
        if len(matches) != 1:
            reason = "ambiguous" if len(matches) > 1 else "unavailable"
            raise ValueError(f"Requested population {selector!r} is {reason} in round {rid!r}.")
        selected.append(matches.iloc[0])
    return _PopulationSelection(
        str(rid), key, mask, mapping,
        tuple(dict.fromkeys(selected)) if target_groups else tuple(mapping["population_id"]),
    )


@dataclass(frozen=True)
class _ActivityResource:
    name: str
    net: pd.DataFrame
    method: str
    min_targets: int
    provenance: dict[str, Any]


def _load_activity_resources(cfg: SampleEnrichmentConfig) -> list[_ActivityResource]:
    """Load each requested resource through the shared loaders, without skipping failures."""
    import decoupler as dc

    resources = []
    for name in ("msigdb", "progeny", "dorothea"):
        if not getattr(cfg, f"run_{name}"):
            continue
        details: dict[str, Any] = {"decoupler_version": str(getattr(dc, "__version__", "unavailable"))}
        if name == "msigdb":
            files, keywords, release = au._resolve_msigdb_gene_sets_cached(cfg.msigdb_gene_sets)
            if not files:
                raise RuntimeError("MSigDB did not resolve any requested collections.")
            expected = [
                str(Path(item).expanduser().resolve()) if item.lower().endswith(".gmt") else item.upper()
                for item in cfg.msigdb_gene_sets
            ]
            unresolved = sorted(set(expected) - set(map(str, keywords)))
            if unresolved:
                raise RuntimeError(f"Requested MSigDB collections were not resolved: {unresolved}")
            pieces = []
            for path in files:
                piece = au._load_msigdb_decoupler_net_cached([path])
                if piece is None or piece.empty:
                    raise RuntimeError(f"Requested MSigDB collection {path!r} could not be loaded.")
                pieces.append(piece)
            net = pd.concat(pieces, ignore_index=True)
            details.update({
                "resource_version": release or "unavailable",
                "organism": "human" if all(".Hs." in str(path) for path in files) else "unspecified",
                "requested_collections": list(cfg.msigdb_gene_sets),
                "retained_collections": [str(path) for path in files], "resolved_keywords": keywords,
            })
        elif name == "progeny":
            net = au._load_progeny_net(dc, organism=cfg.progeny_organism, top_n=cfg.progeny_top_n)
            details.update({"organism": cfg.progeny_organism, "top_n": cfg.progeny_top_n})
        else:
            net = au._load_dorothea_net(dc, organism=cfg.dorothea_organism, confidence=cfg.dorothea_confidence)
            details.update({"organism": cfg.dorothea_organism, "confidence": list(cfg.dorothea_confidence)})
        if not isinstance(net, pd.DataFrame) or net.empty or not {"source", "target"}.issubset(net.columns):
            raise RuntimeError(f"Requested {name} resource is empty or malformed.")
        details.setdefault("resource_version", str(net.attrs.get("version", "unavailable")))
        net = net.copy()
        if net[["source", "target"]].isna().any().any():
            raise RuntimeError(f"Requested {name} resource has missing source or target identifiers.")
        net["source"] = net["source"].astype(str)
        net["target"] = net["target"].astype(str)
        if "weight" not in net:
            if "mor" in net:
                net["weight"] = net["mor"]
            elif name == "msigdb":
                net["weight"] = 1.0
            else:
                raise RuntimeError(f"Requested {name} resource has no weights.")
        net["weight"] = pd.to_numeric(net["weight"], errors="raise")
        if not np.isfinite(net["weight"]).all():
            raise RuntimeError(f"Requested {name} resource has non-finite weights.")
        snapshot = net[["source", "target", "weight"]].sort_values(["source", "target", "weight"]).to_csv(index=False)
        details["network_sha256"] = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
        details["requested_activity_count"] = int(net["source"].nunique())
        method = getattr(cfg, f"{name}_method") or cfg.decoupler_method
        min_targets = getattr(cfg, f"{name}_min_n_targets") or cfg.decoupler_min_n_targets
        resources.append(_ActivityResource(name, net, method, min_targets, details))
    return resources


@dataclass(frozen=True)
class _ScoredActivities:
    scores: pd.DataFrame
    audit: pd.DataFrame


def _score_populations(
    prepared: _PreparedPseudobulks, selection: _PopulationSelection,
    resources: Sequence[_ActivityResource], cfg: SampleEnrichmentConfig,
) -> _ScoredActivities:
    """Score each population separately and retain excluded activities in the audit."""
    score_frames, audit_rows = [], []
    labels = selection.mapping.set_index("population_id")["population_label"]
    for population in selection.selected_ids:
        rows = prepared.qc.index[prepared.qc["population_id"].eq(population) & prepared.qc["included"]]
        expression = prepared.expression.loc[rows]
        for resource in resources:
            overlap = resource.net.loc[resource.net["target"].isin(expression.columns)].groupby("source")["target"].nunique()
            sources = resource.net["source"].drop_duplicates().sort_values().tolist()
            overlap = overlap.reindex(sources, fill_value=0)
            retained = overlap.index[overlap >= resource.min_targets].tolist()
            retained_set = set(retained)
            method_info: dict[str, Any] = {"requested_method": resource.method}
            if len(rows) and retained:
                # Dense materialization is limited to one population at a time.
                matrix = expression.sparse.to_dense().T
                estimate = au._dc_run_method(
                    method=resource.method, mat=matrix, net=resource.net,
                    min_n=resource.min_targets, consensus_methods=cfg.decoupler_consensus_methods,
                )
                if (
                    not estimate.index.is_unique or not estimate.columns.is_unique
                    or set(estimate.index) != retained_set or set(estimate.columns) != set(rows)
                    or not np.isfinite(estimate.to_numpy(dtype=float)).all()
                ):
                    raise RuntimeError(f"{resource.name} returned invalid activities for population {population!r}.")
                method_info = dict(estimate.attrs.get("method_provenance", {}))
                if not method_info:
                    raise RuntimeError(f"{resource.name} scoring returned no method provenance.")
                long = estimate.rename_axis(index="activity", columns="pb_id").reset_index().melt(
                    id_vars="activity", var_name="pb_id", value_name="score",
                )
                long["replicate_id"] = long["pb_id"].map(prepared.qc["replicate_id"])
                long["population_id"] = population
                long["population_label"] = labels[population]
                long["resource"] = resource.name
                score_frames.append(long)
            provenance = json.dumps(resource.provenance, sort_keys=True, default=str)
            for source in sources:
                status = "no_eligible_libraries" if not len(rows) else (
                    "ok" if source in retained_set else "insufficient_target_overlap"
                )
                audit_rows.append({
                    "population_id": population, "population_label": labels[population],
                    "resource": resource.name, "activity": source, "status": status,
                    "target_overlap": int(overlap[source]), "min_targets": resource.min_targets,
                    "n_libraries": len(rows), "n_genes": expression.shape[1],
                    "requested_method": resource.method,
                    "method_provenance": json.dumps(method_info, sort_keys=True),
                    "resource_version": resource.provenance["resource_version"],
                    "organism": resource.provenance["organism"],
                    "network_sha256": resource.provenance["network_sha256"],
                    "resource_provenance": provenance,
                })
    columns = ["pb_id", "replicate_id", "population_id", "population_label", "resource", "activity", "score"]
    scores = pd.concat(score_frames, ignore_index=True)[columns] if score_frames else pd.DataFrame(columns=columns)
    return _ScoredActivities(scores=scores, audit=pd.DataFrame(audit_rows))


def _prepare_sample_inputs(
    adata: ad.AnnData, cfg: SampleEnrichmentConfig,
) -> tuple[_PopulationSelection, _PreparedPseudobulks]:
    """Prepare all round populations before restricting activity scoring targets."""
    selection = _resolve_populations(adata, round_id=cfg.round_id, target_groups=cfg.target_groups)
    keep, filter_info = au._apply_gene_filters_to_var_names(
        adata, gene_filter=cfg.gene_filter, resource_name="sample enrichment",
    )
    working = adata if selection.round_cell_mask.all() else adata[selection.round_cell_mask]
    prepared = _prepare_pseudobulks(
        working, replicate_key=cfg.replicate_key, population_key=selection.population_key,
        condition_key=cfg.condition_key, covariates=cfg.covariates,
        counts_layer=cfg.counts_layer, min_cells_per_replicate_group=cfg.min_cells_per_replicate_group,
    )
    expression = prepared.expression.loc[:, prepared.expression.columns.isin(adata.var_names[keep])]
    provenance = dict(prepared.provenance)
    provenance["gene_filter"] = {
        **provenance["gene_filter"], "retained_genes": expression.shape[1],
        "requested_filters": list(cfg.gene_filter), "annotation_filter": filter_info,
    }
    provenance["round_id"] = selection.round_id
    provenance["cells_outside_round"] = int((~selection.round_cell_mask).sum())
    return selection, replace(prepared, expression=expression, provenance=provenance)
