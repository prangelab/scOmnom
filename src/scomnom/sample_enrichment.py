"""Internal preparation of replicate-population inputs for sample enrichment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import shlex
from typing import Any, Sequence
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .de_utils import pseudobulk_aggregate
from . import annotation_utils as au
from .composition_utils import _resolve_active_cluster_key
from .config import SampleEnrichmentConfig
from . import io_utils
from .logging_utils import init_logging

LOGGER = logging.getLogger(__name__)


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
    _resolve_contrasts(prepared.qc, cfg)
    return selection, replace(prepared, expression=expression, provenance=provenance)


def _resolve_contrasts(qc: pd.DataFrame, cfg: SampleEnrichmentConfig) -> tuple[tuple[str, str], ...]:
    if not cfg.condition_key:
        return ()
    if cfg.condition_key not in qc:
        raise ValueError(f"Condition column {cfg.condition_key!r} is missing.")
    levels = set(_identifier_values(qc[cfg.condition_key], name=cfg.condition_key))
    if not cfg.contrasts:
        if len(levels) != 2 or not cfg.reference:
            raise ValueError("Specify explicit TEST:REFERENCE contrasts, or an explicit reference for two levels.")
        if cfg.reference not in levels:
            raise ValueError(f"Requested reference {cfg.reference!r} is absent from the data.")
        resolved = [(next(iter(levels - {cfg.reference})), cfg.reference)]
    else:
        resolved = []
    for value in cfg.contrasts:
        parts = [part.strip() for part in value.split(":")]
        if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
            raise ValueError(f"Malformed contrast {value!r}; use TEST:REFERENCE.")
        if set(parts) - levels:
            raise ValueError(f"Requested contrast {value!r} contains a level absent from the data.")
        resolved.append(tuple(parts))
    if cfg.subject_key:
        if cfg.subject_key not in qc:
            raise ValueError(f"Subject column {cfg.subject_key!r} is missing.")
        libraries = qc[["replicate_id", cfg.subject_key, cfg.condition_key]].drop_duplicates("replicate_id")
        libraries = libraries.loc[libraries[cfg.subject_key].notna() & libraries[cfg.subject_key].astype(str).str.strip().ne("")]
        _identifier_values(libraries[cfg.subject_key], name=cfg.subject_key)
        for test, reference in resolved:
            paired = libraries.loc[libraries[cfg.condition_key].astype(str).isin([test, reference])]
            if paired.duplicated([cfg.subject_key, cfg.condition_key]).any():
                raise ValueError("Paired sample enrichment does not support multiple replicate libraries per subject-condition.")
    return tuple(dict.fromkeys(resolved))


@dataclass(frozen=True)
class _ActivityDesign:
    matrix: pd.DataFrame
    audit: dict[str, Any]
    exclusions: pd.DataFrame


def _build_activity_design(
    qc: pd.DataFrame, *, test: str, reference: str, cfg: SampleEnrichmentConfig,
) -> _ActivityDesign:
    condition = qc[cfg.condition_key].astype(str)
    in_contrast = condition.isin([test, reference])
    reasons = pd.Series("", index=qc.index, dtype=object)
    reasons.loc[~in_contrast] = "outside_contrast"
    failed_qc = in_contrast & ~qc["included"]
    reasons.loc[failed_qc] = qc.loc[failed_qc, "exclusion_reason"].replace("", "pseudobulk_qc")

    for field in cfg.covariates:
        if field not in qc:
            raise ValueError(f"Requested covariate {field!r} is missing.")
        values = qc[field]
        missing = values.isna() | values.astype(str).str.strip().eq("")
        if pd.api.types.is_numeric_dtype(values) and field != cfg.subject_key:
            missing |= ~np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        for pb_id in qc.index[in_contrast & missing]:
            reasons.loc[pb_id] = ";".join(filter(None, [reasons.loc[pb_id], f"missing_covariate:{field}"]))

    n_subjects = 0
    if cfg.subject_key:
        subjects = qc[cfg.subject_key]
        identified = subjects.notna() & subjects.astype(str).str.strip().ne("")
        _identifier_values(subjects.loc[identified], name=cfg.subject_key)
        pair_rows = qc.loc[in_contrast & identified, [cfg.subject_key, cfg.condition_key]]
        if pair_rows.duplicated().any():
            raise ValueError("Paired sample enrichment does not support multiple replicate libraries per subject-condition.")
        complete = qc.loc[reasons.eq("")].groupby(cfg.subject_key, observed=True)[cfg.condition_key].nunique()
        complete_subjects = complete.index[complete == 2]
        incomplete = reasons.eq("") & ~subjects.isin(complete_subjects)
        reasons.loc[incomplete] = "incomplete_pair"
        n_subjects = len(complete_subjects)

    included = reasons.eq("")
    rows = qc.loc[included]
    n_test = int((condition.loc[included] == test).sum())
    n_reference = int((condition.loc[included] == reference).sum())
    exclusions = qc[["replicate_id", "population_id"]].copy()
    exclusions.insert(0, "pb_id", qc.index)
    exclusions["contrast"] = f"{test}:{reference}"
    exclusions["condition"] = condition
    exclusions["included"] = included
    exclusions["exclusion_reason"] = reasons
    if cfg.subject_key:
        exclusions["subject_id"] = qc[cfg.subject_key]
    exclusions = exclusions.reset_index(drop=True)

    matrix = pd.DataFrame({"intercept": np.ones(len(rows))}, index=rows.index)
    terms: list[dict[str, Any]] = [{"column": "intercept", "field": "intercept", "kind": "intercept"}]
    constant = []
    formula_terms = []
    for number, field in enumerate(cfg.covariates):
        values = rows[field]
        numeric = pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values) and field != cfg.subject_key
        if values.nunique() <= 1:
            constant.append(field)
        if numeric:
            column = f"covariate_{number}"
            matrix[column] = values.astype(float)
            terms.append({"column": column, "field": field, "kind": "numeric"})
            formula_terms.append(f"Q({json.dumps(field)})")
        else:
            labels = _identifier_values(values, name=field)
            levels = sorted(labels.unique())
            for index, level in enumerate(levels[1:], start=1):
                column = f"covariate_{number}_level_{index}"
                matrix[column] = labels.eq(level).astype(float)
                terms.append({"column": column, "field": field, "kind": "categorical", "level": level, "reference": levels[0]})
            formula_terms.append(f"C(Q({json.dumps(field)}))")
    matrix["condition_test"] = condition.loc[included].eq(test).astype(float)
    terms.append({"column": "condition_test", "field": cfg.condition_key, "kind": "condition", "level": test, "reference": reference})
    formula_terms.append(f"C(Q({json.dumps(cfg.condition_key)}), Treatment(reference={json.dumps(reference)}))")
    rank = int(np.linalg.matrix_rank(matrix.to_numpy())) if len(matrix) else 0
    df_resid = len(matrix) - rank
    status = "ok"
    if not {test, reference}.issubset(set(condition)):
        status = "absent_contrast_level"
    elif cfg.subject_key and n_subjects < cfg.min_complete_subjects:
        status = "insufficient_complete_pairs"
    elif min(n_test, n_reference) < cfg.min_replicates_per_level or (not cfg.subject_key and len(rows) < cfg.min_replicates_total):
        status = "insufficient_replicates"
    elif constant:
        status = "covariate_no_variation"
    elif rank < matrix.shape[1]:
        status = "rank_deficient"
    elif df_resid <= 0:
        status = "zero_residual_degrees_of_freedom"
    audit = {
        "design_status": status, "formula": "activity_score ~ " + " + ".join(formula_terms),
        "design_columns": json.dumps(matrix.columns.tolist()), "design_terms": json.dumps(terms),
        "constant_covariates": json.dumps(constant), "rank": rank, "df_resid": df_resid,
        "n_included": len(rows), "n_excluded": len(qc) - len(rows),
        "n_test": n_test, "n_reference": n_reference, "n_subjects": n_subjects,
        "n_outside_contrast": int((~in_contrast).sum()), "covariance": "HC3",
        "inference_distribution": "t", "outcome_sd_ddof": 1,
    }
    return _ActivityDesign(matrix, audit, exclusions)


@dataclass(frozen=True)
class _ActivityModels:
    contrasts: pd.DataFrame
    audit: pd.DataFrame
    exclusions: pd.DataFrame


_MODEL_ESTIMATE_COLUMNS = [
    "effect_raw", "se_raw", "ci_low_raw", "ci_high_raw", "outcome_sd",
    "effect_standardized", "ci_low_standardized", "ci_high_standardized", "pvalue", "fdr",
]
_MODEL_ID_COLUMNS = ["population_id", "population_label", "resource", "activity", "contrast", "test", "reference"]


def _fit_activity_contrasts(
    prepared: _PreparedPseudobulks, scored: _ScoredActivities, cfg: SampleEnrichmentConfig,
) -> _ActivityModels:
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    contrasts = _resolve_contrasts(prepared.qc, cfg)
    estimates, audits, exclusions = [], [], []
    designs: dict[tuple[str, str, str], _ActivityDesign] = {}
    grouping = ["population_id", "resource", "activity"]
    if scored.audit.duplicated(grouping).any():
        raise ValueError("Activity audit contains duplicate population-resource-activity rows.")
    if scored.scores.duplicated([*grouping, "pb_id"]).any():
        raise ValueError("Activity scores contain duplicate library observations.")
    score_groups = {key: table.set_index("pb_id")["score"] for key, table in scored.scores.groupby(grouping, sort=False)}
    for activity in scored.audit.to_dict("records"):
        population = activity["population_id"]
        for test, reference in contrasts:
            design_key = (population, test, reference)
            if design_key not in designs:
                design = _build_activity_design(
                    prepared.qc.loc[prepared.qc["population_id"].eq(population)],
                    test=test, reference=reference, cfg=cfg,
                )
                designs[design_key] = design
                exclusions.append(design.exclusions)
            design = designs[design_key]
            identity = {key: activity[key] for key in _MODEL_ID_COLUMNS[:4]}
            identity.update({"contrast": f"{test}:{reference}", "test": test, "reference": reference})
            result = {
                **identity, **{key: np.nan for key in _MODEL_ESTIMATE_COLUMNS},
                **{key: design.audit[key] for key in ("formula", "n_included", "n_excluded", "n_test", "n_reference", "n_subjects")},
                "status": design.audit["design_status"] if activity["status"] == "ok" else activity["status"],
                "fdr_family_id": json.dumps([population, activity["resource"], test, reference]),
                "fdr_family_size": 0,
            }
            model_warnings = []
            if result["status"] == "ok":
                scores = score_groups.get((population, activity["resource"], activity["activity"]), pd.Series(dtype=float))
                y = pd.to_numeric(scores.reindex(design.matrix.index), errors="raise").to_numpy(dtype=float)
                if not np.isfinite(y).all():
                    result["status"] = "nonfinite_activity"
                else:
                    sd = float(np.std(y, ddof=1))
                    result["outcome_sd"] = sd
                    if not np.isfinite(sd):
                        result["status"] = "nonfinite_activity_variance"
                    elif sd == 0:
                        result["status"] = "zero_variance_activity"
                    else:
                        x = design.matrix.to_numpy(dtype=float)
                        leverage = np.square(np.linalg.qr(x, mode="reduced")[0]).sum(axis=1)
                        if np.any(1 - leverage <= 10 * np.finfo(float).eps):
                            result["status"] = "hc3_undefined_leverage"
                        else:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter("always")
                                try:
                                    fitted = sm.OLS(y, design.matrix, missing="raise").fit(cov_type="HC3", use_t=True)
                                    tolerance = 100 * np.finfo(float).eps * max(1., np.linalg.norm(y))
                                    if np.linalg.norm(fitted.resid) <= tolerance:
                                        result["status"] = "zero_residual_variance"
                                    else:
                                        effect = float(fitted.params["condition_test"])
                                        se = float(fitted.bse["condition_test"])
                                        low, high = fitted.conf_int(alpha=.05).loc["condition_test"].to_numpy(dtype=float)
                                        pvalue = float(fitted.pvalues["condition_test"])
                                        if not np.isfinite([effect, se, low, high, pvalue]).all() or se <= 0 or not 0 <= pvalue <= 1:
                                            result["status"] = "nonfinite_inference"
                                        else:
                                            result.update({
                                                "effect_raw": effect, "se_raw": se, "ci_low_raw": low, "ci_high_raw": high,
                                                "effect_standardized": effect / sd, "ci_low_standardized": low / sd,
                                                "ci_high_standardized": high / sd, "pvalue": pvalue,
                                            })
                                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                                    result["status"] = "fit_failed"
                                    model_warnings.append(f"{type(exc).__name__}: {exc}")
                                model_warnings.extend(str(item.message) for item in caught)
            estimates.append(result)
            audits.append({**identity, **design.audit, "status": result["status"], "warnings": json.dumps(model_warnings)})

    result_table = pd.DataFrame(estimates)
    if not result_table.empty:
        for _, family in result_table.groupby("fdr_family_id", sort=False):
            tested = family.index[family["status"].eq("ok") & family["pvalue"].notna()]
            result_table.loc[family.index, "fdr_family_size"] = len(tested)
            if len(tested):
                result_table.loc[tested, "fdr"] = multipletests(result_table.loc[tested, "pvalue"], method="fdr_bh")[1]
    else:
        result_table = pd.DataFrame(columns=[*_MODEL_ID_COLUMNS, *_MODEL_ESTIMATE_COLUMNS, "status", "fdr_family_id", "fdr_family_size"])
    return _ActivityModels(
        contrasts=result_table, audit=pd.DataFrame(audits),
        exclusions=pd.concat(exclusions, ignore_index=True) if exclusions else pd.DataFrame(),
    )


_TABLE_COLUMNS = {
    "pseudobulk_qc": [
        "pb_id", "replicate_id", "population_id", "population_label", "n_cells",
        "library_size", "detected_genes", "included", "exclusion_reason",
        "selected_population", "included_for_scoring",
    ],
    "activity_scores": ["pb_id", "replicate_id", "population_id", "population_label", "resource", "activity", "score"],
    "activity_contrasts": [
        *_MODEL_ID_COLUMNS, *_MODEL_ESTIMATE_COLUMNS, "formula", "n_included", "n_excluded",
        "n_test", "n_reference", "n_subjects", "status", "fdr_family_id", "fdr_family_size",
    ],
    "model_audit": [
        *_MODEL_ID_COLUMNS, "design_status", "formula", "design_columns", "design_terms",
        "constant_covariates", "rank", "df_resid", "n_included", "n_excluded", "n_test",
        "n_reference", "n_subjects", "n_outside_contrast", "covariance", "inference_distribution",
        "outcome_sd_ddof", "status", "warnings",
    ],
    "model_exclusions": ["pb_id", "replicate_id", "population_id", "contrast", "condition", "subject_id", "included", "exclusion_reason"],
    "resource_provenance": [
        "population_id", "population_label", "resource", "activity", "status", "target_overlap",
        "min_targets", "n_libraries", "n_genes", "requested_method", "method_provenance",
        "resource_version", "organism", "network_sha256", "resource_provenance",
    ],
}
_TABLE_UNITS = {
    "score": "method-specific activity units", "effect_raw": "activity units (test minus reference)",
    "se_raw": "activity units", "ci_low_raw": "activity units", "ci_high_raw": "activity units",
    "outcome_sd": "activity units", "effect_standardized": "outcome SD",
    "ci_low_standardized": "outcome SD", "ci_high_standardized": "outcome SD",
    "pvalue": "probability", "fdr": "BH-adjusted p-value", "n_cells": "cells",
    "library_size": "counts before gene filtering", "detected_genes": "genes",
    "n_included": "replicate libraries", "n_excluded": "replicate libraries (including outside contrast)",
    "n_test": "replicate libraries", "n_reference": "replicate libraries", "n_subjects": "complete subjects",
}


def _sample_output_dir(cfg: SampleEnrichmentConfig) -> Path:
    if cfg.output_dir is not None:
        return Path(cfg.output_dir)
    parent = cfg.input_path.parent
    for candidate in (parent, *parent.parents):
        if candidate.name == "results":
            return candidate
    return parent / "results"


def _sample_output_stem(cfg: SampleEnrichmentConfig, round_id: str) -> str:
    if cfg.output_name is None:
        return f"adata.enrichment_sample_{io_utils.sanitize_identifier(round_id, allow_spaces=False)}"
    stem = cfg.output_name.strip()
    for suffix in (".zarr.tar.zst", ".zarr", ".h5ad"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    if not stem or stem in {".", ".."} or Path(stem).name != stem:
        raise ValueError("output_name must be a dataset filename, without directories.")
    return stem


def _reserve_sample_directory(
    output_dir: Path, round_id: str, stored: dict, formats: Sequence[str],
) -> tuple[str, Path]:
    namespace = f"enrichment_sample_{io_utils.sanitize_identifier(round_id, allow_spaces=False)}"
    root = output_dir / "tables"
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        analysis_id = f"{namespace}_round{number}"
        folder = root / analysis_id
        if analysis_id in stored or any((output_dir / "figures" / fmt / analysis_id).exists() for fmt in formats):
            number += 1
            continue
        try:
            folder.mkdir()
            return analysis_id, folder
        except FileExistsError:
            number += 1


def _sample_tables(
    prepared: _PreparedPseudobulks, selection: _PopulationSelection,
    scored: _ScoredActivities, models: _ActivityModels, cfg: SampleEnrichmentConfig,
) -> dict[str, pd.DataFrame]:
    qc = prepared.qc.reset_index()
    qc["population_label"] = qc["population_id"].map(selection.mapping.set_index("population_id")["population_label"])
    qc["selected_population"] = qc["population_id"].isin(selection.selected_ids)
    qc["included_for_scoring"] = qc["included"] & qc["selected_population"]
    scores = scored.scores.copy()
    metadata_columns = list(dict.fromkeys([*([cfg.condition_key] if cfg.condition_key else []), *cfg.covariates]))
    for key in metadata_columns:
        if key in _TABLE_COLUMNS["activity_scores"]:
            raise ValueError(f"Metadata column {key!r} conflicts with an activity-score output column.")
        scores[key] = scores["pb_id"].map(prepared.qc[key])
    if cfg.replicate_key not in {"replicate_id", *metadata_columns}:
        if cfg.replicate_key in _TABLE_COLUMNS["activity_scores"]:
            raise ValueError(f"Replicate key {cfg.replicate_key!r} conflicts with an activity-score output column.")
        qc[cfg.replicate_key] = qc["replicate_id"]
        scores[cfg.replicate_key] = scores["replicate_id"]
    tables = {
        "pseudobulk_qc": qc, "activity_scores": scores, "activity_contrasts": models.contrasts,
        "model_audit": models.audit, "model_exclusions": models.exclusions,
        "resource_provenance": scored.audit,
    }
    for name, frame in tables.items():
        columns = _TABLE_COLUMNS[name]
        extras = [column for column in frame if column not in columns]
        tables[name] = frame.reindex(columns=[*columns, *extras]).reset_index(drop=True)
        # String row indices are stable across both H5AD and Zarr serialization.
        tables[name].index = tables[name].index.astype(str)
    return tables


def _write_sample_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample_software_versions() -> dict[str, str]:
    from . import __version__

    versions = {"scomnom": __version__}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "decoupler", "anndata"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _persist_sample_figures(payload, output_dir, run_id, formats, activities):
    from . import plot_utils as pu
    from .sample_enrichment_plot_utils import sample_enrichment_artifacts

    previous = (pu.ROOT_FIGDIR, pu.RUN_FIG_SUBDIR, pu.RUN_KEY, pu.FIGURE_FORMATS)
    paths = []
    try:
        pu.setup_scanpy_figs(output_dir / "figures", formats=formats)
        pu.RUN_KEY = run_id
        pu.RUN_FIG_SUBDIR = Path(run_id)
        for artifact in sample_enrichment_artifacts(payload, figdir=Path(run_id), activities=activities):
            try:
                pu.persist_plot_artifacts([artifact])
            finally:
                if artifact.fig is not None:
                    pu.close_plot(artifact.fig)
            paths.extend(str(output_dir / "figures" / fmt / run_id / f"{artifact.stem}.{fmt}") for fmt in formats)
    finally:
        pu.ROOT_FIGDIR, pu.RUN_FIG_SUBDIR, pu.RUN_KEY, pu.FIGURE_FORMATS = previous
    return paths


def _regenerate_sample_figures(adata, cfg, output_dir, command):
    round_id = cfg.round_id or adata.uns.get("active_cluster_round")
    rounds = adata.uns.get("cluster_rounds", {})
    if round_id not in rounds:
        raise ValueError("Figure regeneration requires an existing clustering round.")
    stored = rounds[round_id].get("sample_enrichment", {})
    analysis_id = cfg.analysis_id
    if analysis_id is None:
        if len(stored) != 1:
            raise ValueError(f"Choose --analysis-id from the stored sample analyses: {sorted(stored)}")
        analysis_id = next(iter(stored))
    if analysis_id not in stored:
        raise ValueError(f"Unknown sample analysis_id: {analysis_id!r}")
    payload = stored[analysis_id]
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported sample-enrichment schema version for regeneration.")
    safe_id = io_utils.sanitize_identifier(analysis_id, allow_spaces=False)
    root = output_dir / "figures" / "regeneration"
    root.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        run_id = f"{safe_id}_regeneration_round{number}"
        folder = root / run_id
        if any((output_dir / "figures" / fmt / run_id).exists() for fmt in cfg.figure_formats):
            number += 1
            continue
        try:
            folder.mkdir()
            break
        except FileExistsError:
            number += 1
    manifest = {"schema_version": 1, "status": "running", "source_analysis_id": analysis_id,
                "round_id": round_id, "input_path": str(cfg.input_path.resolve()),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "command": list(command) if command is not None else None,
                "software_versions": _sample_software_versions(),
                "figure_formats": cfg.figure_formats, "plot_activity": list(cfg.plot_activity)}
    manifest_path = folder / "settings.json"
    _write_sample_manifest(manifest_path, manifest)
    try:
        manifest["figure_paths"] = _persist_sample_figures(payload, output_dir, run_id, cfg.figure_formats, cfg.plot_activity)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_sample_manifest(manifest_path, manifest)
    LOGGER.info("Regenerated %d sample figures from %s; manifest: %s", len(manifest["figure_paths"]), analysis_id, manifest_path)


def run_sample_enrichment(
    cfg: SampleEnrichmentConfig, *, command: Sequence[str] | None = None,
) -> ad.AnnData:
    """Internal orchestration for sample enrichment tables and round-native storage."""
    output_dir = _sample_output_dir(cfg).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logging(output_dir / "logs" / "enrichment.sample.log")
    LOGGER.info("Starting sample enrichment from %s", cfg.input_path)
    adata = io_utils.load_dataset(cfg.input_path)
    if cfg.regenerate_figures:
        _regenerate_sample_figures(adata, cfg, output_dir, command)
        return adata
    selection, prepared = _prepare_sample_inputs(adata, cfg)
    stem = _sample_output_stem(cfg, selection.round_id)
    zarr_path = output_dir / f"{stem}.zarr"
    archive_path = output_dir / f"{stem}.zarr.tar.zst"
    h5ad_path = output_dir / f"{stem}.h5ad"
    if cfg.input_path.resolve() in {zarr_path.resolve(), archive_path.resolve(), h5ad_path.resolve()}:
        raise ValueError("Sample enrichment output would replace the input dataset; choose another output_name.")
    round_info = adata.uns["cluster_rounds"][selection.round_id]
    previous = round_info.get("sample_enrichment", {})
    if not isinstance(previous, dict):
        raise ValueError("The round's sample_enrichment namespace is malformed.")
    analysis_id, table_dir = _reserve_sample_directory(output_dir, selection.round_id, previous, cfg.figure_formats)
    resolved = cfg.model_dump(mode="json")
    resolved.update({"round_id": selection.round_id, "output_dir": str(output_dir), "output_name": stem})
    input_path = cfg.input_path.resolve()
    input_stat = input_path.stat() if input_path.exists() else None
    input_provenance = {
        **prepared.provenance, "path": str(input_path), "n_obs": adata.n_obs, "n_vars": adata.n_vars,
        "size_bytes": input_stat.st_size if input_stat and input_path.is_file() else None,
        "mtime_ns": input_stat.st_mtime_ns if input_stat else None,
    }
    software = _sample_software_versions()
    manifest = {
        "schema_version": 1, "analysis_id": analysis_id, "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(), "resolved_config": resolved,
        "command": list(command) if command is not None else None,
        "command_shell": shlex.join(command) if command is not None else None,
        "input_provenance": input_provenance, "software_versions": software,
        "completed_outputs": [],
    }
    manifest_path = table_dir / "settings.json"
    _write_sample_manifest(manifest_path, manifest)
    try:
        resources = _load_activity_resources(cfg)
        scored = _score_populations(prepared, selection, resources, cfg)
        models = _fit_activity_contrasts(prepared, scored, cfg)
        tables = _sample_tables(prepared, selection, scored, models, cfg)
        table_paths = {name: str(table_dir / f"{name}.tsv") for name in tables}
        schema = {
            name: {"columns": frame.columns.tolist(), "units": {column: _TABLE_UNITS[column] for column in frame if column in _TABLE_UNITS}}
            for name, frame in tables.items()
        }
        manifest["table_schema"] = schema
        manifest["table_paths"] = table_paths
        _write_sample_manifest(manifest_path, manifest)
        for name, frame in tables.items():
            destination = Path(table_paths[name])
            temporary = destination.with_suffix(".tsv.tmp")
            frame.to_csv(temporary, sep="\t", index=False, na_rep="NA")
            temporary.replace(destination)
        payload = {
            "schema_version": 1, "analysis_id": analysis_id, "resolved_config": resolved,
            "input_provenance": input_provenance, "normalization": prepared.provenance["normalization"],
            "population_mapping": selection.mapping.assign(selected=selection.mapping["population_id"].isin(selection.selected_ids)).rename(index=str),
            "tables": tables, "table_schema": schema, "software_versions": software,
            "artifacts": {"tables": table_paths, "manifest": str(manifest_path), "figures": []},
        }
        if cfg.make_figures:
            payload["artifacts"]["figures"] = _persist_sample_figures(
                payload, output_dir, analysis_id, cfg.figure_formats, cfg.plot_activity,
            )
        manifest["figure_paths"] = payload["artifacts"]["figures"]
        round_info.setdefault("sample_enrichment", {})[analysis_id] = payload
        io_utils.save_dataset(adata, zarr_path, fmt="zarr")
        if not archive_path.is_file():
            raise OSError(f"Dataset serialization did not create {archive_path}.")
        manifest["completed_outputs"].append(str(archive_path))
        if cfg.save_h5ad:
            io_utils.save_dataset(adata, h5ad_path, fmt="h5ad")
            if not h5ad_path.is_file():
                raise OSError(f"Dataset serialization did not create {h5ad_path}.")
            manifest["completed_outputs"].append(str(h5ad_path))
        manifest["status"] = "complete"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_sample_manifest(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            _write_sample_manifest(manifest_path, manifest)
        except OSError:
            LOGGER.exception("Could not update the failed-run manifest at %s", manifest_path)
        LOGGER.exception("Sample enrichment failed for %s", analysis_id)
        raise
    LOGGER.info("Finished sample enrichment: %s (%d activity observations)", analysis_id, len(tables["activity_scores"]))
    return adata
