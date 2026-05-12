from __future__ import annotations

import logging
import re
import gc
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from .clustering_utils import CLUSTER_LABEL_KEY
from .clustering_utils import (
    _cluster_order_by_size,
    _create_shallow_round_from_parent,
    _make_round_id,
    _next_round_index,
    rebuild_round_from_label_parts,
)
from .config import AdataOpsConfig
from .io_utils import load_dataset, save_dataset, sanitize_identifier
from . import plot_utils
from . import rename_utils
from .rename_utils import rename_idents

LOGGER = logging.getLogger(__name__)

__all__ = [
    "rename_idents",
    "load_subset_mapping_tsv",
    "subset_adata_by_cluster_mapping",
    "subset_dataset_from_tsv",
    "rename_dataset_idents",
    "run_adata_ops",
    "merge_datasets",
]


def _dataset_stem_for_outputs(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".zarr.tar.zst"):
        return name[: -len(".zarr.tar.zst")]
    if lower.endswith(".zarr"):
        return name[: -len(".zarr")]
    if lower.endswith(".h5ad"):
        return name[: -len(".h5ad")]
    return path.stem


def _extract_cnn(label: str) -> str | None:
    s = str(label or "").strip()
    if not s:
        return None
    m = re.search(r"\b(C\d+)\b", s)
    if m:
        return m.group(1)
    m2 = re.search(r"(C\d+)", s)
    if m2:
        return m2.group(1)
    return None


def _resolve_pretty_label_series(adata: ad.AnnData, round_id: str | None = None) -> tuple[pd.Series, str]:
    rid = str(round_id) if round_id else None
    rounds = adata.uns.get("cluster_rounds", {})
    active = adata.uns.get("active_cluster_round", None)
    if rid is None and active is not None:
        rid = str(active)

    if isinstance(rounds, dict) and rid and rid in rounds and isinstance(rounds[rid], dict):
        ann = rounds[rid].get("annotation", {})
        if isinstance(ann, dict):
            pretty_key = ann.get("pretty_cluster_key", None)
            if pretty_key and str(pretty_key) in adata.obs:
                key = str(pretty_key)
                return adata.obs[key].astype(str), key

        fallback = f"{CLUSTER_LABEL_KEY}__{rid}"
        if fallback in adata.obs:
            return adata.obs[fallback].astype(str), fallback

        labels_obs_key = rounds[rid].get("labels_obs_key", None)
        display_map = rounds[rid].get("cluster_display_map", None)
        if (
            labels_obs_key
            and str(labels_obs_key) in adata.obs
            and isinstance(display_map, dict)
            and display_map
        ):
            key = str(labels_obs_key)
            return adata.obs[key].astype(str).map(lambda c: str(display_map.get(str(c), str(c)))), key

    if CLUSTER_LABEL_KEY in adata.obs:
        return adata.obs[CLUSTER_LABEL_KEY].astype(str), CLUSTER_LABEL_KEY

    raise KeyError(
        "Could not resolve a pretty cluster label series for Cnn parsing. "
        "Expected active round annotation pretty key or adata.obs['cluster_label']."
    )


def _resolve_round_id(adata: ad.AnnData, round_id: str | None = None) -> str:
    rid = str(round_id) if round_id else None
    if rid is None:
        active = adata.uns.get("active_cluster_round", None)
        rid = str(active) if active else None
    if rid is None:
        raise KeyError("No round_id provided and adata.uns['active_cluster_round'] is None.")
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or rid not in rounds:
        raise KeyError(f"Round {rid!r} not found in adata.uns['cluster_rounds'].")
    return rid


def _resolve_round_pretty_key(adata: ad.AnnData, round_id: str | None = None) -> str:
    rid = _resolve_round_id(adata, round_id)
    rounds = adata.uns["cluster_rounds"]
    rinfo = rounds[rid]
    ann = rinfo.get("annotation", {}) if isinstance(rinfo.get("annotation", {}), dict) else {}
    pretty_key = ann.get("pretty_cluster_key", None)
    if pretty_key and str(pretty_key) in adata.obs:
        return str(pretty_key)
    fallback = f"{CLUSTER_LABEL_KEY}__{rid}"
    if fallback in adata.obs:
        return fallback
    if CLUSTER_LABEL_KEY in adata.obs:
        return CLUSTER_LABEL_KEY
    raise KeyError(f"Could not resolve pretty label key for round {rid!r}.")


def _extract_label_part(label: str) -> str:
    s = str(label or "").strip()
    if not s:
        return "Unknown"
    s = re.sub(r"^\s*C\d+\s*:\s*", "", s)
    return s.strip() or "Unknown"


def _resolve_child_source_series(
    adata: ad.AnnData,
    *,
    round_id: str | None = None,
    source_field: str | None = None,
) -> tuple[pd.Series, str, str]:
    rid = _resolve_round_id(adata, round_id)
    if source_field is not None:
        key = str(source_field)
        if key not in adata.obs:
            raise KeyError(f"Child source field {key!r} not found in adata.obs.")
    else:
        key = _resolve_round_pretty_key(adata, rid)
    return adata.obs[key].astype(str), key, rid


def _emit_rename_round_plots(
    adata: ad.AnnData,
    *,
    output_root: Path,
    round_id: str,
) -> None:
    plot_utils.setup_scanpy_figs(output_root / "figures")
    try:
        if "X_umap" not in adata.obsm:
            for embedding_key in ("X_integrated", "X_scANVI", "X_scVI", "X_pca", "Unintegrated"):
                if embedding_key not in adata.obsm:
                    continue
                sc.pp.neighbors(adata, use_rep=embedding_key)
                sc.tl.umap(adata)
                break
    except Exception as e:
        LOGGER.warning("Rename-only: failed to ensure UMAP; skipping UMAP plots. (%s)", e)
        return

    rounds = adata.uns.get("cluster_rounds", {})
    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"
    try:
        if isinstance(rounds, dict) and round_id in rounds:
            ann = rounds[round_id].get("annotation", {})
            if isinstance(ann, dict):
                pk = ann.get("pretty_cluster_key", None)
                if pk and str(pk) in adata.obs:
                    pretty_key = str(pk)
    except Exception:
        pass

    if pretty_key not in adata.obs or "X_umap" not in adata.obsm:
        LOGGER.warning("Rename-only: pretty label key '%s' not found; skipping UMAP plots.", pretty_key)
        return

    figdir_cluster = Path("rename") / str(round_id) / "clustering"
    artifacts = plot_utils.umap_by_two_legend_styles(
        adata,
        key=pretty_key,
        figdir=figdir_cluster,
        stem="umap_pretty_cluster_label",
        title=pretty_key,
        base_figsize=(6.0, 6.0),
    )
    plot_utils.persist_plot_artifacts(artifacts)


def rename_dataset_idents(
    adata_or_path: ad.AnnData | Path | str,
    rename_mapping_tsv: Path | str,
    *,
    output_root: Path | str,
    output_name: str | None = None,
    output_format: str | None = None,
    round_id: str | None = None,
    target_round_id: str | None = None,
    update_existing_round: bool = False,
    round_name: str = "manual_rename",
    collapse_same_labels: bool = False,
    set_active: bool = True,
) -> tuple[dict[str, Path], pd.DataFrame]:
    source_path: Path | None = None
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        dataset_stem = "adata"
    else:
        source_path = Path(adata_or_path)
        adata = load_dataset(source_path)
        dataset_stem = _dataset_stem_for_outputs(source_path)

    mapping = rename_utils.load_rename_mapping(Path(rename_mapping_tsv))
    if update_existing_round:
        if not target_round_id:
            raise ValueError("update_existing_round=True requires target_round_id.")
        if round_id is not None:
            raise ValueError("round_id cannot be used with update_existing_round=True.")
        target_round_id = _resolve_round_id(adata, target_round_id)
        target_round = adata.uns.get("cluster_rounds", {}).get(target_round_id, None)
        if not isinstance(target_round, dict):
            raise KeyError(f"Target round {target_round_id!r} not found.")
        if str(target_round.get("round_type", "")) != "manual_rename":
            raise ValueError(f"target_round_id {target_round_id!r} is not a manual_rename round.")
        parent_round_id = None
        manual_rename_payload = target_round.get("manual_rename", None)
        if isinstance(manual_rename_payload, dict):
            parent_round_id = manual_rename_payload.get("parent_round_id", None)
        if parent_round_id is None:
            parent_round_id = target_round.get("parent_round_id", None)
        if parent_round_id is None:
            raise KeyError(f"Target round {target_round_id!r} is missing parent_round_id metadata.")
        parent_round_id = str(parent_round_id)
    else:
        if target_round_id is not None:
            raise ValueError("target_round_id can only be used with update_existing_round=True.")
        parent_round_id = _resolve_round_id(adata, round_id)
    new_round_id = rename_idents(
        adata,
        mapping=mapping,
        parent_round_id=parent_round_id,
        new_round_id=target_round_id,
        round_name=str(round_name),
        collapse_same_labels=collapse_same_labels,
        update_existing_round=update_existing_round,
        set_active=set_active,
        notes="Manual rename of pretty labels.",
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _emit_rename_round_plots(
        adata,
        output_root=output_root,
        round_id=new_round_id,
    )

    fmt = str(output_format).lower().strip() if output_format else None
    if fmt is None:
        if source_path is not None and source_path.suffix.lower() == ".h5ad":
            fmt = "h5ad"
        else:
            fmt = "zarr"
    if fmt not in {"zarr", "h5ad"}:
        raise ValueError("output_format must be 'zarr' or 'h5ad'.")

    stem = str(output_name).strip() if output_name else f"{dataset_stem}.renamed"
    out_path = output_root / f"{stem}.{fmt}"
    save_dataset(adata, out_path, fmt=fmt)

    summary_df = pd.DataFrame(
        [
            {
                "new_round_id": str(new_round_id),
                "parent_round_id": str(parent_round_id),
                "n_renamed_clusters": int(len(mapping)),
                "collapse_same_labels": bool(collapse_same_labels),
                "update_existing_round": bool(update_existing_round),
                "output_path": str(out_path),
            }
        ]
    )
    LOGGER.info(
        "rename: new_round_id=%s parent_round_id=%s renamed_clusters=%d collapse_same_labels=%s update_existing_round=%s set_active=%s",
        new_round_id,
        parent_round_id,
        len(mapping),
        collapse_same_labels,
        update_existing_round,
        set_active,
    )
    return {"renamed": out_path}, summary_df


def _refresh_round_metadata_after_subset(adata: ad.AnnData) -> None:
    rounds = adata.uns.get("cluster_rounds", None)
    if not isinstance(rounds, dict):
        return

    for rid, rinfo in rounds.items():
        if not isinstance(rinfo, dict):
            continue
        labels_obs_key = rinfo.get("labels_obs_key", None)
        if not labels_obs_key or str(labels_obs_key) not in adata.obs:
            continue
        labels_obs_key = str(labels_obs_key)
        labels = adata.obs[labels_obs_key].astype(str)
        cluster_order = _cluster_order_by_size(labels)

        vc = labels.value_counts(dropna=False)
        rinfo["cluster_sizes"] = {str(k): int(v) for k, v in vc.items()}
        rinfo["cluster_order"] = list(cluster_order)

        dm = rinfo.get("cluster_display_map", None)
        if isinstance(dm, dict):
            rinfo["cluster_display_map"] = {str(c): str(dm.get(str(c), str(c))) for c in cluster_order}

        ann = rinfo.get("annotation", None)
        if isinstance(ann, dict):
            pretty_key = ann.get("pretty_cluster_key", None)
            if pretty_key and str(pretty_key) in adata.obs:
                pretty = adata.obs[str(pretty_key)].astype(str)
                tmp = pd.DataFrame(
                    {"cluster": labels.to_numpy(), "pretty": pretty.to_numpy()},
                    index=adata.obs_names,
                )
                display_map = (
                    tmp.groupby("cluster", observed=True)["pretty"].first().astype(str).to_dict()
                )
                rinfo["cluster_display_map"] = {
                    str(c): str(display_map.get(str(c), str(c))) for c in cluster_order
                }

        dec = rinfo.get("decoupler", None)
        if isinstance(dec, dict):
            dec["cluster_order"] = list(cluster_order)
            if isinstance(rinfo.get("cluster_display_map", None), dict):
                dec["cluster_display_map"] = dict(rinfo["cluster_display_map"])

        rinfo["cluster_id_map"] = {str(c): str(c) for c in cluster_order}
        rinfo["cluster_renumbering"] = {str(c): str(c) for c in cluster_order}
        rounds[rid] = rinfo

    adata.uns["cluster_rounds"] = rounds


def _resolve_batch_key_for_plots(adata: ad.AnnData) -> str | None:
    candidates: list[str] = []
    uns_batch = adata.uns.get("batch_key", None)
    if uns_batch is not None:
        candidates.append(str(uns_batch))
    candidates.extend(["sample_id", "sample", "batch", "donor"])
    for key in candidates:
        if key in adata.obs:
            return str(key)
    return None


def _resolve_cluster_id_and_label_series_for_merge(
    adata: ad.AnnData,
    *,
    round_id: str | None = None,
    cluster_key: str | None = None,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    rounds = adata.uns.get("cluster_rounds", None)
    if isinstance(rounds, dict) and rounds:
        rid = _resolve_round_id(adata, round_id)
        rinfo = rounds[rid]
        labels_obs_key = str(rinfo.get("labels_obs_key", rinfo.get("cluster_key", "leiden")))
        if labels_obs_key not in adata.obs:
            raise KeyError(f"Round {rid!r} labels_obs_key {labels_obs_key!r} not in adata.obs.")
        cluster_ids = adata.obs[labels_obs_key].astype(str)
        pretty_key = None
        ann = rinfo.get("annotation", {})
        if isinstance(ann, dict):
            pk = ann.get("pretty_cluster_key", None)
            if pk is not None and str(pk) in adata.obs:
                pretty_key = str(pk)
        if pretty_key is not None:
            cluster_labels = adata.obs[pretty_key].astype(str)
        else:
            dm = rinfo.get("cluster_display_map", None)
            if isinstance(dm, dict) and dm:
                cluster_labels = cluster_ids.map(lambda c: str(dm.get(str(c), str(c)))).astype(str)
            else:
                cluster_labels = cluster_ids.astype(str)
        return cluster_ids, cluster_labels, {
            "mode": "round",
            "round_id": str(rid),
            "id_key": str(labels_obs_key),
            "label_key": str(pretty_key) if pretty_key else None,
        }

    fallback_key = str(cluster_key).strip() if cluster_key is not None else ""
    if fallback_key and fallback_key in adata.obs:
        key = fallback_key
    elif "leiden" in adata.obs:
        key = "leiden"
    else:
        raise KeyError(
            "Could not resolve cluster labels for merge subset mode. "
            "Provide --cluster-key for non-scOmnom inputs or ensure obs['leiden'] exists."
        )
    series = adata.obs[key].astype(str)
    return series, series, {"mode": "obs", "round_id": None, "id_key": key, "label_key": key}


def _load_subset_merge_tsv(path: Path | str) -> dict[str, list[str]]:
    rows: list[tuple[str, str]] = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) != 2:
            raise ValueError(
                f"subset-merge TSV must have exactly 2 tab-delimited columns; got {len(parts)} on line {lineno}."
            )
        ds, token = parts
        if not ds or not token:
            raise ValueError(f"subset-merge TSV has empty dataset/token on line {lineno}.")
        rows.append((ds, token))
    if not rows:
        raise ValueError("subset-merge TSV is empty.")
    out: dict[str, list[str]] = {}
    for ds, token in rows:
        out.setdefault(str(ds), []).append(str(token))
    return out


def _resolve_subset_tokens_for_dataset(
    *,
    dataset_name: str,
    tokens: list[str],
    cluster_ids: pd.Series,
    cluster_labels: pd.Series,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    ids = cluster_ids.astype(str)
    labels = cluster_labels.astype(str)
    all_ids = set(ids.unique().tolist())
    selected_ids: set[str] = set()
    selection_rows: list[dict[str, Any]] = []

    for token in tokens:
        tok = str(token).strip()
        if not tok:
            continue
        matched_as = None
        resolved_id = None
        resolved_label = None
        if tok in all_ids:
            matched_as = "id"
            resolved_id = tok
            labels_for_id = labels[ids == resolved_id]
            resolved_label = str(labels_for_id.iloc[0]) if len(labels_for_id) else tok
        else:
            label_match_ids = sorted(set(ids[labels == tok].tolist()))
            if len(label_match_ids) == 1:
                matched_as = "label"
                resolved_id = str(label_match_ids[0])
                resolved_label = tok
            elif len(label_match_ids) > 1:
                raise ValueError(
                    f"subset-merge token {tok!r} in dataset {dataset_name!r} matches multiple cluster IDs via label."
                )
            else:
                cnn_match_ids = sorted(
                    set(
                        ids[
                            labels.map(lambda x: _extract_cnn(str(x)) == tok)
                        ].tolist()
                    )
                )
                if len(cnn_match_ids) == 1:
                    matched_as = "cnn"
                    resolved_id = str(cnn_match_ids[0])
                    labels_for_id = labels[ids == resolved_id]
                    resolved_label = str(labels_for_id.iloc[0]) if len(labels_for_id) else tok
                elif len(cnn_match_ids) > 1:
                    raise ValueError(
                        f"subset-merge token {tok!r} in dataset {dataset_name!r} matches multiple cluster IDs via Cnn."
                    )
                else:
                    raise ValueError(
                        f"subset-merge token {tok!r} in dataset {dataset_name!r} did not match any cluster ID/label."
                    )

        selected_ids.add(str(resolved_id))
        n_cells = int((ids == str(resolved_id)).sum())
        selection_rows.append(
            {
                "dataset": str(dataset_name),
                "token": str(tok),
                "matched_as": str(matched_as),
                "resolved_cluster_id": str(resolved_id),
                "resolved_cluster_label": str(resolved_label),
                "selected_cells": int(n_cells),
            }
        )

    mask = ids.isin(selected_ids).to_numpy()
    return mask, selection_rows


def _reset_stale_embedding_state(adata: ad.AnnData) -> None:
    for key in ("X_umap", "X_pca"):
        if key in adata.obsm:
            del adata.obsm[key]
    for key in list(adata.obsp.keys()):
        del adata.obsp[key]
    for key in ("neighbors", "pca", "umap", "cluster_rounds", "active_cluster_round"):
        if key in adata.uns:
            del adata.uns[key]
    if "highly_variable" in adata.var:
        del adata.var["highly_variable"]


def _recompute_basic_embedding(adata: ad.AnnData) -> None:
    _reset_stale_embedding_state(adata)
    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    if n_obs < 3 or n_vars < 3:
        LOGGER.warning("merge: too few cells/genes for PCA+UMAP recompute (n_obs=%d n_vars=%d).", n_obs, n_vars)
        return

    n_top = int(min(2000, max(200, n_vars)))
    used_hvg = False
    hvg_layer = "counts_cb" if "counts_cb" in adata.layers else None
    try:
        if hvg_layer is not None:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3", layer=hvg_layer)
        else:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat")
        used_hvg = "highly_variable" in adata.var
    except Exception as e:
        LOGGER.warning("merge: HVG recompute failed (%s); proceeding without HVG mask.", e)

    n_comps = int(min(50, n_vars - 1, n_obs - 1))
    if n_comps < 2:
        LOGGER.warning("merge: insufficient rank for PCA recompute (n_comps=%d).", n_comps)
        return
    sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=bool(used_hvg))
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata)


@plot_utils.collect_plot_artifacts
def _plot_merge_source_counts(adata: ad.AnnData, *, key: str, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    if key not in adata.obs:
        return
    counts = adata.obs[key].astype(str).value_counts()
    if counts.empty:
        return
    labels = counts.index.tolist()
    values = counts.values.astype(int)
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 4))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Cells")
    ax.set_title("Cells per source dataset")
    fig.tight_layout()
    plot_utils.record_plot_artifact("merge_source_dataset_counts", figdir, fig=fig)


def _emit_merge_plots(
    adata: ad.AnnData,
    *,
    output_root: Path,
    cluster_label_key: str,
    source_key: str,
) -> None:
    figures_root = output_root / "figures"
    plot_utils.setup_scanpy_figs(figures_root)
    figdir = Path("merge") / "clustering"
    artifacts = []
    source_plot_key = "merge_source_dataset_short" if "merge_source_dataset_short" in adata.obs else source_key
    artifacts.extend(_plot_merge_source_counts(adata, key=source_plot_key, figdir=figdir))
    if cluster_label_key in adata.obs:
        artifacts.extend(plot_utils.plot_cluster_sizes(adata, cluster_label_key, figdir, stem="merge_cluster_sizes"))
    if "X_umap" in adata.obsm:
        if source_plot_key in adata.obs:
            artifacts.extend(
                plot_utils.umap_by_two_legend_styles(
                    adata,
                    key=source_plot_key,
                    figdir=figdir,
                    stem="umap_source_dataset",
                    title=source_plot_key,
                )
            )
        if cluster_label_key in adata.obs:
            artifacts.extend(
                plot_utils.umap_by_two_legend_styles(
                    adata,
                    key=cluster_label_key,
                    figdir=figdir,
                    stem="umap_merge_clusters",
                    title=cluster_label_key,
                )
            )
    plot_utils.persist_plot_artifacts(artifacts)


def merge_datasets(
    input_paths: Sequence[Path | str],
    *,
    output_root: Path | str,
    output_name: str | None = None,
    output_format: str | None = None,
    dataset_short_labels: Sequence[str] | None = None,
    subset_merge_tsv: Path | str | None = None,
    round_id: str | None = None,
    cluster_key: str | None = None,
    join: str = "outer",
    recompute_embedding: bool = True,
) -> tuple[dict[str, Path], pd.DataFrame]:
    if len(input_paths) < 2:
        raise ValueError("merge requires at least two input paths.")
    paths = [Path(p) for p in input_paths]
    names = [_dataset_stem_for_outputs(p) for p in paths]
    if dataset_short_labels is None or len(tuple(dataset_short_labels)) == 0:
        short_labels = [f"dataset{i+1}" for i in range(len(paths))]
    else:
        short_labels = [str(x).strip() for x in dataset_short_labels]
        if len(short_labels) != len(paths):
            raise ValueError(
                f"--dataset-short-label count ({len(short_labels)}) must match number of inputs ({len(paths)})."
            )
        if any(not x for x in short_labels):
            raise ValueError("--dataset-short-label values must be non-empty.")
    if len(set(short_labels)) != len(short_labels):
        dup = sorted({x for x in short_labels if short_labels.count(x) > 1})
        raise ValueError(f"dataset short labels must be unique; duplicates: {dup}")

    if len(set(names)) != len(names):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"merge input basenames must be unique; duplicates: {dup}")

    subset_map = _load_subset_merge_tsv(subset_merge_tsv) if subset_merge_tsv is not None else None
    if subset_map is not None:
        unknown = sorted(set(subset_map.keys()) - set(names))
        if unknown:
            raise ValueError(f"subset-merge references unknown dataset basenames: {unknown}")

    merged: ad.AnnData | None = None
    selection_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    source_key = "merge_source_dataset"
    cluster_id_key = "merge_cluster_id"
    cluster_label_key = "merge_cluster_label"

    for path, dataset_name, dataset_short in zip(paths, names, short_labels):
        adata = load_dataset(path)
        ids, labels, resolver_meta = _resolve_cluster_id_and_label_series_for_merge(
            adata,
            round_id=round_id,
            cluster_key=cluster_key,
        )
        if subset_map is not None:
            tokens = subset_map.get(dataset_name, [])
            if not tokens:
                continue
            mask, rows = _resolve_subset_tokens_for_dataset(
                dataset_name=dataset_name,
                tokens=list(tokens),
                cluster_ids=ids,
                cluster_labels=labels,
            )
            if int(mask.sum()) == 0:
                raise ValueError(f"subset-merge selected zero cells for dataset {dataset_name!r}.")
            selection_rows.extend(rows)
        else:
            mask = np.ones(adata.n_obs, dtype=bool)

        sub = adata[mask].copy()
        source_cell_ids = [str(x) for x in sub.obs_names.tolist()]
        sub.obs["merge_source_cell_id"] = source_cell_ids
        sub.obs_names = pd.Index([f"{dataset_name}__{x}" for x in source_cell_ids], dtype="object")
        sub.obs[source_key] = pd.Categorical([dataset_name] * sub.n_obs)
        sub.obs["merge_source_dataset_short"] = pd.Categorical([dataset_short] * sub.n_obs)
        sub.obs["merge_source_cluster_id"] = ids[mask].to_numpy(dtype="object")
        sub.obs["merge_source_cluster_label"] = labels[mask].to_numpy(dtype="object")
        source_cnn = (
            sub.obs["merge_source_cluster_id"]
            .astype(str)
            .map(lambda v: f"C{int(v):02d}" if re.fullmatch(r"\d+", str(v)) else str(v))
        )
        source_label_clean = (
            sub.obs["merge_source_cluster_label"]
            .astype(str)
            .map(lambda x: re.sub(r"^\s*C\d+\s*:\s*", "", str(x)).strip())
        )
        combined_label = pd.Series(
            [
                f"{dataset_short} {cid} {lab}".strip()
                for cid, lab in zip(
                    source_cnn.tolist(),
                    source_label_clean.tolist(),
                )
            ],
            index=sub.obs_names,
            dtype="object",
        )
        sub.obs["merge_source_cluster_composite"] = combined_label.to_numpy(dtype="object")
        sub.obs["merge_source_cluster_id"] = sub.obs["merge_source_cluster_id"].astype("category")
        sub.obs["merge_source_cluster_label"] = sub.obs["merge_source_cluster_label"].astype("category")
        sub.obs["merge_source_cluster_composite"] = sub.obs["merge_source_cluster_composite"].astype("category")
        selected_n = int(sub.n_obs)
        if merged is None:
            merged = sub
        else:
            merged = ad.concat([merged, sub], axis=0, join=str(join), merge="same", uns_merge="first")
            del sub
            gc.collect()

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "dataset_short_label": dataset_short,
                "input_path": str(path),
                "selected_cells": selected_n,
                "total_cells": int(adata.n_obs),
                "resolver_mode": str(resolver_meta.get("mode")),
                "resolver_round_id": resolver_meta.get("round_id"),
                "resolver_id_key": resolver_meta.get("id_key"),
                "resolver_label_key": resolver_meta.get("label_key"),
            }
        )

    if merged is None:
        raise ValueError("merge produced zero selected cells across all inputs.")

    composite_key = "merge_source_cluster_composite"
    if composite_key in merged.obs:
        comps = merged.obs[composite_key].astype(str)
        order = _cluster_order_by_size(comps)
        merged_cluster_ids = [f"C{i:02d}" for i in range(len(order))]
        id_map = {str(comp): cid for comp, cid in zip(order, merged_cluster_ids)}
        merged.obs[cluster_id_key] = comps.map(id_map).astype("category")
        merged.obs[cluster_label_key] = merged.obs.apply(
            lambda r: f"{id_map.get(str(r[composite_key]), 'C??')}: {str(r[composite_key])}",
            axis=1,
        ).astype("category")
    else:
        merged.obs[cluster_id_key] = pd.Categorical(["C00"] * merged.n_obs)
        merged.obs[cluster_label_key] = pd.Categorical(["C00: merged"] * merged.n_obs)

    if recompute_embedding:
        _recompute_basic_embedding(merged)

    provenance = {
        "input_paths": [str(p) for p in paths],
        "input_dataset_names": [str(n) for n in names],
        "input_dataset_short_labels": [str(x) for x in short_labels],
        "inputs_table": pd.DataFrame(
            {
                "dataset": [str(n) for n in names],
                "dataset_short_label": [str(x) for x in short_labels],
                "input_path": [str(p) for p in paths],
            }
        ),
        "join": str(join),
        "round_id": None if round_id is None else str(round_id),
        "cluster_key": None if cluster_key is None else str(cluster_key),
        "subset_merge_tsv": None if subset_merge_tsv is None else str(subset_merge_tsv),
        "subset_selections_table": pd.DataFrame(selection_rows),
        "dataset_summary_table": pd.DataFrame(dataset_rows),
        "recompute_embedding": bool(recompute_embedding),
    }
    merged.uns["merge"] = provenance

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _emit_merge_plots(
        merged,
        output_root=output_root,
        cluster_label_key=cluster_label_key,
        source_key=source_key,
    )

    source_path0 = paths[0]
    fmt = str(output_format).lower().strip() if output_format else None
    if fmt is None:
        if source_path0.suffix.lower() == ".h5ad":
            fmt = "h5ad"
        else:
            fmt = "zarr"
    if fmt not in {"zarr", "h5ad"}:
        raise ValueError("output_format must be 'zarr' or 'h5ad'.")

    stem = str(output_name).strip() if output_name else "adata.merged"
    out_path = output_root / f"{stem}.{fmt}"
    save_dataset(merged, out_path, fmt=fmt)

    summary_df = pd.DataFrame(dataset_rows).sort_values("dataset").reset_index(drop=True)
    return {"merged": out_path}, summary_df


def _ensure_umap_for_annotation_merge(adata: ad.AnnData) -> None:
    if "X_umap" in adata.obsm:
        return
    for embedding_key in ("X_integrated", "X_scANVI", "X_scVI", "X_pca", "Unintegrated"):
        if embedding_key not in adata.obsm:
            continue
        try:
            sc.pp.neighbors(adata, use_rep=embedding_key)
            sc.tl.umap(adata)
            LOGGER.info("annotation-merge: computed UMAP from %s", embedding_key)
            return
        except Exception as e:
            LOGGER.warning("annotation-merge: failed to compute UMAP from %s (%s)", embedding_key, e)
    LOGGER.info("annotation-merge: no usable embedding found for UMAP plot generation.")


def _emit_annotation_merge_plots(
    adata: ad.AnnData,
    *,
    output_root: Path,
    round_id: str,
) -> None:
    figures_root = output_root / "figures"
    plot_utils.setup_scanpy_figs(figures_root)
    figdir = Path("merge-annotation") / round_id / "clustering"
    pretty_key = f"{CLUSTER_LABEL_KEY}__{round_id}"
    label_key = pretty_key if pretty_key in adata.obs else str(
        adata.uns.get("cluster_rounds", {}).get(round_id, {}).get("labels_obs_key", CLUSTER_LABEL_KEY)
    )
    batch_key = _resolve_batch_key_for_plots(adata)

    artifacts = []
    _ensure_umap_for_annotation_merge(adata)
    if "X_umap" in adata.obsm:
        try:
            artifacts.extend(
                plot_utils.plot_cluster_umaps(
                    adata=adata,
                    label_key=label_key,
                    batch_key=batch_key,
                    figdir=figdir,
                )
            )
        except Exception as e:
            LOGGER.warning("annotation-merge: failed to create UMAP plots (%s)", e)
        if pretty_key in adata.obs:
            try:
                artifacts.extend(
                    plot_utils.umap_by_two_legend_styles(
                        adata,
                        key=pretty_key,
                        figdir=figdir,
                        stem="umap_pretty_cluster_label",
                        title=pretty_key,
                    )
                )
            except Exception as e:
                LOGGER.warning("annotation-merge: failed to create pretty-label UMAP plots (%s)", e)

    try:
        artifacts.extend(plot_utils.plot_cluster_sizes(adata, label_key, figdir))
    except Exception as e:
        LOGGER.warning("annotation-merge: failed to create cluster size plot (%s)", e)

    if all(col in adata.obs for col in ("n_genes_by_counts", "total_counts", "pct_counts_mt")):
        try:
            artifacts.extend(plot_utils.plot_cluster_qc_summary(adata, label_key, figdir))
        except Exception as e:
            LOGGER.warning("annotation-merge: failed to create cluster QC summary (%s)", e)

    embedding_key = None
    for candidate in ("X_integrated", "X_scANVI", "X_scVI", "X_pca", "Unintegrated", "X_umap"):
        if candidate in adata.obsm:
            embedding_key = candidate
            break
    if embedding_key is not None:
        try:
            artifacts.extend(
                plot_utils.plot_cluster_silhouette_by_cluster(
                    adata,
                    label_key,
                    embedding_key,
                    figdir,
                )
            )
        except Exception as e:
            LOGGER.warning("annotation-merge: failed to create silhouette plot (%s)", e)

    if batch_key is not None:
        try:
            artifacts.extend(plot_utils.plot_cluster_batch_composition(adata, label_key, batch_key, figdir))
        except Exception as e:
            LOGGER.warning("annotation-merge: failed to create batch composition plot (%s)", e)

    plot_utils.persist_plot_artifacts(artifacts)


def annotation_merge_datasets(
    parent_or_path: ad.AnnData | Path | str,
    *,
    child_paths: Sequence[Path | str],
    output_root: Path | str,
    output_format: str | None = None,
    round_id: str | None = None,
    child_round_id: str | None = None,
    child_source_field: str | None = None,
    target_round_id: str | None = None,
    update_existing_round: bool = False,
    annotation_merge_round_name: str = "subset_annotation",
) -> tuple[dict[str, Path], pd.DataFrame]:
    source_path: Path | None = None
    if isinstance(parent_or_path, ad.AnnData):
        adata = parent_or_path
        dataset_stem = "adata"
    else:
        source_path = Path(parent_or_path)
        adata = load_dataset(source_path)
        dataset_stem = _dataset_stem_for_outputs(source_path)

    if not child_paths:
        raise ValueError("annotation-merge requires at least one child dataset path.")

    child_paths_resolved = [Path(p) for p in child_paths]
    fmt = str(output_format).lower().strip() if output_format else None
    if fmt is None:
        if source_path is not None and source_path.suffix.lower() == ".h5ad":
            fmt = "h5ad"
        else:
            fmt = "zarr"
    if fmt not in {"zarr", "h5ad"}:
        raise ValueError("output_format must be 'zarr' or 'h5ad'.")

    if update_existing_round:
        if not target_round_id:
            raise ValueError("update_existing_round=True requires target_round_id.")
        working_round_id = _resolve_round_id(adata, target_round_id)
        rounds = adata.uns["cluster_rounds"]
        rinfo = rounds[working_round_id]
        if str(rinfo.get("round_type", "")) != "subset_annotation":
            raise ValueError(f"target_round_id {working_round_id!r} is not a subset_annotation round.")
        base_round_id = working_round_id
    else:
        base_round_id = _resolve_round_id(adata, round_id)
        base_rinfo = adata.uns["cluster_rounds"][base_round_id]
        if target_round_id is not None:
            raise ValueError("target_round_id can only be used with update_existing_round=True.")
        next_round_id = _make_round_id(_next_round_index(adata), annotation_merge_round_name)
        labels_obs_key_new = f"{base_rinfo.get('cluster_key', 'leiden')}__{next_round_id}"
        if labels_obs_key_new in adata.obs:
            raise ValueError(f"labels_obs_key '{labels_obs_key_new}' already exists in adata.obs.")
        parent_labels_obs_key = str(base_rinfo.get("labels_obs_key", base_rinfo.get("cluster_key", "leiden")))
        adata.obs[labels_obs_key_new] = adata.obs[parent_labels_obs_key].astype(str).astype("category")
        working_round_id = _create_shallow_round_from_parent(
            adata,
            parent_round_id=base_round_id,
            round_name=annotation_merge_round_name,
            new_round_id=next_round_id,
            round_type="subset_annotation",
            kind="SUBSET_ANNOTATION",
            notes="Subset annotation overlay from child objects.",
            set_active=False,
            cluster_key=str(base_rinfo.get("cluster_key", "leiden")),
            labels_obs_key=labels_obs_key_new,
            inherit_fields=("annotation",),
        )

    base_pretty_series, base_pretty_key = _resolve_pretty_label_series(adata, round_id=base_round_id)
    merged_parts = base_pretty_series.map(_extract_label_part).astype(str).copy()

    child_rows: list[dict[str, Any]] = []
    seen_cells: set[str] = set()

    for child_path in child_paths_resolved:
        child = load_dataset(child_path)
        child_series, child_key_used, child_resolved_round_id = _resolve_child_source_series(
            child,
            round_id=child_round_id,
            source_field=child_source_field,
        )
        child_cells = [str(x) for x in child.obs_names.tolist()]
        child_cell_set = set(child_cells)
        missing = sorted(child_cell_set - set(map(str, adata.obs_names.tolist())))
        if missing:
            raise ValueError(
                f"Child dataset {child_path} contains cells absent from parent: {','.join(missing[:10])}"
            )
        overlap = sorted(seen_cells & child_cell_set)
        if overlap:
            raise ValueError(
                f"Child datasets overlap on parent cells: {','.join(overlap[:10])}"
            )
        seen_cells.update(child_cell_set)
        merged_parts.loc[child.obs_names] = child_series.map(_extract_label_part).astype(str).values
        child_rows.append(
            {
                "child_path": str(child_path),
                "child_round_id": str(child_resolved_round_id),
                "child_source_field": str(child_key_used),
                "n_cells": int(child.n_obs),
            }
        )

    merge_meta = {
        "base_round_id": str(base_round_id),
        "child_round_id": None if child_round_id is None else str(child_round_id),
        "child_source_field": (
            str(child_source_field)
            if child_source_field is not None
            else "annotation.pretty_cluster_key"
        ),
        "n_children": int(len(child_rows)),
        "n_overlay_cells": int(len(seen_cells)),
        "children": child_rows,
    }
    rebuild_round_from_label_parts(
        adata,
        round_id=working_round_id,
        label_parts=merged_parts,
        round_type="subset_annotation",
        metadata_key="subset_annotation",
        metadata_value=merge_meta,
        set_active=True,
    )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _emit_annotation_merge_plots(
        adata,
        output_root=output_root,
        round_id=working_round_id,
    )
    safe_round = sanitize_identifier(working_round_id, max_len=80, allow_spaces=False)
    out_path = output_root / f"{dataset_stem}__annotation_merge_{safe_round}.{fmt}"
    save_dataset(adata, out_path, fmt=fmt)

    summary_df = pd.DataFrame(
        [
            {
                "round_id": str(working_round_id),
                "base_round_id": str(base_round_id),
                "n_children": int(len(child_rows)),
                "n_overlay_cells": int(len(seen_cells)),
                "output_path": str(out_path),
            }
        ]
    )
    LOGGER.info(
        "annotation-merge: round=%s base=%s children=%d overlay_cells=%d",
        working_round_id,
        base_round_id,
        len(child_rows),
        len(seen_cells),
    )
    return {"merged": out_path}, summary_df


def load_subset_mapping_tsv(path: Path | str) -> dict[str, list[str]]:
    """
    Load 2-column, tab-delimited mapping:
      col1: Cnn cluster code
      col2: subset name
    """
    tsv_path = Path(path)
    df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str)
    if df.shape[1] != 2:
        raise ValueError("subset mapping file must have exactly 2 columns (no header).")

    rows = []
    for cnn, subset in df.itertuples(index=False):
        c = str(cnn).strip()
        s = str(subset).strip()
        if not c or not s:
            raise ValueError("subset mapping contains empty cluster code or subset name.")
        if re.fullmatch(r"C\d+", c) is None:
            raise ValueError(f"subset mapping key must be strict 'Cnn' format, got: {c!r}")
        rows.append((c, s))

    if not rows:
        raise ValueError("subset mapping file is empty.")

    cluster_to_subset: dict[str, str] = {}
    for c, s in rows:
        if c in cluster_to_subset and cluster_to_subset[c] != s:
            raise ValueError(
                f"cluster {c!r} is mapped to multiple subsets: "
                f"{cluster_to_subset[c]!r} and {s!r}"
            )
        cluster_to_subset[c] = s

    subset_to_clusters: dict[str, list[str]] = {}
    for c, s in cluster_to_subset.items():
        subset_to_clusters.setdefault(s, []).append(c)

    for s in list(subset_to_clusters.keys()):
        subset_to_clusters[s] = sorted(subset_to_clusters[s])

    return dict(sorted(subset_to_clusters.items(), key=lambda kv: kv[0]))


def subset_adata_by_cluster_mapping(
    adata: ad.AnnData,
    subset_to_clusters: dict[str, list[str]],
    *,
    round_id: str | None = None,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    if not subset_to_clusters:
        raise ValueError("subset_to_clusters is empty.")

    pretty_series, label_key_used = _resolve_pretty_label_series(adata, round_id=round_id)
    cell_cnn = pretty_series.map(_extract_cnn)
    if cell_cnn.isna().all():
        raise RuntimeError(
            f"Could not parse any Cnn codes from label key {label_key_used!r}. "
            "Expected labels containing tokens like 'C03'."
        )

    outputs: dict[str, ad.AnnData] = {}
    rows: list[dict[str, Any]] = []
    available = {x for x in cell_cnn.dropna().astype(str).unique().tolist() if x}
    requested = {str(c) for clusters in subset_to_clusters.values() for c in clusters}
    missing = sorted(requested - available)
    if missing:
        avail_preview = ",".join(sorted(available)) if available else "<none>"
        raise ValueError(
            "Requested Cnn clusters were not found in resolved labels "
            f"({label_key_used!r}): {','.join(missing)}. "
            f"Available Cnn values: {avail_preview}."
        )

    for subset_name, clusters in subset_to_clusters.items():
        wanted = {str(c) for c in clusters}
        mask = cell_cnn.isin(wanted).to_numpy()
        sub = adata[mask].copy()
        sub.uns["Compartment"] = str(subset_name)
        _refresh_round_metadata_after_subset(sub)
        outputs[str(subset_name)] = sub
        rows.append(
            {
                "subset_name": str(subset_name),
                "n_cells": int(sub.n_obs),
                "n_clusters": int(len(wanted)),
                "clusters": ",".join(sorted(wanted)),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("subset_name").reset_index(drop=True)
    return outputs, summary_df


def subset_dataset_from_tsv(
    adata_or_path: ad.AnnData | Path | str,
    subset_mapping_tsv: Path | str,
    *,
    output_root: Path | str = Path("results"),
    output_format: str | None = None,
    round_id: str | None = None,
) -> tuple[dict[str, Path], pd.DataFrame]:
    """
    Split an input dataset into named subsets based on a Cnn -> subset TSV.
    """
    source_path: Path | None = None
    if isinstance(adata_or_path, ad.AnnData):
        adata = adata_or_path
        dataset_stem = "adata"
    else:
        source_path = Path(adata_or_path)
        adata = load_dataset(source_path)
        dataset_stem = _dataset_stem_for_outputs(source_path)

    subset_to_clusters = load_subset_mapping_tsv(subset_mapping_tsv)
    subset_map, summary_df = subset_adata_by_cluster_mapping(
        adata,
        subset_to_clusters,
        round_id=round_id,
    )

    fmt = str(output_format).lower().strip() if output_format else None
    if fmt is None:
        if source_path is not None and source_path.suffix.lower() == ".h5ad":
            fmt = "h5ad"
        else:
            fmt = "zarr"
    if fmt not in {"zarr", "h5ad"}:
        raise ValueError("output_format must be 'zarr' or 'h5ad'.")

    output_root = Path(output_root)
    out_dataset_dir = output_root / "subsets"
    out_tables_dir = output_root / "tables"
    out_dataset_dir.mkdir(parents=True, exist_ok=True)
    out_tables_dir.mkdir(parents=True, exist_ok=True)

    out_paths: dict[str, Path] = {}
    for subset_name, sub in subset_map.items():
        safe_subset = sanitize_identifier(subset_name, max_len=80, allow_spaces=False)
        out_path = out_dataset_dir / f"{dataset_stem}__subset_{safe_subset}.{fmt}"
        save_dataset(sub, out_path, fmt=fmt)
        out_paths[subset_name] = out_path

    summary_path = out_tables_dir / f"{dataset_stem}__subset_summary.tsv"
    summary_df.to_csv(summary_path, sep="\t", index=False)

    LOGGER.info("Subset summary:")
    for _, row in summary_df.iterrows():
        LOGGER.info(
            "  subset=%s n_cells=%d clusters=%s",
            str(row["subset_name"]),
            int(row["n_cells"]),
            str(row["clusters"]),
        )
    print(summary_df.to_string(index=False))
    LOGGER.info("Wrote subset summary table: %s", summary_path)

    return out_paths, summary_df


def run_adata_ops(cfg: AdataOpsConfig) -> tuple[dict[str, Path], pd.DataFrame]:
    op = str(cfg.operation).strip().lower()
    if op == "subset":
        if cfg.subset_mapping_tsv is None:
            raise ValueError("subset operation requires subset_mapping_tsv.")
        return subset_dataset_from_tsv(
            cfg.input_path,
            cfg.subset_mapping_tsv,
            output_root=cfg.resolved_output_dir,
            output_format=cfg.output_format,
            round_id=cfg.round_id,
        )
    if op == "rename":
        if cfg.rename_idents_file is None:
            raise ValueError("rename operation requires rename_idents_file.")
        return rename_dataset_idents(
            cfg.input_path,
            cfg.rename_idents_file,
            output_root=cfg.resolved_output_dir,
            output_name=cfg.output_name,
            output_format=cfg.output_format,
            round_id=cfg.round_id,
            target_round_id=cfg.target_round_id,
            update_existing_round=cfg.update_existing_round,
            round_name=cfg.rename_round_name,
            collapse_same_labels=cfg.rename_collapse_same_labels,
            set_active=cfg.rename_set_active,
        )
    if op == "annotation_merge":
        return annotation_merge_datasets(
            cfg.input_path,
            child_paths=cfg.child_paths,
            output_root=cfg.resolved_output_dir,
            output_format=cfg.output_format,
            round_id=cfg.round_id,
            child_round_id=cfg.child_round_id,
            child_source_field=cfg.child_source_field,
            target_round_id=cfg.target_round_id,
            update_existing_round=cfg.update_existing_round,
            annotation_merge_round_name=cfg.annotation_merge_round_name,
        )
    if op == "merge":
        paths = tuple(cfg.input_paths) if cfg.input_paths else (cfg.input_path,)
        return merge_datasets(
            paths,
            output_root=cfg.resolved_output_dir,
            output_name=cfg.output_name,
            output_format=cfg.output_format,
            dataset_short_labels=cfg.dataset_short_labels,
            subset_merge_tsv=cfg.subset_merge_tsv,
            round_id=cfg.round_id,
            cluster_key=cfg.cluster_key,
            join=cfg.join,
            recompute_embedding=cfg.recompute_embedding,
        )
    raise ValueError(f"Unsupported adata-ops operation: {cfg.operation!r}")
