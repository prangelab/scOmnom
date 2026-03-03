from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from .clustering_utils import CLUSTER_LABEL_KEY
from .config import AdataOpsConfig
from .io_utils import load_dataset, save_dataset, sanitize_identifier

LOGGER = logging.getLogger(__name__)


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


def _cluster_order_by_size(labels: pd.Series) -> list[str]:
    s = labels.astype(str)
    vc = s.value_counts(dropna=False)
    df = pd.DataFrame({"cluster": vc.index.astype(str), "n": vc.values.astype(int)})
    df["cluster_sort"] = df["cluster"].astype(str)
    df = df.sort_values(["n", "cluster_sort"], ascending=[False, True], kind="mergesort")
    return df["cluster"].astype(str).tolist()


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
        dataset_stem = source_path.stem

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
    out_dataset_dir = output_root / "datasets" / "subsets"
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
    if op != "subset":
        raise ValueError(f"Unsupported adata-ops operation: {cfg.operation!r}")
    return subset_dataset_from_tsv(
        cfg.input_path,
        cfg.subset_mapping_tsv,
        output_root=cfg.resolved_output_dir,
        output_format=cfg.output_format,
        round_id=cfg.round_id,
    )
