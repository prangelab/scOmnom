#!/usr/bin/env python3
"""Add unique child labels before subset annotation merge-back."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scomnom.io_utils import load_dataset, save_dataset


def _active_round_id(adata) -> str:
    round_id = adata.uns.get("active_cluster_round")
    if not round_id:
        raise KeyError("No active_cluster_round found in adata.uns.")
    return str(round_id)


def _round_keys(adata, round_id: str | None) -> tuple[str, str, str]:
    resolved = str(round_id) if round_id else _active_round_id(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if resolved not in rounds:
        available = ", ".join(map(str, rounds.keys()))
        raise KeyError(f"Round {resolved!r} not found. Available rounds: {available}")
    info = rounds[resolved]
    labels_key = str(info.get("labels_obs_key", ""))
    if labels_key not in adata.obs:
        raise KeyError(f"Round {resolved!r} labels_obs_key {labels_key!r} not present in adata.obs.")
    ann = info.get("annotation", {}) if isinstance(info.get("annotation", {}), dict) else {}
    pretty_key = str(ann.get("pretty_cluster_key", ""))
    if not pretty_key or pretty_key not in adata.obs:
        pretty_key = labels_key
    return resolved, labels_key, pretty_key


def _split_pretty_label(cluster_id: str, pretty: str) -> tuple[str, str]:
    text = str(pretty or "").strip()
    match = re.match(r"^\s*(C\d+)\s*:\s*(.+?)\s*$", text)
    if match:
        return match.group(1), match.group(2).strip() or "Unknown"
    try:
        code = f"C{int(str(cluster_id)):02d}"
    except ValueError:
        code = str(cluster_id)
    return code, text or "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--round-id", default=None)
    parser.add_argument("--source-field", default="tutorial_unique_merge_label")
    parser.add_argument("--label-prefix", default="T/NK")
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = load_dataset(args.input_path)
    round_id, labels_key, pretty_key = _round_keys(adata, args.round_id)
    labels = adata.obs[labels_key].astype(str)
    pretty = adata.obs[pretty_key].astype(str)

    rows = []
    cluster_to_merge_label: dict[str, str] = {}
    for cluster_id in sorted(labels.unique(), key=lambda value: (len(str(value)), str(value))):
        mask = labels == cluster_id
        top_pretty = str(pretty[mask].value_counts().index[0])
        code, label_part = _split_pretty_label(str(cluster_id), top_pretty)
        merge_label = f"{args.label_prefix} {code} - {label_part}"
        cluster_to_merge_label[str(cluster_id)] = merge_label
        rows.append(
            {
                "child_cluster_id": str(cluster_id),
                "child_cluster_code": code,
                "n_cells": int(mask.sum()),
                "child_pretty_label": top_pretty,
                "merge_label": merge_label,
            }
        )

    table = pd.DataFrame(rows)
    if table["merge_label"].duplicated().any():
        dupes = table.loc[table["merge_label"].duplicated(keep=False), "merge_label"].tolist()
        raise RuntimeError(f"Non-unique merge labels generated: {dupes}")

    adata.obs[args.source_field] = labels.map(cluster_to_merge_label).astype("category")
    table.to_csv(args.table_path, sep="\t", index=False)

    lines = [
        "# Annotation Merge Child Source",
        "",
        f"Input: `{args.input_path}`",
        f"Round: `{round_id}`",
        f"Labels key: `{labels_key}`",
        f"Pretty key: `{pretty_key}`",
        f"Created field: `{args.source_field}`",
        f"Label prefix: `{args.label_prefix}`",
        f"Cells: {adata.n_obs}",
        f"Child clusters: {len(table)}",
        "",
        "| Child Cluster | Cells | Child Pretty Label | Merge Label |",
        "| --- | ---: | --- | --- |",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.child_cluster_code} | {row.n_cells} | {row.child_pretty_label} | {row.merge_label} |")
    lines.append("")
    lines.append("Merge labels include child cluster codes to prevent accidental collapse of identical labels.")
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_dataset(adata, args.output_path, fmt="zarr")
    print(f"round_id={round_id}")
    print(f"source_field={args.source_field}")
    print(f"output={args.output_path}")


if __name__ == "__main__":
    main()
