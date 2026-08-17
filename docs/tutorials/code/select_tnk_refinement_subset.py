#!/usr/bin/env python3
"""Create a T/NK refinement subset mapping from global scOmnom labels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scomnom.io_utils import load_dataset


INCLUDE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bT\s*cell",
        r"\bT_cells?\b",
        r"\bCD4\b",
        r"\bCD8\b",
        r"\bTcm\b",
        r"\bTem\b",
        r"\bTreg\b",
        r"\bMAIT\b",
        r"gamma.?delta",
        r"\bgdT\b",
        r"\bNKT\b",
        r"\bNK\b",
        r"cytotoxic",
        r"natural killer",
    ]
]

EXCLUDE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bB\s*cell",
        r"\bB_cells?\b",
        r"plasma",
        r"monocyte",
        r"macrophage",
        r"dendritic",
        r"\bDC\d?\b",
        r"platelet",
        r"megakaryocyte",
        r"erythro",
    ]
]


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


def _is_target_label(label: str) -> bool:
    if any(pattern.search(label) for pattern in EXCLUDE_PATTERNS):
        return False
    return any(pattern.search(label) for pattern in INCLUDE_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--round-id", default=None)
    parser.add_argument("--subset-name", default="tnk_refinement")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    adata = load_dataset(args.input_path)
    round_id, labels_key, pretty_key = _round_keys(adata, args.round_id)
    labels = adata.obs[labels_key].astype(str)
    pretty = adata.obs[pretty_key].astype(str)

    rows = []
    for cluster in sorted(labels.unique(), key=lambda value: (len(str(value)), str(value))):
        mask = labels == cluster
        pretty_counts = pretty[mask].value_counts()
        top_pretty = str(pretty_counts.index[0]) if len(pretty_counts) else str(cluster)
        selected = _is_target_label(top_pretty)
        rows.append(
            {
                "cluster": str(cluster),
                "n_cells": int(mask.sum()),
                "top_pretty_label": top_pretty,
                "selected_for_refinement": bool(selected),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = table_dir / "tnk_refinement_cluster_selection.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    selected = summary.loc[summary["selected_for_refinement"], ["cluster"]].copy()
    if selected.empty:
        raise RuntimeError(
            "No T/NK-like clusters selected. "
            f"Inspect {summary_path} and adjust the refinement rules."
        )
    selected["subset_name"] = args.subset_name
    mapping_path = table_dir / "tnk_refinement_subset_mapping.tsv"
    selected.to_csv(mapping_path, sep="\t", header=False, index=False)

    report_path = args.output_dir / "tnk_refinement_subset.md"
    lines = [
        "# T/NK Refinement Subset",
        "",
        f"Input: `{args.input_path}`",
        f"Round: `{round_id}`",
        f"Labels key: `{labels_key}`",
        f"Pretty key: `{pretty_key}`",
        f"Subset name: `{args.subset_name}`",
        f"Selected clusters: {len(selected)}",
        f"Selected cells: {int(summary.loc[summary['selected_for_refinement'], 'n_cells'].sum())}",
        "",
        "| Cluster | Cells | Label | Selected |",
        "| --- | ---: | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.cluster} | {row.n_cells} | {row.top_pretty_label} | {row.selected_for_refinement} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"round_id={round_id}")
    print(f"mapping={mapping_path}")
    print(f"summary={summary_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
