#!/usr/bin/env python3
"""Add compartment and supercompartment annotations to a refined PBMC object."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scomnom.io_utils import load_dataset, save_dataset


RULES = [
    ("T/NK", "lymphoid", [r"\bT\b", r"\bCD4\b", r"\bCD8\b", r"\bNK\b", r"\bNKT\b", r"MAIT", r"cytotoxic"]),
    ("B/plasma", "lymphoid", [r"\bB\b", r"plasma", r"plasmablast"]),
    ("myeloid", "myeloid", [r"monocyte", r"macrophage", r"dendritic", r"\bDC\b", r"\bDC\d\b"]),
    ("platelet/megakaryocyte", "other", [r"platelet", r"megakaryocyte"]),
    ("erythroid", "other", [r"erythro", r"\bRBC\b"]),
]


def _active_round_id(adata) -> str:
    round_id = adata.uns.get("active_cluster_round")
    if not round_id:
        raise KeyError("No active_cluster_round found in adata.uns.")
    return str(round_id)


def _pretty_key(adata, round_id: str | None) -> tuple[str, str]:
    resolved = str(round_id) if round_id else _active_round_id(adata)
    rounds = adata.uns.get("cluster_rounds", {})
    if resolved not in rounds:
        available = ", ".join(map(str, rounds.keys()))
        raise KeyError(f"Round {resolved!r} not found. Available rounds: {available}")
    info = rounds[resolved]
    ann = info.get("annotation", {}) if isinstance(info.get("annotation", {}), dict) else {}
    key = str(ann.get("pretty_cluster_key", ""))
    if not key or key not in adata.obs:
        key = str(info.get("labels_obs_key", ""))
    if key not in adata.obs:
        raise KeyError(f"No usable label key found for round {resolved!r}.")
    return resolved, key


def _assign(label: str) -> tuple[str, str]:
    text = str(label)
    for compartment, supercompartment, patterns in RULES:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return compartment, supercompartment
    return "other immune", "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--round-id", default=None)
    parser.add_argument("--compartment-key", default="compartment")
    parser.add_argument("--supercompartment-key", default="supercompartment")
    parser.add_argument("--table-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = load_dataset(args.input_path)
    round_id, label_key = _pretty_key(adata, args.round_id)
    labels = adata.obs[label_key].astype(str)

    assigned = labels.map(_assign)
    adata.obs[args.compartment_key] = pd.Categorical([value[0] for value in assigned])
    adata.obs[args.supercompartment_key] = pd.Categorical([value[1] for value in assigned])

    table = (
        pd.DataFrame(
            {
                "label": labels,
                "compartment": adata.obs[args.compartment_key].astype(str).values,
                "supercompartment": adata.obs[args.supercompartment_key].astype(str).values,
            },
            index=adata.obs_names,
        )
        .groupby(["label", "compartment", "supercompartment"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["compartment", "supercompartment", "n_cells"], ascending=[True, True, False])
    )
    table.to_csv(args.table_path, sep="\t", index=False)

    lines = [
        "# Custom Annotation Layers",
        "",
        f"Input: `{args.input_path}`",
        f"Round: `{round_id}`",
        f"Label key: `{label_key}`",
        f"Compartment key: `{args.compartment_key}`",
        f"Supercompartment key: `{args.supercompartment_key}`",
        f"Cells: {adata.n_obs}",
        "",
        "| Compartment | Supercompartment | Cells |",
        "| --- | --- | ---: |",
    ]
    counts = (
        pd.DataFrame(
            {
                "compartment": adata.obs[args.compartment_key].astype(str).values,
                "supercompartment": adata.obs[args.supercompartment_key].astype(str).values,
            }
        )
        .groupby(["compartment", "supercompartment"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
    )
    for row in counts.itertuples(index=False):
        lines.append(f"| {row.compartment} | {row.supercompartment} | {row.n_cells} |")
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_dataset(adata, args.output_path, fmt="zarr")
    print(f"round_id={round_id}")
    print(f"label_key={label_key}")
    print(f"output={args.output_path}")


if __name__ == "__main__":
    main()
