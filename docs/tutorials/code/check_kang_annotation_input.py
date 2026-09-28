"""Check filtered counts and reviewed state mapping before Kang annotation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mapping(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
    if frame.shape[1] != 2 or frame.empty:
        raise ValueError("Mapping must contain two tab-delimited columns, without a header.")
    frame = frame.apply(lambda column: column.str.strip())
    if frame.iloc[:, 0].duplicated().any() or (frame == "").any().any():
        raise ValueError("Mapping contains duplicate codes or empty values.")
    return dict(frame.itertuples(index=False, name=None))


def canonical_counts(matrix):
    counts = sparse.csr_matrix(matrix, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError("counts_raw must contain finite, nonnegative values.")
    if not np.equal(counts.data, np.rint(counts.data)).all():
        raise ValueError("counts_raw is not integer-valued; stop before DE. Do not round normalized values.")
    return counts


def validate(adata, filtered, mapping, round_id, frozen_reference=None):
    for obj in (adata, filtered):
        if not obj.obs_names.is_unique or not obj.var_names.is_unique:
            raise ValueError("Cell and gene identifiers must be unique.")
        if "counts_raw" not in obj.layers or "counts_cb" in obj.layers:
            raise ValueError("This Kang example requires filtered-input counts_raw without counts_cb.")
    rows = filtered.obs_names.get_indexer(adata.obs_names)
    cols = filtered.var_names.get_indexer(adata.var_names)
    if (rows < 0).any() or (cols < 0).any():
        raise ValueError("Cells or genes are absent from the filtered checkpoint.")
    for key in ("sample_id", "donor_id", "condition"):
        if key not in adata.obs or key not in filtered.obs:
            raise ValueError(f"Missing sample metadata: {key}")
        if not np.array_equal(adata.obs[key].astype(str), filtered.obs[key].iloc[rows].astype(str)):
            raise ValueError(f"Sample metadata differs from the filtered checkpoint: {key}")
    actual = canonical_counts(adata.layers["counts_raw"])
    expected = canonical_counts(filtered.layers["counts_raw"])[rows][:, cols].tocsr()
    expected.sort_indices()
    if not (np.array_equal(actual.indptr, expected.indptr)
            and np.array_equal(actual.indices, expected.indices)
            and np.array_equal(actual.data, expected.data)):
        raise ValueError("counts_raw differs from the aligned filtered checkpoint.")
    info = adata.uns["cluster_rounds"][round_id]
    labels = adata.obs[info["labels_obs_key"]].astype(str)
    order = list(map(str, info["cluster_order"]))
    if len(order) != len(set(order)) or set(order) != set(labels):
        raise ValueError("Stored cluster order does not cover the state partition exactly.")
    codes = labels.map({label: f"C{i:02d}" for i, label in enumerate(order)})
    if set(mapping) != set(codes):
        raise ValueError("Mapping must cover every state exactly once.")
    partition = pd.DataFrame({"cell": adata.obs_names, "state": codes.to_numpy()}).sort_values("cell")
    fingerprint = hashlib.sha256(partition.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    if frozen_reference is not None:
        if round_id != frozen_reference["parent_round"] or list(adata.shape) != frozen_reference["shape"]:
            raise ValueError("Input does not match the frozen Kang round and shape.")
        if fingerprint != frozen_reference["partition_sha256"]:
            raise ValueError("State membership differs from the frozen Kang mapping. Review markers and supply your own mapping.")
    return {"status": "PASS", "shape": list(adata.shape), "parent_round": round_id,
            "states": len(mapping), "broad_identities": len(set(mapping.values())),
            "counts_total": float(actual.data.sum(dtype=np.float64)),
            "counts_equal_filtered_checkpoint": True, "partition_sha256": fingerprint,
            "mapping_review": "frozen_partition_verified" if frozen_reference is not None else "user_reviewed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frozen-reference", type=Path)
    mode.add_argument("--reviewed-mapping", action="store_true",
                      help="Confirm that this mapping was reviewed for your own state partition.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    mapping = read_mapping(args.mapping)
    reference = None
    if args.frozen_reference:
        reference = json.loads(args.frozen_reference.read_text())
        if sha256(args.mapping) != reference["mapping_sha256"]:
            raise ValueError("Mapping file differs from the frozen reference.")
    from scomnom import load_dataset

    result = validate(load_dataset(args.input), load_dataset(args.filtered),
                      mapping, args.round_id, reference)
    result["input_sha256"] = sha256(args.input)
    result["filtered_sha256"] = sha256(args.filtered)
    result["mapping_sha256"] = sha256(args.mapping)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
