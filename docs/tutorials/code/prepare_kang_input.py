"""Stage checksum-pinned Kang batch-2 processed counts for scOmnom."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

import numpy as np
import pandas as pd
from scipy import io as scipy_io, sparse


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/"
SOURCES = {
    "GSE96583_RAW.tar": (76195840, "e5d41a3248a813f99d68fd5c9eb9773de7f46a83680a67f4a02d683b8955fe80"),
    "GSE96583_batch2.genes.tsv.gz": (277054, "93aa4e9b530ef9d6411ca129b416324c5cc1cc5a01a1fa6ed4f4a845480ed3ca"),
    "GSE96583_batch2.total.tsne.df.tsv.gz": (756342, "1d57e72e92ca8695250e88cc0f1c3fa8c0be1175d974f8b427c58f1274dc6c09"),
}
SAMPLES = {
    "ctrl": ("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz"),
    "stim": ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz"),
}
ARCHIVE_MEMBERS = {
    "GSM2560245_A.mat.gz", "GSM2560245_barcodes.tsv.gz",
    "GSM2560246_B.mat.gz", "GSM2560246_barcodes.tsv.gz",
    "GSM2560247_C.mat.gz", "GSM2560247_barcodes.tsv.gz",
    *(name for pair in SAMPLES.values() for name in pair),
}
DONORS = {"101", "1015", "1016", "1039", "107", "1244", "1256", "1488"}
EXPECTED_UNMATCHED = {"ctrl": (0, 0), "stim": (313, 313)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size != size or sha256(path) != digest:
        raise ValueError(f"Source differs from the validated GEO file: {path.name}")


def acquire(source_dir: Path, download: bool) -> list[dict]:
    source_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (size, digest) in SOURCES.items():
        path = source_dir / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; use --download to fetch the three pinned GEO files.")
            partial = path.with_name(path.name + ".partial")
            created = False
            try:
                print(f"Downloading {name}", flush=True)
                with partial.open("xb") as handle:
                    created = True
                    with urllib.request.urlopen(BASE_URL + name, timeout=120) as response:
                        written = 0
                        while block := response.read(1024 * 1024):
                            written += len(block)
                            if written > size:
                                raise ValueError(f"Unexpected download size: {name}")
                            handle.write(block)
                verify_source(partial, size, digest)
                partial.rename(path)
            except BaseException:
                # Remove only this invocation's partial download.
                if created:
                    partial.unlink(missing_ok=True)
                raise
        verify_source(path, size, digest)
        records.append({"file": name, "url": BASE_URL + name, "bytes": size, "sha256": digest})
    return records


def validate_archive(archive: tarfile.TarFile) -> dict:
    members = {}
    for member in archive.getmembers():
        name = member.name.removeprefix("./")
        if not member.isfile() or name not in ARCHIVE_MEMBERS or name in members:
            raise ValueError(f"Unexpected, duplicate, or unsafe archive member: {member.name}")
        members[name] = member
    if set(members) != ARCHIVE_MEMBERS:
        raise ValueError("GEO archive membership differs from the validated processed-count archive.")
    return members


@contextmanager
def member_stream(archive, member):
    with archive.extractfile(member) as raw, gzip.GzipFile(fileobj=raw) as stream:
        yield stream


def read_matrix(archive, members, matrix_name, barcode_name, n_genes):
    with member_stream(archive, members[barcode_name]) as handle:
        barcodes = pd.Index(pd.read_csv(handle, sep="\t", header=None, dtype=str, keep_default_na=False)[0])
    if not barcodes.is_unique or (barcodes == "").any():
        raise ValueError("Source barcodes must be unique and nonempty within each condition.")
    with member_stream(archive, members[matrix_name]) as handle:
        matrix = sparse.csc_matrix(scipy_io.mmread(handle))
    if matrix.shape == (len(barcodes), n_genes):
        matrix = matrix.T.tocsc()
    if matrix.shape != (n_genes, len(barcodes)):
        raise ValueError(f"Matrix/feature/barcode dimensions disagree: {matrix_name}")
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any() or not np.equal(matrix.data, np.rint(matrix.data)).all():
        raise ValueError(f"Non-count values in {matrix_name}")
    matrix = matrix.astype(np.int64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix, barcodes


def read_metadata(path):
    frame = pd.read_csv(path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    frame.index.name = "barcode"
    frame = frame.reset_index()
    if not {"barcode", "ind", "stim", "multiplets"}.issubset(frame.columns):
        raise ValueError("Missing barcode, donor, condition, or singlet metadata.")
    if frame.duplicated(["stim", "barcode"]).any() or set(frame["stim"]) != set(SAMPLES):
        raise ValueError("Duplicate condition/barcode identities or unexpected conditions.")
    singlets = frame[frame["multiplets"].str.lower().eq("singlet")]
    if set(singlets["ind"]) != DONORS:
        raise ValueError("Singlet donor identities differ from the eight validated donors.")
    return frame


def match_metadata(condition_meta, barcodes, condition):
    missing_counts = condition_meta.index[~condition_meta.index.isin(barcodes)]
    missing_metadata = barcodes[~barcodes.isin(condition_meta.index)]
    if (len(missing_counts), len(missing_metadata)) != EXPECTED_UNMATCHED[condition]:
        raise ValueError(f"Barcode mismatch differs from the frozen GEO join contract: {condition}")
    positions = np.flatnonzero(barcodes.isin(condition_meta.index))
    exclusions = [{"condition": condition, "barcode": barcode, "reason": reason}
                  for values, reason in ((missing_counts, "metadata_without_matrix_barcode"),
                                         (missing_metadata, "matrix_without_metadata_barcode"))
                  for barcode in values]
    return positions, condition_meta.loc[barcodes[positions]].copy(), exclusions


@contextmanager
def gzip_output(path):
    with path.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        yield stream


def write_table(path, frame):
    with gzip_output(path) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            frame.to_csv(text, sep="\t", index=False, header=False, lineterminator="\n")


def prepare(source_dir: Path, output_dir: Path, *, download=False):
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; select a new directory: {output_dir}")
    sources = acquire(source_dir, download)
    features = pd.read_csv(source_dir / "GSE96583_batch2.genes.tsv.gz", sep="\t", header=None, dtype=str, keep_default_na=False)
    if features.shape != (35635, 2) or not features[0].is_unique or (features == "").any().any():
        raise ValueError("Unexpected source feature table.")
    features[2] = "Gene Expression"
    metadata = read_metadata(source_dir / "GSE96583_batch2.total.tsne.df.tsv.gz")
    if len(metadata) != 29065:
        raise ValueError("Unexpected source metadata cell count.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        rows, cell_rows, conditions, join_exclusions = [], [], [], []
        with tarfile.open(source_dir / "GSE96583_RAW.tar", "r:") as archive:
            members = validate_archive(archive)
            for condition, (matrix_name, barcode_name) in SAMPLES.items():
                matrix, barcodes = read_matrix(archive, members, matrix_name, barcode_name, len(features))
                condition_meta = metadata.loc[metadata["stim"].eq(condition)].set_index("barcode")
                matched_positions, matched_meta, excluded = match_metadata(condition_meta, barcodes, condition)
                join_exclusions.extend(excluded)
                mask = matched_meta["multiplets"].str.lower().eq("singlet").to_numpy()
                selected = matrix[:, matched_positions[mask]]
                selected_meta = matched_meta.loc[mask].copy()
                selected_meta["donor_id"] = "ind" + selected_meta["ind"]
                selected_meta["sample_id"] = selected_meta["donor_id"] + "_" + condition
                sample_total = sample_nnz = 0
                for sample_id, indices in selected_meta.groupby("sample_id", sort=True).indices.items():
                    subset = selected[:, indices]
                    submeta = selected_meta.iloc[indices]
                    target = staging / "kang_ifnb_10x" / f"{sample_id}.filtered_feature_bc_matrix"
                    target.mkdir(parents=True)
                    with gzip_output(target / "matrix.mtx.gz") as handle:
                        scipy_io.mmwrite(handle, subset.tocoo(), field="integer")
                    write_table(target / "features.tsv.gz", features)
                    write_table(target / "barcodes.tsv.gz", pd.DataFrame(submeta.index))
                    total = int(subset.sum())
                    sample_total += total
                    sample_nnz += subset.nnz
                    rows.append({"sample_id": sample_id, "donor_id": submeta["donor_id"].iloc[0],
                                 "condition": condition, "source_sample": f"batch2_{condition}",
                                 "dataset": "kang_ifnb", "organism": "human", "tissue": "PBMC",
                                 "n_cells": len(submeta), "total_counts": total, "nnz": subset.nnz})
                    cell_rows.append(pd.DataFrame({"barcode": submeta.index, "sample_id": sample_id,
                                      "donor_id": submeta["donor_id"].to_numpy(), "condition": condition}))
                if sample_total != int(selected.sum()) or sample_nnz != selected.nnz:
                    raise ValueError("Donor split did not preserve selected counts/nonzeros.")
                conditions.append({"condition": condition, "source_barcodes": len(barcodes),
                    "metadata_matched": len(matched_meta), "without_metadata": len(barcodes) - len(matched_meta),
                    "metadata_without_matrix_barcode": len(condition_meta) - len(matched_meta),
                    "excluded_non_singlets": int((~mask).sum()), "retained_singlets": len(selected_meta),
                    "retained_total_counts": sample_total, "retained_nnz": sample_nnz})
                print(f"Staged {condition}: {len(selected_meta)} singlets", flush=True)
        samples = pd.DataFrame(rows).sort_values(["donor_id", "condition"])
        cells = pd.concat(cell_rows, ignore_index=True)
        if len(cells) != 24366 or len(samples) != 16 or cells.duplicated(["sample_id", "barcode"]).any():
            raise ValueError("Output differs from the validated 24,366-singlet/16-sample contract.")
        if samples.groupby("donor_id")["condition"].apply(set).map(lambda values: values != {"ctrl", "stim"}).any():
            raise ValueError("Every donor must have control and stimulated samples.")
        samples.drop(columns=["total_counts", "nnz"]).to_csv(staging / "metadata.tsv", sep="\t", index=False)
        cells.to_csv(staging / "cell_identity.tsv", sep="\t", index=False)
        samples.to_csv(staging / "sample_counts.tsv", sep="\t", index=False)
        pd.DataFrame(join_exclusions, columns=["condition", "barcode", "reason"]).to_csv(
            staging / "barcode_join_exclusions.tsv", sep="\t", index=False)
        files = [{"path": str(path.relative_to(staging)), "bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(staging.rglob("*")) if path.is_file()]
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "GSE96583",
            "input_mode": "filtered", "source_type": "GEO processed UMI matrices; not unfiltered droplets or sequencing reads",
            "sources": sources, "archive_members": sorted(members), "conditions": conditions,
            "n_cells": len(cells), "n_genes": len(features), "n_samples": len(samples), "n_donors": 8,
            "total_counts": int(samples["total_counts"].sum()), "nnz": int(samples["nnz"].sum()),
            "selection": "Batch 2, metadata-matched barcodes marked singlet; no extra QC or normalization",
            "filtered_sample_dir": "kang_ifnb_10x", "metadata_tsv": "metadata.tsv",
            "author_annotations_used": ["donor assignment", "condition", "multiplets == singlet"],
            "author_annotations_not_exported": ["cell", "cluster", "tsne1", "tsne2"],
            "output_files": files}
        (staging / "staging_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError(output_dir)
        staging.rename(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("source_geo"))
    parser.add_argument("--output-dir", type=Path, default=Path("input"))
    parser.add_argument("--download", action="store_true", help="Download missing pinned processed GEO files; otherwise operate offline.")
    args = parser.parse_args()
    manifest = prepare(args.source_dir.resolve(), args.output_dir.resolve(), download=args.download)
    print(json.dumps({key: manifest[key] for key in ("n_cells", "n_genes", "n_samples", "n_donors", "total_counts", "nnz")}, indent=2))


if __name__ == "__main__":
    main()
