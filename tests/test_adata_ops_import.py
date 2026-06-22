from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scomnom.adata_ops import import_external_adata
from scomnom.io_utils import load_dataset


def _make_import_adata() -> ad.AnnData:
    X = np.log1p(
        np.array(
            [
                [10.0, 0.0, 2.0],
                [4.0, 1.0, 0.0],
                [7.0, 3.0, 1.0],
            ],
            dtype=float,
        )
    )
    adata = ad.AnnData(X=X)
    adata.obs_names = ["c1", "c2", "c3"]
    adata.var_names = ["g1", "g2", "g3"]
    adata.layers["raw"] = np.array(
        [
            [10, 0, 2],
            [4, 1, 0],
            [7, 3, 1],
        ],
        dtype=np.int32,
    )
    adata.obs["sample_id"] = ["s1", "s1", "s2"]
    adata.obs["patient_id"] = ["p1", "p1", "p2"]
    adata.obs["full_clustering"] = ["Mono", "Mono", "T cell"]
    adata.obsm["X_pca_harmony"] = np.array(
        [
            [0.1, 0.0],
            [0.2, 0.1],
            [1.0, 0.5],
        ],
        dtype=float,
    )
    return adata


def test_import_external_adata_creates_counts_raw_and_round(tmp_path: Path) -> None:
    adata = _make_import_adata()

    out_map, summary = import_external_adata(
        adata,
        output_root=tmp_path,
        output_name="imported_test",
        output_format="h5ad",
    )

    out_path = out_map["imported"]
    assert out_path == tmp_path / "imported_test.h5ad"
    assert out_path.exists()

    loaded = load_dataset(out_path)
    np.testing.assert_allclose(loaded.X, adata.X)
    np.testing.assert_array_equal(loaded.layers["counts_raw"], adata.layers["raw"])
    assert loaded.uns["batch_key"] == "sample_id"

    round_id = loaded.uns["active_cluster_round"]
    assert isinstance(round_id, str)
    assert round_id.startswith("r0_imported")

    round_info = loaded.uns["cluster_rounds"][round_id]
    pretty_key = round_info["annotation"]["pretty_cluster_key"]
    assert round_info["inputs"]["source_cluster_key"] == "full_clustering"
    assert round_info["annotation"]["source_cluster_key"] == "full_clustering"
    assert pretty_key in loaded.obs
    assert "cluster_label" in loaded.obs
    assert summary.loc[0, "source_count_layer"] == "raw"
    assert summary.loc[0, "stored_count_layer"] == "counts_raw"
    assert summary.loc[0, "cluster_key"] == "full_clustering"
    assert summary.loc[0, "embedding_key"] == "X_pca_harmony"


def test_import_external_adata_can_use_X_as_count_source(tmp_path: Path) -> None:
    adata = ad.AnnData(X=np.array([[1.0, 0.0], [2.0, 3.0]], dtype=float))
    adata.obs_names = ["a", "b"]
    adata.var_names = ["g1", "g2"]

    out_map, summary = import_external_adata(
        adata,
        output_root=tmp_path,
        output_name="imported_from_x",
        output_format="h5ad",
        source_count_layer="X",
    )

    loaded = load_dataset(out_map["imported"])
    np.testing.assert_array_equal(loaded.layers["counts_raw"], np.asarray(adata.X))
    assert loaded.uns["import_provenance"]["source_count_layer"] == "X"
    assert pd.isna(summary.loc[0, "cluster_key"])
