from __future__ import annotations

import anndata as ad
import numpy as np

from scomnom.composition_utils import _resolve_active_cluster_key
from scomnom.markers_and_de import _run_namespace_for_round


def _make_adata_with_round(round_id: str | None) -> ad.AnnData:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    if round_id is not None:
        adata.uns["active_cluster_round"] = round_id
    return adata


def test_run_namespace_for_round_uses_explicit_round_id() -> None:
    adata = _make_adata_with_round("r3_refined_idents")
    got = _run_namespace_for_round(adata, prefix="de", round_id="r5_archetypes")
    assert got == "de_r5_archetypes"


def test_run_namespace_for_round_falls_back_to_active_round() -> None:
    adata = _make_adata_with_round("r4_subset_annotation")
    got = _run_namespace_for_round(adata, prefix="markers", round_id=None)
    assert got == "markers_r4_subset_annotation"


def test_run_namespace_for_round_without_round_uses_prefix_only() -> None:
    adata = _make_adata_with_round(None)
    got = _run_namespace_for_round(adata, prefix="da", round_id=None)
    assert got == "da"


def test_resolve_active_cluster_key_prefers_round_labels_obs_key() -> None:
    adata = ad.AnnData(X=np.zeros((2, 2)))
    adata.obs["leiden"] = ["0", "1"]
    adata.obs["leiden__r5_archetypes"] = ["0", "0"]
    adata.uns["active_cluster_round"] = "r3_refined_idents"
    adata.uns["cluster_rounds"] = {
        "r3_refined_idents": {
            "cluster_key": "leiden",
            "labels_obs_key": "leiden",
        },
        "r5_archetypes": {
            "cluster_key": "leiden",
            "labels_obs_key": "leiden__r5_archetypes",
        },
    }

    got = _resolve_active_cluster_key(adata, round_id="r5_archetypes")
    assert got == "leiden__r5_archetypes"
