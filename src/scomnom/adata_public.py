from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import anndata as ad
import pandas as pd

from .adata_ops import subset_adata_by_cluster_mapping as _subset_adata_by_cluster_mapping
from .annotation_utils import run_decoupler_for_round
from .markers_and_de import _compute_de_enrichment_from_dir, _compute_module_score_on_adata
from .rename_utils import rename_idents


def subset_adata_by_cluster_mapping(
    adata: ad.AnnData,
    subset_mapping: Mapping[str, str] | Mapping[str, list[str]],
    round_id: str | None = None,
) -> tuple[dict[str, ad.AnnData], pd.DataFrame]:
    """
    Public API wrapper for subset creation.

    Accepted mapping forms:
    1) Cnn -> subset (preferred, mirrors subset TSV):
       {"C00": "Hepatocyte", "C01": "Endothelial"}
    2) subset -> [Cnn, ...]:
       {"Hepatocyte": ["C00", "C02"], "Immune": ["C04", "C06"]}
    """
    if not isinstance(subset_mapping, Mapping) or not subset_mapping:
        raise ValueError("subset_mapping must be a non-empty mapping.")

    values = list(subset_mapping.values())
    first = values[0] if values else None

    if isinstance(first, str):
        # Cnn -> subset
        subset_to_clusters: dict[str, list[str]] = {}
        for cnn, subset in subset_mapping.items():
            c = str(cnn).strip()
            s = str(subset).strip()
            if not c or not s:
                raise ValueError("subset_mapping contains empty keys or values.")
            subset_to_clusters.setdefault(s, []).append(c)
    else:
        # assume subset -> [Cnn, ...]
        subset_to_clusters = {}
        for subset, clusters in subset_mapping.items():
            s = str(subset).strip()
            if not s:
                raise ValueError("subset_mapping contains an empty subset name.")
            if not isinstance(clusters, (list, tuple, set)):
                raise ValueError("subset -> clusters mapping values must be list/tuple/set of Cnn strings.")
            subset_to_clusters[s] = [str(c).strip() for c in clusters if str(c).strip()]
            if not subset_to_clusters[s]:
                raise ValueError(f"subset_mapping has no clusters for subset {s!r}.")

    return _subset_adata_by_cluster_mapping(
        adata=adata,
        subset_to_clusters=subset_to_clusters,
        round_id=round_id,
    )


def enrichment_cluster(
    adata: ad.AnnData,
    *,
    round_id: str | None = None,
    condition_key: str | None = None,
    gene_filter: Sequence[str] = (),
    decoupler_pseudobulk_agg: str = "mean",
    decoupler_use_raw: bool = True,
    decoupler_method: str = "consensus",
    decoupler_consensus_methods: Sequence[str] | None = ("ulm", "mlm", "wsum"),
    decoupler_min_n_targets: int = 5,
    msigdb_gene_sets: Sequence[str] | None = ("HALLMARK", "REACTOME"),
    msigdb_method: str = "consensus",
    msigdb_min_n_targets: int = 5,
    run_progeny: bool = True,
    progeny_method: str = "consensus",
    progeny_min_n_targets: int = 5,
    progeny_top_n: int = 100,
    progeny_organism: str = "human",
    run_dorothea: bool = True,
    dorothea_method: str = "consensus",
    dorothea_min_n_targets: int = 5,
    dorothea_confidence: Sequence[str] | None = ("A", "B", "C"),
    dorothea_organism: str = "human",
) -> dict:
    """
    Run round-native enrichment on an in-memory AnnData object.

    The selected round's decoupler payload is updated in place on ``adata`` and
    returned for convenience.
    """
    rid = round_id or str(adata.uns.get("active_cluster_round", "") or "")
    if not rid:
        raise RuntimeError("enrichment_cluster: no round_id provided and no active_cluster_round found.")

    cfg = SimpleNamespace(
        condition_key=condition_key,
        gene_filter=tuple(str(x) for x in gene_filter),
        decoupler_pseudobulk_agg=str(decoupler_pseudobulk_agg),
        decoupler_use_raw=bool(decoupler_use_raw),
        decoupler_method=str(decoupler_method),
        decoupler_consensus_methods=list(decoupler_consensus_methods) if decoupler_consensus_methods is not None else None,
        decoupler_min_n_targets=int(decoupler_min_n_targets),
        msigdb_gene_sets=list(msigdb_gene_sets) if msigdb_gene_sets is not None else [],
        msigdb_method=str(msigdb_method),
        msigdb_min_n_targets=int(msigdb_min_n_targets),
        run_progeny=bool(run_progeny),
        progeny_method=str(progeny_method),
        progeny_min_n_targets=int(progeny_min_n_targets),
        progeny_top_n=int(progeny_top_n),
        progeny_organism=str(progeny_organism),
        run_dorothea=bool(run_dorothea),
        dorothea_method=str(dorothea_method),
        dorothea_min_n_targets=int(dorothea_min_n_targets),
        dorothea_confidence=list(dorothea_confidence) if dorothea_confidence is not None else [],
        dorothea_organism=str(dorothea_organism),
    )
    run_decoupler_for_round(adata, cfg, round_id=rid)
    rounds = adata.uns.get("cluster_rounds", {})
    return dict(rounds.get(rid, {}).get("decoupler", {}))


def enrichment_de_from_tables(
    input_dir: str | Path,
    *,
    gene_filter: Sequence[str] = (),
    de_decoupler_source: str = "auto",
    de_decoupler_stat_col: str = "stat",
    decoupler_method: str = "consensus",
    decoupler_consensus_methods: Sequence[str] | None = ("ulm", "mlm", "wsum"),
    decoupler_min_n_targets: int = 5,
    msigdb_gene_sets: Sequence[str] | None = ("HALLMARK", "REACTOME"),
    msigdb_method: str = "consensus",
    msigdb_min_n_targets: int = 5,
    run_progeny: bool = True,
    progeny_method: str = "consensus",
    progeny_min_n_targets: int = 5,
    progeny_top_n: int = 100,
    progeny_organism: str = "human",
    run_dorothea: bool = True,
    dorothea_method: str = "consensus",
    dorothea_min_n_targets: int = 5,
    dorothea_confidence: Sequence[str] | None = ("A", "B", "C"),
    dorothea_organism: str = "human",
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """
    Run enrichment directly from exported DE result tables on disk.

    Returns a nested payload bundle keyed as
    ``condition_key -> contrast -> source -> payload``.
    """
    cfg = SimpleNamespace(
        gene_filter=tuple(str(x) for x in gene_filter),
        de_decoupler_source=str(de_decoupler_source),
        de_decoupler_stat_col=str(de_decoupler_stat_col),
        decoupler_method=str(decoupler_method),
        decoupler_consensus_methods=list(decoupler_consensus_methods) if decoupler_consensus_methods is not None else None,
        decoupler_min_n_targets=int(decoupler_min_n_targets),
        msigdb_gene_sets=list(msigdb_gene_sets) if msigdb_gene_sets is not None else [],
        msigdb_method=str(msigdb_method),
        msigdb_min_n_targets=int(msigdb_min_n_targets),
        run_progeny=bool(run_progeny),
        progeny_method=str(progeny_method),
        progeny_min_n_targets=int(progeny_min_n_targets),
        progeny_top_n=int(progeny_top_n),
        progeny_organism=str(progeny_organism),
        run_dorothea=bool(run_dorothea),
        dorothea_method=str(dorothea_method),
        dorothea_min_n_targets=int(dorothea_min_n_targets),
        dorothea_confidence=list(dorothea_confidence) if dorothea_confidence is not None else [],
        dorothea_organism=str(dorothea_organism),
    )
    return _compute_de_enrichment_from_dir(Path(input_dir), cfg=cfg)


def module_score(
    adata: ad.AnnData,
    *,
    module_files: Sequence[str | Path],
    module_set_name: str | None = None,
    round_id: str | None = None,
    condition_key: str | None = None,
    module_score_method: str = "scanpy",
    module_score_use_raw: bool = False,
    module_score_layer: str | None = None,
    module_score_ctrl_size: int = 50,
    module_score_n_bins: int = 25,
    module_score_random_state: int = 0,
    module_score_max_umaps: int = 12,
) -> dict:
    """
    Run module scoring on an in-memory AnnData object.

    The selected round's module-score payload is updated in place on ``adata`` and
    returned for convenience.
    """
    rid = round_id or str(adata.uns.get("active_cluster_round", "") or "")
    if not rid:
        raise RuntimeError("module_score: no round_id provided and no active_cluster_round found.")

    cfg = SimpleNamespace(
        round_id=rid,
        condition_key=condition_key,
        module_files=tuple(str(Path(x)) for x in module_files),
        module_set_name=(str(module_set_name).strip() if module_set_name else Path(module_files[0]).stem),
        module_score_method=str(module_score_method),
        module_score_use_raw=bool(module_score_use_raw),
        module_score_layer=(str(module_score_layer) if module_score_layer else None),
        module_score_ctrl_size=int(module_score_ctrl_size),
        module_score_n_bins=int(module_score_n_bins),
        module_score_random_state=int(module_score_random_state),
        module_score_max_umaps=int(module_score_max_umaps),
    )
    payload, _score_keys, _rid = _compute_module_score_on_adata(adata, cfg)
    return dict(payload)

__all__ = [
    "rename_idents",
    "subset_adata_by_cluster_mapping",
    "enrichment_cluster",
    "enrichment_de_from_tables",
    "module_score",
]
