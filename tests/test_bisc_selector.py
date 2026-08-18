import numpy as np
import pytest

import scomnom.clustering_utils as cu


def make_metrics(
    *,
    resolutions=(0.1, 0.2, 0.3, 0.4, 0.5),
    silhouette=None,
    adjacent_ari=0.9,
    cluster_counts=None,
    cluster_sizes=None,
    bio_homogeneity=None,
    bio_fragmentation=None,
    bio_ari=None,
    n_bio_labels=None,
):
    resolutions = [float(r) for r in resolutions]
    if silhouette is None:
        silhouette = {r: 0.5 for r in resolutions}
    if cluster_counts is None:
        cluster_counts = {r: 4 for r in resolutions}
    if cluster_sizes is None:
        cluster_sizes = {
            r: np.full(int(cluster_counts[r]), 100, dtype=int)
            for r in resolutions
        }
    if np.isscalar(adjacent_ari):
        adjacent_ari = [float(adjacent_ari)] * (len(resolutions) - 1)
    ari_adjacent = {
        (left, right): float(value)
        for left, right, value in zip(resolutions[:-1], resolutions[1:], adjacent_ari)
    }
    labels = {r: np.zeros(8, dtype=int) for r in resolutions}
    return cu.ResolutionMetrics(
        resolutions=resolutions,
        silhouette={float(r): float(value) for r, value in silhouette.items()},
        cluster_counts={float(r): int(value) for r, value in cluster_counts.items()},
        cluster_sizes={float(r): np.asarray(value, dtype=int) for r, value in cluster_sizes.items()},
        labels_per_resolution=labels,
        ari_adjacent=ari_adjacent,
        bio_homogeneity=bio_homogeneity,
        bio_fragmentation=bio_fragmentation,
        bio_ari=bio_ari,
        n_bio_labels=n_bio_labels,
    )


def make_config(**overrides):
    values = {
        "stability_threshold": 0.85,
        "min_plateau_len": 2,
        "max_cluster_jump_frac": 0.4,
        "min_cluster_size": 20,
        "tiny_cluster_size": 20,
        "w_stab": 0.0,
        "w_sil": 1.0,
        "w_tiny": 0.0,
        "w_hom": 0.0,
        "w_frag": 0.0,
        "w_bioari": 0.0,
        "use_bio": False,
    }
    values.update(overrides)
    return cu.ResolutionSelectionConfig(**values)


def set_plateaus(monkeypatch, *plateaus):
    monkeypatch.setattr(cu, "_detect_plateaus", lambda metrics, config, stability: list(plateaus))


def test_selects_plateau_with_highest_mean_stability(monkeypatch):
    metrics = make_metrics(resolutions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    set_plateaus(
        monkeypatch,
        cu.Plateau([0.2, 0.3], mean_stability=0.90),
        cu.Plateau([0.4, 0.5], mean_stability=0.95),
    )

    result = cu.select_best_resolution(metrics, make_config())

    assert result.best_resolution == 0.4


def test_plateau_stability_tie_prefers_longer_plateau(monkeypatch):
    metrics = make_metrics(resolutions=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
    set_plateaus(
        monkeypatch,
        cu.Plateau([0.2, 0.3], mean_stability=0.90),
        cu.Plateau([0.4, 0.5, 0.6], mean_stability=0.90),
    )

    result = cu.select_best_resolution(metrics, make_config())

    assert result.best_resolution == 0.4


def test_three_percent_parsimony_selects_lowest_near_optimal_resolution(monkeypatch):
    silhouette = {0.1: 0.0, 0.2: 0.971, 0.3: 0.98, 0.4: 1.0, 0.5: 0.0}
    metrics = make_metrics(silhouette=silhouette)
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3, 0.4], mean_stability=0.90))

    result = cu.select_best_resolution(metrics, make_config())

    assert result.best_resolution == 0.2


def test_score_ties_are_resolved_to_lower_resolution(monkeypatch):
    metrics = make_metrics()
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3, 0.4], mean_stability=0.90))

    result = cu.select_best_resolution(metrics, make_config())

    assert result.best_resolution == 0.2


def test_biological_weights_can_change_selected_resolution(monkeypatch):
    silhouette = {0.1: 0.0, 0.2: 1.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0}
    bio_ari = {0.1: 0.0, 0.2: 0.0, 0.3: 1.0, 0.4: 0.0, 0.5: 0.0}
    metrics = make_metrics(
        silhouette=silhouette,
        bio_homogeneity={r: 0.5 for r in silhouette},
        bio_fragmentation={r: 0.5 for r in silhouette},
        bio_ari=bio_ari,
    )
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3], mean_stability=0.90))

    structural = cu.select_best_resolution(metrics, make_config())
    biological = cu.select_best_resolution(
        metrics,
        make_config(use_bio=True, w_sil=0.1, w_bioari=1.0),
    )

    assert structural.best_resolution == 0.2
    assert biological.best_resolution == 0.3


def test_flat_biological_components_retain_parsimonious_selection(monkeypatch):
    silhouette = {0.1: 0.0, 0.2: 0.971, 0.3: 1.0, 0.4: 0.0, 0.5: 0.0}
    flat_biology = {r: 0.5 for r in silhouette}
    metrics = make_metrics(
        silhouette=silhouette,
        bio_homogeneity=flat_biology,
        bio_fragmentation=flat_biology,
        bio_ari=flat_biology,
    )
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3], mean_stability=0.90))

    result = cu.select_best_resolution(
        metrics,
        make_config(use_bio=True, w_hom=0.15, w_frag=0.10, w_bioari=0.15),
    )

    assert result.best_resolution == 0.2


def test_missing_biological_components_use_structural_scores(monkeypatch):
    silhouette = {0.1: 0.0, 0.2: 1.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0}
    metrics = make_metrics(silhouette=silhouette)
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3], mean_stability=0.90))

    result = cu.select_best_resolution(
        metrics,
        make_config(use_bio=True, w_sil=1.0, w_bioari=1.0),
    )

    assert result.best_resolution == 0.2


def test_structural_selection_ignores_available_biological_metrics(monkeypatch):
    silhouette = {0.1: 0.0, 0.2: 1.0, 0.3: 0.0, 0.4: 0.0, 0.5: 0.0}
    bio_values = {0.1: 0.0, 0.2: 0.0, 0.3: 1.0, 0.4: 0.0, 0.5: 0.0}
    with_bio = make_metrics(
        silhouette=silhouette,
        bio_homogeneity=bio_values,
        bio_fragmentation=bio_values,
        bio_ari=bio_values,
    )
    without_bio = make_metrics(silhouette=silhouette)
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3], mean_stability=0.90))

    result_with = cu.select_best_resolution(with_bio, make_config(use_bio=False))
    result_without = cu.select_best_resolution(without_bio, make_config(use_bio=False))

    assert result_with.best_resolution == result_without.best_resolution == 0.2
    assert result_with.scores == result_without.scores


def test_biological_cluster_count_cap_excludes_overfragmented_candidate(monkeypatch):
    resolutions = (0.1, 0.2, 0.3, 0.4, 0.5)
    counts = {0.1: 4, 0.2: 6, 0.3: 8, 0.4: 12, 0.5: 14}
    bio_ari = {0.1: 0.0, 0.2: 0.2, 0.3: 0.8, 0.4: 1.0, 0.5: 0.0}
    metrics = make_metrics(
        resolutions=resolutions,
        cluster_counts=counts,
        bio_homogeneity={r: 0.5 for r in resolutions},
        bio_fragmentation={r: 0.5 for r in resolutions},
        bio_ari=bio_ari,
        n_bio_labels=4,
    )
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3, 0.4], mean_stability=0.90))

    result = cu.select_best_resolution(
        metrics,
        make_config(use_bio=True, w_sil=0.0, w_bioari=1.0),
    )

    assert result.best_resolution == 0.3
    assert metrics.cluster_counts[result.best_resolution] <= 2.5 * metrics.n_bio_labels


def test_no_plateau_fallback_selects_stability_knee():
    metrics = make_metrics(adjacent_ari=(0.7, 0.95, 0.95, 0.7))

    result = cu.select_best_resolution(
        metrics,
        make_config(stability_threshold=0.99, min_plateau_len=3),
    )

    assert result.plateaus == []
    assert result.best_resolution == 0.3


def test_minimum_feasible_stability_filters_plateau_candidates(monkeypatch):
    metrics = make_metrics(adjacent_ari=(0.5, 0.5, 0.7, 0.7))
    set_plateaus(
        monkeypatch,
        cu.Plateau([0.2], mean_stability=0.99),
        cu.Plateau([0.4], mean_stability=0.90),
    )

    result = cu.select_best_resolution(metrics, make_config())

    assert result.stability[0.2] < cu._BISC_MIN_FEASIBLE_STABILITY
    assert result.stability[0.4] >= cu._BISC_MIN_FEASIBLE_STABILITY
    assert result.best_resolution == 0.4


def test_later_feasible_plateau_is_used_when_best_plateau_becomes_infeasible(monkeypatch):
    resolutions = (0.1, 0.2, 0.3, 0.4, 0.5)
    counts = {0.1: 4, 0.2: 12, 0.3: 6, 0.4: 8, 0.5: 14}
    metrics = make_metrics(
        resolutions=resolutions,
        cluster_counts=counts,
        bio_homogeneity={r: 0.5 for r in resolutions},
        bio_fragmentation={r: 0.5 for r in resolutions},
        bio_ari={r: 0.5 for r in resolutions},
        n_bio_labels=4,
    )
    set_plateaus(
        monkeypatch,
        cu.Plateau([0.2], mean_stability=0.99),
        cu.Plateau([0.3, 0.4], mean_stability=0.90),
    )

    result = cu.select_best_resolution(metrics, make_config(use_bio=True))

    assert result.best_resolution == 0.3


def test_no_feasible_plateau_uses_no_plateau_fallback(monkeypatch):
    resolutions = (0.1, 0.2, 0.3, 0.4, 0.5)
    counts = {0.1: 4, 0.2: 12, 0.3: 6, 0.4: 8, 0.5: 14}
    metrics = make_metrics(
        resolutions=resolutions,
        cluster_counts=counts,
        bio_homogeneity={r: 0.5 for r in resolutions},
        bio_fragmentation={r: 0.5 for r in resolutions},
        bio_ari={r: 0.5 for r in resolutions},
        n_bio_labels=4,
    )
    set_plateaus(monkeypatch, cu.Plateau([0.2], mean_stability=0.99))

    result = cu.select_best_resolution(metrics, make_config(use_bio=True))

    assert result.best_resolution == 0.3


def test_tiny_cluster_penalty_discourages_tiny_cluster_burden(monkeypatch):
    sizes = {
        0.1: np.array([100, 100]),
        0.2: np.array([100, 100]),
        0.3: np.array([1, 199]),
        0.4: np.array([100, 100]),
        0.5: np.array([100, 100]),
    }
    counts = {r: len(values) for r, values in sizes.items()}
    metrics = make_metrics(cluster_counts=counts, cluster_sizes=sizes)
    set_plateaus(monkeypatch, cu.Plateau([0.2, 0.3], mean_stability=0.90))

    result = cu.select_best_resolution(
        metrics,
        make_config(w_sil=0.0, w_tiny=1.0),
    )

    assert result.tiny_cluster_penalty[0.2] == 1.0
    assert result.tiny_cluster_penalty[0.3] < result.tiny_cluster_penalty[0.2]
    assert result.best_resolution == 0.2


def test_absolute_minimum_cluster_size_controls_plateau_membership():
    resolutions = (0.1, 0.2)
    config = make_config(min_plateau_len=2)
    stability = {0.1: 0.9, 0.2: 0.9}

    accepted = make_metrics(
        resolutions=resolutions,
        cluster_counts={0.1: 2, 0.2: 2},
        cluster_sizes={0.1: np.array([5, 100]), 0.2: np.array([5, 100])},
    )
    rejected = make_metrics(
        resolutions=resolutions,
        cluster_counts={0.1: 2, 0.2: 2},
        cluster_sizes={0.1: np.array([5, 100]), 0.2: np.array([4, 100])},
    )

    assert len(cu._detect_plateaus(accepted, config, stability)) == 1
    assert cu._detect_plateaus(rejected, config, stability) == []


def test_fixed_rule_snapshot_records_validated_values():
    assert cu._bisc_fixed_rule_snapshot() == {
        "minimum_feasible_stability": 0.60,
        "parsimony_tolerance": 0.03,
        "max_clusters_per_biological_label": 2.5,
        "absolute_minimum_cluster_size": 5,
    }


def test_resolution_lens_endpoints_are_excluded(monkeypatch):
    silhouette = {0.1: 1.0, 0.2: 0.2, 0.3: 0.8, 0.4: 1.0}
    metrics = make_metrics(resolutions=(0.1, 0.2, 0.3, 0.4), silhouette=silhouette)
    set_plateaus(monkeypatch, cu.Plateau([0.1, 0.2, 0.3, 0.4], mean_stability=0.90))

    result = cu.select_best_resolution(metrics, make_config())

    assert result.best_resolution == 0.3


def test_empty_interior_resolution_lens_raises_value_error(monkeypatch):
    metrics = make_metrics(resolutions=(0.1, 0.2))
    set_plateaus(monkeypatch)

    with pytest.raises(ValueError, match="interior"):
        cu.select_best_resolution(metrics, make_config())
