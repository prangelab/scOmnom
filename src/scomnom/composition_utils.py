from __future__ import annotations

import logging
import re
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components
from typing import Optional, Sequence

LOGGER = logging.getLogger(__name__)
_MIN_GLM_SAMPLES_PER_LEVEL = 2
_MIN_GLM_LEVELS = 2


def _resolve_active_cluster_key(adata: ad.AnnData, *, round_id: Optional[str]) -> str:
    rid = round_id
    if rid is None:
        rid0 = adata.uns.get("active_cluster_round", None)
        rid = str(rid0) if rid0 else None
    rounds = adata.uns.get("cluster_rounds", {})
    if not rid or not isinstance(rounds, dict) or rid not in rounds:
        raise RuntimeError(
            "composition: active cluster round not resolved. "
            f"Resolved round_id={rid!r}, active_round={adata.uns.get('active_cluster_round', None)!r}."
        )
    rinfo = rounds[rid]
    labels_obs_key = rinfo.get("labels_obs_key", None)
    if labels_obs_key and str(labels_obs_key) in adata.obs:
        return str(labels_obs_key)

    cluster_key = rinfo.get("cluster_key", None)
    if not cluster_key or str(cluster_key) not in adata.obs:
        raise RuntimeError(
            f"composition: labels_obs_key/cluster_key not found in adata.obs for round_id={rid!r}."
        )
    return str(cluster_key)


def prepare_counts_and_metadata(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    sample_key: str,
    condition_key: str,
    covariates: Sequence[str],
    restrict_mask: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = adata.obs
    if restrict_mask is not None:
        if len(restrict_mask) != adata.n_obs:
            raise RuntimeError("composition: restrict_mask length does not match adata.n_obs")
        obs = obs.loc[np.asarray(restrict_mask)]
    counts = pd.crosstab(
        obs[str(sample_key)],
        obs[str(cluster_key)],
        dropna=False,
    )
    meta_cols = [str(sample_key), str(condition_key), *[str(c) for c in covariates]]
    missing = [c for c in meta_cols if c not in obs]
    if missing:
        raise RuntimeError(f"composition: missing covariate columns in adata.obs: {missing}")
    metadata = (
        obs.loc[:, meta_cols]
        .drop_duplicates(subset=[str(sample_key)])
        .set_index(str(sample_key))
    )
    common = counts.index.intersection(metadata.index)
    counts = counts.loc[common]
    metadata = metadata.loc[common]
    return counts, metadata


def _choose_reference_most_stable(
    counts: pd.DataFrame,
    *,
    min_mean_prop: float,
) -> str:
    totals = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(totals, axis=0)
    mean_prop = props.mean(axis=0)
    keep = mean_prop[mean_prop >= float(min_mean_prop)].index.tolist()
    if not keep:
        keep = mean_prop.sort_values(ascending=False).head(1).index.tolist()
    props = props.loc[:, keep]
    center = props.median(axis=0)
    mad = (props.sub(center, axis=1)).abs().median(axis=0)
    if mad.isna().all():
        ref = mean_prop.sort_values(ascending=False).index[0]
        return str(ref)
    ref = mad.sort_values(ascending=True).index[0]
    return str(ref)


def _validate_min_samples_per_level(
    metadata: pd.DataFrame,
    *,
    condition_key: str,
    min_samples: int,
) -> None:
    vc = metadata[str(condition_key)].value_counts(dropna=False)
    if (vc < int(min_samples)).any():
        bad = vc[vc < int(min_samples)]
        raise RuntimeError(
            "composition: too few samples per condition level. "
            f"Minimum required={int(min_samples)}. "
            f"Levels below minimum: {bad.to_dict()}"
        )


def run_sccoda_model(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    sample_key: str,
    condition_key: str,
    covariates: Sequence[str],
    reference_cell_type: str,
    fdr: float,
    num_samples: int,
    num_warmup: int,
) -> pd.DataFrame:
    try:
        import pertpy as pt
    except Exception as e:
        raise RuntimeError(f"composition: failed to import pertpy: {e}")
    rng_key = 42

    cov_cols = [str(condition_key), *[str(c) for c in covariates]]
    sccoda = pt.tl.Sccoda()
    mdata = sccoda.load(
        adata,
        type="cell_level",
        generate_sample_level=True,
        cell_type_identifier=str(cluster_key),
        sample_identifier=str(sample_key),
        covariate_obs=cov_cols,
    )

    terms = " + ".join([str(condition_key), *[str(c) for c in covariates]])
    formula = str(terms)
    mdata = sccoda.prepare(
        mdata,
        formula=formula,
        reference_cell_type=str(reference_cell_type),
    )
    sccoda.run_nuts(
        mdata,
        modality_key="coda",
        num_samples=int(num_samples),
        num_warmup=int(num_warmup),
        rng_key=rng_key,
    )
    sccoda.set_fdr(mdata, est_fdr=float(fdr))
    effects = sccoda.get_effect_df(mdata, modality_key="coda")
    if isinstance(effects.index, pd.MultiIndex):
        effect_terms = effects.index.map(lambda x: str(x[0]))
        effect_clusters = effects.index.map(lambda x: str(x[-1]))
        effects.index = effects.index.map(lambda x: "|".join(map(str, x)))
    else:
        effects.index = effects.index.astype(str)
        split_index = effects.index.to_series().str.split("|", n=1, expand=True)
        effect_terms = split_index.iloc[:, 0].astype(str).to_numpy()
        effect_clusters = (
            split_index.iloc[:, -1].astype(str).to_numpy()
            if split_index.shape[1] > 1
            else effects.index.to_numpy()
        )
    effects["term"] = effect_terms
    effects["cluster"] = effect_clusters
    return effects


def run_glm_composition(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_key: str,
    covariates: Sequence[str],
    reference_level: Optional[str],
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise RuntimeError(f"composition: failed to import statsmodels: {e}")

    meta = metadata.copy()
    cond = str(condition_key)
    covar_cols = [str(c) for c in covariates]
    meta = meta[[cond] + covar_cols].dropna()
    if meta.empty:
        return pd.DataFrame()
    levels = meta[cond].dropna().unique().tolist()
    if len(levels) < _MIN_GLM_LEVELS:
        LOGGER.info("composition: GLM skipped for %s (n_levels=%d)", cond, len(levels))
        return pd.DataFrame()
    vc = meta[cond].value_counts(dropna=False)
    if (vc < _MIN_GLM_SAMPLES_PER_LEVEL).any():
        LOGGER.warning(
            "composition: GLM skipped for %s (min samples per level=%d; counts=%s)",
            cond,
            _MIN_GLM_SAMPLES_PER_LEVEL,
            vc.to_dict(),
        )
        return pd.DataFrame()
    meta[cond] = meta[cond].astype("category")
    if reference_level is not None and reference_level in meta[cond].cat.categories:
        meta[cond] = meta[cond].cat.reorder_categories(
            [reference_level] + [c for c in meta[cond].cat.categories if c != reference_level],
            ordered=True,
        )

    counts = counts.loc[meta.index]
    totals = counts.sum(axis=1)
    valid = totals > 0
    if not valid.all():
        counts = counts.loc[valid]
        meta = meta.loc[valid]
        totals = totals.loc[valid]
    if counts.empty or meta.empty:
        return pd.DataFrame()

    design = pd.get_dummies(meta[[cond] + covar_cols], drop_first=True, dtype=float)
    if design.empty:
        return pd.DataFrame()
    design = sm.add_constant(design, has_constant="add")
    results = []
    for cl in counts.columns:
        y = counts[cl].astype(float)
        failures = totals.astype(float) - y
        if (failures < 0).any():
            LOGGER.warning("composition: GLM skipped cluster %s because counts exceed sample totals", cl)
            continue
        if (totals == 0).all():
            continue
        try:
            endog = np.column_stack([y.to_numpy(dtype=float), failures.to_numpy(dtype=float)])
            model = sm.GLM(
                endog,
                design,
                family=sm.families.Binomial(),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit()
            fit_warnings = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
        except Exception as e:
            LOGGER.warning("composition: GLM skipped cluster %s because fitting failed: %s", cl, e)
            continue

        for term in design.columns:
            if term == "const":
                continue
            coef = fit.params.get(term, np.nan)
            bse = getattr(fit, "bse", None)
            se = bse.get(term, np.nan) if hasattr(bse, "get") else np.nan
            z = coef / se if se and np.isfinite(se) else np.nan
            ci_low = coef - 1.96 * se if se and np.isfinite(se) else np.nan
            ci_high = coef + 1.96 * se if se and np.isfinite(se) else np.nan
            pvals = getattr(fit, "pvalues", None)
            pval = pvals.get(term, np.nan) if hasattr(pvals, "get") else np.nan
            effect = coef / np.log(2) if np.isfinite(coef) else np.nan
            results.append(
                {
                    "cluster": str(cl),
                    "term": str(term),
                    "coef": float(coef),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "z": float(z) if np.isfinite(z) else np.nan,
                    "pval": float(pval) if np.isfinite(pval) else np.nan,
                    "effect": float(effect) if np.isfinite(effect) else np.nan,
                    "fit_warning": "; ".join(fit_warnings),
                    "n_fit_warnings": int(len(fit_warnings)),
                }
            )

    out = pd.DataFrame(results)
    if out.empty:
        return out
    from statsmodels.stats.multitest import multipletests
    _, fdr, _, _ = multipletests(out["pval"].to_numpy(), method="fdr_bh")
    out["fdr"] = fdr
    return out


def run_clr_mannwhitney(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_key: str,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    from itertools import combinations

    cond = metadata[str(condition_key)].astype(str)
    totals = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(totals, axis=0)
    clr = np.log(props + float(pseudocount))
    clr = clr.sub(clr.mean(axis=1), axis=0)

    levels = sorted(cond.dropna().unique().tolist())
    if len(levels) < 2:
        raise RuntimeError(
            "composition: CLR backend requires at least 2 condition levels. "
            f"Found levels={levels}."
        )

    all_blocks = []
    for ref_level, test_level in combinations(levels, 2):
        ref_mask = cond == ref_level
        test_mask = cond == test_level

        rows = []
        for cl in counts.columns:
            ref_vals = clr.loc[ref_mask, cl]
            test_vals = clr.loc[test_mask, cl]
            if ref_vals.empty or test_vals.empty:
                continue
            try:
                pval = stats.mannwhitneyu(ref_vals, test_vals, alternative="two-sided")[1]
            except Exception:
                pval = np.nan

            ref_prop = props.loc[ref_mask, cl].mean()
            test_prop = props.loc[test_mask, cl].mean()
            log2fc = np.log2((test_prop + float(pseudocount)) / (ref_prop + float(pseudocount)))

            rows.append(
                {
                    "cluster": str(cl),
                    "level_ref": str(ref_level),
                    "level_test": str(test_level),
                    "log2fc_test_vs_ref": float(log2fc),
                    "clr_mean_ref": float(ref_vals.mean()),
                    "clr_mean_test": float(test_vals.mean()),
                    "pval": float(pval) if np.isfinite(pval) else np.nan,
                }
            )

        block = pd.DataFrame(rows)
        if block.empty:
            continue
        _, fdr, _, _ = multipletests(block["pval"].to_numpy(), method="fdr_bh")
        block["fdr"] = fdr
        block["pair"] = f"{ref_level}_vs_{test_level}"
        all_blocks.append(block)

    if not all_blocks:
        return pd.DataFrame()

    out = pd.concat(all_blocks, axis=0, ignore_index=True)
    return out.sort_values(["pair", "pval"])


def _milo_effect_review_flags(
    effects: pd.Series,
    support: pd.Series,
    *,
    extreme_log2fc: float,
    minimum_support: int,
) -> tuple[pd.Series, pd.Series]:
    effect_values = pd.to_numeric(effects, errors="coerce")
    support_values = pd.to_numeric(support, errors="coerce")
    extreme = effect_values.abs() >= float(extreme_log2fc)
    marginal_support = support_values <= int(minimum_support)
    labels = pd.Series("ok", index=effects.index, dtype=object)
    labels.loc[extreme] = "extreme_log2fc"
    labels.loc[marginal_support] = "minimum_sample_support"
    labels.loc[extreme & marginal_support] = "extreme_log2fc;minimum_sample_support"
    return (extreme | marginal_support).astype(bool), labels


def _group_milo_contrast(
    block: pd.DataFrame,
    *,
    membership: csr_matrix,
    neighborhood_names: Sequence[str],
    graph_obs: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    sample_alias: str,
    condition_alias: str,
    condition_labels: dict[str, str],
    alpha: float,
    min_overlap: int,
    max_lfc_delta: float | None,
    broad_coverage_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = block.copy()
    out["region_id"] = pd.NA
    out["region_n_neighborhoods"] = pd.NA
    out["region_n_unique_cells"] = pd.NA
    out["region_cell_fraction"] = np.nan
    out["region_effect_median"] = np.nan
    out["region_effect_q25"] = np.nan
    out["region_effect_q75"] = np.nan
    out["region_min_fdr"] = np.nan
    out["region_cluster_label"] = pd.NA
    out["region_cluster_label_fraction"] = np.nan

    fdr = pd.to_numeric(out["fdr"], errors="coerce")
    effects = pd.to_numeric(out["effect"], errors="coerce")
    significant = fdr <= float(alpha)
    name_to_position = {str(name): idx for idx, name in enumerate(neighborhood_names)}
    sig_rows = out.index[significant & out["cluster"].astype(str).isin(name_to_position)].tolist()
    pair_codes = [out["level_ref_code"].iloc[0], out["level_test_code"].iloc[0]]
    pair_cell_mask = graph_obs[condition_alias].isin(pair_codes).to_numpy(dtype=bool)
    n_pair_cells = int(pair_cell_mask.sum())

    coverage_base = {
        "pair": str(out["pair"].iloc[0]),
        "level_ref": str(out["level_ref"].iloc[0]),
        "level_test": str(out["level_test"].iloc[0]),
        "alpha": float(alpha),
        "n_tested_neighborhoods": int(out.shape[0]),
        "n_significant_neighborhoods": int(len(sig_rows)),
        "n_cells_in_contrast": n_pair_cells,
        "n_regions": 0,
        "n_unique_significant_cells": 0,
        "fraction_unique_significant_cells": 0.0,
        "broad_coverage_fraction": float(broad_coverage_fraction),
        "coverage_requires_review": False,
        "coverage_review_reason": "ok",
    }
    if not sig_rows:
        return out, pd.DataFrame(), pd.DataFrame(), pd.DataFrame([coverage_base])

    sig_names = out.loc[sig_rows, "cluster"].astype(str).tolist()
    sig_positions = [name_to_position[name] for name in sig_names]
    sig_membership = membership[:, sig_positions].tocsr()
    overlap = (sig_membership.T @ sig_membership).tocsr()
    overlap.setdiag(0)
    overlap.eliminate_zeros()

    coo = overlap.tocoo()
    keep_edge = coo.data >= int(min_overlap)
    effect_values = effects.loc[sig_rows].to_numpy(dtype=float)
    keep_edge &= np.sign(effect_values[coo.row]) == np.sign(effect_values[coo.col])
    if max_lfc_delta is not None:
        keep_edge &= np.abs(effect_values[coo.row] - effect_values[coo.col]) <= float(max_lfc_delta)
    adjacency = csr_matrix(
        (
            np.ones(int(np.sum(keep_edge)), dtype=np.int8),
            (coo.row[keep_edge], coo.col[keep_edge]),
        ),
        shape=overlap.shape,
    )
    _, component_labels = connected_components(adjacency, directed=False, return_labels=True)

    component_order = sorted(
        np.unique(component_labels),
        key=lambda component: min(
            sig_names[idx] for idx in np.flatnonzero(component_labels == component)
        ),
    )
    sample_totals = graph_obs.groupby(sample_alias, observed=False).size()
    pair_samples = sample_metadata.index[
        sample_metadata[condition_alias].isin(pair_codes)
    ].astype(str)
    region_rows: list[dict] = []
    sample_rows: list[dict] = []
    union_significant_cells = np.zeros(graph_obs.shape[0], dtype=bool)

    for region_number, component in enumerate(component_order, start=1):
        local_positions = np.flatnonzero(component_labels == component)
        member_rows = [sig_rows[idx] for idx in local_positions]
        member_names = [sig_names[idx] for idx in local_positions]
        member_cell_mask = np.asarray(sig_membership[:, local_positions].sum(axis=1)).ravel() > 0
        member_pair_cell_mask = member_cell_mask & pair_cell_mask
        union_significant_cells |= member_pair_cell_mask
        region_id = f"milo_region_{str(out['pair'].iloc[0])}_{region_number:03d}"
        region_effects = effects.loc[member_rows]
        region_fdr = fdr.loc[member_rows]
        cell_clusters = graph_obs.loc[member_pair_cell_mask, "__milo_cluster"].astype(str)
        cluster_counts = cell_clusters.value_counts()
        dominant_cluster = str(cluster_counts.index[0]) if not cluster_counts.empty else "NA"
        dominant_fraction = (
            float(cluster_counts.iloc[0] / cluster_counts.sum()) if not cluster_counts.empty else np.nan
        )
        n_unique_cells = int(member_pair_cell_mask.sum())
        region_values = {
            "region_id": region_id,
            "pair": str(out["pair"].iloc[0]),
            "level_ref": str(out["level_ref"].iloc[0]),
            "level_test": str(out["level_test"].iloc[0]),
            "region_n_neighborhoods": int(len(member_rows)),
            "region_n_unique_cells": n_unique_cells,
            "region_cell_fraction": float(n_unique_cells / n_pair_cells) if n_pair_cells > 0 else np.nan,
            "region_effect_median": float(region_effects.median()),
            "region_effect_q25": float(region_effects.quantile(0.25)),
            "region_effect_q75": float(region_effects.quantile(0.75)),
            "region_min_fdr": float(region_fdr.min()),
            "region_cluster_label": dominant_cluster,
            "region_cluster_label_fraction": dominant_fraction,
            "n_review_flagged_neighborhoods": int(out.loc[member_rows, "effect_requires_review"].sum()),
            "neighborhoods": ";".join(member_names),
        }
        region_rows.append(region_values)
        for column, value in region_values.items():
            if column in out.columns and column != "neighborhoods":
                out.loc[member_rows, column] = value

        region_sample_counts = (
            graph_obs.loc[member_pair_cell_mask]
            .groupby(sample_alias, observed=False)
            .size()
            .reindex(pair_samples, fill_value=0)
        )
        for sample in pair_samples:
            condition_code = str(sample_metadata.loc[sample, condition_alias])
            n_cells = int(region_sample_counts.loc[sample])
            sample_total = int(sample_totals.get(sample, 0))
            sample_rows.append(
                {
                    "region_id": region_id,
                    "pair": str(out["pair"].iloc[0]),
                    "sample": str(sample),
                    "condition": condition_labels.get(condition_code, condition_code),
                    "n_region_cells": n_cells,
                    "n_sample_cells": sample_total,
                    "region_fraction": float(n_cells / sample_total) if sample_total > 0 else np.nan,
                }
            )

    coverage_base.update(
        {
            "n_regions": int(len(region_rows)),
            "n_unique_significant_cells": int(union_significant_cells.sum()),
            "fraction_unique_significant_cells": (
                float(union_significant_cells.sum() / n_pair_cells) if n_pair_cells > 0 else np.nan
            ),
        }
    )
    coverage_base["coverage_requires_review"] = bool(
        coverage_base["fraction_unique_significant_cells"] >= float(broad_coverage_fraction)
    )
    if coverage_base["coverage_requires_review"]:
        coverage_base["coverage_review_reason"] = "broad_unique_cell_coverage"
    return out, pd.DataFrame(region_rows), pd.DataFrame(sample_rows), pd.DataFrame([coverage_base])


def run_milo_da(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    sample_key: str,
    condition_key: str,
    covariates: Sequence[str],
    embedding_key: str | None = "X_integrated",
    n_seeds: int = 1000,
    k_ref: int = 75,
    max_k: int = 200,
    min_size: int = 50,
    random_state: int = 42,
    min_nonzero_samples_per_level: int = 3,
    alpha: float = 0.05,
    group_regions: bool = True,
    group_min_overlap: int = 1,
    group_max_lfc_delta: float | None = None,
    extreme_log2fc: float = 3.0,
    broad_coverage_fraction: float = 0.5,
    n_permutations: int | None = None,
    effect_shrink_k: float = 10.0,
    solver: str = "pydeseq2",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from importlib.metadata import version
    from itertools import combinations

    try:
        import pertpy as pt
        import scanpy as sc
        from mudata import MuData
    except Exception as e:
        raise RuntimeError(f"composition: failed to import the pertpy Milo stack: {e}")

    solver = str(solver).strip().lower()
    if solver not in {"pydeseq2", "edger"}:
        raise ValueError("composition: Milo solver must be 'pydeseq2' or 'edger'.")
    if int(n_seeds) < 1:
        raise ValueError("composition: milo_n_seeds must be at least 1.")
    if int(k_ref) < 2:
        raise ValueError("composition: milo_k_ref must be at least 2.")
    if int(min_size) < 1:
        raise ValueError("composition: milo_min_size must be at least 1.")
    if int(min_nonzero_samples_per_level) < 1:
        raise ValueError("composition: milo_min_nonzero_samples_per_level must be at least 1.")
    if not 0 < float(alpha) <= 1:
        raise ValueError("composition: alpha must be in (0, 1].")
    if int(group_min_overlap) < 1:
        raise ValueError("composition: milo_group_min_overlap must be at least 1.")
    if group_max_lfc_delta is not None and float(group_max_lfc_delta) < 0:
        raise ValueError("composition: milo_group_max_lfc_delta must be non-negative.")
    if float(extreme_log2fc) <= 0:
        raise ValueError("composition: milo_extreme_log2fc must be positive.")
    if not 0 < float(broad_coverage_fraction) <= 1:
        raise ValueError("composition: milo_broad_coverage_fraction must be in (0, 1].")

    deprecated = []
    if int(max_k) != 200:
        deprecated.append(f"max_k={int(max_k)}")
    if n_permutations not in (None, 0):
        deprecated.append(f"n_permutations={int(n_permutations)}")
    if float(effect_shrink_k) != 10.0:
        deprecated.append(f"effect_shrink_k={float(effect_shrink_k):g}")
    if deprecated:
        LOGGER.warning(
            "composition: deprecated graph compatibility arguments are ignored by Milo: %s",
            ", ".join(deprecated),
        )

    emb_key = str(embedding_key) if embedding_key else "X_integrated"
    if emb_key not in adata.obsm:
        integration = adata.uns.get("integration", {})
        best = integration.get("best_embedding") if isinstance(integration, dict) else None
        if best and str(best) in adata.obsm:
            emb_key = str(best)
        else:
            raise RuntimeError(
                f"composition: embedding_key={emb_key!r} not found in adata.obsm and no fallback found."
            )

    required = [str(cluster_key), str(sample_key), str(condition_key), *[str(c) for c in covariates]]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise RuntimeError(f"composition: missing Milo columns in adata.obs: {missing}")
    if adata.n_obs < 3:
        raise RuntimeError("composition: Milo requires at least 3 cells.")

    source_obs = adata.obs.loc[:, required].copy()
    if source_obs[[str(sample_key), str(condition_key), *[str(c) for c in covariates]]].isna().any().any():
        raise RuntimeError(
            "composition: Milo sample, condition, and covariate columns may not contain missing values."
        )

    sample_meta_cols = [str(condition_key), *[str(c) for c in covariates]]
    ambiguity = source_obs.groupby(str(sample_key), observed=False)[sample_meta_cols].nunique(dropna=False)
    ambiguous = ambiguity.columns[(ambiguity > 1).any(axis=0)].tolist()
    if ambiguous:
        raise RuntimeError(
            "composition: Milo requires one condition/covariate value per sample; "
            f"ambiguous columns={ambiguous}."
        )

    condition_values = source_obs[str(condition_key)]
    if isinstance(condition_values.dtype, pd.CategoricalDtype):
        observed = set(condition_values.astype(str))
        levels = [str(level) for level in condition_values.cat.categories if str(level) in observed]
    else:
        levels = sorted(condition_values.astype(str).unique().tolist())
    if len(levels) < 2:
        raise RuntimeError(
            "composition: Milo requires at least 2 condition levels; "
            f"found levels={levels}."
        )

    sample_alias = "__milo_sample"
    condition_alias = "__milo_condition"
    cluster_alias = "__milo_cluster"
    embedding_alias = "X_milo"
    covariate_aliases = [f"__milo_covariate_{idx}" for idx, _ in enumerate(covariates)]
    level_codes = {level: f"L{idx:03d}" for idx, level in enumerate(levels)}

    graph_obs = pd.DataFrame(index=adata.obs_names.copy())
    graph_obs[sample_alias] = source_obs[str(sample_key)].astype(str).to_numpy()
    coded_condition = source_obs[str(condition_key)].astype(str).map(level_codes)
    graph_obs[condition_alias] = pd.Categorical(
        coded_condition,
        categories=[level_codes[level] for level in levels],
        ordered=True,
    )
    graph_obs[cluster_alias] = pd.Categorical(source_obs[str(cluster_key)].astype(str))
    for source_key, alias in zip(covariates, covariate_aliases):
        graph_obs[alias] = source_obs[str(source_key)].to_numpy()

    graph_adata = ad.AnnData(X=csr_matrix((adata.n_obs, 0), dtype=np.float32), obs=graph_obs)
    graph_adata.obsm[embedding_alias] = np.asarray(adata.obsm[emb_key])
    actual_k = min(int(k_ref), int(adata.n_obs) - 1)
    sc.pp.neighbors(
        graph_adata,
        n_neighbors=actual_k,
        use_rep=embedding_alias,
        key_added="milo",
        random_state=int(random_state),
    )

    target_seeds = min(int(n_seeds), int(adata.n_obs))
    sampling_prop = float(target_seeds / int(adata.n_obs))
    milo = pt.tl.Milo()
    mdata = milo.load(graph_adata)
    milo.make_nhoods(
        mdata["rna"],
        neighbors_key="milo",
        prop=sampling_prop,
        seed=int(random_state),
    )

    membership = mdata["rna"].obsm["nhoods"].tocsr()
    refined_positions = np.flatnonzero(mdata["rna"].obs["nhood_ixs_refined"].to_numpy() == 1)
    if membership.shape[1] != refined_positions.size:
        raise RuntimeError(
            "composition: Milo returned inconsistent neighborhood membership and refined-index counts."
        )
    neighborhood_sizes = np.asarray(membership.sum(axis=0)).ravel().astype(int)
    kth_distances = pd.to_numeric(
        mdata["rna"].obs.iloc[refined_positions]["nhood_kth_distance"], errors="coerce"
    ).to_numpy(dtype=float)
    all_names = [f"nh_{idx:06d}" for idx in range(membership.shape[1])]
    annotation_dummies = pd.get_dummies(graph_obs[cluster_alias])
    annotation_counts = membership.T.dot(csr_matrix(annotation_dummies.to_numpy()))
    annotation_counts = annotation_counts.toarray()
    annotation_totals = annotation_counts.sum(axis=1)
    annotation_fractions = np.divide(
        annotation_counts,
        annotation_totals[:, np.newaxis],
        out=np.zeros_like(annotation_counts, dtype=float),
        where=annotation_totals[:, np.newaxis] > 0,
    )
    dominant_annotation = annotation_dummies.columns.to_numpy()[annotation_fractions.argmax(axis=1)]
    neighborhoods_df = pd.DataFrame(
        {
            "neighborhood": all_names,
            "neighborhood_size": neighborhood_sizes,
            "index_cell": mdata["rna"].obs_names[refined_positions].astype(str),
            "index_cell_position": refined_positions.astype(int),
            "kth_distance": kth_distances,
            "cluster_label": dominant_annotation.astype(str),
            "cluster_label_fraction": annotation_fractions.max(axis=1),
        }
    )
    neighborhoods_df["spatial_weight"] = np.divide(
        1.0,
        kth_distances,
        out=np.zeros_like(kth_distances, dtype=float),
        where=np.isfinite(kth_distances) & (kth_distances > 0),
    )
    neighborhoods_df["passes_min_size"] = neighborhood_sizes >= int(min_size)

    keep = neighborhoods_df["passes_min_size"].to_numpy(dtype=bool)
    kept_positions = refined_positions[keep]
    kept_names = neighborhoods_df.loc[keep, "neighborhood"].tolist()
    mdata["rna"].obsm["nhoods"] = membership[:, keep]
    mdata["rna"].obs["nhood_ixs_refined"] = 0
    mdata["rna"].obs.iloc[
        kept_positions,
        mdata["rna"].obs.columns.get_loc("nhood_ixs_refined"),
    ] = 1

    engine_fields = {
        "engine": "pertpy_milo",
        "solver": solver,
        "pertpy_version": version("pertpy"),
        "embedding_key": emb_key,
        "graph_neighbors_requested": int(k_ref),
        "graph_neighbors_actual": int(actual_k),
        "initial_seed_target": int(target_seeds),
        "initial_seeds_sampled": int(mdata["rna"].obs["nhood_ixs_random"].sum()),
        "sampling_proportion": float(sampling_prop),
        "refined_neighborhoods": int(membership.shape[1]),
        "retained_neighborhoods": int(np.sum(keep)),
        "min_size": int(min_size),
        "min_nonzero_samples_per_level_required": int(min_nonzero_samples_per_level),
        "max_k_deprecated_ignored": int(max_k),
        "n_permutations_deprecated_ignored": int(n_permutations or 0),
        "effect_shrink_k_deprecated_ignored": float(effect_shrink_k),
    }
    for key, value in engine_fields.items():
        neighborhoods_df[key] = value
    neighborhoods_df["tested"] = False
    neighborhoods_df["tested_pair_count"] = 0
    neighborhoods_df["tested_all_pairs"] = False
    neighborhoods_df["group_regions"] = bool(group_regions)
    neighborhoods_df["group_min_overlap"] = int(group_min_overlap)
    neighborhoods_df["group_max_lfc_delta"] = group_max_lfc_delta
    neighborhoods_df["extreme_log2fc_threshold"] = float(extreme_log2fc)
    neighborhoods_df["broad_coverage_fraction"] = float(broad_coverage_fraction)

    if not kept_names:
        return pd.DataFrame(), neighborhoods_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    mdata = milo.count_nhoods(mdata, sample_col=sample_alias)
    mdata["milo"].var_names = kept_names
    milo.annotate_nhoods(mdata, anno_col=cluster_alias)
    sample_adata = mdata["milo"]
    counts_array = sample_adata.X.toarray() if issparse(sample_adata.X) else np.asarray(sample_adata.X)
    counts = pd.DataFrame(counts_array, index=sample_adata.obs_names.astype(str), columns=kept_names)
    retained_membership = membership[:, keep].tocsr()

    sample_metadata = (
        graph_obs[[sample_alias, condition_alias, *covariate_aliases]]
        .drop_duplicates(subset=[sample_alias])
        .set_index(sample_alias)
        .loc[counts.index]
    )
    support = pd.DataFrame(index=counts.columns)
    for level in levels:
        mask = sample_metadata[condition_alias].astype(str) == level_codes[level]
        support[f"n_nonzero_{level}"] = (counts.loc[mask] > 0).sum(axis=0).astype(int)
    support["min_nonzero_per_level"] = support.filter(like="n_nonzero_").min(axis=1).astype(int)
    support["n_nonzero_total"] = (counts > 0).sum(axis=0).astype(int)

    retained_meta = sample_adata.var.copy()
    retained_meta.index = retained_meta.index.astype(str)
    retained_meta = retained_meta.rename(
        columns={
            "nhood_annotation": "cluster_label",
            "nhood_annotation_frac": "cluster_label_fraction",
        }
    )
    neighborhoods_df = neighborhoods_df.set_index("neighborhood")
    update_cols = ["cluster_label", "cluster_label_fraction"]
    neighborhoods_df.loc[kept_names, update_cols] = retained_meta.loc[kept_names, update_cols]
    neighborhoods_df = neighborhoods_df.join(support, how="left")

    design = "~" + " + ".join([*covariate_aliases, condition_alias])
    result_blocks: list[pd.DataFrame] = []
    region_blocks: list[pd.DataFrame] = []
    region_sample_blocks: list[pd.DataFrame] = []
    coverage_blocks: list[pd.DataFrame] = []
    tested_pair_count = pd.Series(0, index=counts.columns, dtype=int)
    pairs = list(combinations(levels, 2))
    for ref_level, test_level in pairs:
        ref_code = level_codes[ref_level]
        test_code = level_codes[test_level]
        ref_mask = sample_metadata[condition_alias].astype(str) == ref_code
        test_mask = sample_metadata[condition_alias].astype(str) == test_code
        pair_support = pd.DataFrame(
            {
                "n_nonzero_ref": (counts.loc[ref_mask] > 0).sum(axis=0).astype(int),
                "n_nonzero_test": (counts.loc[test_mask] > 0).sum(axis=0).astype(int),
            }
        )
        pair_support["min_nonzero_per_level_pair"] = pair_support.min(axis=1).astype(int)
        eligible = pair_support["min_nonzero_per_level_pair"] >= int(min_nonzero_samples_per_level)
        eligible_names = eligible[eligible].index.astype(str).tolist()
        tested_pair_count.loc[eligible_names] += 1
        if not eligible_names:
            LOGGER.info(
                "composition: Milo contrast %s vs %s has no neighborhoods meeting support >= %d.",
                test_level,
                ref_level,
                int(min_nonzero_samples_per_level),
            )
            continue

        pair_mdata = MuData(
            {
                "rna": mdata["rna"],
                "milo": sample_adata[:, eligible_names].copy(),
            }
        )
        contrast = f"{condition_alias}{test_code}-{condition_alias}{ref_code}"
        try:
            milo.da_nhoods(
                pair_mdata,
                design=design,
                model_contrasts=contrast,
                solver=solver,
            )
        except Exception as e:
            raise RuntimeError(
                "composition: Milo inference failed "
                f"(solver={solver}, contrast={test_level} vs {ref_level}): {e}"
            ) from e

        fitted = pair_mdata["milo"].var.copy()
        fitted.index = fitted.index.astype(str)
        fitted = fitted.join(pair_support, how="left")
        log_fc = pd.to_numeric(fitted.get("logFC"), errors="coerce")
        block = pd.DataFrame(
            {
                "cluster": fitted.index,
                "term": f"{condition_key}_{test_level}",
                "pair": f"{ref_level}_vs_{test_level}",
                "level_ref": str(ref_level),
                "level_test": str(test_level),
                "level_ref_code": str(ref_code),
                "level_test_code": str(test_code),
                "coef": log_fc * np.log(2.0),
                "effect_raw": log_fc,
                "effect": log_fc,
                "pval": pd.to_numeric(fitted.get("PValue"), errors="coerce"),
                "fdr_bh": pd.to_numeric(fitted.get("FDR"), errors="coerce"),
                "fdr_spatial": pd.to_numeric(fitted.get("SpatialFDR"), errors="coerce"),
                "n_nonzero_ref": pair_support.loc[fitted.index, "n_nonzero_ref"].to_numpy(),
                "n_nonzero_test": pair_support.loc[fitted.index, "n_nonzero_test"].to_numpy(),
                "min_nonzero_per_level_pair": pair_support.loc[
                    fitted.index, "min_nonzero_per_level_pair"
                ].to_numpy(),
                "n_hypotheses": int(len(eligible_names)),
                "engine": "pertpy_milo",
                "solver": solver,
            }
        )
        block["fdr"] = block["fdr_spatial"]
        block["effect_requires_review"], block["effect_review_reason"] = _milo_effect_review_flags(
            block["effect"],
            block["min_nonzero_per_level_pair"],
            extreme_log2fc=float(extreme_log2fc),
            minimum_support=int(min_nonzero_samples_per_level),
        )
        if group_regions:
            block, regions, region_samples, coverage = _group_milo_contrast(
                block,
                membership=retained_membership,
                neighborhood_names=kept_names,
                graph_obs=graph_obs,
                sample_metadata=sample_metadata,
                sample_alias=sample_alias,
                condition_alias=condition_alias,
                condition_labels={code: level for level, code in level_codes.items()},
                alpha=float(alpha),
                min_overlap=int(group_min_overlap),
                max_lfc_delta=group_max_lfc_delta,
                broad_coverage_fraction=float(broad_coverage_fraction),
            )
            region_blocks.append(regions)
            region_sample_blocks.append(region_samples)
            coverage_blocks.append(coverage)
        else:
            block["region_id"] = pd.NA
            sig_mask = pd.to_numeric(block["fdr"], errors="coerce") <= float(alpha)
            sig_names = block.loc[sig_mask, "cluster"].astype(str)
            name_to_position = {name: idx for idx, name in enumerate(kept_names)}
            sig_positions = [name_to_position[name] for name in sig_names if name in name_to_position]
            if sig_positions:
                unique_cells = np.asarray(retained_membership[:, sig_positions].sum(axis=1)).ravel() > 0
            else:
                unique_cells = np.zeros(graph_obs.shape[0], dtype=bool)
            pair_cell_mask = graph_obs[condition_alias].isin([ref_code, test_code]).to_numpy(dtype=bool)
            unique_pair_cells = unique_cells & pair_cell_mask
            coverage_fraction = (
                float(unique_pair_cells.sum() / pair_cell_mask.sum())
                if pair_cell_mask.any()
                else np.nan
            )
            coverage_requires_review = bool(
                np.isfinite(coverage_fraction)
                and coverage_fraction >= float(broad_coverage_fraction)
            )
            coverage_blocks.append(
                pd.DataFrame(
                    [
                        {
                            "pair": f"{ref_level}_vs_{test_level}",
                            "level_ref": str(ref_level),
                            "level_test": str(test_level),
                            "alpha": float(alpha),
                            "n_tested_neighborhoods": int(block.shape[0]),
                            "n_significant_neighborhoods": int(sig_mask.sum()),
                            "n_cells_in_contrast": int(pair_cell_mask.sum()),
                            "n_regions": np.nan,
                            "n_unique_significant_cells": int(unique_pair_cells.sum()),
                            "fraction_unique_significant_cells": coverage_fraction,
                            "broad_coverage_fraction": float(broad_coverage_fraction),
                            "coverage_requires_review": coverage_requires_review,
                            "coverage_review_reason": (
                                "broad_unique_cell_coverage" if coverage_requires_review else "ok"
                            ),
                        }
                    ]
                )
            )
        result_blocks.append(block)

    neighborhoods_df.loc[kept_names, "tested_pair_count"] = tested_pair_count.astype(int)
    neighborhoods_df.loc[kept_names, "tested"] = tested_pair_count > 0
    neighborhoods_df.loc[kept_names, "tested_all_pairs"] = tested_pair_count == len(pairs)
    neighborhoods_df["n_pairs_total"] = int(len(pairs))
    neighborhoods_df = neighborhoods_df.reset_index()

    if not result_blocks:
        return pd.DataFrame(), neighborhoods_df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    results = pd.concat(result_blocks, axis=0, ignore_index=True)
    regions = pd.concat(region_blocks, axis=0, ignore_index=True) if region_blocks else pd.DataFrame()
    region_samples = (
        pd.concat(region_sample_blocks, axis=0, ignore_index=True) if region_sample_blocks else pd.DataFrame()
    )
    coverage = pd.concat(coverage_blocks, axis=0, ignore_index=True) if coverage_blocks else pd.DataFrame()
    if not regions.empty:
        regions["group_min_overlap"] = int(group_min_overlap)
        regions["group_max_lfc_delta"] = group_max_lfc_delta
        regions["alpha"] = float(alpha)
    return results, neighborhoods_df, regions, region_samples, coverage


def run_graph_da(*args, **kwargs):
    """Deprecated compatibility alias for :func:`run_milo_da`."""
    warnings.warn(
        "run_graph_da() is deprecated; use run_milo_da().",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_milo_da(*args, **kwargs)


def _standardize_composition_results(
    df: pd.DataFrame,
    *,
    backend: str,
    condition_key: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if "cluster" not in out.columns:
        out["cluster"] = out.index.astype(str)

    if "term" not in out.columns:
        out["term"] = str(condition_key)

    if "effect" not in out.columns:
        for cand in ("Final Parameter", "final_parameter", "effect_size", "coef"):
            if cand in out.columns:
                out["effect"] = pd.to_numeric(out[cand], errors="coerce")
                break

    if "pval" not in out.columns:
        for cand in ("pval", "p_value", "pvalue"):
            if cand in out.columns:
                out["pval"] = pd.to_numeric(out[cand], errors="coerce")
                break

    if "fdr" not in out.columns:
        for cand in ("fdr", "FDR", "qval", "q_value"):
            if cand in out.columns:
                out["fdr"] = pd.to_numeric(out[cand], errors="coerce")
                break

    if "ci_low" not in out.columns:
        for cand in ("ci_low", "ci_lower", "lower_ci", "lower"):
            if cand in out.columns:
                out["ci_low"] = pd.to_numeric(out[cand], errors="coerce")
                break

    if "ci_high" not in out.columns:
        for cand in ("ci_high", "ci_upper", "upper_ci", "upper"):
            if cand in out.columns:
                out["ci_high"] = pd.to_numeric(out[cand], errors="coerce")
                break

    if backend == "sccoda":
        if "cluster" in out.columns:
            cl = out["cluster"].astype(str)
            if cl.str.contains(r"\|").any():
                out["cluster"] = cl.str.split("|", n=1, expand=True).iloc[:, -1]
        for cand in ("Inclusion probability", "inclusion_prob", "inclusion_probability"):
            if cand in out.columns:
                out["inclusion_prob"] = pd.to_numeric(out[cand], errors="coerce")
                break
        final_parameter = pd.to_numeric(
            out.get("Final Parameter", pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        out["is_significant"] = final_parameter.notna() & final_parameter.ne(0)

    return out


def _annotate_sccoda_contrasts(
    df: pd.DataFrame,
    *,
    condition_levels: Sequence[str],
) -> pd.DataFrame:
    if df is None or df.empty or not condition_levels:
        return df
    out = df.copy()
    levels = [str(level) for level in condition_levels]
    reference_level = levels[0]

    def _test_level(term: object) -> Optional[str]:
        term_text = str(term)
        term_token = re.sub(r"[^A-Za-z0-9]+", "_", term_text).strip("_").lower()
        candidates = []
        for level in levels[1:]:
            level_token = re.sub(r"[^A-Za-z0-9]+", "_", level).strip("_").lower()
            if (
                term_text.endswith(level)
                or term_text.endswith(f"T.{level}")
                or term_token.endswith(level_token)
            ):
                candidates.append(level)
        return max(candidates, key=len) if candidates else None

    out["level_ref"] = reference_level
    out["level_test"] = out["term"].map(_test_level)
    valid_test = out["level_test"].notna()
    if not valid_test.all():
        LOGGER.warning(
            "composition: could not map %d scCODA coefficient row(s) to a condition contrast.",
            int((~valid_test).sum()),
        )
    out.loc[valid_test, "pair"] = (
        out.loc[valid_test, "level_ref"].astype(str)
        + "_vs_"
        + out.loc[valid_test, "level_test"].astype(str)
    )
    return out


def _build_composition_consensus_summary(
    results_by_method: dict[str, pd.DataFrame],
    *,
    alpha: float,
    condition_key: str,
) -> pd.DataFrame:
    rows = []
    for method, df in results_by_method.items():
        if df is None or df.empty:
            continue
        if "cluster" not in df.columns:
            continue
        sub = df.copy()
        if "term" in sub.columns:
            sub = sub[sub["term"].astype(str).str.startswith(str(condition_key))]
        if "effect" in sub.columns:
            sub["effect"] = pd.to_numeric(sub["effect"], errors="coerce")
        if "pval" in sub.columns:
            sub["pval"] = pd.to_numeric(sub["pval"], errors="coerce")
        if "fdr" in sub.columns:
            sub["fdr"] = pd.to_numeric(sub["fdr"], errors="coerce")

        if str(method) == "milo":
            fdr_values = pd.to_numeric(
                sub.get("fdr", pd.Series(np.nan, index=sub.index)), errors="coerce"
            )
            pval_values = pd.to_numeric(
                sub.get("pval", pd.Series(np.nan, index=sub.index)), errors="coerce"
            )
            sub["_is_sig"] = np.where(
                fdr_values.notna(),
                fdr_values <= float(alpha),
                pval_values <= float(alpha),
            )
            if "region_id" in sub.columns and sub["region_id"].notna().any():
                sub = sub.loc[sub["region_id"].notna()].drop_duplicates(["region_id", "pair"])
                sub["cluster"] = sub["region_cluster_label"].astype(str)
                sub["effect"] = pd.to_numeric(sub["region_effect_median"], errors="coerce")
                sub["fdr"] = pd.to_numeric(sub["region_min_fdr"], errors="coerce")
                sub["_is_sig"] = True
            elif "cluster_label" in sub.columns:
                sub = sub.loc[sub["_is_sig"]].copy()
                sub["cluster"] = sub["cluster_label"].astype(str)

        for _, row in sub.iterrows():
            cluster = str(row.get("cluster"))
            contrast = None
            if "pair" in sub.columns and row.get("pair", None):
                contrast = str(row.get("pair"))
            elif "term" in sub.columns and row.get("term", None):
                contrast = str(row.get("term"))
            effect = row.get("effect", np.nan)
            sign = 1 if effect > 0 else (-1 if effect < 0 else 0)
            is_sig = bool(row.get("_is_sig", False))
            if "_is_sig" not in sub.columns and "is_significant" in sub.columns:
                is_sig = bool(row.get("is_significant", False))
            elif "_is_sig" not in sub.columns and "fdr" in sub.columns and np.isfinite(row.get("fdr", np.nan)):
                is_sig = bool(row.get("fdr") <= alpha)
            elif "_is_sig" not in sub.columns and "pval" in sub.columns and np.isfinite(row.get("pval", np.nan)):
                is_sig = bool(row.get("pval") <= alpha)
            rows.append(
                {
                    "method": str(method),
                    "cluster": cluster,
                    "contrast": contrast if contrast is not None else "NA",
                    "effect": float(effect) if np.isfinite(effect) else np.nan,
                    "sign": int(sign),
                    "is_sig": bool(is_sig),
                }
            )

    if not rows:
        return pd.DataFrame()

    base = pd.DataFrame(rows)
    meth = (
        base.groupby(["cluster", "contrast", "method"], dropna=False)
        .agg(
            method_mean_effect=("effect", "mean"),
            method_sign=("sign", lambda x: int(np.sign(np.nansum(x)))),
            method_sig=("is_sig", "any"),
        )
        .reset_index()
    )
    summary = meth.groupby(["cluster", "contrast"]).agg(
        n_methods=("method", "nunique"),
        n_sig=("method_sig", "sum"),
        mean_effect=("method_mean_effect", "mean"),
        sign_consensus=("method_sign", lambda x: int(np.sign(np.nansum(x)))),
        sign_agree=("method_sign", lambda x: int(len(set([s for s in x if s != 0])) == 1)),
    ).reset_index()

    meth_w = meth.pivot(index=["cluster", "contrast"], columns="method")
    meth_w.columns = [f"{a}_{b}" for a, b in meth_w.columns]
    meth_w = meth_w.reset_index()
    out = summary.merge(meth_w, on=["cluster", "contrast"], how="left")

    for m in ("milo", "clr", "sccoda"):
        sig_col = f"method_sig_{m}"
        sign_col = f"method_sign_{m}"
        if sig_col not in out.columns:
            out[sig_col] = False
        if sign_col not in out.columns:
            out[sign_col] = 0
        # Avoid pandas object-dtype fillna downcasting warnings on newer versions.
        out[sig_col] = pd.array(out[sig_col], dtype="boolean").fillna(False).to_numpy(dtype=bool)
        out[sign_col] = pd.to_numeric(out[sign_col], errors="coerce").fillna(0).astype(int)

    milo_sig = out["method_sig_milo"]
    milo_sign = out["method_sign_milo"]
    clr_agree = (
        out["method_sig_clr"]
        & (out["method_sign_clr"] != 0)
        & (out["method_sign_clr"] == milo_sign)
    )
    sccoda_agree = (
        out["method_sig_sccoda"]
        & (out["method_sign_sccoda"] != 0)
        & (out["method_sign_sccoda"] == milo_sign)
    )
    out["high_confidence_da"] = milo_sig & (clr_agree | sccoda_agree)
    global_sig = out["method_sig_clr"] | out["method_sig_sccoda"]
    global_discordant = global_sig & ~clr_agree & ~sccoda_agree
    out["da_evidence_tier"] = np.select(
        [
            out["high_confidence_da"],
            milo_sig & global_discordant,
            milo_sig,
            global_sig,
        ],
        [
            "cross_scale_supported",
            "cross_scale_discordant",
            "local_milo_only",
            "global_only",
        ],
        default="no_supported_da",
    )

    return out
