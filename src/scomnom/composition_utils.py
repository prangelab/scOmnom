from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Sequence

LOGGER = logging.getLogger(__name__)
_MIN_GLM_SAMPLES_PER_LEVEL = 2


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
    cluster_key = rinfo.get("cluster_key", None)
    if not cluster_key or str(cluster_key) not in adata.obs:
        raise RuntimeError(
            f"composition: cluster_key not found in adata.obs for round_id={rid!r}."
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
        effects.index = effects.index.map(lambda x: "|".join(map(str, x)))
    else:
        effects.index = effects.index.astype(str)
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
    if len(levels) <= 2:
        LOGGER.info("composition: GLM skipped for %s (n_levels=%d; use CLR instead)", cond, len(levels))
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

    design = pd.get_dummies(meta[[cond] + covar_cols], drop_first=True)
    if design.empty:
        return pd.DataFrame()
    design = sm.add_constant(design, has_constant="add")
    results = []
    for cl in counts.columns:
        y = counts[cl].astype(float)
        if (totals == 0).all():
            continue
        try:
            model = sm.GLM(
                y,
                design,
                family=sm.families.Binomial(),
                offset=np.log(totals.replace(0, np.nan)),
            )
            fit = model.fit()
        except Exception:
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


def run_graph_da(
    adata: ad.AnnData,
    *,
    cluster_key: str,
    sample_key: str,
    condition_key: str,
    covariates: Sequence[str],
    embedding_key: str | None = "X_integrated",
    n_seeds: int = 2000,
    k_ref: int = 30,
    max_k: int = 200,
    min_size: int = 20,
    random_state: int = 42,
    min_nonzero_samples_per_level: int = 3,
    n_permutations: int | None = None,
    effect_shrink_k: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise RuntimeError(f"composition: failed to import scikit-learn: {e}")

    emb_key = str(embedding_key) if embedding_key else "X_integrated"
    if emb_key not in adata.obsm:
        best = None
        try:
            integ = adata.uns.get("integration", {})
            if isinstance(integ, dict):
                best = integ.get("best_embedding", None)
        except Exception:
            best = None
        if best and best in adata.obsm:
            emb_key = str(best)
        else:
            raise RuntimeError(
                f"composition: embedding_key={emb_key!r} not found in adata.obsm and no fallback found."
            )

    if n_permutations not in (None, 0):
        LOGGER.warning(
            "composition: GraphDA argument n_permutations is deprecated and ignored; "
            "GraphDA now uses NB-GLM with spatial weighted-BH FDR."
        )

    X = np.asarray(adata.obsm[emb_key])
    n_cells = X.shape[0]
    if n_cells == 0:
        return pd.DataFrame(), pd.DataFrame()

    max_k = int(max(max_k, k_ref + 1))
    nn = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    nn.fit(X)
    dists, idxs = nn.kneighbors(X, return_distance=True)

    k_ref_idx = min(int(k_ref), dists.shape[1] - 1)
    ref_dists = dists[:, k_ref_idx]
    radius = float(np.nanmedian(ref_dists))

    rng = np.random.default_rng(int(random_state))
    n_seeds = int(min(max(1, n_seeds), n_cells))

    # Draw seeds stratified by cluster to avoid under-sampling sparse but valid clusters.
    cluster_series = adata.obs[str(cluster_key)].astype(str)
    cluster_to_idx: dict[str, np.ndarray] = {
        str(cl): np.flatnonzero(cluster_series.to_numpy() == str(cl))
        for cl in pd.Index(cluster_series).astype(str).unique().tolist()
    }
    eligible_clusters = [cl for cl, arr in cluster_to_idx.items() if arr.size >= int(min_size)]
    seed_list: list[int] = []
    if eligible_clusters and n_seeds >= len(eligible_clusters):
        for cl in eligible_clusters:
            arr = cluster_to_idx[cl]
            seed_list.append(int(rng.choice(arr, size=1, replace=False)[0]))
        remaining = n_seeds - len(seed_list)
        if remaining > 0:
            seed_pool = np.setdiff1d(np.arange(n_cells, dtype=int), np.array(seed_list, dtype=int), assume_unique=False)
            if seed_pool.size > 0:
                extra = rng.choice(seed_pool, size=min(remaining, seed_pool.size), replace=False)
                seed_list.extend(int(x) for x in np.asarray(extra, dtype=int))
    if not seed_list:
        seed_idx = rng.choice(n_cells, size=n_seeds, replace=False)
    else:
        seed_idx = np.asarray(seed_list, dtype=int)

    cluster_labels = adata.obs[str(cluster_key)].astype(str).to_numpy()
    neighborhoods = []
    k_nh = int(min(max(int(k_ref), int(min_size)), dists.shape[1] - 1))
    for i, seed in enumerate(seed_idx):
        seed_dists = dists[seed]
        seed_neighbors = idxs[seed]
        members = seed_neighbors[: (k_nh + 1)]
        if members.size < int(min_size):
            continue
        member_labels = cluster_labels[members]
        label_counts = pd.Series(member_labels).value_counts()
        dominant_label = label_counts.index[0] if not label_counts.empty else "NA"
        neighborhoods.append((f"nh_{i}", members, int(seed), float(seed_dists[k_nh]), str(dominant_label)))

    if not neighborhoods:
        return pd.DataFrame(), pd.DataFrame()

    sample_ids = adata.obs[str(sample_key)].astype(str).to_numpy()
    samples = pd.Index(pd.unique(sample_ids))
    counts = pd.DataFrame(0, index=samples, columns=[n for n, _, _, _, _ in neighborhoods], dtype=int)

    for name, members, _seed, _max_dist, _label in neighborhoods:
        member_samples = pd.Index(sample_ids[members])
        vc = member_samples.value_counts()
        counts.loc[vc.index, name] = vc.values

    meta_cols = [str(sample_key), str(condition_key), *[str(c) for c in covariates]]
    missing = [c for c in meta_cols if c not in adata.obs]
    if missing:
        raise RuntimeError(f"composition: missing covariate columns in adata.obs: {missing}")
    metadata = (
        adata.obs.loc[:, meta_cols]
        .drop_duplicates(subset=[str(sample_key)])
        .set_index(str(sample_key))
        .loc[counts.index]
    )

    cond_series = metadata[str(condition_key)].astype(str)
    levels = cond_series.dropna().unique().tolist()
    support = pd.DataFrame(index=counts.columns)
    for lvl in levels:
        mask = cond_series == str(lvl)
        support[f"n_nonzero_{lvl}"] = (counts.loc[mask, :] > 0).sum(axis=0).astype(int)
    if len(levels) >= 2:
        support["min_nonzero_per_level"] = support.filter(like="n_nonzero_").min(axis=1).astype(int)
    else:
        support["min_nonzero_per_level"] = 0
    support["n_nonzero_total"] = (counts > 0).sum(axis=0).astype(int)
    tested = support["min_nonzero_per_level"] >= int(min_nonzero_samples_per_level)
    tested_cols = tested[tested].index.tolist()
    if tested_cols:
        counts_test = counts.loc[:, tested_cols].copy()
        support_test = support.loc[tested_cols].copy()
    else:
        counts_test = pd.DataFrame(index=counts.index)
        support_test = pd.DataFrame(index=pd.Index([], dtype=object))

    neighborhoods_df = pd.DataFrame(
        {
            "neighborhood": [n for n, _, _, _, _ in neighborhoods],
            "neighborhood_size": [len(m) for _, m, _, _, _ in neighborhoods],
            "seed_index": [s for _, _, s, _, _ in neighborhoods],
            "max_dist": [float(d) for _, _, _, d, _ in neighborhoods],
            "cluster_label": [lbl for _, _, _, _, lbl in neighborhoods],
        }
    )
    neighborhoods_df["radius"] = float(radius)
    neighborhoods_df["k_ref"] = int(k_ref)
    neighborhoods_df["n_seeds"] = int(n_seeds)
    neighborhoods_df["embedding_key"] = emb_key
    neighborhoods_df = neighborhoods_df.set_index("neighborhood")
    if not support.empty:
        neighborhoods_df = neighborhoods_df.join(support, how="left")
    neighborhoods_df["tested"] = neighborhoods_df.index.isin(tested_cols)
    neighborhoods_df = neighborhoods_df.reset_index()

    if counts_test.shape[1] == 0:
        return pd.DataFrame(), neighborhoods_df

    sample_offsets = adata.obs.groupby(str(sample_key), observed=False).size()
    sample_offsets = sample_offsets.reindex(counts_test.index).fillna(0).astype(float)
    sample_offsets = sample_offsets.clip(lower=1.0)
    LOGGER.info(
        "composition: GraphDA testing with NB-GLM + spatial weighted-BH FDR "
        "(min_nonzero_samples_per_level=%d, effect_shrink_k=%.3g).",
        int(min_nonzero_samples_per_level),
        float(effect_shrink_k),
    )

    results = _run_graph_nb_glm(
        counts_test,
        metadata,
        condition_key=str(condition_key),
        covariates=covariates,
        sample_offsets=sample_offsets,
    )
    if results.empty:
        results = run_glm_composition(
            counts_test,
            metadata,
            condition_key=str(condition_key),
            covariates=covariates,
            reference_level=None,
        )
        if results.empty:
            results = run_clr_mannwhitney(
                counts_test,
                metadata,
                condition_key=str(condition_key),
            )
            if not results.empty:
                results = results.assign(effect=results["log2fc_test_vs_ref"])

    if not results.empty and "cluster" in results.columns:
        support_cols = ["min_nonzero_per_level", "n_nonzero_total"]
        support_join = support_test.loc[:, [c for c in support_cols if c in support_test.columns]].copy()
        support_join.index = support_join.index.astype(str)
        results = results.merge(
            support_join,
            left_on=results["cluster"].astype(str),
            right_index=True,
            how="left",
        ).drop(columns=["key_0"], errors="ignore")

        if "pval" in results.columns:
            dist_map = neighborhoods_df.set_index("neighborhood")["max_dist"]
            w_raw = pd.to_numeric(
                pd.Index(results["cluster"].astype(str)).map(dist_map.to_dict()),
                errors="coerce",
            )
            w = 1.0 / np.clip(w_raw.astype(float), 1e-8, np.inf)
            results["spatial_weight"] = w
            results["fdr_bh"] = np.nan
            results["fdr_spatial"] = np.nan
            if "term" in results.columns:
                g = results.groupby(results["term"].astype(str), sort=False).groups
            else:
                g = {"all": results.index}
            from statsmodels.stats.multitest import multipletests

            for _, idx in g.items():
                p = pd.to_numeric(results.loc[idx, "pval"], errors="coerce")
                valid = p.replace([np.inf, -np.inf], np.nan).dropna()
                if valid.empty:
                    continue
                _, q_bh, _, _ = multipletests(valid.to_numpy(), method="fdr_bh")
                q_bh_s = pd.Series(np.nan, index=results.loc[idx].index, dtype=float)
                q_bh_s.loc[valid.index] = q_bh
                results.loc[idx, "fdr_bh"] = q_bh_s
                q_sp = _weighted_bh(valid, pd.to_numeric(results.loc[valid.index, "spatial_weight"], errors="coerce"))
                results.loc[idx, "fdr_spatial"] = q_sp
            results["fdr"] = pd.to_numeric(results["fdr_spatial"], errors="coerce")

        if "effect" in results.columns:
            eff = pd.to_numeric(results["effect"], errors="coerce")
            support_total = pd.to_numeric(results.get("n_nonzero_total", np.nan), errors="coerce").fillna(0.0)
            shrink = support_total / (support_total + float(effect_shrink_k))
            results["effect_raw"] = eff
            results["effect_shrunk"] = eff * shrink
            results["effect"] = results["effect_shrunk"]

    return results, neighborhoods_df


def _run_graph_nb_glm(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    condition_key: str,
    covariates: Sequence[str],
    sample_offsets: pd.Series,
) -> pd.DataFrame:
    try:
        import statsmodels.api as sm
    except Exception as e:
        raise RuntimeError(f"composition: failed to import statsmodels: {e}")

    cond = str(condition_key)
    covar_cols = [str(c) for c in covariates]
    cols = [cond, *covar_cols]
    meta = metadata.loc[:, cols].dropna()
    if meta.empty:
        return pd.DataFrame()

    levels = sorted(meta[cond].astype(str).dropna().unique().tolist())
    if len(levels) < 2:
        return pd.DataFrame()
    meta[cond] = pd.Categorical(meta[cond].astype(str), categories=levels, ordered=True)

    common = counts.index.intersection(meta.index).intersection(sample_offsets.index.astype(str))
    if common.empty:
        return pd.DataFrame()
    meta = meta.loc[common]
    counts = counts.loc[common]
    offsets = pd.to_numeric(sample_offsets.reindex(common), errors="coerce").fillna(0.0).clip(lower=1.0)

    design = pd.get_dummies(meta[[cond, *covar_cols]], drop_first=True, dtype=float)
    if design.empty:
        return pd.DataFrame()
    design = sm.add_constant(design, has_constant="add")
    cond_prefix = f"{cond}_"
    cond_terms = [t for t in design.columns if t.startswith(cond_prefix)]
    if not cond_terms:
        return pd.DataFrame()

    results: list[dict] = []
    off = np.log(offsets.to_numpy(dtype=float))
    for nh in counts.columns:
        y = pd.to_numeric(counts[nh], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        try:
            fit = sm.GLM(
                y,
                design,
                family=sm.families.NegativeBinomial(alpha=1.0),
                offset=off,
            ).fit()
        except Exception:
            continue
        for term in cond_terms:
            coef = float(fit.params.get(term, np.nan))
            se = float(fit.bse.get(term, np.nan)) if hasattr(fit, "bse") else np.nan
            pval = float(fit.pvalues.get(term, np.nan)) if hasattr(fit, "pvalues") else np.nan
            z = coef / se if np.isfinite(coef) and np.isfinite(se) and se > 0 else np.nan
            ci_low = coef - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = coef + 1.96 * se if np.isfinite(se) else np.nan
            test_level = str(term[len(cond_prefix):]) if term.startswith(cond_prefix) else str(term)
            results.append(
                {
                    "cluster": str(nh),
                    "term": str(term),
                    "level_ref": str(levels[0]),
                    "level_test": test_level,
                    "coef": coef,
                    "ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
                    "ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
                    "z": float(z) if np.isfinite(z) else np.nan,
                    "pval": pval if np.isfinite(pval) else np.nan,
                    "effect": float(coef / np.log(2)) if np.isfinite(coef) else np.nan,
                }
            )

    return pd.DataFrame(results)


def _weighted_bh(pvals: pd.Series, weights: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvals, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    valid = p.notna() & w.notna() & np.isfinite(p) & np.isfinite(w) & (w > 0)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    if not valid.any():
        return out

    pv = p.loc[valid].to_numpy(dtype=float)
    wv = w.loc[valid].to_numpy(dtype=float)
    wv = wv * (len(wv) / float(wv.sum()))
    pw = pv / wv
    order = np.argsort(pw)
    pw_sorted = pw[order]
    m = float(len(pw_sorted))
    ranks = np.arange(1, len(pw_sorted) + 1, dtype=float)
    adj_sorted = np.minimum.accumulate((pw_sorted * m / ranks)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    out.loc[valid] = adj
    return out


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

        for _, row in sub.iterrows():
            cluster = str(row.get("cluster"))
            contrast = None
            if "pair" in sub.columns and row.get("pair", None):
                contrast = str(row.get("pair"))
            elif "term" in sub.columns and row.get("term", None):
                contrast = str(row.get("term"))
            effect = row.get("effect", np.nan)
            sign = 1 if effect > 0 else (-1 if effect < 0 else 0)
            is_sig = False
            if "fdr" in sub.columns and np.isfinite(row.get("fdr", np.nan)):
                is_sig = bool(row.get("fdr") <= alpha)
            elif "pval" in sub.columns and np.isfinite(row.get("pval", np.nan)):
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
    summary = base.groupby(["cluster", "contrast"]).agg(
        n_methods=("method", "nunique"),
        n_sig=("is_sig", "sum"),
        mean_effect=("effect", "mean"),
        sign_consensus=("sign", lambda x: int(np.sign(np.nansum(x)))),
        sign_agree=("sign", lambda x: int(len(set([s for s in x if s != 0])) == 1)),
    ).reset_index()

    # Method-specific agreement fields to support a stricter "high confidence" DA call.
    meth = (
        base.groupby(["cluster", "contrast", "method"], dropna=False)
        .agg(
            method_mean_effect=("effect", "mean"),
            method_sign=("sign", lambda x: int(np.sign(np.nansum(x)))),
            method_sig=("is_sig", "any"),
        )
        .reset_index()
    )
    meth_w = meth.pivot(index=["cluster", "contrast"], columns="method")
    meth_w.columns = [f"{a}_{b}" for a, b in meth_w.columns]
    meth_w = meth_w.reset_index()
    out = summary.merge(meth_w, on=["cluster", "contrast"], how="left")

    for m in ("graph", "clr", "sccoda"):
        sig_col = f"method_sig_{m}"
        sign_col = f"method_sign_{m}"
        if sig_col not in out.columns:
            out[sig_col] = False
        if sign_col not in out.columns:
            out[sign_col] = 0
        # Avoid pandas object-dtype fillna downcasting warnings on newer versions.
        out[sig_col] = pd.array(out[sig_col], dtype="boolean").fillna(False).to_numpy(dtype=bool)
        out[sign_col] = pd.to_numeric(out[sign_col], errors="coerce").fillna(0).astype(int)

    graph_sig = out["method_sig_graph"]
    graph_sign = out["method_sign_graph"]
    clr_agree = (out["method_sign_clr"] != 0) & (out["method_sign_clr"] == graph_sign)
    sccoda_agree = (out["method_sign_sccoda"] != 0) & (out["method_sign_sccoda"] == graph_sign)
    out["high_confidence_da"] = graph_sig & (clr_agree | sccoda_agree)

    return out
