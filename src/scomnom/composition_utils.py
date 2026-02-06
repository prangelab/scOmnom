from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Sequence


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        adata.obs[str(sample_key)],
        adata.obs[str(cluster_key)],
        dropna=False,
    )
    meta_cols = [str(sample_key), str(condition_key), *[str(c) for c in covariates]]
    missing = [c for c in meta_cols if c not in adata.obs]
    if missing:
        raise RuntimeError(f"composition: missing covariate columns in adata.obs: {missing}")
    metadata = (
        adata.obs.loc[:, meta_cols]
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
    try:
        from jax import random as jrandom
        rng_key = jrandom.PRNGKey(0)
    except Exception:
        rng_key = 0

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
    meta[cond] = meta[cond].astype("category")
    if reference_level is not None and reference_level in meta[cond].cat.categories:
        meta[cond] = meta[cond].cat.reorder_categories(
            [reference_level] + [c for c in meta[cond].cat.categories if c != reference_level],
            ordered=True,
        )

    design = pd.get_dummies(meta[[cond] + [str(c) for c in covariates]], drop_first=True)
    design = sm.add_constant(design, has_constant="add")

    totals = counts.sum(axis=1)
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
            se = fit.bse.get(term, np.nan)
            z = coef / se if se and np.isfinite(se) else np.nan
            ci_low = coef - 1.96 * se if se and np.isfinite(se) else np.nan
            ci_high = coef + 1.96 * se if se and np.isfinite(se) else np.nan
            pval = fit.pvalues.get(term, np.nan)
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
    n_seeds: int = 1000,
    k_ref: int = 50,
    max_k: int = 200,
    min_size: int = 20,
    random_state: int = 42,
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
    seed_idx = rng.choice(n_cells, size=n_seeds, replace=False)

    cluster_labels = adata.obs[str(cluster_key)].astype(str).to_numpy()
    neighborhoods = []
    for i, seed in enumerate(seed_idx):
        seed_dists = dists[seed]
        seed_neighbors = idxs[seed]
        keep = seed_dists <= radius
        members = seed_neighbors[keep]
        if members.size < int(min_size):
            continue
        member_labels = cluster_labels[members]
        label_counts = pd.Series(member_labels).value_counts()
        dominant_label = label_counts.index[0] if not label_counts.empty else "NA"
        neighborhoods.append((f"nh_{i}", members, int(seed), seed_dists[keep].max(), str(dominant_label)))

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

    results = run_glm_composition(
        counts,
        metadata,
        condition_key=str(condition_key),
        covariates=covariates,
        reference_level=None,
    )
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

    return results, neighborhoods_df


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
    )
    summary = summary.reset_index()
    return summary
