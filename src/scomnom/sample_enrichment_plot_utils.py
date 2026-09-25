"""Sample-enrichment figures constructed exclusively from stored result tables."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import PlotArtifact


def _artifact(fig, kind, keys, figdir):
    digest = hashlib.sha256(repr(tuple(keys)).encode()).hexdigest()[:16]
    return PlotArtifact(stem=f"{kind}_{digest}", figdir=Path(figdir), fig=fig,
                        savefig_kwargs={"bbox_inches": "tight"})


def _axes(*, height=4.2, width=6.4):
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    return fig, ax


def _title(ax, title):
    ax.set_title(textwrap.fill(str(title), width=70), fontsize=11, pad=12)


def _samples(scores, row, config, exclusions, title):
    fig, ax = _axes(height=4.8)
    condition = config.get("condition_key")
    levels = [str(row["reference"]), str(row["test"])] if row is not None else (
        sorted(scores[condition].astype(str).unique()) if condition else ["All libraries"]
    )
    shown = scores.copy()
    shown["_level"] = shown[condition].astype(str) if condition else "All libraries"
    shown = shown.loc[shown["_level"].isin(levels)].sort_values("replicate_id")
    included_ids = None
    if row is not None:
        mask = exclusions["population_id"].eq(row["population_id"]) & exclusions["contrast"].eq(row["contrast"])
        included_ids = set(exclusions.loc[mask & exclusions["included"], "pb_id"])
    shown["_included"] = True if included_ids is None else shown["pb_id"].isin(included_ids)
    shown["_x"] = 0.
    for i, level in enumerate(levels):
        indices = shown.index[shown["_level"].eq(level)]
        shown.loc[indices, "_x"] = i + (np.linspace(-.1, .1, len(indices)) if len(indices) > 1 else 0.)
    subject = config.get("subject_key")
    if row is not None and subject:
        for _, pair in shown.loc[shown["_included"]].groupby(subject, observed=True):
            if len(pair) == 2 and set(pair["_level"]) == set(levels):
                pair = pair.assign(_order=pair["_level"].map(dict(zip(levels, range(len(levels)))))).sort_values("_order")
                ax.plot(pair["_x"], pair["score"], color=".75", lw=.8, zorder=1)
    for included, marker, color, label in ((True, "o", "#27647B", "Model included" if row is not None else "Scored library"),
                                            (False, "x", ".5", "Model excluded")):
        subset = shown.loc[shown["_included"].eq(included)]
        if not subset.empty:
            ax.scatter(subset["_x"], subset["score"], marker=marker, color=color, s=30, label=label, zorder=2)
    ax.set_xticks(range(len(levels)), [textwrap.fill(level, 22) for level in levels])
    ax.set_xlim(-.4, len(levels) - .6)
    ax.set_ylabel("Unadjusted activity score")
    if row is not None and row["status"] == "ok":
        ax.set_xlabel(f"Adjusted effect: {row['effect_standardized']:.2f} SD "
                      f"[95% CI {row['ci_low_standardized']:.2f}, {row['ci_high_standardized']:.2f}]\n"
                      f"FDR = {row['fdr']:.3g}; test minus reference", fontsize=9, labelpad=12)
    elif row is not None:
        ax.set_xlabel(f"Adjusted estimate unavailable: {row['status']}", fontsize=9, labelpad=12)
    else:
        ax.set_xlabel("Descriptive scores; no contrast estimate", fontsize=9, labelpad=12)
    _title(ax, title)
    ax.legend(frameon=False, fontsize=8)
    return fig


def sample_enrichment_artifacts(payload, *, figdir, activities=()):
    """Yield figures without saving or modifying the stored inference family.

    Defaults show ten leading forest estimates and three sample activities per
    family, ranked by FDR then absolute effect. Score-only selection is alphabetical.
    """
    tables = payload["tables"]
    scores, contrasts, qc = (tables[key] for key in ("activity_scores", "activity_contrasts", "pseudobulk_qc"))
    config = payload["resolved_config"]
    unknown = set(activities) - set(scores["activity"]) - set(contrasts["activity"])
    if unknown:
        raise ValueError(f"Unknown plot activities: {sorted(unknown)}")
    for population, group in qc.groupby("population_id", sort=True, observed=True):
        fig, ax = _axes()
        for included, marker, color, label in ((True, "o", "#27647B", "QC eligible"), (False, "x", "#A4523B", "QC excluded")):
            subset = group.loc[group["included"].eq(included)]
            if not subset.empty:
                ax.scatter(subset["n_cells"], subset["library_size"], marker=marker, color=color, s=30,
                           label=f"{label} (n={len(subset)})")
        ax.axvline(config["min_cells_per_replicate_group"], color=".6", ls="--", lw=.8, label="Cell threshold")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Cells per library-population pseudobulk")
        ax.set_ylabel("Total counts before gene filtering")
        selected = "selected for scoring" if group["selected_population"].any() else "not selected for scoring"
        _title(ax, f"{group['population_label'].iloc[0]} | {selected}")
        ax.legend(frameon=False, fontsize=8)
        yield _artifact(fig, "qc", [population], figdir)
    for keys, family in contrasts.groupby(["population_id", "resource", "contrast"], sort=True, observed=True):
        title = f"{family['population_label'].iloc[0]} | {keys[1]} | {keys[2]}"
        successful = family.loc[family["status"].eq("ok")].copy()
        ranked = successful.assign(_magnitude=successful["effect_standardized"].abs()).sort_values(
            ["fdr", "_magnitude", "activity"], ascending=[True, False, True])
        if not successful.empty:
            fig, ax = _axes()
            significant = successful["fdr"].le(.05)
            ax.scatter(successful["effect_standardized"], -np.log10(successful["fdr"].clip(lower=np.finfo(float).tiny)),
                       c=np.where(significant, "#27647B", ".65"), s=24)
            ax.axvline(0, color=".7", lw=.8)
            ax.axhline(-np.log10(.05), color=".6", ls="--", lw=.8, label="FDR = 0.05")
            ax.set_xlabel("Adjusted effect (outcome SD; test minus reference)")
            ax.set_ylabel("−log10(FDR)")
            _title(ax, title)
            ax.legend(frameon=False, fontsize=8)
            yield _artifact(fig, "overview", keys, figdir)
            leading = ranked.loc[ranked["activity"].isin(activities)] if activities else ranked.head(10)
            if not leading.empty:
                fig, ax = _axes(height=max(3., .45 * len(leading) + 1.8), width=8.)
                y = np.arange(len(leading))
                ax.hlines(y, leading["ci_low_standardized"], leading["ci_high_standardized"], color="#27647B")
                ax.scatter(leading["effect_standardized"], y, color="#27647B", s=25)
                ax.axvline(0, color=".6", ls="--", lw=.8)
                ax.set_yticks(y, [textwrap.fill(str(a), 34) for a in leading["activity"]])
                for pos, fdr in zip(y, leading["fdr"]):
                    ax.text(1.02, pos, f"FDR {fdr:.3g}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
                ax.invert_yaxis()
                ax.set_xlabel("Adjusted effect and 95% CI (outcome SD)")
                _title(ax, title)
                yield _artifact(fig, "forest", keys, figdir)
        selected = family.loc[family["activity"].isin(activities)] if activities else ranked.head(3)
        for _, row in selected.iterrows():
            subset = scores.loc[scores["population_id"].eq(keys[0]) & scores["resource"].eq(keys[1]) & scores["activity"].eq(row["activity"])]
            if subset.empty:
                continue
            fig = _samples(subset, row, config, tables["model_exclusions"], f"{row['activity']} | {title}")
            yield _artifact(fig, "samples", [*keys, row["activity"]], figdir)
    if contrasts.empty:
        for keys, group in scores.groupby(["population_id", "resource"], sort=True, observed=True):
            selected = sorted(set(group["activity"]) & set(activities)) if activities else sorted(group["activity"].unique())[:3]
            for activity in selected:
                subset = group.loc[group["activity"].eq(activity)]
                title = f"{activity} | {subset['population_label'].iloc[0]} | {keys[1]}"
                fig = _samples(subset, None, config, tables["model_exclusions"], title)
                yield _artifact(fig, "samples", [*keys, activity], figdir)
