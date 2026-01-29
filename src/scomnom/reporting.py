# src/scomnom/reporting.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import html
import json
import re


# =============================================================================
# Small HTML helpers
# =============================================================================
def _safe(x: Any) -> str:
    return html.escape(str(x))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-") or "section"


def _pretty_relpath(p: Path) -> str:
    # Display-only (keep stable, short)
    s = p.as_posix()
    s = re.sub(r"^png/", "", s)
    return s


def _caption_from_filename(fname: str) -> str:
    name = Path(fname).stem
    name = name.replace("_", " ")
    name = name.replace("postfilter", "post-filter")
    name = name.replace("prefilter", "pre-filter")
    name = name.replace("qc", "QC")
    return name.strip().capitalize()


def _details_block(title: str, inner_html: str, *, open_by_default: bool = False, extra_class: str = "") -> str:
    open_attr = " open" if open_by_default else ""
    cls = f"section {extra_class}".strip()
    title_html = title  # already expected to be safe HTML if it contains spans
    return (
        f'<details class="{cls}"{open_attr}>'
        f"<summary>{title_html}</summary>"
        f'<div class="section-body">{inner_html}</div>'
        f"</details>"
    )


def _grid_block(items: List[str]) -> str:
    return f'<div class="grid">{"".join(items)}</div>'


def _toc(sections: List[Tuple[str, str]]) -> str:
    links = "\n".join([f'<a href="#{_safe(sid)}">{_safe(title)}</a>' for sid, title in sections])
    return (
        '<nav class="toc">'
        "<h2>Contents</h2>"
        f"{links}"
        "</nav>"
    )


# =============================================================================
# Plot classification + descriptions
# =============================================================================
def _classify_plot_type(rel_path: Path) -> str:
    """Heuristic plot-type classifier used for grouping in reports."""
    s = rel_path.as_posix().lower()
    stem = rel_path.stem.lower()

    if "decoupler" in s:
        if ("bar" in stem) or ("barplot" in stem) or ("__top" in stem) or ("top" in stem and "bar" in stem):
            return "decoupler_bar"
        if "dotplot" in stem or "dot" in stem:
            return "decoupler_dotplot"
        if "heatmap" in stem:
            return "decoupler_heatmap"
        return "decoupler_other"

    if "umap" in stem or "/umap" in s:
        return "umap"

    if "scib" in s or "integration_metrics" in stem or "results_table" in stem:
        return "scib"

    if "doublet" in s or "doublet" in stem:
        return "doublet"

    if "cellbender" in s or "cellbender" in stem:
        return "cellbender"

    if "qc_scatter" in s or "scatter" in stem:
        return "qc_scatter"

    if "qc_metrics" in s or "violin" in stem or "hist" in stem:
        return "qc_metrics"

    if "cluster" in stem or "/clustering/" in s or "resolution" in stem or "stability" in stem:
        return "clustering"

    if "compaction" in s or "compaction" in stem or "compacted" in stem:
        return "compaction"

    if "annotation" in s or "celltypist" in s or "pretty_cluster" in stem:
        return "annotation"

    return "other"


def _describe_plot(rel_path: Path) -> str:
    """Short, human description shown under each plot."""
    s = rel_path.as_posix().lower()
    stem = rel_path.stem.lower()

    # QC
    if "qc_plots" in s and ("violin" in stem):
        if "mt" in stem:
            return "Per-sample distribution of mitochondrial fraction."
        if "counts" in stem:
            return "Per-sample distribution of library size (UMIs)."
        if "genes" in stem:
            return "Per-sample distribution of detected genes."
        if "ribo" in stem:
            return "Per-sample distribution of ribosomal fraction."
        if "hb" in stem:
            return "Per-sample distribution of hemoglobin fraction."
        return "Per-sample QC distribution."

    if "qc_plots" in s and ("hist" in stem):
        if "pct_mt" in stem or "mt" in stem:
            return "Overall distribution of mitochondrial fraction."
        if "total_counts" in stem:
            return "Overall distribution of library size (UMIs)."
        if "n_genes" in stem:
            return "Overall distribution of detected genes."
        return "Overall QC distribution."

    if "qc_plots" in s and ("complexity" in stem):
        return "Complexity scatter: library size vs genes detected (colored by mt%)."
    if "qc_plots" in s and ("scatter" in stem):
        return "QC scatter plot for outlier detection."

    # Doublets
    if "doublet" in stem:
        if "hist" in stem:
            return "Distribution of doublet scores; red lines indicate per-sample inferred thresholds."
        if "threshold" in stem:
            return "Inferred doublet-score threshold per sample (rate-based calling)."
        if "fraction" in stem:
            return "Observed doublet fraction per sample."
        if "vs_total_counts" in stem:
            return "Doublet score vs library size."
        if "violin" in stem:
            return "Doublet-score distribution per sample; red line indicates inferred threshold."
        if "ecdf" in stem:
            return "ECDF of doublet scores; red lines show inferred thresholds."
        return "Doublet diagnostics."

    # CellBender
    if "cellbender" in stem or "cellbender" in s:
        if "removed_fraction" in stem:
            return "Fraction of counts removed by CellBender (background removal)."
        if "raw_vs_cb" in stem:
            return "Gene-level comparison of raw vs CellBender counts."
        return "CellBender diagnostic comparison."

    # Integration
    if "scib" in s or "scib" in stem or "results_table" in stem:
        return "Integration benchmarking summary (scIB metrics)."
    if "umap" in stem and "vs_unintegrated" in stem:
        return "Side-by-side UMAP comparison against the unintegrated baseline."
    if "umap" in stem:
        return "UMAP visualization of an embedding / integration method."

    # Cluster + annotate
    if "cluster_tree" in stem:
        return "How clusters split/merge across the resolution sweep."
    if "resolution_sweep" in stem:
        return "Resolution sweep diagnostics: silhouette, number of clusters, and penalized score."
    if "stability" in stem and "ari" in stem:
        return "Subsampling stability across repeats (ARI vs full-data clustering)."
    if "cluster_selection" in stem or "stability_curves" in stem:
        return "Metrics used for resolution selection (structural and penalty terms)."
    if "biological_metrics" in stem:
        return "Bio-guided clustering metrics across the resolution sweep."
    if "cluster_sizes" in stem:
        return "Cluster sizes (cells per cluster)."
    if "cluster_qc_summary" in stem:
        return "Mean QC metrics per cluster (diagnostic)."
    if "cluster_batch_composition" in stem:
        return "Batch composition within each cluster (stacked fractions)."
    if "silhouette_by_cluster" in stem:
        return "Silhouette distributions per cluster (quality diagnostic)."
    if "compaction_flow" in stem:
        return "Flow from parent clusters to compacted clusters."
    if "decoupler" in s:
        if "heatmap" in stem:
            return "Decoupler activity heatmap (top pathways/features)."
        if "dotplot" in stem:
            return "Decoupler dotplot summary (activity + magnitude)."
        if "bar" in stem:
            return "Per-cluster top-N decoupler features (barplot)."
        return "Decoupler diagnostic plot."

    return "Diagnostic plot."


def _render_image_card(rel_path_from_fig_root: Path) -> str:
    caption = _caption_from_filename(rel_path_from_fig_root.name)
    desc = _describe_plot(rel_path_from_fig_root)
    meta = _pretty_relpath(rel_path_from_fig_root)
    src = rel_path_from_fig_root.as_posix()
    return (
        '<figure class="card">'
        '<div class="card-img">'
        f'<img loading="lazy" src="{_safe(src)}">'
        "</div>"
        '<figcaption class="card-cap">'
        f'<div class="cap-title">{_safe(caption)}</div>'
        f'<div class="cap-desc">{_safe(desc)}</div>'
        f'<div class="cap-meta">{_safe(meta)}</div>'
        "</figcaption>"
        "</figure>"
    )


# =============================================================================
# CSS
# =============================================================================
def _css() -> str:
    # One CSS for all reports (tight, readable)
    return """
:root {
  --bg: #ffffff;
  --muted: #6b7280;
  --border: #e5e7eb;
  --card: #ffffff;
  --shadow: 0 1px 2px rgba(0,0,0,0.06);
  --shadow2: 0 6px 20px rgba(0,0,0,0.08);
  --radius: 12px;
}

body {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: #111827;
  margin: 0;
  padding: 0;
}

.wrap {
  max-width: 1440px;
  margin: 0 auto;
  padding: 1.25rem 1.25rem 3rem;
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 1.25rem;
}

@media (max-width: 1100px) {
  .wrap { grid-template-columns: 1fr; }
  .toc { position: static !important; }
}

h1 {
  font-size: 1.6rem;
  margin: 0.2rem 0 0.8rem;
  letter-spacing: -0.02em;
}

.meta {
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.9rem 1rem;
  box-shadow: var(--shadow);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  white-space: pre-wrap;
  font-size: 12px;
  line-height: 1.35;
  color: #111827;
}

.toc {
  position: sticky;
  top: 1rem;
  align-self: start;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.8rem;
  background: #fff;
  box-shadow: var(--shadow);
}

.toc h2 {
  font-size: 0.95rem;
  margin: 0 0 0.5rem;
}

.toc a {
  display: block;
  text-decoration: none;
  color: #111827;
  padding: 0.28rem 0.35rem;
  border-radius: 8px;
  font-size: 0.9rem;
}

.toc a:hover { background: #f3f4f6; }

.content { min-width: 0; }

details.section {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: #fff;
  box-shadow: var(--shadow);
  margin: 0 0 0.9rem;
  overflow: hidden;
}

details.section > summary {
  cursor: pointer;
  padding: 0.75rem 0.9rem;
  font-weight: 650;
  list-style: none;
  user-select: none;
}

details.section > summary::-webkit-details-marker { display: none; }

details.section[open] > summary {
  border-bottom: 1px solid var(--border);
  background: #fafafa;
}

.section-body { padding: 0.9rem; }

.summary-table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.25rem 0 0.5rem;
  font-size: 0.95rem;
}

.summary-table th, .summary-table td {
  border: 1px solid var(--border);
  padding: 0.45rem 0.6rem;
  text-align: left;
  vertical-align: top;
}

.summary-table th {
  background: #f3f4f6;
  font-weight: 650;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
  gap: 0.85rem;
  align-items: start;
}

@media (max-width: 520px) {
  .grid { grid-template-columns: 1fr; }
}

.grid.grid2 {
  grid-template-columns: repeat(2, minmax(360px, 1fr));
}

@media (max-width: 900px) {
  .grid.grid2 { grid-template-columns: 1fr; }
}

figure.card {
  margin: 0;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  transition: box-shadow 120ms ease;
}

figure.card:hover { box-shadow: var(--shadow2); }

.card-img {
  padding: 0.6rem;
  background: #fff;
}

figure.card img {
  width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: #fff;
}

.card-cap {
  padding: 0.7rem 0.8rem 0.75rem;
  border-top: 1px solid var(--border);
}

.cap-title {
  font-size: 0.95rem;
  font-weight: 650;
  margin-bottom: 0.25rem;
}

.cap-desc {
  font-size: 0.88rem;
  color: #374151;
  margin-bottom: 0.35rem;
}

.cap-meta {
  font-size: 0.78rem;
  color: var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  overflow-wrap: anywhere;
}

.pill {
  display: inline-block;
  font-size: 0.78rem;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: #fafafa;
  color: #111827;
  margin-left: 0.35rem;
}

.note {
  font-size: 0.92rem;
  color: #374151;
  margin: 0.15rem 0 0.7rem;
}

/* tighter nested subsections */
details.section.subsection > summary {
  padding: 0.60rem 0.80rem;
  font-weight: 620;
}
    """


# =============================================================================
# File collection
# =============================================================================
def _collect_pngs(fig_root: Path) -> List[Path]:
    """
    Returns paths *relative to fig_root* suitable for <img src="...">,
    e.g. Path("png/integration_round1/...png")
    """
    fig_root = Path(fig_root)
    png_root = fig_root / "png"
    if not png_root.exists():
        return []
    imgs_abs = sorted(png_root.rglob("*.png"))
    return [p.relative_to(fig_root) for p in imgs_abs]

# --- ADD near your other helpers in reporting.py (top-level) ---

def _grid2_block(items: List[str]) -> str:
    # fixed 2-col layout for side-by-side comparisons
    return f'<div class="grid grid2">{"".join(items)}</div>'


def _infer_is_annotated_integration_report(rel_imgs: List[Path], adata) -> bool:
    # Primary signal: you stashed the old UMAP
    try:
        if hasattr(adata, "obsm") and "X_umap_pre_annotated" in adata.obsm:
            return True
    except Exception:
        pass

    # Secondary signal: integration provenance
    try:
        runs = getattr(adata, "uns", {}).get("integration", {}).get("runs", [])
        if isinstance(runs, list) and runs:
            last = runs[-1]
            if isinstance(last, dict) and str(last.get("mode", "")).lower() == "annotated_secondary":
                return True
    except Exception:
        pass

    # Tertiary signal: filenames contain "annotated"
    for p in rel_imgs:
        s = p.as_posix().lower()
        if "umap" in s and "annotated" in s:
            return True

    return False


def _pick_pre_post_umap_pngs(rel_imgs: List[Path]) -> tuple[Path | None, Path | None]:
    """
    Best-effort pick of pre/post UMAP figures from saved PNGs.

    Matching rules (case-insensitive, in path string):
      - pre: contains 'pre', 'before', or 'old' AND contains 'umap'
      - post: contains 'annotated' or 'post' AND contains 'umap'

    If that fails:
      - pick first UMAP as pre and second UMAP as post (if available)
    """
    umaps = [p for p in rel_imgs if _classify_plot_type(p) == "umap"]
    if not umaps:
        return None, None

    def _is_pre(p: Path) -> bool:
        s = p.as_posix().lower()
        return ("umap" in s) and any(tok in s for tok in ("pre", "before", "old"))

    def _is_post(p: Path) -> bool:
        s = p.as_posix().lower()
        return ("umap" in s) and any(tok in s for tok in ("annotated", "post"))

    pre = next((p for p in umaps if _is_pre(p)), None)
    post = next((p for p in umaps if _is_post(p)), None)

    if pre is None and len(umaps) >= 1:
        pre = umaps[0]
    if post is None and len(umaps) >= 2:
        # prefer a different one than pre
        post = umaps[1] if umaps[0] == pre and len(umaps) > 1 else umaps[-1]

    if pre == post:
        # avoid duplicating the same image
        post = None

    return pre, post

# =============================================================================
# QC report
# =============================================================================
def _collect_qc_summary(adata) -> Dict[str, Any]:
    obs = adata.obs
    summary: Dict[str, Any] = {
        "n_cells": int(getattr(adata, "n_obs", 0)),
        "n_genes": int(getattr(adata, "n_vars", 0)),
    }

    info = getattr(adata, "uns", {}).get("doublet_calling", None)
    if isinstance(info, dict):
        summary["doublet_observed_rate"] = info.get("observed_global_rate")

    for col in ["total_counts", "n_genes_by_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"]:
        if hasattr(obs, "columns") and col in obs.columns:
            summary[f"median_{col}"] = float(obs[col].median())
            summary[f"mean_{col}"] = float(obs[col].mean())

    return summary


def generate_qc_report(*, fig_root: Path, cfg, version: str, adata) -> None:
    """
    Write: <fig_root>/qc_report.html
    Expects figures under: <fig_root>/png/...
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "qc_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_imgs = _collect_pngs(fig_root)

    sections: Dict[str, List[Path]] = {
        "Overview": [],
        "QC metrics": [],
        "QC scatter": [],
        "Doublets": [],
        "CellBender": [],
        "Other": [],
    }

    for p in rel_imgs:
        t = _classify_plot_type(p)
        if t == "doublet":
            sections["Doublets"].append(p)
        elif t == "cellbender":
            sections["CellBender"].append(p)
        elif t == "qc_metrics":
            sections["QC metrics"].append(p)
        elif t == "qc_scatter":
            sections["QC scatter"].append(p)
        elif "overview" in p.as_posix().lower():
            sections["Overview"].append(p)
        else:
            sections["Other"].append(p)

    cfg_dict = getattr(cfg, "__dict__", {})
    cfg_json = _safe(json.dumps(cfg_dict, indent=2, default=str))

    qc_summary = _collect_qc_summary(adata)
    rows: List[str] = []
    for k, v in qc_summary.items():
        if isinstance(v, float):
            rows.append(f"<tr><td>{_safe(k)}</td><td>{v:.4g}</td></tr>")
        else:
            rows.append(f"<tr><td>{_safe(k)}</td><td>{_safe(v)}</td></tr>")

    summary_table = (
        '<table class="summary-table">'
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    header = (
        '<h1 id="top">scOmnom QC report</h1>'
        '<div class="meta">'
        f"Version:   {_safe(version)}\n"
        f"Timestamp: {_safe(timestamp)}\n\n"
        "Parameters:\n"
        f"{cfg_json}"
        "</div>"
    )

    toc_sections: List[Tuple[str, str]] = [("summary", "Summary statistics")]
    for title in sections.keys():
        if sections[title]:
            toc_sections.append((_slug(title), title))
    toc_html = _toc(toc_sections)

    blocks: List[str] = [header]
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary statistics", summary_table, open_by_default=True))

    notes_by_title: Dict[str, str] = {
        "QC metrics": "Distributions and violins for key QC metrics (pre/post filter where available).",
        "QC scatter": "Scatter diagnostics for outliers and complexity.",
        "Doublets": "SOLO doublet diagnostics and inferred thresholds.",
        "CellBender": "Raw vs CellBender comparisons (when CellBender input was used).",
        "Overview": "High-level outputs (e.g., HVGs/PCA, filter summaries).",
        "Other": "Additional diagnostic figures.",
    }

    for title, imgs in sections.items():
        if not imgs:
            continue
        sid = _slug(title)
        note = notes_by_title.get(title, "")
        note_html = f"<p class='note'>{_safe(note)}</p>" if note else ""
        grid = _grid_block([_render_image_card(p) for p in imgs])
        inner = note_html + grid
        title_html = f"{_safe(title)} <span class='pill'>{len(imgs)} plots</span>"
        blocks.append(f'<div id="{sid}"></div>')
        blocks.append(_details_block(title_html, inner, open_by_default=False))

    html_doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        "<title>scOmnom QC report</title>"
        f"<style>{_css()}</style>"
        "</head><body>"
        f'<div class="wrap">{toc_html}<main class="content">{"".join(blocks)}</main></div>'
        "</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")


# =============================================================================
# Integration report
# =============================================================================
def generate_integration_report(
    *,
    fig_root: Path,
    version: str,
    adata,
    batch_key: str,
    label_key: str,
    methods: List[str],
    selected_embedding: str,
    benchmark_n_jobs: int,
) -> None:
    """
    Write: <fig_root>/integration_report.html
    Expects figures under: <fig_root>/png/...

    Standard integration report only:
      - Integration benchmarking (scIB)
      - UMAPs
      - Other
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "integration_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_imgs = _collect_pngs(fig_root)

    # -----------------------------
    # Collect figures (standard)
    # -----------------------------
    sections: Dict[str, List[Path]] = {
        "Integration benchmarking": [],
        "UMAPs": [],
        "Other": [],
    }

    for p in rel_imgs:
        t = _classify_plot_type(p)
        if t == "scib":
            sections["Integration benchmarking"].append(p)
        elif t == "umap":
            sections["UMAPs"].append(p)
        else:
            sections["Other"].append(p)

    # -----------------------------
    # Summary payload
    # -----------------------------
    summary: Dict[str, Any] = {
        "version": version,
        "timestamp": timestamp,
        "batch_key": batch_key,
        "label_key": label_key,
        "methods_requested": methods,
        "benchmark_n_jobs": benchmark_n_jobs,
        "selected_embedding": selected_embedding,
    }

    rows = [f"<tr><td>{_safe(k)}</td><td>{_safe(v)}</td></tr>" for k, v in summary.items()]
    summary_table = (
        '<table class="summary-table">'
        "<thead><tr><th>Field</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    summary_json = _safe(json.dumps(summary, indent=2, default=str))

    header = (
        '<h1 id="top">scOmnom Integration report</h1>'
        '<div class="meta">'
        f"Version:   {_safe(version)}\n"
        f"Timestamp: {_safe(timestamp)}\n\n"
        f"Selected embedding:\n{_safe(selected_embedding)}\n\n"
        "Summary:\n"
        f"{summary_json}"
        "</div>"
    )

    # -----------------------------
    # TOC (standard)
    # -----------------------------
    toc_sections: List[Tuple[str, str]] = [
        ("summary", "Summary"),
        ("integration-benchmarking", "Integration benchmarking"),
        ("umaps", "UMAPs"),
    ]
    if sections["Other"]:
        toc_sections.append(("other", "Other"))
    toc_html = _toc(toc_sections)

    # -----------------------------
    # Render body (standard)
    # -----------------------------
    blocks: List[str] = [header]
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", summary_table, open_by_default=True))

    if sections["Integration benchmarking"]:
        note = (
            "Benchmarking outputs (e.g., scIB tables/plots). "
            "Use this to understand tradeoffs between batch mixing and biological conservation."
        )
        inner = f"<p class='note'>{_safe(note)}</p>" + _grid_block(
            [_render_image_card(p) for p in sections["Integration benchmarking"]]
        )
        blocks.append('<div id="integration-benchmarking"></div>')
        title_html = (
            f"Integration benchmarking <span class='pill'>{len(sections['Integration benchmarking'])} plots</span>"
        )
        blocks.append(_details_block(title_html, inner, open_by_default=False))

    if sections["UMAPs"]:
        note = "UMAPs for each embedding / method. Expand only when needed."
        inner = f"<p class='note'>{_safe(note)}</p>" + _grid_block(
            [_render_image_card(p) for p in sections["UMAPs"]]
        )
        blocks.append('<div id="umaps"></div>')
        title_html = f"UMAPs <span class='pill'>{len(sections['UMAPs'])} plots</span>"
        blocks.append(_details_block(title_html, inner, open_by_default=False))

    if sections["Other"]:
        inner = _grid_block([_render_image_card(p) for p in sections["Other"]])
        blocks.append('<div id="other"></div>')
        title_html = f"Other <span class='pill'>{len(sections['Other'])} plots</span>"
        blocks.append(_details_block(title_html, inner, open_by_default=False))

    html_doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        "<title>scOmnom Integration report</title>"
        f"<style>{_css()}</style>"
        "</head><body>"
        f'<div class="wrap">{toc_html}<main class="content">{"".join(blocks)}</main></div>'
        "</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")


def generate_annotated_integration_report(
    *,
    fig_root: Path,
    version: str,
    adata,
    batch_key: str,
    final_label_key: str,
    round_id: str,
    # Optional overrides (paths relative to fig_root). Leave None to auto-infer.
    pre_full_png: str | None = None,
    post_full_png: str | None = None,
    pre_short_png: str | None = None,
    post_short_png: str | None = None,
    pre_post_png: str | None = None,
) -> None:
    """
    Write: <fig_root>/integration_report_annotated.html

    Annotated-run report:
      - NO scIB benchmarking
      - Only the 5 annotated-run UMAP figures produced by plot_annotated_run_umaps()

    Expected stems (written by plot_annotated_run_umaps via save_umap_multi):
      - umap_pre__<tag>__fulllegend
      - umap_post__<tag>__fulllegend
      - umap_pre__<tag>__shortlegend
      - umap_post__<tag>__shortlegend
      - umap_pre_vs_post__<tag>__shortlegend
    """
    from datetime import datetime
    from pathlib import Path
    import json
    import re
    from typing import Any, List, Tuple

    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "integration_report_annotated.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_imgs = _collect_pngs(fig_root)

    rid = str(round_id)

    def _sanitize_tag(s: str) -> str:
        s = str(s or "").strip()
        s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
        return s or "NA"

    tag = _sanitize_tag(rid)

    def _as_relpath(x: str | Path | None) -> Path | None:
        if not x:
            return None
        p = Path(x)
        return p if not p.is_absolute() else p.relative_to(fig_root)

    def _find_first_contains(substr: str) -> Path | None:
        # rel_imgs are already relative-to-fig_root paths
        for p in rel_imgs:
            if substr in p.as_posix():
                return p
        return None

    # If user passes explicit files, respect them; else auto-infer by the stems your plotter uses.
    pre_full_path = _as_relpath(pre_full_png) or _find_first_contains(f"umap_pre__{tag}__fulllegend")
    post_full_path = _as_relpath(post_full_png) or _find_first_contains(f"umap_post__{tag}__fulllegend")
    pre_short_path = _as_relpath(pre_short_png) or _find_first_contains(f"umap_pre__{tag}__shortlegend")
    post_short_path = _as_relpath(post_short_png) or _find_first_contains(f"umap_post__{tag}__shortlegend")
    pre_post_path = _as_relpath(pre_post_png) or _find_first_contains(f"umap_pre_vs_post__{tag}__shortlegend")

    # last-resort fallback (if tag mismatch) — pick “any tag”
    if pre_full_path is None:
        pre_full_path = _find_first_contains("umap_pre__") and _find_first_contains("__fulllegend")
    if post_full_path is None:
        post_full_path = _find_first_contains("umap_post__") and _find_first_contains("__fulllegend")
    if pre_short_path is None:
        pre_short_path = _find_first_contains("umap_pre__") and _find_first_contains("__shortlegend")
    if post_short_path is None:
        post_short_path = _find_first_contains("umap_post__") and _find_first_contains("__shortlegend")
    if pre_post_path is None:
        pre_post_path = _find_first_contains("umap_pre_vs_post__") and _find_first_contains("__shortlegend")

    # Order: 2-panel first, then post/pre fulllegend, then post/pre shortlegend
    cards_sorted: list[Path] = []
    for p in [pre_post_path, post_full_path, pre_full_path, post_short_path, pre_short_path]:
        if p is not None:
            cards_sorted.append(p)

    # Deduplicate while keeping order
    seen = set()
    cards_sorted = [p for p in cards_sorted if not (p in seen or seen.add(p))]

    summary: dict[str, Any] = {
        "mode": "annotated_secondary",
        "version": version,
        "timestamp": timestamp,
        "batch_key": str(batch_key),
        "final_label_key": str(final_label_key),
        "round_id": rid,
        "round_tag": tag,
        "active_umap_key": "X_umap",
        "stashed_pre_umap_key": "X_umap__pre_annotated_run",
        "plots_expected": 5,
        "plots_found": len(cards_sorted),
        "plots": [p.as_posix() for p in cards_sorted],
    }

    summary_json = _safe(json.dumps(summary, indent=2, default=str))

    rows = []
    for k, v in summary.items():
        vv = ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)
        rows.append(f"<tr><td>{_safe(k)}</td><td>{_safe(vv)}</td></tr>")

    summary_table = (
        '<table class="summary-table">'
        "<thead><tr><th>Field</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    header = (
        '<h1 id="top">scOmnom Annotated integration report</h1>'
        '<div class="meta">'
        f"Version:   {_safe(version)}\n"
        f"Timestamp: {_safe(timestamp)}\n\n"
        f"Round:     {_safe(rid)}\n\n"
        "Summary:\n"
        f"{summary_json}"
        "</div>"
        "<p class='note'>This report is intentionally minimal: it documents the annotated scANVI re-run and shows the pre/post UMAPs.</p>"
    )

    toc_sections: List[Tuple[str, str]] = [
        ("summary", "Summary"),
        ("umaps", "UMAPs (pre/post)"),
    ]
    toc_html = _toc(toc_sections)

    blocks: List[str] = [header]
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", summary_table, open_by_default=True))

    blocks.append('<div id="umaps"></div>')
    if cards_sorted:
        note = (
            "Figures produced by the annotated-run UMAP helper:\n"
            "• Fulllegend plots show the full pretty labels with a right-side legend.\n"
            "• Shortlegend plots replace legends with on-data Cnn IDs at cluster centroids.\n"
            "• The 2-panel plot compares Pre vs Post side-by-side using short labels."
        )
        inner = (
            f"<p class='note'>{_safe(note)}</p>"
            + '<div class="grid grid2">'
            + "".join([_render_image_card(p) for p in cards_sorted])
            + "</div>"
        )
        title_html = f"UMAPs (annotated run on { _safe(rid) }) <span class='pill'>{len(cards_sorted)} plots</span>"
        blocks.append(_details_block(title_html, inner, open_by_default=True))
    else:
        blocks.append(
            _details_block(
                f"UMAPs (annotated run on { _safe(rid) })",
                "<p class='note'><em>No annotated-run UMAP plots were found under fig_root/png. "
                "Expected files like <code>png/integration/umap_*__&lt;tag&gt;__fulllegend.png</code> and "
                "<code>png/integration/umap_*__&lt;tag&gt;__shortlegend.png</code>.</em></p>",
                open_by_default=True,
            )
        )

    html_doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        "<title>scOmnom Annotated integration report</title>"
        f"<style>{_css()}</style>"
        "</head><body>"
        f'<div class="wrap">{toc_html}<main class="content">{"".join(blocks)}</main></div>'
        "</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")


# =============================================================================
# Cluster + annotate report
# =============================================================================
def _split_round_sections(imgs: List[Path]) -> List[Tuple[str, List[Path]]]:
    clustering: List[Path] = []
    annotation: List[Path] = []
    decoupler: List[Path] = []
    compaction: List[Path] = []
    other: List[Path] = []

    for p in imgs:
        t = _classify_plot_type(p)
        s = p.as_posix().lower()

        if "/clustering/" in s or t == "clustering":
            clustering.append(p)
        elif "/annotation/" in s or t == "annotation":
            annotation.append(p)
        elif "/decoupler/" in s or t.startswith("decoupler"):
            decoupler.append(p)
        elif t == "compaction":
            compaction.append(p)
        else:
            other.append(p)

    keyfn = lambda x: x.as_posix()
    return [
        ("Clustering", sorted(clustering, key=keyfn)),
        ("Annotation", sorted(annotation, key=keyfn)),
        ("Decoupler", sorted(decoupler, key=keyfn)),
        ("Compaction", sorted(compaction, key=keyfn)),
        ("Other", sorted(other, key=keyfn)),
    ]


def _infer_round_cluster_labels_key(adata, rid: str, rinfo: dict, cfg=None) -> str | None:
    """Return the obs column that contains cluster labels for *this* round."""
    obs = getattr(adata, "obs", None)
    if obs is None:
        return None

    # 1) explicit key saved in uns (best)
    k = rinfo.get("labels_obs_key", None)
    if isinstance(k, str) and k in obs:
        return k

    # 2) cluster_key may already be round-specific
    ck = rinfo.get("cluster_key", None)
    if isinstance(ck, str) and ck in obs and "__" in ck:
        return ck

    # 3) try common round-suffixed conventions
    candidates: list[str] = []

    if isinstance(ck, str) and ck:
        candidates.append(f"{ck}__{rid}")  # e.g. leiden__r0...

    # cfg.label_key often used as base (if passed)
    if cfg is not None:
        base = getattr(cfg, "label_key", None)
        if isinstance(base, str) and base:
            candidates.append(f"{base}__{rid}")
            candidates.append(f"{base}_{rid}")  # sometimes people use underscore

    for cand in candidates:
        if cand in obs:
            return cand

    # 4) last resort: plain cluster_key (may be overwritten!)
    if isinstance(ck, str) and ck in obs:
        return ck

    return None


def _infer_n_clusters_for_round(adata, rid: str, rinfo: dict, cfg=None) -> int | None:
    k = _infer_round_cluster_labels_key(adata, rid, rinfo, cfg=cfg)
    if not k:
        return None
    try:
        return int(getattr(adata.obs[k], "astype", lambda x: adata.obs[k])(str).nunique())
    except Exception:
        try:
            return int(adata.obs[k].nunique())
        except Exception:
            return None


def _render_round_summary_table(adata, rid: str, cfg=None) -> str:
    """
    Renders a summary table for a specific clustering round.
    Strictly derives n_clusters from the round-specific labels_obs_key.
    """
    # 1. Access round info safely
    rounds = adata.uns.get("cluster_rounds", {})
    rinfo = rounds.get(rid, {}) if isinstance(rounds, dict) else {}

    # 2. Extract specific round attributes
    cluster_key = rinfo.get("cluster_key")
    best_res = rinfo.get("best_resolution")
    labels_key = rinfo.get("labels_obs_key")

    # 3. Derive cluster count strictly from the labels_obs_key
    n_clusters = ""
    if labels_key and labels_key in adata.obs:
        s = adata.obs[labels_key]
        try:
            # Use categories if available, else unique values
            if hasattr(s, "cat"):
                n_clusters = len(s.cat.categories)
            else:
                n_clusters = s.nunique()
        except Exception:
            pass

    # 4. Helper for HTML rows
    def _row(label, value):
        val_str = str(value) if value is not None else ""
        return f"<tr><td><b>{label}</b></td><td>{val_str}</td></tr>"

    # 5. Build Table
    rows = [
        _row("Round ID", rid),
        _row("Cluster Key", cluster_key),
        _row("Labels Obs Key", labels_key),
        _row("Best Resolution", best_res),
        _row("N Clusters", n_clusters),
        _row("Notes", rinfo.get("notes")),
    ]

    return (
        '<table class="summary-table" style="width:100%; border-collapse: collapse;">'
        '<thead><tr style="text-align: left; border-bottom: 2px solid #ddd;">'
        '<th>Field</th><th>Value</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

def generate_cluster_and_annotate_report(*, fig_root: Path, cfg, version: str, adata) -> None:
    """
    Write: <fig_root>/cluster_and_annotate_report.html
    Expects figures under: <fig_root>/png/...
    Behavior:
      - Organize by clustering state (from adata.uns, NOT from file paths)
      - Clustering state sections collapsed by default (active round open if known)
      - Decoupler barplots ALWAYS collapsed by default
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "cluster_and_annotate_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_imgs = _collect_pngs(fig_root)  # Paths relative to fig_root (as in the rest of your report)

    # ------------------------------------------------------------------
    # Authoritative round list from AnnData
    # ------------------------------------------------------------------
    uns = getattr(adata, "uns", {}) or {}

    rounds_meta = uns.get("cluster_rounds", {})
    if not isinstance(rounds_meta, dict):
        rounds_meta = {}

    active_round = uns.get("active_cluster_round", None)
    active_round = str(active_round) if active_round is not None else None

    # Prefer explicit stored order, but keep only valid rounds
    ordered_rounds: List[str] = []
    ro = uns.get("cluster_round_order", None)
    if isinstance(ro, (list, tuple)) and ro:
        ordered_rounds = [str(x) for x in ro if str(x) in rounds_meta]

    # Fallback to sorted registered rounds
    if not ordered_rounds:
        ordered_rounds = sorted([str(k) for k in rounds_meta.keys()])

    # Ensure active round is shown even if not registered (rare but possible)
    if active_round and active_round not in ordered_rounds:
        ordered_rounds.append(active_round)

    # ------------------------------------------------------------------
    # Map images -> rounds (best-effort), otherwise "unassigned"
    # ------------------------------------------------------------------
    by_round: Dict[str, List[Path]] = {rid: [] for rid in ordered_rounds}
    by_round["unassigned"] = []

    for p in rel_imgs:
        rid = _extract_round_id_from_rel_png(p)
        if rid and rid in by_round:
            by_round[rid].append(p)
        else:
            by_round["unassigned"].append(p)

    # Nice fallback: if EVERYTHING is unassigned but we have exactly one registered round,
    # assign all images to that round (common when the path format changed).
    if by_round["unassigned"] and len(ordered_rounds) == 1:
        only = ordered_rounds[0]
        by_round[only].extend(by_round["unassigned"])
        by_round["unassigned"] = []

    # Drop "unassigned" if empty
    if not by_round["unassigned"]:
        by_round.pop("unassigned", None)

    # ------------------------------------------------------------------
    # Header / meta
    # ------------------------------------------------------------------
    cfg_dict = getattr(cfg, "__dict__", {})
    cfg_json = _safe(json.dumps(cfg_dict, indent=2, default=str))
    rounds_json = _safe(json.dumps(ordered_rounds, indent=2, default=str))

    header = (
        '<h1 id="top">scOmnom cluster-and-annotate report</h1>'
        '<div class="meta">'
        f"Version:   {_safe(version)}\n"
        f"Timestamp: {_safe(timestamp)}\n\n"
        f"Active clustering state: {_safe(active_round)}\n\n"
        "Clustering state order:\n"
        f"{rounds_json}\n\n"
        "Parameters:\n"
        f"{cfg_json}"
        "</div>"
        "<p class='note'>This report is organized by <b>clustering state</b>."
        " Plot families are collapsed by default to keep the page readable.</p>"
    )

    # ------------------------------------------------------------------
    # TOC: summary + registered rounds (+ unassigned if present)
    # ------------------------------------------------------------------
    toc_sections: List[Tuple[str, str]] = [("summary", "Summary")]
    for rid in ordered_rounds:
        toc_sections.append((f"clustering state-{_slug(rid)}", f"Clustering state: {rid}"))
    if "unassigned" in by_round:
        toc_sections.append(("state-unassigned", "Unassigned"))
    toc_html = _toc(toc_sections)

    blocks: List[str] = [header]

    # Summary (open)
    summary_inner = (
        "<p class='note'>Clustering state list and active state (round) pointer (from <code>adata.uns</code>).</p>"
        '<table class="summary-table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>'
        f"<tr><td>active_round</td><td>{_safe(active_round)}</td></tr>"
        f"<tr><td>n_rounds_registered</td><td>{len(ordered_rounds)}</td></tr>"
        f"<tr><td>n_plots_total</td><td>{sum(len(v) for v in by_round.values())}</td></tr>"
        "</tbody></table>"
    )
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", summary_inner, open_by_default=True))

    # ------------------------------------------------------------------
    # Clustering state sections (render even if empty)
    # ------------------------------------------------------------------
    if not ordered_rounds and "unassigned" not in by_round:
        blocks.append("<p class='note'><em>No cluster_and_annotate rounds found in adata.uns, and no figures found.</em></p>")
    else:
        for rid in ordered_rounds:
            imgs = by_round.get(rid, [])

            sid = f"clustering state-{_slug(rid)}"
            blocks.append(f'<div id="{sid}"></div>')

            open_round = bool(active_round) and (rid == active_round)
            sections = _split_round_sections(imgs) if imgs else []

            round_parts: List[str] = []
            round_parts.append(f"<p class='note'>Figures for clustering state <strong>{_safe(rid)}</strong>.</p>")
            round_parts.append(_render_round_summary_table(adata, rid))

            if not imgs:
                round_parts.append("<p class='note'><em>No figures were matched to this clustering state.</em></p>")
            else:
                for title, sec_imgs in sections:
                    if not sec_imgs:
                        continue

                    # Decoupler: split further; barplots collapsed by default
                    if title == "Decoupler":
                        heat = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_heatmap"]
                        dots = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_dotplot"]
                        bars = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_bar"]
                        other = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_other"]

                        if heat:
                            inner = "<p class='note'>Top features/pathways across clusters.</p>" + _grid_block(
                                [_render_image_card(p) for p in heat]
                            )
                            title_html = f"Decoupler heatmaps <span class='pill'>{len(heat)} plots</span>"
                            round_parts.append(_details_block(title_html, inner, open_by_default=False, extra_class="subsection"))

                        if dots:
                            inner = "<p class='note'>Compact overview of activity patterns.</p>" + _grid_block(
                                [_render_image_card(p) for p in dots]
                            )
                            title_html = f"Decoupler dotplots <span class='pill'>{len(dots)} plots</span>"
                            round_parts.append(_details_block(title_html, inner, open_by_default=False, extra_class="subsection"))

                        if bars:
                            inner = (
                                "<p class='note'>Per-cluster top-N features.</p>"
                                + _grid_block([_render_image_card(p) for p in bars])
                            )
                            title_html = f"Decoupler barplots <span class='pill'>{len(bars)} plots</span>"
                            round_parts.append(_details_block(title_html, inner, open_by_default=False, extra_class="subsection"))

                        if other:
                            inner = _grid_block([_render_image_card(p) for p in other])
                            title_html = f"Decoupler other <span class='pill'>{len(other)} plots</span>"
                            round_parts.append(_details_block(title_html, inner, open_by_default=False, extra_class="subsection"))

                        continue

                    # Non-decoupler sections: collapsible, tight, collapsed by default
                    note = ""
                    if title == "Clustering":
                        note = "Resolution sweep, stability, cluster tree, and clustering UMAP diagnostics."
                    elif title == "Annotation":
                        note = "CellTypist and cluster label visualizations and summaries."
                    elif title == "Compaction":
                        note = "Compaction flow and compacted-state diagnostics."

                    note_html = f"<p class='note'>{_safe(note)}</p>" if note else ""
                    inner = note_html + _grid_block([_render_image_card(p) for p in sec_imgs])
                    title_html = f"{_safe(title)} <span class='pill'>{len(sec_imgs)} plots</span>"
                    round_parts.append(_details_block(title_html, inner, open_by_default=False, extra_class="subsection"))

            round_title_html = f"Clustering state: {_safe(rid)} <span class='pill'>{len(imgs)} plots</span>"
            blocks.append(_details_block(round_title_html, "".join(round_parts), open_by_default=open_round))

        # Unassigned bucket (if present)
        if "unassigned" in by_round:
            imgs = by_round["unassigned"]
            blocks.append('<div id="state-unassigned"></div>')
            inner = (
                "<p class='note'>These plots could not be mapped to a registered clustering state from <code>adata.uns['cluster_rounds']</code>.</p>"
                + _grid_block([_render_image_card(p) for p in imgs])
            )
            blocks.append(_details_block(
                f"Unassigned <span class='pill'>{len(imgs)} plots</span>",
                inner,
                open_by_default=False,
            ))

    html_doc = (
        "<!DOCTYPE html><html><head>"
        '<meta charset="utf-8">'
        "<title>scOmnom cluster-and-annotate report</title>"
        f"<style>{_css()}</style>"
        "</head><body>"
        f'<div class="wrap">{toc_html}<main class="content">{"".join(blocks)}</main></div>'
        "</body></html>"
    )
    out_html.write_text(html_doc, encoding="utf-8")


def _extract_round_id_from_rel_png(rel_path_from_fig_root: Path) -> str | None:
    """
    rel_path_from_fig_root looks like:
      png/cluster_and_annotate_roundN/<round_id>/...
    or (older/alternate):
      png/cluster_and_annotate/<round_id>/...
    Returns the <round_id> or None if not detectable.
    """
    parts = list(Path(rel_path_from_fig_root).parts)

    # strip leading "png" if present
    if parts and parts[0] == "png":
        parts = parts[1:]

    if not parts:
        return None

    # Layout A: cluster_and_annotate_roundN/<rid>/...
    if parts[0].startswith("cluster_and_annotate_round"):
        return parts[1] if len(parts) >= 2 else None

    # Layout B: cluster_and_annotate/<rid>/...
    try:
        i = parts.index("cluster_and_annotate")
        return parts[i + 1] if i + 1 < len(parts) else None
    except ValueError:
        return None
