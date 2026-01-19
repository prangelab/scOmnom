# src/scomnom/reporting.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Iterable
import html
import json
import re


# =============================================================================
# Small UX helpers
# =============================================================================

def _safe(x: Any) -> str:
    return html.escape(str(x))


def _slug(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"


def _pretty_relpath(p: Path) -> str:
    # report is written to fig_root/*.html, images live under fig_root/png/...
    s = p.as_posix()
    s = re.sub(r"^png/", "", s)
    return s


def _cfg_to_json(cfg: Any) -> str:
    try:
        if is_dataclass(cfg):
            return json.dumps(asdict(cfg), indent=2, default=str)
    except Exception:
        pass
    try:
        return json.dumps(getattr(cfg, "__dict__", {}), indent=2, default=str)
    except Exception:
        return json.dumps({"cfg": str(cfg)}, indent=2, default=str)


def _caption_from_filename(fname: str) -> str:
    name = Path(fname).stem
    name = name.replace("__", " / ")
    name = name.replace("_", " ")
    name = name.replace("postfilter", "post-filter")
    name = name.replace("prefilter", "pre-filter")
    name = name.replace("qc", "QC")
    name = re.sub(r"\s+", " ", name).strip()
    # small niceties
    name = name.replace("umap", "UMAP")
    name = name.replace("scib", "scIB")
    return name[:1].upper() + name[1:] if name else "Plot"


# =============================================================================
# Plot classification + descriptions
# =============================================================================

def _classify_plot_type(rel_path: Path) -> str:
    """Heuristic classifier used for grouping and collapse policy."""
    s = rel_path.as_posix().lower()
    stem = rel_path.stem.lower()

    # decoupler
    if "decoupler" in s:
        if "heatmap" in stem:
            return "decoupler_heatmap"
        if "dotplot" in stem or "dotplot" in s or "dot" in stem:
            return "decoupler_dotplot"
        if "bar" in stem or "top" in stem and "bar" in stem:
            return "decoupler_bar"
        return "decoupler_other"

    # clustering / rounds
    if "/clustering/" in s or "resolution" in stem or "cluster_tree" in stem or "stability" in stem:
        return "clustering"

    if "compaction" in s or "compaction" in stem or "compacted" in stem:
        return "compaction"

    if "celltypist" in s or "annotation" in s or "pretty_cluster" in stem:
        return "annotation"

    # integration
    if "scib" in s or "integration_metrics" in stem or "results_table" in stem:
        return "scib"

    if "umap" in stem or "/umap" in s:
        return "umap"

    # qc
    if "doublet" in s or "doublet" in stem:
        return "doublet"

    if "cellbender" in s or "cellbender" in stem:
        return "cellbender"

    if "qc_scatter" in s or "complexity" in stem or "scatter" in stem:
        return "qc_scatter"

    if "qc_metrics" in s or "violin" in stem or "hist" in stem:
        return "qc_metrics"

    if "overview" in s:
        return "overview"

    return "other"


def _describe_plot(rel_path: Path) -> str:
    """Short description shown under each plot."""
    s = rel_path.as_posix().lower()
    stem = rel_path.stem.lower()

    # QC
    if "qc_plots" in s and "violin" in stem:
        if "mt" in stem:
            return "Per-sample distribution of mitochondrial fraction."
        if "ribo" in stem:
            return "Per-sample distribution of ribosomal fraction."
        if "hb" in stem:
            return "Per-sample distribution of hemoglobin fraction."
        if "counts" in stem:
            return "Per-sample distribution of library size (UMIs)."
        if "genes" in stem:
            return "Per-sample distribution of detected genes."
        return "Per-sample QC distribution."

    if "qc_plots" in s and "hist" in stem:
        if "mt" in stem:
            return "Overall distribution of mitochondrial fraction."
        if "total_counts" in stem:
            return "Overall distribution of library size (UMIs)."
        if "n_genes" in stem or "genes" in stem:
            return "Overall distribution of detected genes."
        return "Overall QC distribution."

    if "qc_plots" in s and ("complexity" in stem or ("scatter" in stem and "qc" in stem)):
        return "QC scatter diagnostic for outliers and complexity."

    # Doublets
    if "doublet" in stem:
        if "hist" in stem:
            return "Doublet-score distribution; red lines indicate inferred per-sample thresholds."
        if "threshold" in stem:
            return "Inferred doublet-score threshold per sample."
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
        return "CellBender diagnostics."

    # Integration
    if "scib" in s or "results_table" in stem:
        return "Integration benchmarking summary (scIB metrics)."
    if "umap" in stem and "vs_unintegrated" in stem:
        return "Side-by-side UMAP comparison against the unintegrated baseline."
    if "umap" in stem:
        return "UMAP visualization of an embedding / integration method."

    # Clustering
    if "cluster_tree" in stem:
        return "How clusters split/merge across the resolution sweep."
    if "resolution_sweep" in stem:
        return "Resolution sweep: silhouette, number of clusters, and penalized score."
    if "stability" in stem and "ari" in stem:
        return "Subsampling stability across repeats (ARI vs full-data clustering)."
    if "cluster_selection" in stem or "stability_curves" in stem:
        return "Metrics used for resolution selection (structural + penalties)."
    if "biological_metrics" in stem:
        return "Bio-guided clustering metrics across the resolution sweep."
    if "cluster_sizes" in stem:
        return "Cluster sizes (cells per cluster)."
    if "cluster_qc_summary" in stem:
        return "Mean QC metrics per cluster (diagnostic)."
    if "cluster_batch_composition" in stem:
        return "Batch composition within each cluster."
    if "silhouette_by_cluster" in stem:
        return "Silhouette distributions per cluster."
    if "compaction_flow" in stem:
        return "Flow from parent clusters to compacted clusters."

    # Decoupler
    if "decoupler" in s:
        if "heatmap" in stem:
            return "Decoupler activity heatmap (top features/pathways)."
        if "dotplot" in stem:
            return "Decoupler dotplot summary (activity + magnitude)."
        if "bar" in stem:
            return "Per-cluster top-N decoupler features (barplot)."
        return "Decoupler plot."

    return "Diagnostic plot."


# =============================================================================
# HTML building blocks
# =============================================================================

def _css() -> str:
    # tight + readable + stable across many images
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
      padding: 1.15rem 1.15rem 3rem;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 1.1rem;
    }

    @media (max-width: 1100px) {
      .wrap { grid-template-columns: 1fr; }
      .toc { position: static !important; }
    }

    h1 {
      font-size: 1.55rem;
      margin: 0.2rem 0 0.8rem;
      letter-spacing: -0.02em;
    }

    h2 {
      font-size: 1.15rem;
      margin: 0.2rem 0 0.6rem;
      letter-spacing: -0.01em;
    }

    .meta {
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
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
      padding: 0.26rem 0.35rem;
      border-radius: 8px;
      font-size: 0.9rem;
    }

    .toc a:hover {
      background: #f3f4f6;
    }

    .content {
      min-width: 0;
    }

    details.section {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: #fff;
      box-shadow: var(--shadow);
      margin: 0 0 0.85rem;
      overflow: hidden;
    }

    details.section > summary {
      cursor: pointer;
      padding: 0.72rem 0.9rem;
      font-weight: 650;
      list-style: none;
      user-select: none;
    }

    details.section > summary::-webkit-details-marker { display: none; }

    details.section[open] > summary {
      border-bottom: 1px solid var(--border);
      background: #fafafa;
    }

    .section-body {
      padding: 0.85rem;
    }

    .note {
      font-size: 0.92rem;
      color: #374151;
      margin: 0.18rem 0 0.65rem;
    }

    .pill {
      display: inline-block;
      font-size: 0.78rem;
      padding: 0.12rem 0.5rem;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: #fafafa;
      color: #111827;
      margin-left: 0.35rem;
      vertical-align: middle;
    }

    .summary-table {
      width: 100%;
      border-collapse: collapse;
      margin: 0.25rem 0 0.45rem;
      font-size: 0.95rem;
    }

    .summary-table th, .summary-table td {
      border: 1px solid var(--border);
      padding: 0.42rem 0.58rem;
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
      gap: 0.8rem;
      align-items: start;
    }

    @media (max-width: 520px) {
      .grid { grid-template-columns: 1fr; }
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

    figure.card:hover {
      box-shadow: var(--shadow2);
    }

    .card-img {
      padding: 0.55rem;
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
      padding: 0.65rem 0.78rem 0.7rem;
      border-top: 1px solid var(--border);
    }

    .cap-title {
      font-size: 0.94rem;
      font-weight: 650;
      margin-bottom: 0.22rem;
    }

    .cap-desc {
      font-size: 0.87rem;
      color: #374151;
      margin-bottom: 0.33rem;
    }

    .cap-meta {
      font-size: 0.77rem;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      overflow-wrap: anywhere;
    }

    /* Nested sections (still collapsible, but slightly tighter) */
    details.section.subsection > summary {
      padding: 0.55rem 0.8rem;
      font-weight: 600;
    }
    """


def _details_block(title: str, inner_html: str, *, open_by_default: bool = False, extra_class: str = "") -> str:
    open_attr = " open" if open_by_default else ""
    cls = f"section {extra_class}".strip()
    return f"""
    <details class="{cls}"{open_attr}>
      <summary>{title}</summary>
      <div class="section-body">
        {inner_html}
      </div>
    </details>
    """


def _grid_block(items: List[str]) -> str:
    return f"""
    <div class="grid">
      {''.join(items)}
    </div>
    """


def _toc(sections: List[Tuple[str, str]]) -> str:
    links = "\n".join([f'<a href="#{_safe(sid)}">{_safe(title)}</a>' for sid, title in sections])
    return f"""
    <nav class="toc">
      <h2>Contents</h2>
      {links}
    </nav>
    """


def render_image_block(rel_img_path: Path) -> str:
    caption = _caption_from_filename(rel_img_path.name)
    desc = _describe_plot(rel_img_path)
    meta = _pretty_relpath(rel_img_path)
    src = rel_img_path.as_posix()  # must be relative to the report file in fig_root/
    return f"""
    <figure class="card">
      <div class="card-img">
        <img loading="lazy" src="{src}">
      </div>
      <figcaption class="card-cap">
        <div class="cap-title">{_safe(caption)}</div>
        <div class="cap-desc">{_safe(desc)}</div>
        <div class="cap-meta">{_safe(meta)}</div>
      </figcaption>
    </figure>
    """


# =============================================================================
# File collection
# =============================================================================

def _collect_images(fig_root: Path) -> List[Path]:
    """
    Returns PNGs as paths *relative to fig_root*, i.e. `png/.../file.png`.
    """
    fig_root = Path(fig_root).resolve()
    png_root = fig_root / "png"
    if not png_root.exists():
        return []
    imgs_abs = sorted(png_root.rglob("*.png"))
    return [p.relative_to(fig_root) for p in imgs_abs]


# =============================================================================
# QC report
# =============================================================================

def _collect_qc_summary(adata) -> dict:
    obs = getattr(adata, "obs", None)
    summary: Dict[str, Any] = {
        "n_cells": int(getattr(adata, "n_obs", 0)),
        "n_genes": int(getattr(adata, "n_vars", 0)),
    }

    try:
        info = getattr(adata, "uns", {}).get("doublet_calling")
        if isinstance(info, dict):
            summary["doublet_observed_rate"] = info.get("observed_global_rate")
    except Exception:
        pass

    if obs is not None:
        for col in [
            "total_counts",
            "n_genes_by_counts",
            "pct_counts_mt",
            "pct_counts_ribo",
            "pct_counts_hb",
        ]:
            try:
                if col in obs:
                    summary[f"median_{col}"] = float(obs[col].median())
                    summary[f"mean_{col}"] = float(obs[col].mean())
            except Exception:
                continue

    return summary


def generate_qc_report(*, fig_root: Path, cfg: Any, version: str, adata: Any) -> None:
    """
    Output:
      <fig_root>/qc_report.html

    Sections are collapsible and collapsed by default (except Summary).
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "qc_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_images = _collect_images(fig_root)

    sections: Dict[str, List[Path]] = {
        "Overview": [],
        "QC metrics": [],
        "QC scatter": [],
        "Doublets": [],
        "CellBender": [],
        "Other": [],
    }

    for p in rel_images:
        t = _classify_plot_type(p)
        if t == "doublet":
            sections["Doublets"].append(p)
        elif t == "cellbender":
            sections["CellBender"].append(p)
        elif t == "qc_metrics":
            sections["QC metrics"].append(p)
        elif t == "qc_scatter":
            sections["QC scatter"].append(p)
        elif t == "overview":
            sections["Overview"].append(p)
        else:
            sections["Other"].append(p)

    cfg_json = _cfg_to_json(cfg)
    header = f"""
    <h1 id="top">scOmnom QC report</h1>
    <div class="meta">Version:   {_safe(version)}
Timestamp: {_safe(timestamp)}

Parameters:
{_safe(cfg_json)}</div>
    """

    qc_summary = _collect_qc_summary(adata)
    rows = []
    for k, v in qc_summary.items():
        if isinstance(v, float):
            rows.append(f"<tr><td>{_safe(k)}</td><td>{v:.4g}</td></tr>")
        else:
            rows.append(f"<tr><td>{_safe(k)}</td><td>{_safe(v)}</td></tr>")

    summary_html = f"""
    {header}
    <table class="summary-table">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

    toc_sections: List[Tuple[str, str]] = [("summary", "Summary")]
    for title, imgs in sections.items():
        if imgs:
            toc_sections.append((_slug(title), title))

    blocks: List[str] = []
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", summary_html, open_by_default=True))

    notes = {
        "Overview": "High-level outputs (e.g., HVGs/PCA, overall summaries).",
        "QC metrics": "Distributions and violins for key QC metrics (pre/post filter where available).",
        "QC scatter": "Scatter diagnostics for outliers and complexity.",
        "Doublets": "SOLO doublet diagnostics and inferred thresholds.",
        "CellBender": "Raw vs CellBender comparisons (when CellBender input was used).",
        "Other": "",
    }

    for title, imgs in sections.items():
        if not imgs:
            continue
        inner = ""
        note = notes.get(title, "")
        if note:
            inner += f"<p class='note'>{_safe(note)}</p>"
        inner += _grid_block([render_image_block(p) for p in imgs])
        blocks.append(f'<div id="{_slug(title)}"></div>')
        blocks.append(
            _details_block(
                f"{_safe(title)} <span class='pill'>{len(imgs)} plots</span>",
                inner,
                open_by_default=False,
            )
        )

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom QC report</title>
      <style>{_css()}</style>
    </head>
    <body>
      <div class="wrap">
        {_toc(toc_sections)}
        <main class="content">
          {''.join(blocks)}
        </main>
      </div>
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")


# =============================================================================
# Integration report
# =============================================================================

def generate_integration_report(
    *,
    fig_root: Path,
    version: str,
    adata: Any,
    batch_key: str,
    label_key: str,
    methods: List[str],
    selected_embedding: str,
    benchmark_n_jobs: int,
) -> None:
    """
    Output:
      <fig_root>/integration_report.html

    Sections are collapsible and collapsed by default (except Summary).
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "integration_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_images = _collect_images(fig_root)

    sections: Dict[str, List[Path]] = {
        "Benchmarking": [],
        "UMAPs": [],
        "Other": [],
    }

    for p in rel_images:
        t = _classify_plot_type(p)
        if t == "scib":
            sections["Benchmarking"].append(p)
        elif t == "umap":
            sections["UMAPs"].append(p)
        else:
            sections["Other"].append(p)

    summary = {
        "version": version,
        "timestamp": timestamp,
        "batch_key": batch_key,
        "label_key": label_key,
        "methods_requested": methods,
        "benchmark_n_jobs": benchmark_n_jobs,
        "selected_embedding": selected_embedding,
    }

    rows = [f"<tr><td>{_safe(k)}</td><td>{_safe(v)}</td></tr>" for k, v in summary.items()]
    summary_html = f"""
    <h1 id="top">scOmnom Integration report</h1>
    <div class="meta">Version:   {_safe(version)}
Timestamp: {_safe(timestamp)}

Selected embedding:
{_safe(selected_embedding)}</div>

    <table class="summary-table">
      <thead><tr><th>Field</th><th>Value</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

    toc_sections: List[Tuple[str, str]] = [("summary", "Summary")]
    if sections["Benchmarking"]:
        toc_sections.append(("benchmarking", "Benchmarking"))
    if sections["UMAPs"]:
        toc_sections.append(("umaps", "UMAPs"))
    if sections["Other"]:
        toc_sections.append(("other", "Other"))

    blocks: List[str] = []
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", summary_html, open_by_default=True))

    if sections["Benchmarking"]:
        inner = (
            "<p class='note'>scIB benchmarking outputs. Use these to understand tradeoffs between batch mixing "
            "and biological conservation.</p>"
            + _grid_block([render_image_block(p) for p in sections["Benchmarking"]])
        )
        blocks.append('<div id="benchmarking"></div>')
        blocks.append(_details_block(
            f"Benchmarking <span class='pill'>{len(sections['Benchmarking'])} plots</span>",
            inner,
            open_by_default=False,
        ))

    if sections["UMAPs"]:
        inner = (
            "<p class='note'>UMAPs for each embedding / method. These can be many; expand only when needed.</p>"
            + _grid_block([render_image_block(p) for p in sections["UMAPs"]])
        )
        blocks.append('<div id="umaps"></div>')
        blocks.append(_details_block(
            f"UMAPs <span class='pill'>{len(sections['UMAPs'])} plots</span>",
            inner,
            open_by_default=False,
        ))

    if sections["Other"]:
        inner = _grid_block([render_image_block(p) for p in sections["Other"]])
        blocks.append('<div id="other"></div>')
        blocks.append(_details_block(
            f"Other <span class='pill'>{len(sections['Other'])} plots</span>",
            inner,
            open_by_default=False,
        ))

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom Integration report</title>
      <style>{_css()}</style>
    </head>
    <body>
      <div class="wrap">
        {_toc(toc_sections)}
        <main class="content">
          {''.join(blocks)}
        </main>
      </div>
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")


# =============================================================================
# Cluster + annotate report
# =============================================================================

def _extract_round_id(rel_img_path: Path) -> str:
    """
    Try to find `cluster_and_annotate/<round_id>/...` in the relative path.

    Works whether images are under:
      png/<run_round>/cluster_and_annotate/<round_id>/...
    or:
      png/cluster_and_annotate/<round_id>/...
    """
    parts = rel_img_path.parts
    if "cluster_and_annotate" not in parts:
        return "unknown"
    i = list(parts).index("cluster_and_annotate")
    if i + 1 < len(parts):
        return str(parts[i + 1])
    return "unknown"


def _classify_cluster_round_sections(imgs: List[Path]) -> List[Tuple[str, List[Path]]]:
    clustering, annotation, compaction, decoupler, other = [], [], [], [], []

    for p in imgs:
        t = _classify_plot_type(p)
        if t == "clustering":
            clustering.append(p)
        elif t == "annotation":
            annotation.append(p)
        elif t == "compaction":
            compaction.append(p)
        elif t.startswith("decoupler"):
            decoupler.append(p)
        else:
            other.append(p)

    keyfn = lambda x: x.as_posix()
    return [
        ("Clustering", sorted(clustering, key=keyfn)),
        ("Annotation", sorted(annotation, key=keyfn)),
        ("Compaction", sorted(compaction, key=keyfn)),
        ("Decoupler", sorted(decoupler, key=keyfn)),
        ("Other", sorted(other, key=keyfn)),
    ]


def _render_cluster_report_summary(adata: Any, ordered_rounds: List[str]) -> str:
    active_round = None
    try:
        active_round = getattr(adata, "uns", {}).get("active_cluster_round", None)
        active_round = str(active_round) if active_round is not None else None
    except Exception:
        active_round = None

    # best-effort: surface round meta if present
    rounds_meta = {}
    try:
        rounds_meta = getattr(adata, "uns", {}).get("cluster_rounds", {})
        if not isinstance(rounds_meta, dict):
            rounds_meta = {}
    except Exception:
        rounds_meta = {}

    rows = []
    for rid in ordered_rounds:
        rinfo = rounds_meta.get(rid, {}) if isinstance(rounds_meta, dict) else {}
        if not isinstance(rinfo, dict):
            rinfo = {}
        best_res = rinfo.get("best_resolution", "")
        ncl = rinfo.get("n_clusters", "")
        rows.append(
            "<tr>"
            f"<td>{_safe(rid)}{' <span class=\"pill\">active</span>' if active_round == rid else ''}</td>"
            f"<td>{_safe(best_res)}</td>"
            f"<td>{_safe(ncl)}</td>"
            "</tr>"
        )

    return f"""
    <p class="note">
      Report is organized by <b>round</b>. All sections are collapsed by default, and
      <b>Decoupler barplots are always collapsed</b> to keep the page readable.
    </p>

    <table class="summary-table">
      <thead><tr><th>Round</th><th>Best resolution</th><th># clusters</th></tr></thead>
      <tbody>{''.join(rows) if rows else ''}</tbody>
    </table>
    """


def generate_cluster_and_annotate_report(*, fig_root: Path, cfg: Any, version: str, adata: Any) -> None:
    """
    Output:
      <fig_root>/cluster_and_annotate_report.html

    Requirements implemented:
      - Tight layout
      - Short description per plot
      - Sections collapsible and collapsed by default
      - Cluster-and-annotate: decoupler barplots collapsed by default
      - Active round (if known) is expanded; other rounds collapsed
    """
    fig_root = Path(fig_root).resolve()
    out_html = fig_root / "cluster_and_annotate_report.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rel_images = _collect_images(fig_root)

    # Group by round id (best-effort)
    by_round: Dict[str, List[Path]] = defaultdict(list)
    for p in rel_images:
        rid = _extract_round_id(p)
        by_round[rid].append(p)

    # round ordering: prefer adata.uns["cluster_round_order"]
    round_order: List[str] = []
    try:
        ro = getattr(adata, "uns", {}).get("cluster_round_order", None)
        if isinstance(ro, (list, tuple)) and ro:
            round_order = [str(x) for x in ro]
    except Exception:
        round_order = []

    ordered_rounds: List[str] = []
    seen = set()
    for rid in round_order:
        if rid in by_round:
            ordered_rounds.append(rid)
            seen.add(rid)
    for rid in sorted(by_round.keys()):
        if rid not in seen:
            ordered_rounds.append(rid)

    active_round: Optional[str] = None
    try:
        active_round = getattr(adata, "uns", {}).get("active_cluster_round", None)
        active_round = str(active_round) if active_round is not None else None
    except Exception:
        active_round = None

    cfg_json = _cfg_to_json(cfg)
    header = f"""
    <h1 id="top">scOmnom cluster-and-annotate report</h1>
    <div class="meta">Version:   {_safe(version)}
Timestamp: {_safe(timestamp)}

Active round: {_safe(active_round)}

Round order:
{_safe(json.dumps(ordered_rounds, indent=2))}

Parameters:
{_safe(cfg_json)}</div>
    """

    toc_sections: List[Tuple[str, str]] = [("summary", "Summary")]
    for rid in ordered_rounds:
        toc_sections.append((f"round-{_slug(rid)}", f"Round: {rid}"))

    blocks: List[str] = []
    blocks.append('<div id="summary"></div>')
    blocks.append(_details_block("Summary", header + _render_cluster_report_summary(adata, ordered_rounds), open_by_default=True))

    # Policy toggle (kept as a code constant; default is to keep them, but collapsed)
    OMIT_DECOUPLER_BARPLOTS = False

    if not ordered_rounds:
        blocks.append("<p class='note'><em>No cluster_and_annotate figures found under png/.</em></p>")
    else:
        for rid in ordered_rounds:
            imgs = by_round.get(rid, [])
            if not imgs:
                continue

            open_round = bool(active_round) and (rid == active_round)

            # Round title anchor
            blocks.append(f'<div id="round-{_slug(rid)}"></div>')

            # Split into sections
            sections = _classify_cluster_round_sections(imgs)

            inner_parts: List[str] = []
            inner_parts.append(
                f"<p class='note'>Figures for round <strong>{_safe(rid)}</strong>.</p>"
            )

            for title, sec_imgs in sections:
                if not sec_imgs:
                    continue

                if title == "Decoupler":
                    heat = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_heatmap"]
                    dots = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_dotplot"]
                    bars = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_bar"]
                    other = [p for p in sec_imgs if _classify_plot_type(p) == "decoupler_other"]

                    dec_parts: List[str] = []
                    dec_parts.append("<p class='note'>Cluster-level pathway/regulator activity scores.</p>")

                    if heat:
                        dec_parts.append(_details_block(
                            f"Heatmaps <span class='pill'>{len(heat)} plots</span>",
                            "<p class='note'>Top features/pathways across clusters.</p>"
                            + _grid_block([render_image_block(p) for p in heat]),
                            open_by_default=False,
                            extra_class="subsection",
                        ))
                    if dots:
                        dec_parts.append(_details_block(
                            f"Dotplots <span class='pill'>{len(dots)} plots</span>",
                            "<p class='note'>Compact overview of activity patterns.</p>"
                            + _grid_block([render_image_block(p) for p in dots]),
                            open_by_default=False,
                            extra_class="subsection",
                        ))

                    if bars and not OMIT_DECOUPLER_BARPLOTS:
                        # REQUIRED: collapsed by default
                        dec_parts.append(_details_block(
                            f"Barplots <span class='pill'>{len(bars)} plots</span>",
                            "<p class='note'>Per-cluster top-N features. Collapsed by default to reduce clutter.</p>"
                            + _grid_block([render_image_block(p) for p in bars]),
                            open_by_default=False,
                            extra_class="subsection",
                        ))

                    if other:
                        dec_parts.append(_details_block(
                            f"Other <span class='pill'>{len(other)} plots</span>",
                            _grid_block([render_image_block(p) for p in other]),
                            open_by_default=False,
                            extra_class="subsection",
                        ))

                    inner_parts.append(_details_block(
                        f"Decoupler <span class='pill'>{len(sec_imgs)} plots</span>",
                        "".join(dec_parts),
                        open_by_default=False,
                        extra_class="subsection",
                    ))
                    continue

                # Non-decoupler sections
                section_note = {
                    "Clustering": "Resolution selection and clustering diagnostics (sweep, stability, UMAPs).",
                    "Annotation": "CellTypist / cluster label diagnostics and summaries.",
                    "Compaction": "Compaction mapping and compacted-round diagnostics (if enabled).",
                    "Other": "",
                }.get(title, "")

                inner = ""
                if section_note:
                    inner += f"<p class='note'>{_safe(section_note)}</p>"
                inner += _grid_block([render_image_block(p) for p in sec_imgs])

                inner_parts.append(_details_block(
                    f"{_safe(title)} <span class='pill'>{len(sec_imgs)} plots</span>",
                    inner,
                    open_by_default=False,
                    extra_class="subsection",
                ))

            blocks.append(_details_block(
                f"Round: {_safe(rid)} <span class='pill'>{len(imgs)} plots</span>",
                "".join(inner_parts),
                open_by_default=open_round,
            ))

    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom cluster-and-annotate report</title>
      <style>{_css()}</style>
    </head>
    <body>
      <div class="wrap">
        {_toc(toc_sections)}
        <main class="content">
          {''.join(blocks)}
        </main>
      </div>
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")
