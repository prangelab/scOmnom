from pathlib import Path
from datetime import datetime
import html
import json

from collections import defaultdict
from typing import List, Dict


# ======================================================================
# Public API
# ======================================================================

def generate_qc_report(
    *,
    fig_root: Path,
    cfg,
    version: str,
    adata,
):
    """
    Generate a self-contained HTML QC report embedding all plots.

    Output:
      <fig_root>/report.html
    """

    fig_root = fig_root.resolve()
    png_root = fig_root / "png"
    out_html = fig_root / "qc_report.html"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Collect images (PNG only)
    # ------------------------------------------------------------------
    images = sorted(png_root.rglob("*.png"))
    rel_images = [img.relative_to(fig_root) for img in images]

    # ------------------------------------------------------------------
    # High-level section classification
    # ------------------------------------------------------------------
    sections = {
        "Doublets": [],
        "CellBender": [],
        "QC metrics": [],
        "QC scatter": [],
        "Overview": [],
        "Other": [],
    }

    for p in rel_images:
        s = p.as_posix()
        if "doublets" in s:
            sections["Doublets"].append(p)
        elif "cellbender" in s:
            sections["CellBender"].append(p)
        elif "qc_metrics" in s:
            sections["QC metrics"].append(p)
        elif "qc_scatter" in s:
            sections["QC scatter"].append(p)
        elif "overview" in s:
            sections["Overview"].append(p)
        else:
            sections["Other"].append(p)

    # ------------------------------------------------------------------
    # CSS
    # ------------------------------------------------------------------
    css = """
    body {
      font-family: system-ui, -apple-system, sans-serif;
      margin: 2rem;
      max-width: 1400px;
    }

    h1, h2, h3, h4 {
      margin-top: 1.5rem;
    }

    details > summary {
      cursor: pointer;
      font-weight: 600;
      margin: 0.5rem 0;
    }

    .meta {
      background: #f6f8fa;
      border: 1px solid #ddd;
      padding: 1rem;
      border-radius: 6px;
      font-family: monospace;
      white-space: pre-wrap;
    }

    figure {
      margin: 0.5rem;
    }

    figure img {
      max-width: 100%;
      border: 1px solid #ddd;
      border-radius: 4px;
    }

    figcaption {
      font-size: 0.85rem;
      color: #444;
      margin-top: 0.25rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 1rem;
    }

    table.summary {
      border-collapse: collapse;
      margin-top: 1rem;
      margin-bottom: 2rem;
    }

    table.summary th,
    table.summary td {
      border: 1px solid #ccc;
      padding: 0.4rem 0.6rem;
      text-align: left;
    }

    table.summary th {
      background: #f0f0f0;
    }

    .raw-cb-block {
      margin-bottom: 2rem;
    }

    .raw-cb-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      align-items: start;
    }

    .img-missing {
      border: 1px dashed #bbb;
      padding: 2rem;
      text-align: center;
      color: #888;
      font-style: italic;
    }
    """

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    cfg_json = json.dumps(cfg.__dict__, indent=2, default=str)

    header = f"""
    <h1>scOmnom QC report</h1>

    <div class="meta">
    Version:   {version}
    Timestamp: {timestamp}

    Parameters:
    {html.escape(cfg_json)}
    </div>
    """

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    qc_summary = _collect_qc_summary(adata)

    rows = []
    for k, v in qc_summary.items():
        rows.append(
            f"<tr><td>{k}</td><td>{v:.4g}</td></tr>"
            if isinstance(v, float)
            else f"<tr><td>{k}</td><td>{v}</td></tr>"
        )

    summary_html = f"""
    <h2>Summary statistics</h2>
    <table class="summary">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """

    # ------------------------------------------------------------------
    # Render body
    # ------------------------------------------------------------------
    body = [header, summary_html]

    for title, imgs in sections.items():
        if not imgs:
            continue

        body.append(f"<details open><summary><h2>{title}</h2></summary>")

        # --------------------------------------------------------------
        # QC metrics / scatter: PRE vs POST → RAW vs CB
        # --------------------------------------------------------------
        if title in {"QC metrics", "QC scatter"}:
            stage_groups = group_by_stage(imgs)

            for stage, stage_imgs in stage_groups.items():
                body.append(f"<h3>{stage.replace('_', ' ').title()}</h3>")

                raw_cb_groups = group_raw_cb(stage_imgs)

                for base_key, entry in sorted(raw_cb_groups.items()):
                    if entry["raw"] or entry["cb"]:
                        body.append(
                            render_raw_cb_block(
                                base_key=base_key,
                                entry=entry,
                                rel_root=fig_root,
                            )
                        )
                    else:
                        for p in entry["other"]:
                            body.append(render_image_block(p))

        # --------------------------------------------------------------
        # Everything else: flat grid
        # --------------------------------------------------------------
        else:
            body.append('<div class="grid">')
            for p in imgs:
                body.append(render_image_block(p))
            body.append("</div>")

        body.append("</details>")

    # ------------------------------------------------------------------
    # Final HTML
    # ------------------------------------------------------------------
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom QC report</title>
      <style>{css}</style>
    </head>
    <body>
      {''.join(body)}
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")


# ======================================================================
# Helpers
# ======================================================================

def _caption_from_filename(fname: str) -> str:
    name = Path(fname).stem
    name = name.replace("_", " ")
    name = name.replace("postfilter", "post-filter")
    name = name.replace("prefilter", "pre-filter")
    return name.capitalize()


def render_image_block(img_path: Path) -> str:
    caption = _caption_from_filename(img_path.name)
    return f"""
    <figure>
      <img src="{img_path.as_posix()}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def group_by_stage(files: List[Path]) -> Dict[str, List[Path]]:
    groups = defaultdict(list)
    for f in files:
        name = f.name
        if "prefilter" in name:
            groups["pre-filter"].append(f)
        elif "postfilter" in name:
            groups["post-filter"].append(f)
        else:
            groups["other"].append(f)
    return groups


def group_raw_cb(files: List[Path]) -> Dict[str, dict]:
    groups = defaultdict(lambda: {"raw": None, "cb": None, "other": []})
    for f in files:
        stem = f.stem
        if stem.endswith("_raw"):
            groups[stem[:-4]]["raw"] = f
        elif stem.endswith("_cb"):
            groups[stem[:-3]]["cb"] = f
        else:
            groups[stem]["other"].append(f)
    return groups


def render_raw_cb_block(base_key: str, entry: dict, rel_root: Path) -> str:
    def slot(p: Path | None, label: str):
        if p is None:
            return f"<div class='img-missing'>{label}<br>(missing)</div>"
        return f"""
        <figure>
          <img src="{p.as_posix()}">
          <figcaption>{label}</figcaption>
        </figure>
        """

    title = base_key.replace("_", " ")

    return f"""
    <div class="raw-cb-block">
      <h4>{html.escape(title)}</h4>
      <div class="raw-cb-grid">
        {slot(entry["raw"], "Raw")}
        {slot(entry["cb"], "CellBender")}
      </div>
    </div>
    """


def _collect_qc_summary(adata) -> dict:
    obs = adata.obs
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

    info = adata.uns.get("doublet_calling")
    if info is not None:
        summary["doublet_observed_rate"] = info.get("observed_global_rate")

    for col in [
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
        "pct_counts_ribo",
        "pct_counts_hb",
    ]:
        if col in obs:
            summary[f"median_{col}"] = float(obs[col].median())
            summary[f"mean_{col}"] = float(obs[col].mean())

    return summary


# ======================================================================
# Integration report
# ======================================================================
def generate_integration_report(
    *,
    fig_root: Path,
    version: str,
    adata,
    batch_key: str,
    label_key: str,
    methods: list[str],
    selected_embedding: str,
    benchmark_n_jobs: int,
):
    """
    Generate a self-contained HTML integration report embedding all plots.

    Output:
      <fig_root>/integration_report.html
    """

    fig_root = fig_root.resolve()
    png_root = fig_root / "png"
    out_html = fig_root / "integration_report.html"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Collect images (PNG only)
    # ------------------------------------------------------------------
    images = sorted(png_root.rglob("*.png"))
    rel_images = [img.relative_to(fig_root) for img in images]

    # ------------------------------------------------------------------
    # Section classification
    # ------------------------------------------------------------------
    sections: Dict[str, List[Path]] = {
        "Integration benchmarking": [],
        "UMAPs": [],
        "Other": [],
    }

    for p in rel_images:
        s = p.as_posix().lower()
        if "scib" in s:
            sections["Integration benchmarking"].append(p)
        elif "umap" in s:
            sections["UMAPs"].append(p)
        else:
            sections["Other"].append(p)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    header = f"""
    <h1>scOmnom Integration Report</h1>

    <div class="meta">
    Version:   {version}
    Timestamp: {timestamp}

    Selected embedding:
    {html.escape(selected_embedding)}
    </div>
    """

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    summary = {
        "version": version,
        "timestamp": timestamp,
        "batch_key": batch_key,
        "label_key": label_key,
        "methods_requested": methods,
        "benchmark_n_jobs": benchmark_n_jobs,
        "selected_embedding": selected_embedding,
    }

    rows = []
    for k, v in summary.items():
        rows.append(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>")

    summary_html = f"""
    <h2>Summary</h2>
    <table class="summary">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """

    # ------------------------------------------------------------------
    # Render body
    # ------------------------------------------------------------------
    body = [header, summary_html]

    for title, imgs in sections.items():
        if not imgs:
            continue

        body.append(f"<details open><summary><h2>{title}</h2></summary>")
        body.append('<div class="grid">')

        for p in imgs:
            body.append(render_image_block(p))

        body.append("</div></details>")

    # ------------------------------------------------------------------
    # Final HTML
    # ------------------------------------------------------------------
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom Integration Report</title>
      <style>{_REPORT_CSS}</style>
    </head>
    <body>
      {''.join(body)}
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")


# ----------------------------------------------------------------------
# Shared CSS (reuse QC report style verbatim)
# ----------------------------------------------------------------------
_REPORT_CSS = """
body {
  font-family: system-ui, -apple-system, sans-serif;
  margin: 2rem;
  max-width: 1400px;
}

h1, h2, h3, h4 {
  margin-top: 1.5rem;
}

details > summary {
  cursor: pointer;
  font-weight: 600;
  margin: 0.5rem 0;
}

.meta {
  background: #f6f8fa;
  border: 1px solid #ddd;
  padding: 1rem;
  border-radius: 6px;
  font-family: monospace;
  white-space: pre-wrap;
}

figure {
  margin: 0.5rem;
}

figure img {
  max-width: 100%;
  border: 1px solid #ddd;
  border-radius: 4px;
}

figcaption {
  font-size: 0.85rem;
  color: #444;
  margin-top: 0.25rem;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 1rem;
}

table.summary {
  border-collapse: collapse;
  margin-top: 1rem;
  margin-bottom: 2rem;
}

table.summary th,
table.summary td {
  border: 1px solid #ccc;
  padding: 0.4rem 0.6rem;
  text-align: left;
}

table.summary th {
  background: #f0f0f0;
}
"""


from __future__ import annotations

from pathlib import Path
from datetime import datetime
import html
import json

from collections import defaultdict
from typing import Any, Dict, List, Tuple


# ======================================================================
# Cluster+Annotate report
# ======================================================================

def generate_cluster_and_annotate_report(
    *,
    fig_root: Path,
    cfg,
    version: str,
    adata,
) -> None:
    """
    Generate a self-contained HTML report for the cluster-and-annotate module
    embedding all plots.

    Output:
      <fig_root>/cluster_and_annotate_report.html

    Assumes the standard fig layout:
      <fig_root>/png/**.png
      <fig_root>/pdf/**.pdf
    """
    fig_root = Path(fig_root).resolve()
    png_root = fig_root / "png"
    out_html = fig_root / "cluster_and_annotate_report.html"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Collect images (PNG only)
    # ------------------------------------------------------------------
    images = sorted(png_root.rglob("*.png"))
    rel_images = [img.relative_to(fig_root) for img in images]

    # ------------------------------------------------------------------
    # Round grouping (based on path: .../cluster_and_annotate/<round_id>/...)
    # ------------------------------------------------------------------
    by_round: Dict[str, List[Path]] = defaultdict(list)

    for p in rel_images:
        parts = p.parts
        # Expect: png/cluster_and_annotate/<round_id>/...
        rid = _extract_round_id_from_relpath(parts)
        by_round[rid].append(p)

    # Use stored round order if available
    round_order = []
    try:
        ro = adata.uns.get("cluster_round_order", None)
        if isinstance(ro, (list, tuple)) and ro:
            round_order = [str(x) for x in ro]
    except Exception:
        round_order = []

    # Ensure stable order with unknowns appended
    ordered_rounds = []
    seen = set()
    for rid in round_order:
        if rid in by_round:
            ordered_rounds.append(rid)
            seen.add(rid)
    for rid in sorted(by_round.keys()):
        if rid not in seen:
            ordered_rounds.append(rid)

    active_round = adata.uns.get("active_cluster_round", None)
    active_round = str(active_round) if active_round is not None else None

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    cfg_json = json.dumps(getattr(cfg, "__dict__", {}), indent=2, default=str)

    header = f"""
    <h1>scOmnom cluster-and-annotate report</h1>

    <div class="meta">
    Version:   {html.escape(str(version))}
    Timestamp: {html.escape(str(timestamp))}

    Active round: {html.escape(str(active_round))}

    Round order:
    {html.escape(json.dumps(ordered_rounds, indent=2))}

    Parameters:
    {html.escape(cfg_json)}
    </div>
    """

    # ------------------------------------------------------------------
    # Summary (round table)
    # ------------------------------------------------------------------
    summary_html = _render_cluster_round_summary(adata, ordered_rounds)

    # ------------------------------------------------------------------
    # Render body
    # ------------------------------------------------------------------
    body: List[str] = [header, summary_html]

    if not ordered_rounds:
        body.append("<p><em>No cluster_and_annotate figures found under png/.</em></p>")
    else:
        for rid in ordered_rounds:
            imgs = by_round.get(rid, [])
            if not imgs:
                continue

            rid_label = html.escape(str(rid))
            open_attr = " open" if (active_round and rid == active_round) else " open"

            body.append(f"<details{open_attr}><summary><h2>Round: {rid_label}</h2></summary>")

            sections = _classify_cluster_annotate_images(imgs)

            for title, sec_imgs in sections:
                if not sec_imgs:
                    continue

                body.append(f"<h3>{html.escape(title)}</h3>")
                body.append('<div class="grid">')
                for p in sec_imgs:
                    body.append(render_image_block(p))
                body.append("</div>")

            body.append("</details>")

    # ------------------------------------------------------------------
    # Final HTML
    # ------------------------------------------------------------------
    html_doc = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>scOmnom cluster-and-annotate report</title>
      <style>{_REPORT_CSS}</style>
    </head>
    <body>
      {''.join(body)}
    </body>
    </html>
    """

    out_html.write_text(html_doc, encoding="utf-8")


# ======================================================================
# Helpers (cluster+annotate report)
# ======================================================================

def _extract_round_id_from_relpath(parts: Tuple[str, ...]) -> str:
    """
    Extract round_id from a relative png path.

    Expected patterns:
      png/cluster_and_annotate/<round_id>/...
      png/<anything>/<round_id>/...   (fallback)

    Returns "unknown" if not found.
    """
    try:
        # common: ("png", "cluster_and_annotate", "<rid>", ...)
        if len(parts) >= 3 and parts[0] == "png" and parts[1] == "cluster_and_annotate":
            return str(parts[2])

        # fallback: find "cluster_and_annotate" anywhere
        if "cluster_and_annotate" in parts:
            i = list(parts).index("cluster_and_annotate")
            if i + 1 < len(parts):
                return str(parts[i + 1])

        return "unknown"
    except Exception:
        return "unknown"


def _classify_cluster_annotate_images(imgs: List[Path]) -> List[Tuple[str, List[Path]]]:
    """
    Group round images into stable, readable sections.
    """
    clustering: List[Path] = []
    annotation: List[Path] = []
    decoupler: List[Path] = []
    compaction: List[Path] = []
    other: List[Path] = []

    for p in imgs:
        s = p.as_posix().lower()

        # folder-based hints
        if "/clustering/" in s:
            clustering.append(p)
            continue
        if "/annotation/" in s:
            annotation.append(p)
            continue
        if "/decoupler/" in s:
            decoupler.append(p)
            continue
        if "compaction" in s or "compacted" in s and ("flow" in s or "merge" in s or "component" in s):
            compaction.append(p)
            continue

        # filename-based hints
        if "umap" in s or "cluster_tree" in s or "cluster_flow" in s:
            clustering.append(p)
        elif "celltypist" in s or "cluster_label" in s and "umap" in s:
            annotation.append(p)
        elif "heatmap" in s or "dotplot" in s or "barplot" in s or "decoupler" in s:
            decoupler.append(p)
        elif "compaction" in s or "compacted" in s:
            compaction.append(p)
        else:
            other.append(p)

    # Keep deterministic ordering within sections
    clustering = sorted(clustering, key=lambda x: x.as_posix())
    annotation = sorted(annotation, key=lambda x: x.as_posix())
    decoupler = sorted(decoupler, key=lambda x: x.as_posix())
    compaction = sorted(compaction, key=lambda x: x.as_posix())
    other = sorted(other, key=lambda x: x.as_posix())

    return [
        ("Clustering", clustering),
        ("Annotation", annotation),
        ("Decoupler", decoupler),
        ("Compaction", compaction),
        ("Other", other),
    ]


def _render_cluster_round_summary(adata, ordered_rounds: List[str]) -> str:
    """
    Render a small summary table per round (best-effort).
    """
    rounds = adata.uns.get("cluster_rounds", {})
    if not isinstance(rounds, dict) or not ordered_rounds:
        return """
        <h2>Summary</h2>
        <p><em>No round metadata available.</em></p>
        """

    rows = []
    for rid in ordered_rounds:
        r = rounds.get(rid, None)
        if not isinstance(r, dict):
            continue

        labels_obs_key = r.get("labels_obs_key", "")
        cluster_key = r.get("cluster_key", "")
        n_clusters = ""
        try:
            if labels_obs_key and labels_obs_key in adata.obs:
                n_clusters = int(adata.obs[str(labels_obs_key)].astype(str).nunique())
        except Exception:
            n_clusters = ""

        parent = r.get("parent_round_id", "") or r.get("parent", "") or ""
        notes = r.get("notes", "") or ""

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(rid))}</td>"
            f"<td>{html.escape(str(parent))}</td>"
            f"<td>{html.escape(str(labels_obs_key))}</td>"
            f"<td>{html.escape(str(cluster_key))}</td>"
            f"<td>{html.escape(str(n_clusters))}</td>"
            f"<td>{html.escape(str(notes))}</td>"
            "</tr>"
        )

    if not rows:
        return """
        <h2>Summary</h2>
        <p><em>No round metadata rows.</em></p>
        """

    return f"""
    <h2>Summary</h2>
    <table class="summary">
      <thead>
        <tr>
          <th>round_id</th>
          <th>parent</th>
          <th>labels_obs_key</th>
          <th>cluster_key</th>
          <th>n_clusters</th>
          <th>notes</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """
