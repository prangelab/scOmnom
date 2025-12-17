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
    out_html = fig_root / "report.html"

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
