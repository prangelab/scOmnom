from __future__ import annotations
import typer
from pathlib import Path
from .load_and_qc import run_load_and_qc
from .integrate import run_integration
from .cluster_and_annotate import run_clustering
from .config import LoadAndQCConfig

app = typer.Typer(help="scOmnom CLI")

@app.command()
def load_and_qc(
    sample_dir: Path = typer.Option(..., help="Path with <sample>.raw_feature_bc_matrix folders"),
    output_dir: Path = typer.Option(..., help="Output directory"),
    metadata_tsv: Path = typer.Option(None, help="Sample-level metadata TSV", dir_okay=False, file_okay=True, exists=False),
    cellbender_dir: Path = typer.Option(None, help="Path with <sample>.cellbender_filtered.output folders"),
    n_jobs: int = typer.Option(4),
    min_cells: int = typer.Option(3),
    min_genes: int = typer.Option(200),
    min_cells_per_sample: int = typer.Option(20),
    max_pct_mt: float = typer.Option(5.0),
    make_figures: bool = typer.Option(True),
):
    cfg = LoadAndQCConfig(
        sample_dir=sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        output_dir=output_dir,
        n_jobs=n_jobs,
        min_cells=min_cells,
        min_genes=min_genes,
        min_cells_per_sample=min_cells_per_sample,
        max_pct_mt=max_pct_mt,
        make_figures=make_figures,
    )
    run_load_and_qc(cfg, logfile=output_dir / "pipeline.log")

@app.command()
def integrate():
    run_integration()

@app.command()
def cluster_and_annotate():
    run_clustering()

if __name__ == "__main__":
    app()