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
    sample_dir: Path = typer.Option(...,help="Sample directory path"),
    output_dir: Path = typer.Option(...,help="Output directory path"),
    metadata_tsv: Path = typer.Option(None, help="Metadata tsv file path"),
    cellbender_dir: Path = typer.Option(None,help="Cellbender directory path"),
    n_jobs: int = typer.Option(4,help="Number of (scrublet) jobs to run in parallel"),
    min_cells: int = typer.Option(3,help="Cut-off for minimum number of cells per gene"),
    min_genes: int = typer.Option(200,help="Cut-off for minimum number of genes per cell"),
    min_genes_prefilter: int = typer.Option(100, help="Prefilter: remove barcodes with fewer genes before QC"),
    min_umis_prefilter: int = typer.Option(50, help="Prefilter: remove barcodes with fewer UMIs before QC"),
    min_cells_per_sample: int = typer.Option(20,help="Cut-off for minimal number of cells per sample"),
    max_pct_mt: float = typer.Option(5.0, help="Percentage cut-off for mitchondrial reads"),
    make_figures: bool = typer.Option(True,help="Should we plot the figures"),
    batch_key: str = typer.Option("sample", help="Observation column identifying batches/samples"),
    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix", help="Raw feature matrix file name pattern"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output", help="Cellbender filtered matrix folder name pattern"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out.h5",help="Cellbender filtered matrix h5 file name suffix"),
):
    cfg = LoadAndQCConfig(
        sample_dir=sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        output_dir=output_dir,
        n_jobs=n_jobs,
        min_cells=min_cells,
        min_genes=min_genes,
        min_genes_prefilter=min_genes_prefilter,
        min_umis_prefilter=min_umis_prefilter,
        min_cells_per_sample=min_cells_per_sample,
        max_pct_mt=max_pct_mt,
        make_figures=make_figures,
        batch_key=batch_key,
        raw_pattern=raw_pattern,
        cellbender_pattern=cellbender_pattern,
        cellbender_h5_suffix=cellbender_h5_suffix,
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