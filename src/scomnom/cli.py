from __future__ import annotations
import typer
from pathlib import Path
from .load_and_qc import run_load_and_qc
from .integrate import run_integration
from .cluster_and_annotate import run_clustering
from .config import LoadAndQCConfig

app = typer.Typer(help="scOmnom CLI")

@app.command()
@app.command()
def load_and_qc(
    raw_sample_dir: Path = typer.Option(None, help="Path with <sample>.raw_feature_bc_matrix folders"),
    filtered_sample_dir: Path = typer.Option(None, help="Path with <sample>.filtered_feature_bc_matrix folders (Cell Ranger output)"),
    cellbender_dir: Path = typer.Option(None, help="Path with <sample>.cellbender_filtered.output folders"),
    output_dir: Path = typer.Option(None, help="Output directory"),
    metadata_tsv: Path = typer.Option(None, help="Sample-level metadata TSV"),
    n_jobs: int = typer.Option(4),
    min_cells: int = typer.Option(3),
    min_genes: int = typer.Option(200),
    min_cells_per_sample: int = typer.Option(20),
    max_pct_mt: float = typer.Option(5.0),
    make_figures: bool = typer.Option(True),
    figure_format: list[str] = typer.Option(
        ["png", "pdf"],
        help="Figure formats to save. Repeat option for multiple formats, e.g. --figure-format png --figure-format pdf",
    ),
    batch_key: str = typer.Option("sample"),
    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix"),
    filtered_pattern: str = typer.Option("*.filtered_feature_bc_matrix"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out.h5"),
    min_genes_prefilter: int = typer.Option(50),
    min_umis_prefilter: int = typer.Option(20),
):
    if raw_sample_dir and filtered_sample_dir:
        raise typer.BadParameter("Cannot specify both --raw-sample-dir and --filtered-sample-dir")
    if filtered_sample_dir and cellbender_dir:
        raise typer.BadParameter(
            "Invalid input: CellBender outputs cannot be combined with Cell Ranger filtered matrices. "
            "Use --raw-sample-dir instead."
        )
    cfg = LoadAndQCConfig(
        raw_sample_dir=raw_sample_dir,
        filtered_sample_dir=filtered_sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        output_dir=output_dir,
        n_jobs=n_jobs,
        min_cells=min_cells,
        min_genes=min_genes,
        min_cells_per_sample=min_cells_per_sample,
        max_pct_mt=max_pct_mt,
        make_figures=make_figures,
        figure_formats=figure_format,
        batch_key=batch_key,
        raw_pattern=raw_pattern,
        filtered_pattern=filtered_pattern,
        cellbender_pattern=cellbender_pattern,
        cellbender_h5_suffix=cellbender_h5_suffix,
        min_genes_prefilter=min_genes_prefilter,
        min_umis_prefilter=min_umis_prefilter,
    )

    run_load_and_qc(cfg, logfile=output_dir / "pipeline.log")

from .config import IntegrationConfig
from .integrate import run_integration

@app.command()
def integrate(
    input_path: Path = typer.Option(
        ...,
        help="Input h5ad produced by load_and_qc (typically adata.preprocessed.h5ad)"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Output integrated h5ad. Defaults to <input_stem>.integrated.h5ad"
    ),
    methods: Optional[List[str]] = typer.Option(
        None,
        help=(
            "Integration methods to run. Repeat option for multiple.\n"
            "Supported: Scanorama, LIGER, Harmony, scVI, scANVI.\n"
            "Default: all except scANVI."
        ),
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        help="Batch column in .obs (default: auto-detect)"
    ),
    label_key: str = typer.Option(
        "leiden",
        help="Label/cluster column for scib-metrics (default: leiden)"
    ),
    use_gpu: bool = typer.Option(
        False,
        help="Run scVI/scANVI on GPU"
    ),
    include_scanvi: bool = typer.Option(
        False,
        help="Include scANVI in benchmarking (slow without GPU)"
    ),
    benchmark_n_jobs: int = typer.Option(
        1,
        help="Parallel workers for scib-metrics"
    ),
):
    """
    Run integration + scib-metrics benchmarking and write an integrated h5ad.
    """

    logfile = input_path.parent / "integration.log"

    cfg = IntegrationConfig(
        input_path=input_path,
        output_path=output_path,
        methods=methods,
        batch_key=batch_key,
        label_key=label_key,
        use_gpu=use_gpu,
        include_scanvi=include_scanvi,
        benchmark_n_jobs=benchmark_n_jobs,
        logfile=logfile,
    )

    run_integration(cfg)


@app.command()
def cluster_and_annotate():
    run_clustering()

if __name__ == "__main__":
    app()