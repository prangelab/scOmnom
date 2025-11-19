from __future__ import annotations
from typing import Optional, List
import typer
from pathlib import Path
from .load_and_filter import run_load_and_filter
from .integrate import run_integration
from .cluster_and_annotate import run_clustering
from .config import LoadAndQCConfig
from .cell_qc import run_cell_qc
from .logging_utils import init_logging

ALLOWED_METHODS = {"scVI", "Scanorama", "Harmony"}
app = typer.Typer(help="scOmnom CLI")

# Globally supress some warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique",
    category=UserWarning,
    module="anndata"
)
warnings.filterwarnings(
    "ignore",
    message=".*not compatible with tight_layout.*",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
    module="lightning.pytorch"
)


def _normalize_methods(methods):
    """
    Normalize and validate integration method inputs.

    Supports:
    --methods A --methods B
    --methods A,B,C
    or mixed.

    Returns a list of validated method names or None.
    """
    if methods is None:
        return None

    # Expand comma-separated entries
    expanded = []
    for m in methods:
        expanded.extend([x.strip() for x in m.split(",") if x.strip()])

    # Validate
    invalid = [m for m in expanded if m not in ALLOWED_METHODS]
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_METHODS))
        bad = ", ".join(invalid)
        raise ValueError(
            f"Invalid method(s): {bad}. Allowed methods: {allowed}"
        )

    return expanded



# ----------------------------------------------------------
# Standalone: scomnom cell-qc
# ----------------------------------------------------------
@app.command("cell-qc", help="Generate QC comparisons between raw, Cell Ranger–filtered, and CellBender matrices.")
def cell_qc(
    # Input directories (0–3 provided)
    raw: Optional[Path] = typer.Option(
        None, "--raw", "-r",
        help="Directory containing raw 10x matrices (*.raw_feature_bc_matrix)"
    ),
    filtered: Optional[Path] = typer.Option(
        None, "--filtered", "-f",
        help="Directory containing filtered 10x matrices (*.filtered_feature_bc_matrix)"
    ),
    cellbender: Optional[Path] = typer.Option(
        None, "--cellbender", "-c",
        help="Directory containing CellBender outputs (*.cellbender_filtered.output)"
    ),

    # Output
    output_dir: Path = typer.Option(
        ..., "--out", "-o",
        help="Output directory containing figures/"
    ),

    # optional: control figure formats
    figure_formats: List[str] = typer.Option(
        ["png", "pdf"],
        "--format",
        help="Figure formats to export (png, pdf, svg)"
    ),
):
    """
    Standalone QC module for comparing multiple input count matrices.

    Supports:
    - Raw 10x data (before and after knee+GMM filtering)
    - Cell Ranger filtered matrices
    - CellBender matrices

    Generates:
    - Read count comparisons
    - Knee/GMM UMI plots
    - Cell count per sample comparisons

    Useful for evaluating barcode calling and dataset quality **before** running full preprocessing.
    """

    from .config import CellQCConfig
    from .cell_qc import run_cell_qc

    logfile = output_dir / "cell_qc.log"
    init_logging(logfile)

    # --- sanity checks ---
    if (raw is None) and (filtered is None) and (cellbender is None):
        raise typer.BadParameter(
            "Provide at least one input: --raw, --filtered, or --cellbender"
        )

    # create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- build config ---
    cfg = CellQCConfig(
        output_dir=output_dir,
        figdir_name="figures",
        figure_formats=figure_formats,
        raw_sample_dir=raw,
        filtered_sample_dir=filtered,
        cellbender_dir=cellbender,
        make_figures=True,
    )

    # --- run the module ---
    run_cell_qc(cfg)

@app.command(
    "load-and-filter",
    help="Run the full scOmnom preprocessing pipeline to load and filter a dataset"
)
def load_and_filter(
    raw_sample_dir: Path = typer.Option(None, help="Path with <sample>.raw_feature_bc_matrix folders"),
    filtered_sample_dir: Path = typer.Option(None, help="Path with <sample>.filtered_feature_bc_matrix folders (Cell Ranger output)"),
    cellbender_dir: Path = typer.Option(None, help="Path with <sample>.cellbender_filtered.output folders"),
    output_dir: Path = typer.Option(
        ...,
        "--out",
        "-o",
        help="Output directory (required). Will contain h5ad + figures/",
    ),

    metadata_tsv: Path = typer.Option(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Sample-level metadata TSV (required)",
    ),
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
    batch_key: str = typer.Option(None),
    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix"),
    filtered_pattern: str = typer.Option("*.filtered_feature_bc_matrix"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out.h5"),
    min_genes_prefilter: int = typer.Option(50),
    min_umis_prefilter: int = typer.Option(20),
):
    """
        Run the complete scOmnom preprocessing pipeline.

        This includes:
        - Loading raw, Cell Ranger filtered, or CellBender matrices (exactly one source)
        - Sample merging
        - Metadata integration
        - QC metric computation
        - Doublet detection
        - Filtering (genes, cells, mitochondrial content)
        - Normalization and log transform
        - HVG selection, PCA, neighbors, UMAP
        - Leiden clustering
        - Full QC figure panel
        - Writing the final preprocessed h5ad

        Typical usage:
            scOmnom load-and-filter --raw-sample-dir data/raw --metadata metadata.tsv --out results/
        """
    if raw_sample_dir and filtered_sample_dir:
        raise typer.BadParameter("Cannot specify both --raw-sample-dir and --filtered-sample-dir")
    if filtered_sample_dir and cellbender_dir:
        raise typer.BadParameter(
            "Invalid input: CellBender outputs cannot be combined with Cell Ranger filtered matrices. "
            "Use --raw-sample-dir instead."
        )

    logfile = output_dir / "load-and-filter.log"
    init_logging(logfile)

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

    run_load_and_filter(cfg, logfile)

from .config import IntegrationConfig
from .integrate import run_integration

@app.command(
    "integrate",
    help="Run batch correction and scIB benchmarking on a preprocessed h5ad."
)
def integrate(
    input_path: Path = typer.Option(
        ...,
        help="Input h5ad produced by load_and_filter (typically adata.preprocessed.h5ad)"
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Output integrated h5ad. Defaults to <input_stem>.integrated.h5ad"
    ),
    methods: Optional[List[str]] = typer.Option(
        None,
        "--methods",
        "-m",
        help=(
            "Integration methods to run. Repeat option for multiple.\n"
            "Supported: Scanorama, Harmony, scVI, scANVI.\n"
            "Default: all except scANVI."
        ),
        case_sensitive=False,
        show_default=False,
        autocompletion=lambda: ALLOWED_METHODS,
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        help="Batch column in .obs (default: auto-detect)"
    ),
    label_key: str = typer.Option(
        "leiden",
        help="Label/cluster column for scib-metrics (default: leiden)"
    ),
    benchmark_n_jobs: int = typer.Option(
        4,
        help="Parallel workers for scib-metrics"
    ),
):
    """
        Perform integration on a preprocessed AnnData object.

        Features:
        - Run selected integration methods (Scanorama, Harmony, scVI, scANVI)
        - Compute scIB metrics for objective method comparison
        - Write an integrated dataset and metrics report

        Use after:
            scOmnom load-and-filter
        """
    methods = _normalize_methods(methods)

    logfile = input_path.parent / "integrate.log"
    init_logging(logfile)

    cfg = IntegrationConfig(
        input_path=input_path,
        output_path=output_path,
        methods=methods,
        batch_key=batch_key,
        label_key=label_key,
        benchmark_n_jobs=benchmark_n_jobs,
        logfile=logfile,
    )

    run_integration(cfg)


@app.command(
    "cluster-and-annotate",
    help="Generate clusters and annotations on an integrated dataset."
)
def cluster_and_annotate():
    """
       Run clustering and annotation on an already preprocessed h5ad.

       This step:
       - Loads the preprocessed AnnData
       - Performs Leiden clustering (and optionally other clusterings)
       - Generates UMAP/cluster plots
       - Optionally performs marker-based annotation (if configured)
       - Writes an updated annotated h5ad

       Use after:
           scOmnom load-and-filter
       """

    logfile = output_dir / "cluster-and-annotate.log"
    init_logging(logfile)

    run_clustering()

if __name__ == "__main__":
    app()