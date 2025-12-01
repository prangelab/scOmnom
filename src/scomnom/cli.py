from __future__ import annotations
from typing import Optional, List
import typer
from pathlib import Path
import warnings

from .cell_qc import run_cell_qc
from .load_and_filter import run_load_and_filter
from .integrate import run_integration
from .cluster_and_annotate import run_clustering

from .config import CellQCConfig
from .config import LoadAndQCConfig
from .config import IntegrationConfig
from .config import ClusterAnnotateConfig

from .logging_utils import init_logging


ALLOWED_METHODS = {"scVI", "scANVI", "Scanorama", "Harmony", "BBKNN"}
app = typer.Typer(help="scOmnom CLI")

# Globally suppress some warnings
warnings.filterwarnings(
    "ignore",
    message="Variable names are not unique",
    category=UserWarning,
    module="anndata",
)
warnings.filterwarnings(
    "ignore",
    message=".*not compatible with tight_layout.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*does not have many workers which may be a bottleneck.*",
    category=UserWarning,
    module="lightning.pytorch",
)
warnings.filterwarnings(
    "ignore",
    message=".*already log-transformed.*",
    category=UserWarning,
)


def _normalize_methods(methods):
    """
    Normalize and validate integration method inputs.
    Supports:
      --methods A --methods B
      --methods A,B,C
    """
    if methods is None:
        return None

    expanded = []
    for m in methods:
        expanded.extend([x.strip() for x in m.split(",") if x.strip()])

    invalid = [m for m in expanded if m not in ALLOWED_METHODS]
    if invalid:
        allowed = ", ".join(sorted(ALLOWED_METHODS))
        bad = ", ".join(invalid)
        raise ValueError(f"Invalid method(s): {bad}. Allowed methods: {allowed}")

    return expanded


# ======================================================================
#  cell-qc
# ======================================================================
@app.command("cell-qc", help="Generate QC comparisons between raw, CR-filtered, and CellBender matrices.")
def cell_qc(
    raw: Optional[Path] = typer.Option(
        None, "--raw", "-r",
        help="Directory with raw 10x matrices (*.raw_feature_bc_matrix)",
    ),
    filtered: Optional[Path] = typer.Option(
        None, "--filtered", "-f",
        help="Directory with CellRanger filtered matrices (*.filtered_feature_bc_matrix)",
    ),
    cellbender: Optional[Path] = typer.Option(
        None, "--cellbender", "-c",
        help="Directory with CellBender outputs (*.cellbender_filtered.output)",
    ),

    output_dir: Path = typer.Option(
        ..., "--out", "-o",
        help="Output directory containing figures/",
    ),

    figure_formats: List[str] = typer.Option(
        ["png", "pdf"],
        "--format", "-F",
        help="Figure formats to export (png, pdf, svg)",
    ),
):
    logfile = output_dir / "cell_qc.log"
    init_logging(logfile)

    if (raw is None) and (filtered is None) and (cellbender is None):
        raise typer.BadParameter("Provide at least one of --raw, --filtered, or --cellbender")

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CellQCConfig(
        output_dir=output_dir,
        figdir_name="figures",
        figure_formats=figure_formats,
        raw_sample_dir=raw,
        filtered_sample_dir=filtered,
        cellbender_dir=cellbender,
        make_figures=True,
    )
    run_cell_qc(cfg)


# ======================================================================
#  load-and-filter
# ======================================================================
@app.command("load-and-filter", help="Run the full scOmnom preprocessing pipeline.")
def load_and_filter(
    raw_sample_dir: Path = typer.Option(
        None, "--raw-sample-dir", "-r",
        help="Path with <sample>.raw_feature_bc_matrix folders",
    ),
    filtered_sample_dir: Path = typer.Option(
        None, "--filtered-sample-dir", "-f",
        help="Path with <sample>.filtered_feature_bc_matrix folders",
    ),
    cellbender_dir: Path = typer.Option(
        None, "--cellbender-dir", "-c",
        help="Path with <sample>.cellbender_filtered.output folders",
    ),

    output_dir: Path = typer.Option(
        ..., "--out", "-o",
        help="Output directory (required). Will contain h5ad + figures/",
    ),

    metadata_tsv: Path = typer.Option(
        ..., "--metadata-tsv", "-m",
        exists=True,
        help="Sample metadata TSV (required)",
    ),

    n_jobs: int = typer.Option(4),
    min_cells: int = typer.Option(3),
    min_genes: int = typer.Option(200),
    min_cells_per_sample: int = typer.Option(20),
    max_pct_mt: float = typer.Option(5.0),

    make_figures: bool = typer.Option(True),
    figure_format: List[str] = typer.Option(
        ["png", "pdf"],
        "--figure-format", "-F",
        help="Repeat for multiple formats",
    ),

    batch_key: str = typer.Option(None, "--batch-key", "-b"),

    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix"),
    filtered_pattern: str = typer.Option("*.filtered_feature_bc_matrix"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out.h5"),
):
    if raw_sample_dir and filtered_sample_dir:
        raise typer.BadParameter("Cannot specify both raw and filtered sample dirs")
    if filtered_sample_dir and cellbender_dir:
        raise typer.BadParameter("Cannot combine filtered and CellBender inputs")

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
    )

    run_load_and_filter(cfg, logfile)


# ======================================================================
#  integrate
# ======================================================================
@app.command("integrate", help="Run batch correction + scIB benchmarking.")
def integrate(
    input_path: Path = typer.Option(
        ..., "--input-path", "-i",
        help="Preprocessed h5ad from load-and-filter",
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output-path", "-o",
        help="Output integrated h5ad",
    ),

    methods: Optional[List[str]] = typer.Option(
        None, "--methods", "-m",
        help="Repeat or comma-separated list",
        case_sensitive=False,
    ),

    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="Batch column in .obs",
    ),
    label_key: str = typer.Option(
        "leiden", "--label-key", "-l",
        help="Label column for scIB metrics",
    ),

    benchmark_n_jobs: int = typer.Option(4),
):
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


# ======================================================================
#  cluster-and-annotate
# ======================================================================
@app.command("cluster-and-annotate", help="Cluster resolution sweep + optional CellTypist annotation.")
def cluster_and_annotate(
    input_path: Optional[Path] = typer.Option(
        None, "--input", "-i",
        help="Integrated h5ad file",
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output clustered/annotated h5ad",
    ),

    embedding_key: str = typer.Option(
        "X_integrated", "--embedding-key", "-e",
        help="Embedding key in .obsm",
    ),
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="Batch/sample column",
    ),
    label_key: str = typer.Option(
        "leiden", "--label-key", "-l",
        help="Final cluster label key",
    ),

    res_min: float = typer.Option(0.2, "--res-min", "-rmin"),
    res_max: float = typer.Option(2.0, "--res-max", "-rmax"),
    n_resolutions: int = typer.Option(10, "--n-resolutions", "-nres"),
    penalty_alpha: float = typer.Option(0.02),

    stability_repeats: int = typer.Option(5, "--stability-repeats", "-sr"),
    subsample_frac: float = typer.Option(0.8, "--subsample-frac", "-sf"),
    random_state: int = typer.Option(42, "--random-state", "-rs"),

    tiny_cluster_size: int = typer.Option(20),
    min_cluster_size: int = typer.Option(20),
    min_plateau_len: int = typer.Option(3),
    max_cluster_jump_frac: float = typer.Option(0.4),
    stability_threshold: float = typer.Option(0.85),

    w_stab: float = typer.Option(0.50),
    w_sil: float = typer.Option(0.35),
    w_tiny: float = typer.Option(0.15),

    celltypist_model: Optional[str] = typer.Option(
        "Immune_All_Low.pkl", "--celltypist-model", "-M",
        help="CellTypist model path/name",
    ),
    celltypist_majority_voting: bool = typer.Option(True),
    annotation_csv: Optional[Path] = typer.Option(
        None, "--annotation-csv", "-A",
        help="Write per-cluster annotation table",
    ),

    list_models: bool = typer.Option(
        False, "--list-models",
    ),
    download_models: bool = typer.Option(
        False, "--download-models",
    ),

    make_figures: bool = typer.Option(True),
    figure_format: List[str] = typer.Option(
        ["png", "pdf"], "--figure-format", "-F",
    ),
    figdir_name: str = typer.Option("figures", "--figdir-name", "-D"),
):
    # ---------------------------------------------------------
    # Handle --list-models
    # ---------------------------------------------------------
    if list_models:
        from .io_utils import get_available_celltypist_models
        import os
        typer.echo("\nAvailable CellTypist models:\n")

        models = get_available_celltypist_models()
        if not models:
            typer.echo("Unable to fetch model list (offline?).")
            raise typer.Exit()

        cache_dir = Path("~/.celltypist/data/models").expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Cache directory: {cache_dir}\n")

        for m in models:
            name = m["name"]
            cached = (cache_dir / name).exists()
            status = "✔ cached" if cached else "✘ not cached"
            typer.echo(f"  - {name:<35} [{status}]")

        raise typer.Exit()

    # ---------------------------------------------------------
    # Handle --download-models
    # ---------------------------------------------------------
    if download_models:
        from .io_utils import download_all_celltypist_models
        typer.echo("Downloading ALL CellTypist models...\n")
        try:
            download_all_celltypist_models()
        except Exception as e:
            typer.echo(f"Failed: {e}")
            raise typer.Exit(1)
        typer.echo("Done.")
        raise typer.Exit()

    # ---------------------------------------------------------
    # Normal mode
    # ---------------------------------------------------------
    if input_path is None:
        raise typer.BadParameter("Missing --input / -i")

    log_path = (
        output_path.parent / "cluster-and-annotate.log"
        if output_path is not None
        else input_path.parent / "cluster-and-annotate.log"
    )
    init_logging(log_path)

    cfg = ClusterAnnotateConfig(
        input_path=input_path,
        output_path=output_path,
        embedding_key=embedding_key,
        batch_key=batch_key,
        label_key=label_key,

        res_min=res_min,
        res_max=res_max,
        n_resolutions=n_resolutions,
        penalty_alpha=penalty_alpha,

        stability_repeats=stability_repeats,
        subsample_frac=subsample_frac,
        random_state=random_state,

        tiny_cluster_size=tiny_cluster_size,
        min_cluster_size=min_cluster_size,
        min_plateau_len=min_plateau_len,
        max_cluster_jump_frac=max_cluster_jump_frac,
        stability_threshold=stability_threshold,

        w_stab=w_stab,
        w_sil=w_sil,
        w_tiny=w_tiny,

        celltypist_model=celltypist_model,
        celltypist_majority_voting=celltypist_majority_voting,
        annotation_csv=annotation_csv,

        make_figures=make_figures,
        figure_formats=figure_format,
        figdir_name=figdir_name,

        logfile=log_path,
    )

    run_clustering(cfg)


if __name__ == "__main__":
    app()
