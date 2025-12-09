from __future__ import annotations
from typing import Optional, List
import typer
from pathlib import Path
import warnings

from .load_data import run_load_data
from .load_and_filter import run_load_and_filter
from .integrate import run_integration
from .cluster_and_annotate import run_clustering

from .config import LoadDataConfig, QCFilterConfig, LoadAndQCConfig, IntegrationConfig, ClusterAnnotateConfig
from .logging_utils import init_logging


ALLOWED_METHODS = {"scVI", "scANVI", "Scanorama", "Harmony", "BBKNN"}
app = typer.Typer(help="scOmnom CLI — high-throughput scRNA-seq preprocessing and analysis pipeline.")

# Globally suppress noisy warnings
warnings.filterwarnings("ignore", message="Variable names are not unique", category=UserWarning, module="anndata")
warnings.filterwarnings("ignore", message=".*not compatible with tight_layout.*", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*does not have many workers.*", category=UserWarning, module="lightning.pytorch")
warnings.filterwarnings("ignore", message=".*already log-transformed.*", category=UserWarning)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _normalize_methods(methods):
    """
    Normalize and validate integration method inputs.
    Supports e.g. --methods A,B --methods C.
    """
    if methods is None:
        return None

    expanded = []
    for m in methods:
        expanded.extend([x.strip() for x in m.split(",") if x.strip()])

    invalid = [m for m in expanded if m not in ALLOWED_METHODS]
    if invalid:
        raise ValueError(f"Invalid method(s): {', '.join(invalid)}. "
                         f"Allowed: {', '.join(sorted(ALLOWED_METHODS))}")

    return expanded


def _methods_completion(ctx: typer.Context, args: List[str], incomplete: str) -> List[str]:
    prefix = incomplete.lower()
    return [m for m in sorted(ALLOWED_METHODS) if m.lower().startswith(prefix)]


def _celltypist_models_completion(ctx: typer.Context, args: List[str], incomplete: str) -> List[str]:
    from .io_utils import get_available_celltypist_models
    try:
        models = get_available_celltypist_models()
    except Exception:
        return []

    prefix = incomplete.lower()
    return [m.get("name", "") for m in models if m.get("name", "").lower().startswith(prefix)]


def _gene_sets_completion(
    ctx: typer.Context,
    args: List[str],
    incomplete: str,
) -> List[str]:
    """
    Autocomplete for --ssgsea-gene-sets.

    Suggests:
    1. Common MSigDB collection keywords (HALLMARK, REACTOME, BIOCARTA, ...)
    2. Any locally cached or project .gmt files
    3. Prefix-matching only (case-insensitive)
    """
    prefix = incomplete.lower()

    # -----------------------------------------------------
    # 1. Known MSigDB collections (broadest modern set)
    # -----------------------------------------------------
    standard_sets = [
        # Hallmark
        "HALLMARK",

        # MSigDB curated (C2)
        "REACTOME",
        "BIOCARTA",
        "KEGG",

        # GO collections
        "GO_BP",
        "GO_MF",
        "GO_CC",

        # Immunologic signatures
        "IMMUNO",

        # Oncogenic signatures
        "ONCOGENIC",

        # Transcription factor targets
        "TF_TARGETS",

        # MicroRNA targets
        "MIR_TARGETS",

        # WikiPathways (now part of MSigDB 2024+)
        "WIKIPATHWAYS",
    ]

    suggestions = [s for s in standard_sets if s.lower().startswith(prefix)]

    # -----------------------------------------------------
    # 2. Include local *.gmt files if they match prefix
    # -----------------------------------------------------
    known_dirs = [
        Path.cwd(),
        Path.home(),
        Path.home() / "msigdb_cache",
        Path.home() / ".cache/scomnom/msigdb",
    ]

    for d in known_dirs:
        if not d.exists():
            continue
        for f in d.glob("*.gmt"):
            if f.name.lower().startswith(prefix):
                suggestions.append(str(f))

    # Deduplicate but preserve order
    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


# ======================================================================
#  load-data
# ======================================================================
@app.command("load-data", help="Load data, attach metadata, merge samples, and save merged .zarr (+ optional .h5ad)")

def load_data(
    # exactly one of these three
    raw_sample_dir: Path = typer.Option(
        None, "--raw-sample-dir", "-r",
        help="Directory containing <sample>.raw_feature_bc_matrix folders."
    ),
    filtered_sample_dir: Path = typer.Option(
        None, "--filtered-sample-dir", "-f",
        help="Directory containing <sample>.filtered_feature_bc_matrix folders."
    ),
    cellbender_dir: Path = typer.Option(
        None, "--cellbender-dir", "-c",
        help="Directory containing <sample>.cellbender_filtered.output folders."
    ),

    # required
    metadata_tsv: Path = typer.Option(
        ..., "--metadata-tsv", "-m",
        help="TSV with per-sample metadata indexed by sample_id."
    ),

    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="Column name in metadata_tsv to use as batch/sample ID. "
    ),

    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o",
        help="Directory for merged output Zarr."
    ),

    # optional
    output_name: str = typer.Option(
        "adata.merged", "--output-name",
        help="Base name for merged Zarr ('.zarr' appended automatically)."
    ),
    n_jobs: int = typer.Option(
        4, "--n-jobs",
        help="Parallel workers for loading samples & writing Zarrs."
    ),
    save_h5ad: bool = typer.Option(
            False,
            "--save-h5ad/--no-save-h5ad",
            help="Also write an .h5ad copy (requires loading matrix fully into RAM).",
        ),
):
    """
    Load single-cell data from exactly one input source (RAW, filtered, or CellBender),
    attach metadata, write per-sample Zarrs, merge, and save final dataset.
    """

    logfile = output_dir / "load-data.log"
    init_logging(logfile)

    cfg = LoadDataConfig(
        raw_sample_dir=raw_sample_dir,
        filtered_sample_dir=filtered_sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        batch_key=batch_key,
        output_dir=output_dir,
        output_name=output_name,
        n_jobs=n_jobs,
        save_h5ad=save_h5ad,
    )

    run_load_data(cfg, logfile)


# ======================================================================
#  qc-and-filter
# ======================================================================
@app.command(
    "qc-and-filter",
    help=(
        "QC and filter a merged dataset (.zarr or .h5ad) produced by load-data.\n\n"
        "Applies min_genes/min_cells/max_pct_mt/min_cells_per_sample filters and "
        "writes adata.filtered.zarr (+ optional .h5ad)."
    ),
)
def qc_and_filter(
    input_path: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to merged dataset from load-data (.zarr directory or .h5ad file).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help=(
            "Directory for filtered output. "
            "Defaults to the parent directory of --input."
        ),
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        "--batch-key",
        "-b",
        help=(
            "Column in adata.obs to treat as batch/sample ID. "
            "If omitted, io_utils.infer_batch_key() chooses."
        ),
    ),
    min_cells: int = typer.Option(
        3,
        "--min-cells",
        help="[QC] Minimum cells per gene.",
    ),
    min_genes: int = typer.Option(
        500,
        "--min-genes",
        help="[QC] Minimum genes per cell. Lower this to ~200 for snRNA-seq.",
    ),
    min_cells_per_sample: int = typer.Option(
        20,
        "--min-cells-per-sample",
        help="[QC] Minimum cells per sample.",
    ),
    max_pct_mt: float = typer.Option(
        5.0,
        "--max-pct-mt",
        help="[QC] Max mitochondrial percentage. Increase this to ~30–50 for snRNA-seq.",
    ),
    make_figures: bool = typer.Option(
        True,
        "--make-figures/--no-make-figures",
        help="Generate QC plots.",
    ),
    figure_format: List[str] = typer.Option(
        ["png", "pdf"],
        "--figure-format",
        help="Figure format(s) for plots (can be given multiple times).",
    ),
    save_h5ad: bool = typer.Option(
        False,
        "--save-h5ad/--no-save-h5ad",
        help="Also write an .h5ad copy of the filtered dataset.",
    ),
):
    """
    QC-and-filter stage operating on the merged dataset produced by `load-data`.
    """

    # Logfile in the output directory
    logfile = output_dir / "qc-and-filter.log"
    init_logging(logfile)

    # Build config (Pydantic will validate paths and defaults)
    cfg = QCFilterConfig(
        input_path=input_path,
        output_dir=output_dir,
        batch_key=batch_key,
        min_cells=min_cells,
        min_genes=min_genes,
        min_cells_per_sample=min_cells_per_sample,
        max_pct_mt=max_pct_mt,
        make_figures=make_figures,
        figure_formats=figure_format,
        save_h5ad=save_h5ad,
    )

    run_qc_and_filter(cfg, logfile)



# ======================================================================
#  load-and-filter
# ======================================================================
@app.command("load-and-filter", help="Full scOmnom preprocessing pipeline.")
def load_and_filter(
    # -------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------
    raw_sample_dir: Optional[Path] = typer.Option(
        None, "--raw-sample-dir", "-r",
        help="[I/O] Path containing <sample>.raw_feature_bc_matrix folders.",
    ),
    filtered_sample_dir: Optional[Path] = typer.Option(
        None, "--filtered-sample-dir", "-f",
        help="[I/O] Path with <sample>.filtered_feature_bc_matrix folders.",
    ),
    cellbender_dir: Optional[Path] = typer.Option(
        None, "--cellbender-dir", "-c",
        help="[I/O] Path with <sample>.cellbender_filtered.output folders.",
    ),
    output_dir: Path = typer.Option(
        ..., "--out", "-o",
        help="[I/O] Output directory for anndata and figures/",
    ),
    metadata_tsv: Path = typer.Option(
        ..., "--metadata-tsv", "-m", exists=True,
        help="[I/O] TSV with sample metadata.",
    ),
    n_jobs: int = typer.Option(
        None,
        "--n-jobs",
        help="Number of CPU cores to use (default: value from config, typically 4).",
    ),
    save_h5ad: bool = typer.Option(
            False,
            "--save-h5ad/--no-save-h5ad",
            help="Also write a .h5ad copy of the merged dataset (WARNING: loads full matrix into RAM).",
    ),

    # -------------------------------------------------------------
    # QC thresholds
    # -------------------------------------------------------------
    min_cells: int = typer.Option(3, help="[QC] Minimum cells per gene."),
    min_genes: int = typer.Option(500, help="[QC] Minimum genes per cell. Lower this to ~200 for snRNA-seq"),
    min_cells_per_sample: int = typer.Option(20, help="[QC] Minimum cells per sample."),
    max_pct_mt: float = typer.Option(5.0, help="[QC] Max mitochondrial percentage. Increase this to ~30-50% for snRNA-seq"),

    # -------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------
    make_figures: bool = typer.Option(True, help="[Figures] Whether to create QC plots."),
    figure_format: List[str] = typer.Option(
        ["png", "pdf"], "--figure-format", "-F",
        help="[Figures] Formats to save."
    ),

    # -------------------------------------------------------------
    # HVG / PCA / batch
    # -------------------------------------------------------------
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="[PCA/Batch] Batch column in .obs.",
    ),

    # -------------------------------------------------------------
    # Input pattern overrides (advanced)
    # -------------------------------------------------------------
    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix"),
    filtered_pattern: str = typer.Option("*.filtered_feature_bc_matrix"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out.h5"),
):
    if raw_sample_dir and filtered_sample_dir:
        raise typer.BadParameter("Cannot specify both --raw-sample-dir and --filtered-sample-dir")
    if filtered_sample_dir and cellbender_dir:
        raise typer.BadParameter("CellBender outputs cannot be mixed with Cell Ranger filtered matrices.")

    logfile = output_dir / "load-and-filter.log"
    init_logging(logfile)

    cfg = LoadAndQCConfig(
        raw_sample_dir=raw_sample_dir,
        filtered_sample_dir=filtered_sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        output_dir=output_dir,
        save_h5ad=save_h5ad,
        n_jobs=4,
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

    if n_jobs is not None:
        cfg.n_jobs = n_jobs

    run_load_and_filter(cfg, logfile)


# ======================================================================
#  integrate
# ======================================================================
@app.command("integrate", help="Run integration and scIB benchmarking.")
def integrate(
    # -------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------
    input_path: Path = typer.Option(
        ..., "--input-path", "-i",
        help="[I/O] Preprocessed h5ad from load-and-filter.",
    ),
    output_path: Optional[Path] = typer.Option(
        None, "--output-path", "-o",
        help="[I/O] Output integrated h5ad.",
    ),

    # -------------------------------------------------------------
    # Integration
    # -------------------------------------------------------------
    methods: Optional[List[str]] = typer.Option(
        None, "--methods", "-m",
        help="[Integration] Methods to run (Scanorama, Harmony, scVI, scANVI, BBKNN).",
        case_sensitive=False,
        autocompletion=_methods_completion,
    ),
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="[Integration] Batch column in .obs.",
    ),

    # -------------------------------------------------------------
    # Benchmarking
    # -------------------------------------------------------------
    label_key: str = typer.Option(
        "leiden", "--label-key", "-l",
        help="[scIB] Cluster label for benchmarking.",
    ),
    benchmark_n_jobs: int = typer.Option(4, help="[scIB] Parallel workers."),
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
@app.command(
    "cluster-and-annotate",
    help="Perform clustering (resolution sweep + stability) and optional CellTypist + ssGSEA annotation.",
)
def cluster_and_annotate(
    # --- I/O ---
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Integrated h5ad file produced by `scOmnom integrate`.",
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--out",
        "-o",
        help="Output h5ad. Defaults to <input>.clustered.annotated.h5ad",
    ),

    # --- Embeddings / keys ---
    embedding_key: str = typer.Option(
        "X_integrated",
        "--embedding-key",
        "-e",
        help="Embedding key in .obsm to use for neighbors and silhouette scoring.",
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        "--batch-key",
        "-b",
        help="Batch/sample column in adata.obs (default: auto-detect).",
    ),
    label_key: str = typer.Option(
        "leiden",
        "--label-key",
        "-l",
        help="Final cluster key stored in adata.obs.",
    ),

    # --- Resolution sweep ---
    res_min: float = typer.Option(0.1, "--res-min"),
    res_max: float = typer.Option(2.5, "--res-max"),
    n_resolutions: int = typer.Option(25, "--n-resolutions"),
    penalty_alpha: float = typer.Option(0.02),

    # --- Stability ---
    stability_repeats: int = typer.Option(5),
    subsample_frac: float = typer.Option(0.8),
    random_state: int = typer.Option(42),
    tiny_cluster_size: int = typer.Option(20),
    min_cluster_size: int = typer.Option(20),
    min_plateau_len: int = typer.Option(3),
    max_cluster_jump_frac: float = typer.Option(0.4),
    stability_threshold: float = typer.Option(0.85),

    w_stab: float = typer.Option(0.50),
    w_sil: float = typer.Option(0.35),
    w_tiny: float = typer.Option(0.15),

    # --- CellTypist annotation ---
    celltypist_model: Optional[str] = typer.Option(
        "Immune_All_Low.pkl",
        "--celltypist-model",
        "-M",
        autocompletion=_celltypist_models_completion,
        help="Path or name of CellTypist model. If None, skip annotation.",
    ),
    celltypist_majority_voting: bool = typer.Option(True),
    annotation_csv: Optional[Path] = typer.Option(
        None,
        "--annotation-csv",
        "-A",
        help="Optional CSV with per-cluster annotations.",
    ),

    # --- Model management ---
    list_models: bool = typer.Option(False, "--list-models"),
    download_models: bool = typer.Option(False, "--download-models"),

    # --- ssGSEA ---
    run_ssgsea: bool = typer.Option(
        True, help="Run ssGSEA enrichment per cell."
    ),
    ssgsea_gene_sets_cli: Optional[str] = typer.Option(
        None,
        "--ssgsea-gene-sets",
        help="Comma-separated MSigDB keywords (e.g. 'HALLMARK,REACTOME') or paths to .gmt files.",
        autocompletion=_gene_sets_completion,
    ),
    ssgsea_use_raw: bool = typer.Option(True),
    ssgsea_min_size: int = typer.Option(10),
    ssgsea_max_size: int = typer.Option(500),
    ssgsea_sample_norm_method: str = typer.Option("rank"),
    ssgsea_nproc: Optional[int] = typer.Option(
        None,
        help="Parallel workers. Default = CPU cores - 1.",
    ),

    # --- Figures ---
    make_figures: bool = typer.Option(True),
    figure_format: List[str] = typer.Option(["png", "pdf"]),
    figdir_name: str = typer.Option("figures"),
):
    """
    Run clustering + annotation (CellTypist + ssGSEA).
    """

    # ---------------------------------------------------------
    # Handle --list-models / --download-models
    # ---------------------------------------------------------
    if list_models:
        from .io_utils import get_available_celltypist_models
        models = get_available_celltypist_models()

        typer.echo("\nAvailable CellTypist models:\n")
        for m in models:
            typer.echo(f"  - {m['name']}")
        raise typer.Exit()

    if download_models:
        from .io_utils import download_all_celltypist_models
        download_all_celltypist_models()
        raise typer.Exit()

    # ---------------------------------------------------------
    # Validate required input
    # ---------------------------------------------------------
    if input_path is None:
        raise typer.BadParameter("Missing required option --input / -i")

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------
    log_path = (
        output_path.parent / "cluster-and-annotate.log"
        if output_path
        else input_path.parent / "cluster-and-annotate.log"
    )
    init_logging(log_path)

    # ---------------------------------------------------------
    # Parse ssGSEA gene set list
    # ---------------------------------------------------------
    if ssgsea_gene_sets_cli is None:
        # allow default Hallmark + Reactome from config
        gene_sets_list = None
    else:
        # split user-specified list
        gene_sets_list = [
            x.strip()
            for x in ssgsea_gene_sets_cli.split(",")
            if x.strip()
        ]

    # nproc
    import multiprocessing
    nproc = (
        max(1, multiprocessing.cpu_count() - 1)
        if ssgsea_nproc is None
        else ssgsea_nproc
    )

    # ---------------------------------------------------------
    # Build config
    # ---------------------------------------------------------
    kwargs = dict(
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

        run_ssgsea=run_ssgsea,
        ssgsea_use_raw=ssgsea_use_raw,
        ssgsea_min_size=ssgsea_min_size,
        ssgsea_max_size=ssgsea_max_size,
        ssgsea_sample_norm_method=ssgsea_sample_norm_method,
        ssgsea_nproc=nproc,

        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_format,

        logfile=log_path,
    )

    # Only insert explicitly if user provided it
    if gene_sets_list is not None:
        kwargs["ssgsea_gene_sets"] = gene_sets_list

    cfg = ClusterAnnotateConfig(**kwargs)

    # ---------------------------------------------------------
    # Run the module
    # ---------------------------------------------------------
    run_clustering(cfg)



if __name__ == "__main__":
    app()
