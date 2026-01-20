from __future__ import annotations
from typing import Optional, List, Literal, Dict
import typer
from pathlib import Path
import warnings

from .load_and_filter import run_load_and_filter
from .integrate import run_integrate
from .cluster_and_annotate import run_clustering

from .config import LoadAndFilterConfig, IntegrateConfig, ClusterAnnotateConfig
import logging
from .logging_utils import init_logging


ALLOWED_METHODS = {"scVI", "scANVI", "Harmony", "Scanorama", "BBKNN"}
ALLOWED_DECOUPLER_METHODS = {"ulm", "mlm", "wsum", "aucell"}
app = typer.Typer(help="scOmnom CLI — high-throughput scRNA-seq preprocessing and analysis pipeline.")

# Globally suppress noisy warnings
warnings.filterwarnings(action="ignore", message="Variable names are not unique", category=UserWarning, module="anndata")
warnings.filterwarnings(action="ignore", message=".*not compatible with tight_layout.*", category=UserWarning)
warnings.filterwarnings(action="ignore", message="pkg_resources is deprecated as an API", category=UserWarning)
warnings.filterwarnings(action="ignore", message=r".*does not have many workers.*", category=UserWarning, module="lightning.pytorch")
warnings.filterwarnings(action="ignore", message=".*already log-transformed.*", category=UserWarning)
warnings.filterwarnings(action="ignore", message="Argument `use_highly_variable` is deprecated", category=FutureWarning, module="scanpy")


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
        raise ValueError(
            f"Invalid method(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_METHODS))}"
        )

    return expanded


def _methods_completion(ctx: typer.Context, args: List[str], incomplete: str) -> List[str]:
    prefix = incomplete.lower()
    return [m for m in sorted(ALLOWED_METHODS) if m.lower().startswith(prefix)]


def validate_decoupler_consensus_methods(
    value: Optional[List[str]],
) -> Optional[List[str]]:
    """
    Validate and normalize --decoupler-consensus-methods.

    - lowercases
    - deduplicates (order-preserving)
    - checks against supported decoupler methods
    """
    if value is None:
        return None

    seen = set()
    methods: list[str] = []

    for m in value:
        if m is None:
            continue
        m_norm = m.strip().lower()
        if not m_norm:
            continue
        if m_norm not in ALLOWED_DECOUPLER_METHODS:
            raise typer.BadParameter(
                f"Invalid decoupler consensus method '{m}'. "
                f"Allowed methods: {sorted(ALLOWED_DECOUPLER_METHODS)}"
            )
        if m_norm not in seen:
            seen.add(m_norm)
            methods.append(m_norm)

    if not methods:
        raise typer.BadParameter(
            "--decoupler-consensus-methods must contain at least one valid method."
        )

    return methods


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
    Autocomplete for --msigdb-gene-sets.

    Suggests:
    1. Common MSigDB collection keywords (HALLMARK, REACTOME, BIOCARTA, ...)
    2. Any locally cached or project .gmt files
    3. Prefix-matching only (case-insensitive)
    """
    prefix = incomplete.lower()

    standard_sets = [
        "HALLMARK",
        "REACTOME",
        "BIOCARTA",
        "KEGG",
        "GO_BP",
        "GO_MF",
        "GO_CC",
        "IMMUNO",
        "ONCOGENIC",
        "TF_TARGETS",
        "MIR_TARGETS",
        "WIKIPATHWAYS",
    ]

    suggestions = [s for s in standard_sets if s.lower().startswith(prefix)]

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

    seen = set()
    unique = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique


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
        help=(
            "[I/O] Path with CellBender filtered outputs.\n"
            "Used alone: CellBender-only mode (no raw counts).\n"
            "Used with --raw-sample-dir: enables raw vs CellBender QC comparison."
        ),
    ),
    output_dir: Path = typer.Option(
        ..., "--out", "-o",
        help="[I/O] Output directory for anndata and figures/",
    ),
    metadata_tsv: Optional[Path] = typer.Option(
        None, "--metadata-tsv", "-m", exists=True,
        help="[I/O] TSV with sample metadata (not required with --apply-doublet-score).",
    ),
    n_jobs: int = typer.Option(
        None,
        "--n-jobs",
        help="Number of CPU cores to use.",
    ),
    save_h5ad: bool = typer.Option(
        False,
        "--save-h5ad/--no-save-h5ad",
        help="Also write a .h5ad copy (WARNING: loads full matrix into RAM).",
    ),

    # -------------------------------------------------------------
    # QC thresholds
    # -------------------------------------------------------------
    min_cells: int = typer.Option(3, help="[QC] Minimum cells per gene."),
    min_genes: int = typer.Option(500, help="[QC] Minimum genes per cell."),
    min_cells_per_sample: int = typer.Option(20, help="[QC] Minimum cells per sample."),
    max_pct_mt: float = typer.Option(5.0, help="[QC] Max mitochondrial percentage."),
    n_top_genes: int = typer.Option(2000, help="Number of highly variable genes to select"),

    # -------------------------------------------------------------
    # QC upper-cut filters (MAD + quantile)
    # -------------------------------------------------------------
    max_genes_mad: float = typer.Option(
        5.0,
        "--max-genes-mad",
        help="[QC] Upper cutoff for genes per cell as median + k*MAD (default: 5).",
    ),
    max_genes_quantile: float = typer.Option(
        0.999,
        "--max-genes-quantile",
        help="[QC] Upper quantile cutoff for genes per cell (default: 0.999).",
    ),
    max_counts_mad: float = typer.Option(
        5.0,
        "--max-counts-mad",
        help="[QC] Upper cutoff for total UMI counts as median + k*MAD (default: 5).",
    ),
    max_counts_quantile: float = typer.Option(
        0.999,
        "--max-counts-quantile",
        help="[QC] Upper quantile cutoff for total UMI counts (default: 0.999).",
    ),

    # -------------------------------------------------------------
    # Doublet detection (SOLO)
    # -------------------------------------------------------------
    expected_doublet_rate: float = typer.Option(
        0.1,
        "--expected-doublet-rate",
        help="Used when --doublet-mode rate",
    ),
    apply_doublet_score: bool = typer.Option(
        False,
        "--apply-doublet-score",
        help="Resume from pre-doublet AnnData instead of rerunning QC + SOLO.",
    ),
    apply_doublet_score_path: Optional[Path] = typer.Option(
        None,
        "--apply-doublet-score-path",
        help="Path to pre-doublet AnnData (.zarr or .h5ad). Defaults to <out>/adata.merged.zarr.",
    ),

    # -------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------
    make_figures: bool = typer.Option(True, help="[Figures] Whether to create QC plots."),
    figure_formats: List[str] = typer.Option(
        ["png", "pdf"], "--figure-formats", "-F",
        help="[Figures] Formats to save."
    ),

    # -------------------------------------------------------------
    # Batch
    # -------------------------------------------------------------
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="Batch/sample column in metadata.",
    ),

    # -------------------------------------------------------------
    # Input patterns
    # -------------------------------------------------------------
    raw_pattern: str = typer.Option("*.raw_feature_bc_matrix"),
    filtered_pattern: str = typer.Option("*.filtered_feature_bc_matrix"),
    cellbender_pattern: str = typer.Option("*.cellbender_filtered.output"),
    cellbender_h5_suffix: str = typer.Option(".cellbender_out_filtered.h5"),
    cellbender_barcode_suffix: str = typer.Option(
        ".cellbender_out_cell_barcodes.csv",
        "--cellbender-barcode-suffix",
        help="[I/O] Suffix for CellBender barcode file "
             "(e.g. '_cellbender_out_cell_barcodes.csv').",
    ),
):
    logfile = output_dir / "load-and-filter.log"
    init_logging(logfile)
    logging.getLogger(__name__).info("Logging initialized")
    logging.getLogger("scomnom.load_and_filter").info("load_and_filter logger active")

    if apply_doublet_score:
        if apply_doublet_score_path is None:
            apply_doublet_score_path = output_dir / "adata.merged.zarr"

    cfg = LoadAndFilterConfig(
        raw_sample_dir=raw_sample_dir,
        filtered_sample_dir=filtered_sample_dir,
        cellbender_dir=cellbender_dir,
        metadata_tsv=metadata_tsv,
        output_dir=output_dir,
        save_h5ad=save_h5ad,
        n_jobs=n_jobs or 4,
        min_cells=min_cells,
        min_genes=min_genes,
        min_cells_per_sample=min_cells_per_sample,
        max_pct_mt=max_pct_mt,
        n_top_genes=n_top_genes,
        max_genes_mad=max_genes_mad,
        max_genes_quantile=max_genes_quantile,
        max_counts_mad=max_counts_mad,
        max_counts_quantile=max_counts_quantile,
        expected_doublet_rate=expected_doublet_rate,
        apply_doublet_score=apply_doublet_score,
        apply_doublet_score_path=apply_doublet_score_path,
        make_figures=make_figures,
        figure_formats=figure_formats,
        batch_key=batch_key,
        raw_pattern=raw_pattern,
        filtered_pattern=filtered_pattern,
        cellbender_pattern=cellbender_pattern,
        cellbender_h5_suffix=cellbender_h5_suffix,
        cellbender_barcode_suffix=cellbender_barcode_suffix,
        logfile=logfile,
    )

    run_load_and_filter(cfg)


# ======================================================================
#  integrate
# ======================================================================
@app.command("integrate", help="Integration + benchmarking.")
def integrate(
    input_path: Path = typer.Option(
        ..., "--input-path", "-i",
        help="[I/O] Dataset produced by load-and-filter (.zarr or .h5ad)."
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="[I/O] Output directory (default = input parent).",
    ),
    output_name: str = typer.Option(
        "adata.integrated",
        "--output-name",
        help="[I/O] Base name for integrated dataset.",
    ),
    save_h5ad: bool = typer.Option(
        False,
        "--save-h5ad/--no-save-h5ad",
        help="Optionally write an .h5ad copy.",
    ),

    figdir_name: str = typer.Option(
        "figures",
        "--figdir-name",
        help="[Figures] Name of figure directory.",
    ),
    figure_formats: List[str] = typer.Option(
        ["png", "pdf"],
        "--figure-formats",
        "-F",
        help="[Figures] Output figure formats.",
    ),

    methods: Optional[List[str]] = typer.Option(
        None, "--methods", "-m",
        help="[Integration] Methods: BBKNN, Harmony, Scanorama, scVI, scANVI.",
        case_sensitive=False,
    ),
    batch_key: Optional[str] = typer.Option(
        None, "--batch-key", "-b",
        help="[Integration] Batch/sample column in .obs.",
    ),
    label_key: str = typer.Option(
        "leiden", "--label-key", "-l",
        help="[scIB] Label key for benchmarking and scANVI.",
    ),

    benchmark_n_jobs: int = typer.Option(
        16,
        "--benchmark-n-jobs",
        help="[scIB] Parallel workers.",
    ),
    benchmark_threshold: int = 100000,
    benchmark_n_cells: int = 100000,
    benchmark_random_state: int = 42,
):
    logfile = (output_dir or input_path.parent) / "integrate.log"
    init_logging(logfile)

    cfg = IntegrateConfig(
        input_path=input_path,
        output_dir=output_dir or input_path.parent,
        output_name=output_name,
        save_h5ad=save_h5ad,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        methods=methods,
        batch_key=batch_key,
        label_key=label_key,
        benchmark_n_jobs=benchmark_n_jobs,
        benchmark_threshold=benchmark_threshold,
        benchmark_n_cells=benchmark_n_cells,
        benchmark_random_state=benchmark_random_state,
        logfile=logfile,
    )

    run_integrate(cfg)


# ======================================================================
#  cluster-and-annotate
# ======================================================================
@app.command(
    "cluster-and-annotate",
    help="Clustering (resolution sweep + stability) and optional CellTypist + decoupler annotation (MSigDB/DoRothEA/PROGENy).",
)
def cluster_and_annotate(
    # -----------------------------
    # I/O (integrate-style)
    # -----------------------------
    input_path: Optional[Path] = typer.Option(
        None,
        "--input-path",
        "-i",
        help="[I/O] Integrated dataset produced by `scomnom integrate` (.zarr or .h5ad).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default = input parent).",
    ),
    output_name: str = typer.Option(
        "adata.clustered.annotated",
        "--output-name",
        help="[I/O] Base name for clustered/annotated dataset.",
    ),
    save_h5ad: bool = typer.Option(
        False,
        "--save-h5ad/--no-save-h5ad",
        help="[I/O] Optionally write an .h5ad copy (in addition to .zarr).",
    ),

    # -----------------------------
    # Figures
    # -----------------------------
    figdir_name: str = typer.Option(
        "figures",
        "--figdir-name",
        help="[Figures] Name of figure directory.",
    ),
    figure_formats: List[str] = typer.Option(
        ["png", "pdf"],
        "--figure-formats",
        "-F",
        help="[Figures] Output figure formats.",
    ),
    make_figures: bool = typer.Option(
        True,
        "--make-figures/--no-make-figures",
        help="[Figures] Enable/disable figure generation.",
    ),

    # -----------------------------
    # Embeddings / keys
    # -----------------------------
    embedding_key: str = typer.Option(
        "X_integrated",
        "--embedding-key",
        "-e",
        help="[Clustering] Embedding key in .obsm for neighbors + scoring.",
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        "--batch-key",
        "-b",
        help="[Clustering] Batch/sample column in adata.obs (default: auto-detect).",
    ),
    label_key: str = typer.Option(
        "leiden",
        "--label-key",
        "-l",
        help="[Clustering] Final cluster key stored in adata.obs.",
    ),

    # -----------------------------
    # Bio-guided clustering weights
    # -----------------------------
    bio_guided_clustering: bool = typer.Option(
        True,
        "--bio-guided/--no-bio-guided",
        help="[Bio] Enable/disable biological metrics guidance during resolution selection.",
    ),
    w_hom: float = typer.Option(0.15, "--w-hom", help="[Bio] Weight for bio homogeneity (if available)."),
    w_frag: float = typer.Option(0.10, "--w-frag", help="[Bio] Weight for bio fragmentation (if available)."),
    w_bioari: float = typer.Option(0.15, "--w-bioari", help="[Bio] Weight for bio ARI (if available)."),

    # -----------------------------
    # Resolution sweep
    # -----------------------------
    res_min: float = typer.Option(0.1, "--res-min"),
    res_max: float = typer.Option(2.5, "--res-max"),
    n_resolutions: int = typer.Option(25, "--n-resolutions"),
    penalty_alpha: float = typer.Option(0.02, "--penalty-alpha"),

    # -----------------------------
    # Stability / selection
    # -----------------------------
    stability_repeats: int = typer.Option(5, "--stability-repeats"),
    subsample_frac: float = typer.Option(0.8, "--subsample-frac"),
    random_state: int = typer.Option(42, "--random-state"),
    tiny_cluster_size: int = typer.Option(20, "--tiny-cluster-size"),
    min_cluster_size: int = typer.Option(20, "--min-cluster-size"),
    min_plateau_len: int = typer.Option(3, "--min-plateau-len"),
    max_cluster_jump_frac: float = typer.Option(0.4, "--max-cluster-jump-frac"),
    stability_threshold: float = typer.Option(0.85, "--stability-threshold"),
    w_stab: float = typer.Option(0.50, "--w-stab"),
    w_sil: float = typer.Option(0.35, "--w-sil"),
    w_tiny: float = typer.Option(0.15, "--w-tiny"),

    # -----------------------------
    # CellTypist
    # -----------------------------
    celltypist_model: Optional[str] = typer.Option(
        "Immune_All_Low.pkl",
        "--celltypist-model",
        "-M",
        autocompletion=_celltypist_models_completion,
        help="[CellTypist] Path or name of model. If None, skip annotation.",
    ),
    celltypist_majority_voting: bool = typer.Option(
        True,
        "--celltypist-majority-voting/--no-celltypist-majority-voting",
        help="[CellTypist] Use majority voting.",
    ),
    annotation_csv: Optional[Path] = typer.Option(
        None,
        "--annotation-csv",
        "-A",
        help="[CellTypist] Optional CSV with per-cluster annotations.",
    ),

    # -----------------------------
    # Bio mask (CellTypist confidence gate)
    # -----------------------------
    bio_mask_mode: Literal["entropy_margin", "none"] = typer.Option(
        "entropy_margin",
        "--bio-mask-mode",
        help="[Bio] CellTypist confidence mask mode for bio metrics: entropy_margin or none.",
    ),
    bio_entropy_abs_limit: float = typer.Option(
        0.5,
        "--bio-entropy-abs-limit",
        help="[Bio] Absolute entropy ceiling. Cells with entropy <= this always pass.",
    ),
    bio_entropy_quantile: float = typer.Option(
        0.7,
        "--bio-entropy-quantile",
        help="[Bio] Entropy quantile. Final entropy cut is max(abs_limit, q-threshold).",
    ),
    bio_margin_min: float = typer.Option(
        0.10,
        "--bio-margin-min",
        help="[Bio] Minimum margin (top1 - top2) to pass mask.",
    ),
    bio_mask_min_cells: int = typer.Option(
        500,
        "--bio-mask-min-cells",
        help="[Bio] Disable bio mask if fewer than this many cells pass (safety).",
    ),
    bio_mask_min_frac: float = typer.Option(
        0.05,
        "--bio-mask-min-frac",
        help="[Bio] Disable bio mask if fewer than this fraction of cells pass (safety).",
    ),

    # Pretty label gating (cluster-level CT majority)
    pretty_label_min_masked_cells: int = typer.Option(
        25,
        "--pretty-label-min-masked-cells",
        help="[CellTypist] Minimum masked cells in a cluster to assign cluster-level label; else Unknown.",
    ),
    pretty_label_min_masked_frac: float = typer.Option(
        0.10,
        "--pretty-label-min-masked-frac",
        help="[CellTypist] Minimum masked fraction in a cluster to assign cluster-level label; else Unknown.",
    ),

    # -----------------------------
    # Model management
    # -----------------------------
    list_models: bool = typer.Option(False, "--list-models"),
    download_models: bool = typer.Option(False, "--download-models"),

    # -----------------------------
    # Decoupler (cluster-level nets)
    # -----------------------------
    run_decoupler: bool = typer.Option(
        True,
        "--run-decoupler/--no-run-decoupler",
        help="[Decoupler] Run decoupler-based annotation (MSigDB, DoRothEA, PROGENy).",
    ),

    decoupler_pseudobulk_agg: str = typer.Option(
        "mean",
        "--decoupler-pseudobulk-agg",
        help="[Decoupler] Pseudobulk aggregation: mean or median.",
    ),
    decoupler_use_raw: bool = typer.Option(
        True,
        "--decoupler-use-raw/--no-decoupler-use-raw",
        help="[Decoupler] Prefer raw-like source (counts layers/raw) for pseudobulk when available.",
    ),

    decoupler_method: str = typer.Option(
        "consensus",
        "--decoupler-method",
        help="[Decoupler] Default method for decoupler runs (can be overridden per net).",
    ),
    decoupler_consensus_methods: Optional[List[str]] = typer.Option(
        None,
        "--decoupler-consensus-methods",
        help="[Decoupler] List of consensus methods (e.g. --decoupler-consensus-methods ulm --decoupler-consensus-methods mlm).",
        callback=validate_decoupler_consensus_methods,
    ),
    decoupler_min_n_targets: int = typer.Option(
        5,
        "--decoupler-min-n-targets",
        help="[Decoupler] Minimum targets per source.",
    ),

    # MSigDB
    msigdb_gene_sets_cli: Optional[str] = typer.Option(
        None,
        "--msigdb-gene-sets",
        help="[MSigDB] Comma-separated MSigDB keywords (e.g. 'HALLMARK,REACTOME') or paths to .gmt files.",
        autocompletion=_gene_sets_completion,
    ),
    msigdb_method: str = typer.Option(
        "consensus",
        "--msigdb-method",
        help="[MSigDB] Method override for MSigDB nets.",
    ),
    msigdb_min_n_targets: int = typer.Option(
        5,
        "--msigdb-min-n-targets",
        help="[MSigDB] Minimum targets per pathway.",
    ),

    # DoRothEA / PROGENy toggles + methods
    run_dorothea: bool = typer.Option(True, "--run-dorothea/--no-run-dorothea"),
    dorothea_method: str = typer.Option("consensus", "--dorothea-method"),
    dorothea_min_n_targets: int = typer.Option(5, "--dorothea-min-n-targets"),
    dorothea_confidence: str = typer.Option("A,B,C", "--dorothea-confidence"),
    dorothea_organism: str = typer.Option("human", "--dorothea-organism"),

    run_progeny: bool = typer.Option(True, "--run-progeny/--no-run-progeny"),
    progeny_method: str = typer.Option("consensus", "--progeny-method"),
    progeny_min_n_targets: int = typer.Option(5, "--progeny-min-n-targets"),
    progeny_top_n: int = typer.Option(100, "--progeny-top-n"),
    progeny_organism: str = typer.Option("human", "--progeny-organism"),

    # -----------------------------
    # Compaction
    # -----------------------------
    enable_compacting: bool = typer.Option(
        True,
        "--enable-compacting/--no-enable-compacting",
        help="[Compaction] Create an additional compacted clustering round using multiview agreement.",
    ),
    compact_min_cells: int = typer.Option(
        0,
        "--compact-min-cells",
        help="[Compaction] Exclude clusters smaller than this from compaction decisions (0 disables).",
    ),
    compact_zscore_scope: Literal["within_celltypist_label", "global"] = typer.Option(
        "global",
        "--compact-zscore-scope",
        help="[Compaction] Z-score scope for similarity comparisons.",
    ),
    compact_grouping: Literal["connected_components", "clique"] = typer.Option(
        "connected_components",
        "--compact-grouping",
        help="[Compaction] How to form compaction groups from pass edges.",
    ),
    compact_skip_unknown_celltypist_groups: bool = typer.Option(
        False,
        "--compact-skip-unknown/--no-compact-skip-unknown",
        help="[Compaction] Skip compaction for Unknown/UNKNOWN CellTypist cluster-label groups.",
    ),
    thr_progeny: float = typer.Option(
        0.98,
        "--thr-progeny",
        help="[Compaction] Similarity threshold for PROGENy activities (cosine on z-scored activities).",
    ),
    thr_dorothea: float = typer.Option(
        0.98,
        "--thr-dorothea",
        help="[Compaction] Similarity threshold for DoRothEA activities (cosine on z-scored activities).",
    ),
    thr_msigdb_default: float = typer.Option(
        0.98,
        "--thr-msigdb-default",
        help="[Compaction] Default similarity threshold for each MSigDB GMT block.",
    ),
    thr_msigdb_by_gmt: Optional[str] = typer.Option(
        None,
        "--thr-msigdb-by-gmt",
        help="[Compaction] Optional per-GMT thresholds as 'HALLMARK=0.99,REACTOME=0.985'.",
    ),
    msigdb_required: bool = typer.Option(
        True,
        "--msigdb-required/--no-msigdb-required",
        help="[Compaction] Require MSigDB activity_by_gmt for compaction decisions.",
    ),
):
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
        raise typer.BadParameter("Missing required option --input-path / -i")

    # ---------------------------------------------------------
    # Resolve output dir + logging
    # ---------------------------------------------------------
    out_dir = output_dir or input_path.parent
    log_path = out_dir / "cluster-and-annotate.log"
    init_logging(log_path)

    # ---------------------------------------------------------
    # Parse MSigDB gene sets
    # ---------------------------------------------------------
    if msigdb_gene_sets_cli is None:
        gene_sets_list = None
    else:
        gene_sets_list = [x.strip() for x in msigdb_gene_sets_cli.split(",") if x.strip()]

    # ---------------------------------------------------------
    # Parse thr_msigdb_by_gmt "A=0.99,B=0.985" -> dict
    # ---------------------------------------------------------
    thr_msigdb_by_gmt_dict: Optional[Dict[str, float]] = None
    if thr_msigdb_by_gmt:
        d: Dict[str, float] = {}
        for part in str(thr_msigdb_by_gmt).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise typer.BadParameter(
                    "--thr-msigdb-by-gmt must be 'GMT=thr,GMT=thr' (e.g. HALLMARK=0.99,REACTOME=0.985)"
                )
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                raise typer.BadParameter("Empty GMT key in --thr-msigdb-by-gmt")
            try:
                d[k] = float(v)
            except Exception:
                raise typer.BadParameter(f"Non-numeric threshold for GMT {k!r}: {v!r}")
        thr_msigdb_by_gmt_dict = d or None

    # ---------------------------------------------------------
    # Build config
    # ---------------------------------------------------------
    kwargs = dict(
        input_path=input_path,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,

        embedding_key=embedding_key,
        batch_key=batch_key,
        label_key=label_key,

        bio_guided_clustering=bio_guided_clustering,
        w_hom=w_hom,
        w_frag=w_frag,
        w_bioari=w_bioari,

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

        bio_mask_mode=bio_mask_mode,
        bio_entropy_abs_limit=bio_entropy_abs_limit,
        bio_entropy_quantile=bio_entropy_quantile,
        bio_margin_min=bio_margin_min,
        bio_mask_min_cells=bio_mask_min_cells,
        bio_mask_min_frac=bio_mask_min_frac,

        pretty_label_min_masked_cells=pretty_label_min_masked_cells,
        pretty_label_min_masked_frac=pretty_label_min_masked_frac,

        run_decoupler=run_decoupler,
        decoupler_pseudobulk_agg=decoupler_pseudobulk_agg,
        decoupler_use_raw=decoupler_use_raw,
        decoupler_method=decoupler_method,
        decoupler_min_n_targets=decoupler_min_n_targets,
        msigdb_method=msigdb_method,
        msigdb_min_n_targets=msigdb_min_n_targets,

        run_dorothea=run_dorothea,
        dorothea_method=dorothea_method,
        dorothea_min_n_targets=dorothea_min_n_targets,
        dorothea_confidence=[x.strip() for x in dorothea_confidence.split(",") if x.strip()],
        dorothea_organism=dorothea_organism,

        run_progeny=run_progeny,
        progeny_method=progeny_method,
        progeny_min_n_targets=progeny_min_n_targets,
        progeny_top_n=progeny_top_n,
        progeny_organism=progeny_organism,

        enable_compacting=enable_compacting,
        compact_min_cells=compact_min_cells,
        compact_zscore_scope=compact_zscore_scope,
        compact_grouping=compact_grouping,
        compact_skip_unknown_celltypist_groups=compact_skip_unknown_celltypist_groups,
        thr_progeny=thr_progeny,
        thr_dorothea=thr_dorothea,
        thr_msigdb_default=thr_msigdb_default,
        thr_msigdb_by_gmt=thr_msigdb_by_gmt_dict,
        msigdb_required=msigdb_required,

        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,

        logfile=log_path,
    )

    # only set if provided (so config defaults can apply)
    if decoupler_consensus_methods is not None:
        kwargs["decoupler_consensus_methods"] = decoupler_consensus_methods
    if gene_sets_list is not None:
        kwargs["msigdb_gene_sets"] = gene_sets_list

    cfg = ClusterAnnotateConfig(**kwargs)

    run_clustering(cfg)


if __name__ == "__main__":
    app()
