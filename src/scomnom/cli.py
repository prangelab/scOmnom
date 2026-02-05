from __future__ import annotations
from typing import Optional, List, Literal, Dict, Sequence, Tuple
import typer
from pathlib import Path
import warnings
from enum import Enum
from pandas.errors import PerformanceWarning

from .load_and_filter import run_load_and_filter
from .integrate import run_integrate
from .cluster_and_annotate import run_clustering
from .markers_and_de import run_cluster_vs_rest, run_within_cluster

from .config import LoadAndFilterConfig, IntegrateConfig, ClusterAnnotateConfig, MarkersAndDEConfig
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
warnings.filterwarnings(action="ignore", message="DataFrame is highly fragmented", category=PerformanceWarning, module="scanpy")

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
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Dataset produced by load-and-filter (.zarr or .h5ad).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
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
    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
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
    # ------------------------------------------------------------------
    # Integration + scIB
    # ------------------------------------------------------------------
    methods: Optional[List[str]] = typer.Option(
        None,
        "--methods",
        "-m",
        help="[Integration] Methods: BBKNN, Harmony, Scanorama, scVI, scANVI.",
        case_sensitive=False,
    ),
    batch_key: Optional[str] = typer.Option(
        None,
        "--batch-key",
        "-b",
        help="[Integration] Batch/sample column in .obs.",
    ),
    label_key: str = typer.Option(
        "leiden",
        "--label-key",
        "-l",
        help="[scIB] Label key for benchmarking. (scANVI supervision labels are controlled separately.)",
    ),
    benchmark_n_jobs: int = typer.Option(
        16,
        "--benchmark-n-jobs",
        help="[scIB] Parallel workers.",
    ),
    benchmark_threshold: int = typer.Option(
        100000,
        "--benchmark-threshold",
        help="[scIB] If n_cells exceeds this, benchmark on a stratified subsample.",
    ),
    benchmark_n_cells: int = typer.Option(
        100000,
        "--benchmark-n-cells",
        help="[scIB] Target cell count for stratified benchmarking subsample.",
    ),
    benchmark_random_state: int = typer.Option(
        42,
        "--benchmark-random-state",
        help="[scIB] RNG seed for stratified benchmarking subsample.",
    ),
    # ------------------------------------------------------------------
    # scANVI supervision (NEW)
    # ------------------------------------------------------------------
    scanvi_label_source: str = typer.Option(
        "bisc_light",
        "--scanvi-label-source",
        help="[scANVI] How to generate supervision labels for scANVI: 'leiden' or 'bisc_light'.",
        case_sensitive=False,
    ),
    scanvi_labels_key: str = typer.Option(
        "leiden",
        "--scanvi-labels-key",
        help="[scANVI] If scanvi_label_source='leiden', use this adata.obs key as labels_key.",
    ),
    scanvi_prelabels_key: str = typer.Option(
        "scanvi_prelabels",
        "--scanvi-prelabels-key",
        help="[scANVI] If scanvi_label_source='bisc_light', write prelabels to this adata.obs key and use it for scANVI.",
    ),
    scanvi_preflight_resolutions: Optional[List[float]] = typer.Option(
        None,
        "--scanvi-preflight-resolutions",
        help="[scANVI/BISC-light] Sparse Leiden resolution grid (repeatable option). If omitted, uses defaults.",
    ),
    scanvi_max_prelabel_clusters: int = typer.Option(
        25,
        "--scanvi-max-prelabel-clusters",
        help="[scANVI/BISC-light] Cap the number of prelabel clusters (increase for atlas-scale data).",
    ),
    scanvi_preflight_min_stability: float = typer.Option(
        0.60,
        "--scanvi-preflight-min-stability",
        help="[scANVI/BISC-light] Minimum smoothed adjacent-ARI stability to be considered feasible.",
    ),
    scanvi_preflight_parsimony_eps: float = typer.Option(
        0.03,
        "--scanvi-preflight-parsimony-eps",
        help="[scANVI/BISC-light] Parsimony epsilon: pick lowest resolution within (1-eps) of best composite score.",
    ),
    scanvi_w_stability: float = typer.Option(
        0.50,
        "--scanvi-w-stability",
        help="[scANVI/BISC-light] Weight for stability term in composite score.",
    ),
    scanvi_w_silhouette: float = typer.Option(
        0.35,
        "--scanvi-w-silhouette",
        help="[scANVI/BISC-light] Weight for centroid-silhouette term in composite score.",
    ),
    scanvi_w_tiny: float = typer.Option(
        0.15,
        "--scanvi-w-tiny",
        help="[scANVI/BISC-light] Weight for tiny-cluster penalty term in composite score.",
    ),
    # ------------------------------------------------------------------
    # Guardrails (batch-trap / tiny clusters)
    # ------------------------------------------------------------------
    scanvi_batch_trap_threshold: float = typer.Option(
        0.90,
        "--scanvi-batch-trap-threshold",
        help="[scANVI] If a cluster is >= this fraction one batch, mark it 'Unknown' for scANVI supervision.",
    ),
    scanvi_batch_trap_min_cells: int = typer.Option(
        200,
        "--scanvi-batch-trap-min-cells",
        help="[scANVI] Only apply batch-trap logic to clusters with at least this many cells.",
    ),
    scanvi_tiny_cluster_min_cells: int = typer.Option(
        30,
        "--scanvi-tiny-cluster-min-cells",
        help="[scANVI] Cluster size below which clusters are marked 'Unknown' for scANVI supervision.",
    ),

    # ------------------------------------------------------------------
    # Annotated secondary integration (SECOND PASS; explicit + guarded)
    # ------------------------------------------------------------------
    annotated_run: bool = typer.Option(
        False,
        "--annotated-run/--no-annotated-run",
        help="[Annotated secondary] Run secondary integration using final annotated labels from cluster-and-annotate. "
             "This mode is explicit and should be used with care.",
    ),
    scib_truth_label_key: str = typer.Option(
        "leiden",
        "--scib-truth-label-key",
        help="[scIB] Label key treated as 'truth' for scIB. "
             "Defaults to 'leiden'. Set to 'celltypist' to use CellTypist cell-level labels "
             "(resolved from the selected cluster round, default=active).",
        case_sensitive=False,
    ),
    annotated_run_cluster_round: Optional[str] = typer.Option(
        None,
        "--annotated-run-cluster-round",
        help="[Annotated secondary] Source cluster round id (default: active_cluster_round).",
    ),
    annotated_run_final_label_key: Optional[str] = typer.Option(
        None,
        "--annotated-run-final-label-key",
        help="[Annotated secondary] Override final label key in adata.obs. "
             "Default: rounds[round_id]['annotation']['pretty_cluster_key'] or 'cluster_label'.",
    ),
    annotated_run_confidence_mask_key: str = typer.Option(
        "celltypist_confident_entropy_margin",
        "--annotated-run-confidence-mask-key",
        help="[Annotated secondary] adata.obs key to store the entropy_margin confidence mask.",
    ),
    annotated_run_scanvi_labels_key: str = typer.Option(
        "scanvi_labels__annotated",
        "--annotated-run-scanvi-labels-key",
        help="[Annotated secondary] adata.obs key to store scANVI supervision labels "
             "(final_label where confident, else 'Unknown').",
    ),
    bio_entropy_abs_limit: float = typer.Option(
        0.5,
        "--bio-entropy-abs-limit",
        help="[Annotated secondary] Entropy absolute limit for CellTypist confidence mask.",
    ),
    bio_entropy_quantile: float = typer.Option(
        0.7,
        "--bio-entropy-quantile",
        help="[Annotated secondary] Entropy quantile for CellTypist confidence mask (cut uses max(abs, quantile)).",
    ),
    bio_margin_min: float = typer.Option(
        0.10,
        "--bio-margin-min",
        help="[Annotated secondary] Minimum top1-top2 probability margin for CellTypist confidence mask.",
    ),

):
    outdir = output_dir or input_path.parent
    logfile = outdir / "integrate.log"
    init_logging(logfile)

    # Guardrail: annotated run is explicit, and only computes scANVI__annotated.
    # We still allow benchmarking against any pre-existing embeddings in the input AnnData.
    if annotated_run:
        if methods is not None:
            lowered = [str(m).lower() for m in methods]
            if lowered not in (["scanvi"], ["scanvi__annotated"], ["scanvi_annotated"]):
                LOGGER.warning(
                    "ANNOTATED RUN: ignoring --methods=%r and forcing scANVI-only computation.",
                    methods,
                )
        methods = ["scanvi"]  # compute only annotated scANVI embedding in integrate.py


    cfg = IntegrateConfig(
        input_path=input_path,
        output_dir=outdir,
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
        # scANVI supervision
        scanvi_label_source=str(scanvi_label_source).lower(),
        scanvi_labels_key=scanvi_labels_key,
        scanvi_prelabels_key=scanvi_prelabels_key,
        scanvi_preflight_resolutions=scanvi_preflight_resolutions,
        scanvi_max_prelabel_clusters=scanvi_max_prelabel_clusters,
        scanvi_preflight_min_stability=scanvi_preflight_min_stability,
        scanvi_preflight_parsimony_eps=scanvi_preflight_parsimony_eps,
        scanvi_w_stability=scanvi_w_stability,
        scanvi_w_silhouette=scanvi_w_silhouette,
        scanvi_w_tiny=scanvi_w_tiny,
        # guardrails
        scanvi_batch_trap_threshold=scanvi_batch_trap_threshold,
        scanvi_batch_trap_min_cells=scanvi_batch_trap_min_cells,
        scanvi_tiny_cluster_min_cells=scanvi_tiny_cluster_min_cells,
        logfile=logfile,
        # annotated secondary integration
        annotated_run=annotated_run,
        scib_truth_label_key=str(scib_truth_label_key).lower(),
        annotated_run_cluster_round=annotated_run_cluster_round,
        annotated_run_final_label_key=annotated_run_final_label_key,
        annotated_run_confidence_mask_key=annotated_run_confidence_mask_key,
        annotated_run_scanvi_labels_key=annotated_run_scanvi_labels_key,
        bio_entropy_abs_limit=bio_entropy_abs_limit,
        bio_entropy_quantile=bio_entropy_quantile,
        bio_margin_min=bio_margin_min,

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
        decoupler_bar_split_signed=decoupler_bar_split_signed,
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
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


# ======================================================================
#  markers-and-de
# ======================================================================
class RunWhich(str, Enum):
    both = "both"
    cell = "cell"
    pseudobulk = "pseudobulk"


def _parse_csv_repeat(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    out: List[str] = []
    for s in items:
        if s:
            out.extend([x.strip() for x in str(s).split(",") if x.strip()])
    seen = set()
    uniq: List[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


markers_and_de_app = typer.Typer(
    help="Discovery markers + DE (cluster-vs-rest and within-cluster contrasts)."
)
app.add_typer(markers_and_de_app, name="markers-and-de")


def _build_cfg(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    run: RunWhich,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    group_key: Optional[str],
    label_source: str,
    round_id: Optional[str],
    replicate_key: Optional[str],
    min_pct: float,
    min_diff_pct: float,
    # cell markers (cluster-vs-rest)
    cell_method: str,
    cell_n_genes: int,
    cell_rankby_abs: bool,
    cell_use_raw: bool,
    cell_downsample_threshold: int,
    cell_downsample_max_per_group: int,
    random_state: int,
    # pb
    pb_counts_layer: Optional[List[str]],
    pb_allow_x_counts: bool,
    pb_min_cells_per_replicate_group: int,
    pb_alpha: float,
    pb_store_key: str,
    pb_max_genes: Optional[int],
    pb_min_counts_per_lib: int,
    pb_min_lib_pct: float,
    pb_covariates: Optional[List[str]],
    prune_uns_de: bool,
    # within-cluster condition keys
    condition_keys: Optional[List[str]],
    # de-decoupler
    de_decoupler_source: str,
    de_decoupler_stat_col: str,
    decoupler_method: str,
    decoupler_consensus_methods: Optional[List[str]],
    decoupler_min_n_targets: int,
    decoupler_bar_split_signed: bool,
    decoupler_bar_top_n_up: Optional[int],
    decoupler_bar_top_n_down: Optional[int],
    msigdb_gene_sets: Optional[List[str]],
    msigdb_method: str,
    msigdb_min_n_targets: int,
    run_progeny: bool,
    progeny_method: str,
    progeny_min_n_targets: int,
    progeny_top_n: int,
    progeny_organism: str,
    run_dorothea: bool,
    dorothea_method: str,
    dorothea_min_n_targets: int,
    dorothea_confidence: Optional[List[str]],
    dorothea_organism: str,
    # plots
    plot_lfc_thresh: float,
    plot_volcano_top_label_n: int,
    plot_top_n_per_cluster: int,
    plot_dotplot_top_n_per_cluster: int,
    plot_max_genes_total: int,
    plot_use_raw: bool,
    plot_layer: Optional[str],
    plot_umap_ncols: int,
) -> MarkersAndDEConfig:
    out_dir = output_dir or input_path.parent
    log_path = out_dir / "markers-and-de.log"
    init_logging(log_path)

    layers = _parse_csv_repeat(pb_counts_layer) or ["counts_cb", "counts_raw"]
    covariates = tuple(_parse_csv_repeat(pb_covariates) or ())
    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    msigdb_sets = list(msigdb_gene_sets) if msigdb_gene_sets else None
    doro_conf = list(dorothea_confidence) if dorothea_confidence else None

    return MarkersAndDEConfig(
        input_path=input_path,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run=str(run.value),
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        # grouping
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        # replicate: single user-facing choice
        sample_key=replicate_key,
        batch_key=None,
        # shared filters
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        # cell markers (cluster-vs-rest)
        markers_key="cluster_markers_wilcoxon",
        markers_method=str(cell_method).lower(),
        markers_n_genes=cell_n_genes,
        markers_rankby_abs=cell_rankby_abs,
        markers_use_raw=cell_use_raw,
        markers_downsample_threshold=cell_downsample_threshold,
        markers_downsample_max_per_group=cell_downsample_max_per_group,
        random_state=random_state,
        # pseudobulk
        counts_layers=tuple(layers),
        allow_X_counts=pb_allow_x_counts,
        min_cells_target=pb_min_cells_per_replicate_group,
        alpha=pb_alpha,
        store_key=pb_store_key,
        pb_max_genes=pb_max_genes,
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=tuple(_parse_csv_repeat(pb_covariates) or ()),
        prune_uns_de=prune_uns_de,
        condition_keys=cond_keys,
        de_decoupler_source=str(de_decoupler_source),
        de_decoupler_stat_col=str(de_decoupler_stat_col),
        decoupler_method=str(decoupler_method),
        decoupler_consensus_methods=decoupler_consensus_methods,
        decoupler_min_n_targets=int(decoupler_min_n_targets),
        decoupler_bar_split_signed=bool(decoupler_bar_split_signed),
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
        msigdb_gene_sets=msigdb_sets if msigdb_sets is not None else ["HALLMARK", "REACTOME"],
        msigdb_method=str(msigdb_method),
        msigdb_min_n_targets=int(msigdb_min_n_targets),
        run_progeny=bool(run_progeny),
        progeny_method=str(progeny_method),
        progeny_min_n_targets=int(progeny_min_n_targets),
        progeny_top_n=int(progeny_top_n),
        progeny_organism=str(progeny_organism),
        run_dorothea=bool(run_dorothea),
        dorothea_method=str(dorothea_method),
        dorothea_min_n_targets=int(dorothea_min_n_targets),
        dorothea_confidence=doro_conf if doro_conf is not None else ["A", "B", "C"],
        dorothea_organism=str(dorothea_organism),
        # plots
        plot_lfc_thresh=plot_lfc_thresh,
        plot_volcano_top_label_n=plot_volcano_top_label_n,
        plot_top_n_per_cluster=plot_top_n_per_cluster,
        plot_dotplot_top_n_genes=plot_dotplot_top_n_per_cluster,
        plot_max_genes_total=plot_max_genes_total,
        plot_use_raw=plot_use_raw,
        plot_layer=plot_layer,
        plot_umap_ncols=plot_umap_ncols,
    )


def _default_output_name(input_path: Path, suffix: str) -> str:
    name = input_path.name
    for ext in (".zarr", ".h5ad", ".h5", ".hdf5"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return f"{name}.{suffix}"


@markers_and_de_app.command(
    "markers",
    help="Markers: Define marker genes for each cluster (vs all others.)",
)
def cluster_vs_rest(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),

    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),

    run: RunWhich = typer.Option(RunWhich.both, "--run", case_sensitive=False),

    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    replicate_key: Optional[str] = typer.Option(None, "--replicate-key"),

    min_pct: float = typer.Option(0.25, "--min-pct"),
    min_diff_pct: float = typer.Option(0.25, "--min-diff-pct"),

    cell_method: str = typer.Option("wilcoxon", "--cell-method"),
    cell_n_genes: int = typer.Option(300, "--cell-n-genes"),
    cell_rankby_abs: bool = typer.Option(True, "--cell-rankby-abs/--no-cell-rankby-abs"),
    cell_use_raw: bool = typer.Option(False, "--cell-use-raw/--no-cell-use-raw"),
    cell_downsample_threshold: int = typer.Option(500_000, "--cell-downsample-threshold"),
    cell_downsample_max_per_group: int = typer.Option(2_000, "--cell-downsample-max-per-group"),
    random_state: int = typer.Option(42, "--random-state"),

    pb_counts_layer: List[str] = typer.Option(["counts_cb", "counts_raw"], "--pb-counts-layer"),
    pb_allow_x_counts: bool = typer.Option(True, "--pb-allow-x-counts/--no-pb-allow-x-counts"),
    pb_min_cells_per_replicate_group: int = typer.Option(20, "--pb-min-cells-per-replicate-group"),
    pb_alpha: float = typer.Option(0.05, "--pb-alpha"),
    pb_store_key: str = typer.Option("scomnom_de", "--pb-store-key"),
    pb_max_genes: Optional[int] = typer.Option(None, "--pb-max-genes"),
    pb_min_counts_per_lib: int = typer.Option(0, "--pb-min-counts-per-lib"),
    pb_min_lib_pct: float = typer.Option(0.0, "--pb-min-lib-pct"),
    pb_covariates: List[str] = typer.Option([], "--pb-covariates"),
    prune_uns_de: bool = typer.Option(True, "--prune-uns-de/--no-prune-uns-de"),

    de_decoupler_source: str = typer.Option(
        "auto",
        "--de-decoupler-source",
        help="DE-decoupler input source: auto, all, pseudobulk, cell, none.",
    ),
    de_decoupler_stat_col: str = typer.Option("stat", "--de-decoupler-stat-col"),
    decoupler_method: str = typer.Option("consensus", "--decoupler-method"),
    decoupler_consensus_methods: Optional[List[str]] = typer.Option(
        None,
        "--decoupler-consensus-methods",
        callback=validate_decoupler_consensus_methods,
    ),
    decoupler_min_n_targets: int = typer.Option(5, "--decoupler-min-n-targets"),
    decoupler_bar_split_signed: bool = typer.Option(
        True,
        "--decoupler-bar-split-signed/--no-decoupler-bar-split-signed",
    ),
    decoupler_bar_top_n_up: Optional[int] = typer.Option(None, "--decoupler-bar-top-n-up"),
    decoupler_bar_top_n_down: Optional[int] = typer.Option(None, "--decoupler-bar-top-n-down"),
    msigdb_gene_sets_cli: Optional[str] = typer.Option(
        None,
        "--msigdb-gene-sets",
        help="[MSigDB] Comma-separated MSigDB keywords or paths to .gmt files.",
        autocompletion=_gene_sets_completion,
    ),
    msigdb_method: str = typer.Option("consensus", "--msigdb-method"),
    msigdb_min_n_targets: int = typer.Option(5, "--msigdb-min-n-targets"),

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

    plot_lfc_thresh: float = typer.Option(1.0, "--plot-lfc-thresh"),
    plot_volcano_top_label_n: int = typer.Option(15, "--plot-volcano-top-label-n"),
    plot_top_n_per_cluster: int = typer.Option(10, "--plot-top-n-per-cluster"),
    plot_dotplot_top_n_per_cluster: int = typer.Option(15, "--plot-dotplot-top-n-per-cluster"),
    plot_max_genes_total: int = typer.Option(80, "--plot-max-genes-total"),
    plot_use_raw: bool = typer.Option(False, "--plot-use-raw/--no-plot-use-raw"),
    plot_layer: Optional[str] = typer.Option(None, "--plot-layer"),
    plot_umap_ncols: int = typer.Option(3, "--plot-umap-ncols"),
):
    if output_name is None:
        output_name = _default_output_name(input_path, "markers")
    de_source = str(de_decoupler_source or "auto").lower()
    if de_source not in ("auto", "all", "pseudobulk", "cell", "none"):
        raise typer.BadParameter("Invalid --de-decoupler-source. Use: auto, all, pseudobulk, cell, none.")

    cfg = _build_cfg(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        run=run,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        group_key=group_key,
        label_source=label_source,
        round_id=round_id,
        replicate_key=replicate_key,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        cell_method=cell_method,
        cell_n_genes=cell_n_genes,
        cell_rankby_abs=cell_rankby_abs,
        cell_use_raw=cell_use_raw,
        cell_downsample_threshold=cell_downsample_threshold,
        cell_downsample_max_per_group=cell_downsample_max_per_group,
        random_state=random_state,
        pb_counts_layer=pb_counts_layer,
        pb_allow_x_counts=pb_allow_x_counts,
        pb_min_cells_per_replicate_group=pb_min_cells_per_replicate_group,
        pb_alpha=pb_alpha,
        pb_store_key=pb_store_key,
        pb_max_genes=pb_max_genes,
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=tuple(_parse_csv_repeat(pb_covariates) or ()),
        prune_uns_de=prune_uns_de,
        condition_keys=[],
        de_decoupler_source="none",
        de_decoupler_stat_col="stat",
        decoupler_method="consensus",
        decoupler_consensus_methods=None,
        decoupler_min_n_targets=5,
        msigdb_gene_sets=None,
        msigdb_method="consensus",
        msigdb_min_n_targets=5,
        run_progeny=True,
        progeny_method="consensus",
        progeny_min_n_targets=5,
        progeny_top_n=100,
        progeny_organism="human",
        run_dorothea=True,
        dorothea_method="consensus",
        dorothea_min_n_targets=5,
        dorothea_confidence=["A", "B", "C"],
        dorothea_organism="human",
        plot_lfc_thresh=plot_lfc_thresh,
        plot_volcano_top_label_n=plot_volcano_top_label_n,
        plot_top_n_per_cluster=plot_top_n_per_cluster,
        plot_dotplot_top_n_per_cluster=plot_dotplot_top_n_per_cluster,
        plot_max_genes_total=plot_max_genes_total,
        plot_use_raw=plot_use_raw,
        plot_layer=plot_layer,
        plot_umap_ncols=plot_umap_ncols,
    )

    # enforce mode semantics
    cfg.condition_key = None
    cfg.condition_contrasts = ()
    cfg.contrast_key = None
    cfg.contrast_contrasts = ()
    cfg.contrast_conditional_de = False

    run_cluster_vs_rest(cfg)


@markers_and_de_app.command(
    "de",
    help="Within-cluster contrasts: compare condition levels within each group.",
)
def within_cluster(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),

    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),

    run: RunWhich = typer.Option(RunWhich.both, "--run", case_sensitive=False),

    group_key: Optional[Path] = typer.Option(None, "--group-key"),  # NOTE: fixed below to Optional[str]
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    replicate_key: Optional[str] = typer.Option(None, "--replicate-key"),

    condition_key: Optional[str] = typer.Option(None, "--condition-key"),
    condition_keys: List[str] = typer.Option([], "--condition-keys"),
    contrasts: Optional[List[str]] = typer.Option(None, "--contrasts"),

    min_pct: float = typer.Option(0.25, "--min-pct"),
    min_diff_pct: float = typer.Option(0.25, "--min-diff-pct"),

    # kept for config compatibility; not used by within-cluster orchestrator for cell-level
    cell_method: str = typer.Option("wilcoxon", "--cell-method"),
    cell_n_genes: int = typer.Option(300, "--cell-n-genes"),
    cell_rankby_abs: bool = typer.Option(True, "--cell-rankby-abs/--no-cell-rankby-abs"),
    cell_use_raw: bool = typer.Option(False, "--cell-use-raw/--no-cell-use-raw"),
    cell_downsample_threshold: int = typer.Option(500_000, "--cell-downsample-threshold"),
    cell_downsample_max_per_group: int = typer.Option(2_000, "--cell-downsample-max-per-group"),
    random_state: int = typer.Option(42, "--random-state"),

    pb_counts_layer: List[str] = typer.Option(["counts_cb", "counts_raw"], "--pb-counts-layer"),
    pb_allow_x_counts: bool = typer.Option(True, "--pb-allow-x-counts/--no-pb-allow-x-counts"),
    pb_min_cells_per_replicate_group: int = typer.Option(20, "--pb-min-cells-per-replicate-group"),
    pb_alpha: float = typer.Option(0.05, "--pb-alpha"),
    pb_store_key: str = typer.Option("scomnom_de", "--pb-store-key"),
    pb_max_genes: Optional[int] = typer.Option(None, "--pb-max-genes"),
    pb_min_counts_per_lib: int = typer.Option(0, "--pb-min-counts-per-lib"),
    pb_min_lib_pct: float = typer.Option(0.0, "--pb-min-lib-pct"),
    pb_covariates: List[str] = typer.Option([], "--pb-covariates"),
    prune_uns_de: bool = typer.Option(True, "--prune-uns-de/--no-prune-uns-de"),

    de_decoupler_source: str = typer.Option(
        "auto",
        "--de-decoupler-source",
        help="DE-decoupler input source: auto, all, pseudobulk, cell, none.",
    ),
    de_decoupler_stat_col: str = typer.Option("stat", "--de-decoupler-stat-col"),
    decoupler_method: str = typer.Option("consensus", "--decoupler-method"),
    decoupler_consensus_methods: Optional[List[str]] = typer.Option(
        None,
        "--decoupler-consensus-methods",
        callback=validate_decoupler_consensus_methods,
    ),
    decoupler_min_n_targets: int = typer.Option(5, "--decoupler-min-n-targets"),
    msigdb_gene_sets_cli: Optional[str] = typer.Option(
        None,
        "--msigdb-gene-sets",
        help="[MSigDB] Comma-separated MSigDB keywords or paths to .gmt files.",
        autocompletion=_gene_sets_completion,
    ),
    msigdb_method: str = typer.Option("consensus", "--msigdb-method"),
    msigdb_min_n_targets: int = typer.Option(5, "--msigdb-min-n-targets"),

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

    plot_lfc_thresh: float = typer.Option(1.0, "--plot-lfc-thresh"),
    plot_volcano_top_label_n: int = typer.Option(15, "--plot-volcano-top-label-n"),
    plot_top_n_per_cluster: int = typer.Option(10, "--plot-top-n-per-cluster"),
    plot_dotplot_top_n_per_cluster: int = typer.Option(15, "--plot-dotplot-top-n-per-cluster"),
    plot_max_genes_total: int = typer.Option(80, "--plot-max-genes-total"),
    plot_use_raw: bool = typer.Option(False, "--plot-use-raw/--no-plot-use-raw"),
    plot_layer: Optional[str] = typer.Option(None, "--plot-layer"),
    plot_umap_ncols: int = typer.Option(3, "--plot-umap-ncols"),
):
    # fix typo from signature (keep CLI option stable)
    group_key = None if group_key is None else str(group_key)
    if output_name is None:
        output_name = _default_output_name(input_path, "de")

    if msigdb_gene_sets_cli is None:
        msigdb_gene_sets = None
    else:
        msigdb_gene_sets = [x.strip() for x in msigdb_gene_sets_cli.split(",") if x.strip()]

    de_source = str(de_decoupler_source or "auto").lower()
    if de_source not in ("auto", "all", "pseudobulk", "cell", "none"):
        raise typer.BadParameter("Invalid --de-decoupler-source. Use: auto, all, pseudobulk, cell, none.")

    cfg = _build_cfg(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        run=run,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        group_key=group_key,  # now str|None
        label_source=label_source,
        round_id=round_id,
        replicate_key=replicate_key,
        min_pct=min_pct,
        min_diff_pct=min_diff_pct,
        cell_method=cell_method,
        cell_n_genes=cell_n_genes,
        cell_rankby_abs=cell_rankby_abs,
        cell_use_raw=cell_use_raw,
        cell_downsample_threshold=cell_downsample_threshold,
        cell_downsample_max_per_group=cell_downsample_max_per_group,
        random_state=random_state,
        pb_counts_layer=pb_counts_layer,
        pb_allow_x_counts=pb_allow_x_counts,
        pb_min_cells_per_replicate_group=pb_min_cells_per_replicate_group,
        pb_alpha=pb_alpha,
        pb_store_key=pb_store_key,
        pb_max_genes=pb_max_genes,
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=pb_covariates,
        prune_uns_de=prune_uns_de,
        condition_keys=condition_keys,
        de_decoupler_source=de_source,
        de_decoupler_stat_col=de_decoupler_stat_col,
        decoupler_method=decoupler_method,
        decoupler_consensus_methods=decoupler_consensus_methods,
        decoupler_min_n_targets=decoupler_min_n_targets,
        decoupler_bar_split_signed=decoupler_bar_split_signed,
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
        msigdb_gene_sets=msigdb_gene_sets,
        msigdb_method=msigdb_method,
        msigdb_min_n_targets=msigdb_min_n_targets,
        run_progeny=run_progeny,
        progeny_method=progeny_method,
        progeny_min_n_targets=progeny_min_n_targets,
        progeny_top_n=progeny_top_n,
        progeny_organism=progeny_organism,
        run_dorothea=run_dorothea,
        dorothea_method=dorothea_method,
        dorothea_min_n_targets=dorothea_min_n_targets,
        dorothea_confidence=[x.strip() for x in dorothea_confidence.split(",") if x.strip()],
        dorothea_organism=dorothea_organism,
        plot_lfc_thresh=plot_lfc_thresh,
        plot_volcano_top_label_n=plot_volcano_top_label_n,
        plot_top_n_per_cluster=plot_top_n_per_cluster,
        plot_dotplot_top_n_per_cluster=plot_dotplot_top_n_per_cluster,
        plot_max_genes_total=plot_max_genes_total,
        plot_use_raw=plot_use_raw,
        plot_layer=plot_layer,
        plot_umap_ncols=plot_umap_ncols,
    )

    parsed = _parse_csv_repeat(contrasts)
    cond_keys = _parse_csv_repeat(condition_keys)
    if not cond_keys and condition_key:
        cond_keys = [str(condition_key)]
    if not cond_keys:
        raise typer.BadParameter("Provide at least one --condition-key or --condition-keys.")

    cfg.condition_key = None
    cfg.condition_keys = tuple(cond_keys)
    cfg.condition_contrasts = tuple(parsed)

    # drive cell-level within-cluster contrasts (NO pseudobulk guard in orchestrator)
    cfg.contrast_key = None
    cfg.contrast_contrasts = tuple(parsed)
    cfg.contrast_conditional_de = run in (RunWhich.cell, RunWhich.both)

    run_within_cluster(cfg)


if __name__ == "__main__":
    app()
