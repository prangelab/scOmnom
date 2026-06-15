from __future__ import annotations
from typing import Optional, List, Literal, Dict, Sequence, Tuple
import typer
from pathlib import Path
import warnings
import re
from enum import Enum
from pandas.errors import PerformanceWarning

from .load_and_filter import run_load_and_filter
from .integrate import run_integrate
from .adata_ops import run_adata_ops
from .cluster_and_annotate import run_clustering
from .markers_and_de import (
    run_cluster_vs_rest,
    run_within_cluster,
    run_composition,
    run_enrichment_cluster,
    run_enrichment_de as run_enrichment_de_from_tables,
    run_module_score,
    run_liana_ccc,
    run_liana_paired_rescore,
    run_mebocost_ccc,
    run_mebocost_paired_rescore,
    run_nichenet_ccc,
)

from .config import (
    LoadAndFilterConfig,
    IntegrateConfig,
    AdataOpsConfig,
    ClusterAnnotateConfig,
    MarkersAndDEConfig,
)
import logging
from .logging_utils import init_logging


ALLOWED_METHODS = {"scVI", "scANVI", "Harmony", "Scanorama", "BBKNN"}
ALLOWED_DECOUPLER_METHODS = {"ulm", "mlm", "wsum", "aucell"}
ALLOWED_COMP_METHODS = {"sccoda", "glm", "clr", "graph"}
ALLOWED_LIANA_METHODS = {"rank_aggregate", "cellphonedb", "connectome", "natmi", "sca", "logfc"}
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


def _parse_optional_float_or_none(value: Optional[str], *, param_name: str) -> Optional[float]:
    if value is None:
        return None

    raw = value.strip()
    if raw == "":
        raise typer.BadParameter(f"{param_name} cannot be empty.")
    if raw.lower() in {"none", "null", "na", "n/a"}:
        return None

    try:
        return float(raw)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{param_name} must be a float or 'none'."
        ) from exc


def validate_liana_methods(
    value: Optional[List[str]],
) -> Optional[List[str]]:
    if value is None:
        return None

    seen = set()
    methods: list[str] = []
    for raw in value:
        if raw is None:
            continue
        for part in str(raw).split(","):
            method = part.strip().lower()
            if not method:
                continue
            if method not in ALLOWED_LIANA_METHODS:
                raise typer.BadParameter(
                    f"Invalid LIANA method '{method}'. Allowed methods: {sorted(ALLOWED_LIANA_METHODS)}"
                )
            if method not in seen:
                seen.add(method)
                methods.append(method)

    if not methods:
        raise typer.BadParameter("Provide at least one valid --liana-method.")

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
    min_counts: Optional[int] = typer.Option(
        None,
        "--min-counts",
        help="[QC] Minimum total UMI counts per cell. Default: None.",
    ),
    min_counts_mad: str = typer.Option(
        "5.0",
        "--min-counts-mad",
        help="[QC] Lower cutoff for total UMI counts as median - k*MAD. Use 'none' to disable. Default: 5.0.",
    ),
    min_counts_quantile: str = typer.Option(
        "0.01",
        "--min-counts-quantile",
        help="[QC] Lower quantile cutoff for total UMI counts. Use 'none' to disable. Default: 0.01.",
    ),
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
        help="Path to pre-doublet AnnData (.zarr, .zarr.tar.zst, or .h5ad). Defaults to <out>/adata.merged.zarr.",
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
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "load-and-filter.log"
    init_logging(logfile)
    logging.getLogger(__name__).info("Logging initialized")
    logging.getLogger("scomnom.load_and_filter").info("load_and_filter logger active")

    if apply_doublet_score:
        if apply_doublet_score_path is None:
            apply_doublet_score_path = output_dir / "adata.merged.zarr"

    min_counts_mad_parsed = _parse_optional_float_or_none(
        min_counts_mad,
        param_name="--min-counts-mad",
    )
    min_counts_quantile_parsed = _parse_optional_float_or_none(
        min_counts_quantile,
        param_name="--min-counts-quantile",
    )

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
        min_counts=min_counts,
        min_counts_mad=min_counts_mad_parsed,
        min_counts_quantile=min_counts_quantile_parsed,
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
        help="[I/O] Dataset produced by load-and-filter (.zarr, .zarr.tar.zst, or .h5ad).",
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
    multi_gpu: bool = typer.Option(
        False,
        "--multi-gpu/--single-gpu",
        help="[scVI/scANVI] Use all available CUDA devices via DDP.",
    ),
    # ------------------------------------------------------------------
    # scANVI supervision (NEW)
    # ------------------------------------------------------------------
    scanvi_label_source: str = typer.Option(
        "bisc",
        "--scanvi-label-source",
        help="[scANVI] How to generate supervision labels for scANVI: 'leiden' or 'bisc'.",
        case_sensitive=False,
    ),
    scanvi_labels_key: str = typer.Option(
        "leiden",
        "--scanvi-labels-key",
        help="[scANVI] If scanvi_label_source='leiden', use this adata.obs key as labels_key.",
    ),
    # ------------------------------------------------------------------
    # CellTypist (shared)
    # ------------------------------------------------------------------
    celltypist_model: Optional[str] = typer.Option(
        "Immune_All_Low.pkl",
        "--celltypist-model",
        help="[CellTypist] Path or name of model. If None, skip annotation.",
        autocompletion=_celltypist_models_completion,
    ),
    celltypist_majority_voting: bool = typer.Option(
        True,
        "--celltypist-majority-voting/--no-celltypist-majority-voting",
        help="[CellTypist] Use majority voting.",
    ),
    celltypist_label_key: str = typer.Option(
        "celltypist_label",
        "--celltypist-label-key",
        help="[CellTypist] adata.obs key to store cell-level labels.",
    ),
    celltypist_cluster_label_key: str = typer.Option(
        "celltypist_cluster_label",
        "--celltypist-cluster-label-key",
        help="[CellTypist] adata.obs key to store cluster-level labels.",
    ),
    bio_mask_mode: Literal["entropy_margin", "none"] = typer.Option(
        "entropy_margin",
        "--bio-mask-mode",
        help="[Bio] CellTypist confidence mask mode for bio metrics: entropy_margin or none.",
    ),
    bio_entropy_abs_limit: float = typer.Option(
        0.5,
        "--bio-entropy-abs-limit",
        help="[Bio] Entropy absolute limit for CellTypist confidence mask.",
    ),
    bio_entropy_quantile: float = typer.Option(
        0.7,
        "--bio-entropy-quantile",
        help="[Bio] Entropy quantile for CellTypist confidence mask (cut uses max(abs, quantile)).",
    ),
    bio_margin_min: float = typer.Option(
        0.10,
        "--bio-margin-min",
        help="[Bio] Minimum top1-top2 probability margin for CellTypist confidence mask.",
    ),
    bio_mask_min_cells: int = typer.Option(
        500,
        "--bio-mask-min-cells",
        help="[Bio] Disable bio mask if fewer than this many cells pass.",
    ),
    bio_mask_min_frac: float = typer.Option(
        0.05,
        "--bio-mask-min-frac",
        help="[Bio] Disable bio mask if fewer than this fraction of cells pass.",
    ),
    pretty_label_min_masked_cells: int = typer.Option(
        25,
        "--pretty-label-min-masked-cells",
        help="[CellTypist] Min masked cells in a cluster to assign cluster-level label; else Unknown.",
    ),
    pretty_label_min_masked_frac: float = typer.Option(
        0.10,
        "--pretty-label-min-masked-frac",
        help="[CellTypist] Min masked fraction in a cluster to assign cluster-level label; else Unknown.",
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
        "celltypist",
        "--scib-truth-label-key",
        help="[scIB] Label key treated as 'truth' for scIB. "
             "Defaults to 'celltypist'. Set to 'leiden' or 'final' to use existing labels.",
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
):
    outdir = output_dir or input_path.parent
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "integrate.log"
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
        multi_gpu=multi_gpu,
        # scANVI supervision
        scanvi_label_source=str(scanvi_label_source).lower(),
        scanvi_labels_key=scanvi_labels_key,
        logfile=logfile,
        # CellTypist
        celltypist_model=celltypist_model,
        celltypist_majority_voting=celltypist_majority_voting,
        celltypist_label_key=celltypist_label_key,
        celltypist_cluster_label_key=celltypist_cluster_label_key,
        # Bio mask
        bio_mask_mode=bio_mask_mode,
        bio_mask_min_cells=bio_mask_min_cells,
        bio_mask_min_frac=bio_mask_min_frac,
        pretty_label_min_masked_cells=pretty_label_min_masked_cells,
        pretty_label_min_masked_frac=pretty_label_min_masked_frac,
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
#  adata-ops
# ======================================================================
adata_ops_app = typer.Typer(
    help="AnnData object operations.",
)
app.add_typer(adata_ops_app, name="adata-ops")


@adata_ops_app.command(
    "subset",
    help="Subset input AnnData into named outputs based on a Cnn -> subset TSV.",
)
def adata_ops_subset(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Input dataset (.zarr, .zarr.tar.zst, or .h5ad).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output root directory (default: input parent).",
    ),
    subset_mapping_tsv: Path = typer.Option(
        ...,
        "--subset-mapping-tsv",
        "-s",
        help="[Subset] Two-column, tab-delimited file (no header): Cnn<tab>subset_name.",
    ),
    output_format: Optional[Literal["zarr", "h5ad"]] = typer.Option(
        None,
        "--output-format",
        help="[I/O] Output format for subsetted datasets. Default: match input when possible.",
    ),
    round_id: Optional[str] = typer.Option(
        None,
        "--round-id",
        help="[Subset] Optional cluster round id used to resolve pretty labels for Cnn parsing.",
    ),
):
    out_dir = output_dir or input_path.parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "adata-ops.log"
    init_logging(logfile)

    cfg = AdataOpsConfig(
        input_path=input_path,
        output_dir=out_dir,
        operation="subset",
        subset_mapping_tsv=subset_mapping_tsv,
        output_format=output_format,
        round_id=round_id,
        logfile=logfile,
    )
    run_adata_ops(cfg)


@adata_ops_app.command(
    "rename",
    help="Rename pretty cluster labels by creating a new derived round.",
)
def adata_ops_rename(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Input dataset (.zarr, .zarr.tar.zst, or .h5ad).",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: sibling results/ directory, or input parent if already inside results/).",
    ),
    output_name: Optional[str] = typer.Option(
        None,
        "--output-name",
        help="[I/O] Base name for renamed output dataset.",
    ),
    rename_idents_file: Path = typer.Option(
        ...,
        "--rename-idents-file",
        help="[Rename] Two-column, tab-delimited file (no header): Cnn<tab>New Label.",
    ),
    output_format: Optional[Literal["zarr", "h5ad"]] = typer.Option(
        None,
        "--output-format",
        help="[I/O] Output format for renamed dataset. Default: match input when possible.",
    ),
    round_id: Optional[str] = typer.Option(
        None,
        "--round-id",
        help="[Rename] Parent cluster round id (default: active_cluster_round).",
    ),
    target_round_id: Optional[str] = typer.Option(
        None,
        "--target-round-id",
        help="[Rename] Existing manual_rename round to update.",
    ),
    update_existing_round: bool = typer.Option(
        False,
        "--update-existing-round",
        help="[Rename] Update an existing manual_rename round instead of creating a new one.",
    ),
    rename_round_name: str = typer.Option(
        "manual_rename",
        "--rename-round-name",
        help="[Rename] Name for the new manual rename round id.",
    ),
    rename_collapse_same_labels: bool = typer.Option(
        False,
        "--collapse-same-labels/--no-collapse-same-labels",
        help="[Rename] Merge renamed clusters that share the same target label and renumber by size.",
    ),
    rename_set_active: bool = typer.Option(
        True,
        "--set-active/--no-set-active",
        help="[Rename] Make the new rename round the active round.",
    ),
):
    out_dir = output_dir or input_path.parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "adata-ops.log"
    init_logging(logfile)

    cfg = AdataOpsConfig(
        input_path=input_path,
        output_dir=out_dir,
        operation="rename",
        output_name=output_name,
        rename_idents_file=rename_idents_file,
        output_format=output_format,
        round_id=round_id,
        target_round_id=target_round_id,
        update_existing_round=update_existing_round,
        rename_round_name=rename_round_name,
        rename_collapse_same_labels=rename_collapse_same_labels,
        rename_set_active=rename_set_active,
        logfile=logfile,
    )
    run_adata_ops(cfg)


@adata_ops_app.command(
    "annotation-merge",
    help="Overlay one or more subset child annotation rounds back into a parent dataset.",
)
def adata_ops_annotation_merge(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Parent dataset (.zarr, .zarr.tar.zst, or .h5ad).",
    ),
    child_paths: List[Path] = typer.Option(
        ...,
        "--child-path",
        help="[Merge] Child subset dataset path. Repeat for multiple children.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: input parent).",
    ),
    output_format: Optional[Literal["zarr", "h5ad"]] = typer.Option(
        None,
        "--output-format",
        help="[I/O] Output format for merged dataset. Default: match input when possible.",
    ),
    round_id: Optional[str] = typer.Option(
        None,
        "--round-id",
        help="[Merge] Base parent round id. Default: active_cluster_round.",
    ),
    child_round_id: Optional[str] = typer.Option(
        None,
        "--child-round-id",
        help="[Merge] Child round id to read from each child dataset. Default: active_cluster_round.",
    ),
    child_source_field: Optional[str] = typer.Option(
        None,
        "--child-source-field",
        help="[Merge] Explicit child obs field to overlay. Default: child round pretty_cluster_key.",
    ),
    target_round_id: Optional[str] = typer.Option(
        None,
        "--target-round-id",
        help="[Merge] Existing subset_annotation round to update.",
    ),
    update_existing_round: bool = typer.Option(
        False,
        "--update-existing-round",
        help="[Merge] Update an existing subset_annotation round instead of creating a new one.",
    ),
    annotation_merge_round_name: str = typer.Option(
        "subset_annotation",
        "--annotation-merge-round-name",
        help="[Merge] Suffix for a newly created annotation-merge round.",
    ),
):
    out_dir = output_dir or input_path.parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "adata-ops.log"
    init_logging(logfile)

    cfg = AdataOpsConfig(
        input_path=input_path,
        output_dir=out_dir,
        operation="annotation_merge",
        output_format=output_format,
        round_id=round_id,
        child_paths=tuple(child_paths),
        child_round_id=child_round_id,
        child_source_field=child_source_field,
        target_round_id=target_round_id,
        update_existing_round=update_existing_round,
        annotation_merge_round_name=annotation_merge_round_name,
        logfile=logfile,
    )
    run_adata_ops(cfg)


@adata_ops_app.command(
    "merge",
    help="Merge two or more AnnData datasets with optional cluster-based subset selection per input.",
)
def adata_ops_merge(
    input_paths: List[Path] = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Input dataset path. Repeat for multiple datasets.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: parent of first input).",
    ),
    output_name: Optional[str] = typer.Option(
        None,
        "--output-name",
        help="[I/O] Base name for merged output dataset (default: adata.merged).",
    ),
    output_format: Optional[Literal["zarr", "h5ad"]] = typer.Option(
        None,
        "--output-format",
        help="[I/O] Output format for merged dataset. Default: match first input when possible.",
    ),
    dataset_short_labels: Optional[List[str]] = typer.Option(
        None,
        "--dataset-short-label",
        help="[Merge] Short dataset label per input (repeat in same order as -i). Default: dataset1,dataset2,...",
    ),
    subset_merge_tsv: Optional[Path] = typer.Option(
        None,
        "--subset-merge",
        help="[Merge] Two-column TSV: dataset_basename<TAB>cluster_token. If set, only listed selections are merged.",
    ),
    round_id: Optional[str] = typer.Option(
        None,
        "--round-id",
        help="[Merge] Cluster round id used for cluster token resolution (default: active round).",
    ),
    cluster_key: Optional[str] = typer.Option(
        None,
        "--cluster-key",
        help="[Merge] Fallback obs key for cluster labels when round metadata is unavailable.",
    ),
    join: Literal["outer", "inner"] = typer.Option(
        "outer",
        "--join",
        help="[Merge] Feature join mode across inputs.",
    ),
    recompute_embedding: bool = typer.Option(
        True,
        "--recompute-embedding/--no-recompute-embedding",
        help="[Merge] Recompute PCA/neighbors/UMAP after merge (default: enabled).",
    ),
):
    if len(input_paths) < 2:
        raise typer.BadParameter("merge requires at least two --input-path values.")
    out_dir = output_dir or input_paths[0].parent
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "adata-ops.log"
    init_logging(logfile)

    cfg = AdataOpsConfig(
        input_path=input_paths[0],
        input_paths=tuple(input_paths),
        dataset_short_labels=tuple(dataset_short_labels or ()),
        output_dir=out_dir,
        operation="merge",
        output_name=output_name,
        output_format=output_format,
        subset_merge_tsv=subset_merge_tsv,
        round_id=round_id,
        cluster_key=cluster_key,
        join=join,
        recompute_embedding=recompute_embedding,
        logfile=logfile,
    )
    run_adata_ops(cfg)


@adata_ops_app.command(
    "metadata-import",
    help="Import or replace one or more obs metadata columns from an external table using strict key alignment.",
)
def adata_ops_metadata_import(
    input_path: Path = typer.Option(
        ...,
        "--input-path",
        "-i",
        help="[I/O] Input dataset (.zarr, .zarr.tar.zst, or .h5ad).",
    ),
    metadata_file: Path = typer.Option(
        ...,
        "--metadata-file",
        "-m",
        help="[Metadata] External metadata table (.tsv by default, .csv supported).",
    ),
    metadata_key: str = typer.Option(
        ...,
        "--metadata-key",
        help="[Metadata] Column in the metadata table used for key alignment.",
    ),
    obs_key: Optional[str] = typer.Option(
        None,
        "--obs-key",
        help="[Metadata] obs column used for key alignment. Default: obs_names.",
    ),
    columns: List[str] = typer.Option(
        [],
        "--column",
        help="[Metadata] Column(s) to import. Repeat to restrict import. Default: import all columns except --metadata-key.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: input parent).",
    ),
    output_name: Optional[str] = typer.Option(
        None,
        "--output-name",
        help="[I/O] Base name for metadata-imported output dataset.",
    ),
    output_format: Optional[Literal["zarr", "h5ad"]] = typer.Option(
        None,
        "--output-format",
        help="[I/O] Output format for updated dataset. Default: .h5ad for .h5ad inputs, otherwise compressed .zarr.tar.zst.",
    ),
):
    cfg = AdataOpsConfig(
        input_path=input_path,
        output_dir=output_dir,
        operation="metadata_import",
        output_name=output_name,
        output_format=output_format,
        metadata_file=metadata_file,
        metadata_key=metadata_key,
        obs_key=obs_key,
        metadata_columns=tuple(columns),
    )

    log_dir = cfg.resolved_output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "adata-ops.log"
    init_logging(logfile)
    cfg.logfile = logfile
    run_adata_ops(cfg)


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
        help="[I/O] Integrated dataset produced by `scomnom integrate` (.zarr, .zarr.tar.zst, or .h5ad).",
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
    force_celltypist_recompute: bool = typer.Option(
        False,
        "--force-celltypist-recompute/--reuse-celltypist",
        help="[CellTypist] Force a fresh CellTypist run with --celltypist-model instead of reusing stored outputs.",
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
    download_gene_models: bool = typer.Option(False, "--download-gene-models"),

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
    decoupler_bar_split_signed: bool = typer.Option(
        False,
        "--decoupler-bar-split-signed/--no-decoupler-bar-split-signed",
        help="[Decoupler] Split barplots into top up/down activities (DoRothEA/MSigDB).",
    ),
    decoupler_bar_top_n_up: Optional[int] = typer.Option(
        None,
        "--decoupler-bar-top-n-up",
        help="[Decoupler] Top N positive activities for split barplots.",
    ),
    decoupler_bar_top_n_down: Optional[int] = typer.Option(
        None,
        "--decoupler-bar-top-n-down",
        help="[Decoupler] Top N negative activities for split barplots.",
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
    # Handle --list-models / --download-models / --download-gene-models
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

    if download_gene_models:
        from .io_utils import download_gene_models
        download_gene_models(species="hsapiens")
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
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "cluster-and-annotate.log"
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
        force_celltypist_recompute=force_celltypist_recompute,
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


def _default_results_dir_for_input(input_path: Path) -> Path:
    parent = input_path.parent
    if parent.name == "results":
        return parent
    for ancestor in parent.parents:
        if ancestor.name == "results":
            return ancestor
    return parent / "results"


def _default_results_dir_for_input_dir(input_dir: Path) -> Path:
    if input_dir.name == "results":
        return input_dir
    for ancestor in input_dir.parents:
        if ancestor.name == "results":
            return ancestor
    return input_dir.parent / "results"


markers_and_de_app = typer.Typer(
    help="Discovery markers + DE (cluster-vs-rest and within-cluster contrasts).",
    invoke_without_command=True,
)
enrichment_app = typer.Typer(
    help="Pathway and TF enrichment from either clustering rounds or DE result tables.",
)
ccc_app = typer.Typer(
    help="Cell-cell communication analysis backends.",
)
app.add_typer(markers_and_de_app, name="markers-and-de")
markers_and_de_app.add_typer(enrichment_app, name="enrichment")
markers_and_de_app.add_typer(ccc_app, name="ccc")


@markers_and_de_app.callback()
def markers_and_de_callback(
    download_gene_models: bool = typer.Option(False, "--download-gene-models"),
    gene_species: str = typer.Option("hsapiens", "--gene-species"),
):
    if download_gene_models:
        from .io_utils import download_gene_models
        download_gene_models(species=str(gene_species))
        raise typer.Exit()


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
    max_workers: int,
    pb_min_counts_per_lib: int,
    pb_min_lib_pct: float,
    pb_covariates: Optional[List[str]],
    prune_uns_de: bool,
    # within-cluster condition keys
    condition_keys: Optional[List[str]],
    target_groups: Optional[List[str]],
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
    run_gsea: bool = True,
    gsea_min_size: int = 10,
    gsea_max_size: int = 500,
    gsea_eps: float = 1e-10,
    gsea_rank_col: Optional[str] = None,
    joint_enrichment_alpha: float = 0.05,
    joint_enrichment_top_n: int = 20,
    joint_enrichment_require_concordant: bool = True,
    joint_enrichment_require_gsea_sig: bool = True,
    joint_enrichment_leading_edge_top_n: int = 8,
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
    gene_filter: Optional[List[str]] = None,
    plot_gene_filter: Optional[List[str]],
    plot_sample_annotation_keys: Optional[List[str]],
    regenerate_figures: bool,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.log"
    init_logging(log_path)

    layers = _parse_csv_repeat(pb_counts_layer) or ["counts_cb", "counts_raw"]
    covariates = tuple(_parse_csv_repeat(pb_covariates) or ())
    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    target_groups_parsed = tuple(_parse_csv_repeat(target_groups) or ())
    gene_filters = tuple(gene_filter or ())
    plot_filters = tuple(plot_gene_filter or ())
    plot_annot_keys = tuple(plot_sample_annotation_keys or ())
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
        regenerate_figures=regenerate_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        gene_filter=gene_filters,
        plot_gene_filter=plot_filters,
        plot_sample_annotation_keys=plot_annot_keys,
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
        max_workers=int(max_workers),
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=tuple(_parse_csv_repeat(pb_covariates) or ()),
        prune_uns_de=prune_uns_de,
        condition_keys=cond_keys,
        target_groups=target_groups_parsed,
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
        run_gsea=bool(run_gsea),
        gsea_min_size=int(gsea_min_size),
        gsea_max_size=int(gsea_max_size),
        gsea_eps=float(gsea_eps),
        gsea_rank_col=None if gsea_rank_col is None else str(gsea_rank_col),
        joint_enrichment_alpha=float(joint_enrichment_alpha),
        joint_enrichment_top_n=int(joint_enrichment_top_n),
        joint_enrichment_require_concordant=bool(joint_enrichment_require_concordant),
        joint_enrichment_require_gsea_sig=bool(joint_enrichment_require_gsea_sig),
        joint_enrichment_leading_edge_top_n=int(joint_enrichment_leading_edge_top_n),
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


def _build_cfg_composition(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    regenerate_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    replicate_key: Optional[str],
    condition_keys: Optional[List[str]],
    covariates: Optional[List[str]],
    methods: Optional[List[str]],
    reference: str,
    min_mean_prop: float,
    min_cells_per_sample_cluster: int,
    alpha: float,
    graph_n_seeds: int,
    graph_k_ref: int,
    graph_max_k: int,
    graph_min_size: int,
    graph_random_state: int,
    graph_min_nonzero_samples_per_level: int,
    graph_n_permutations: int,
    graph_effect_shrink_k: float,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.composition.log"
    init_logging(log_path)

    covars = tuple(_parse_csv_repeat(covariates) or ())
    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    methods_list = _parse_csv_repeat(methods) or ["sccoda", "glm", "clr", "graph"]
    bad = [m for m in methods_list if m not in ALLOWED_COMP_METHODS]
    if bad:
        raise typer.BadParameter(
            f"Invalid --method value(s): {bad}. Allowed: {sorted(ALLOWED_COMP_METHODS)}"
        )

    return MarkersAndDEConfig(
        input_path=input_path,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=regenerate_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=None,
        label_source="pretty",
        round_id=round_id,
        sample_key=replicate_key,
        batch_key=None,
        condition_key=None,
        condition_keys=cond_keys,
        condition_contrasts=(),
        composition_methods=tuple(methods_list),
        composition_reference=str(reference),
        composition_min_mean_prop=float(min_mean_prop),
        composition_min_cells_per_sample_cluster=int(min_cells_per_sample_cluster),
        composition_alpha=float(alpha),
        composition_covariates=covars,
        composition_graph_n_seeds=int(graph_n_seeds),
        composition_graph_k_ref=int(graph_k_ref),
        composition_graph_max_k=int(graph_max_k),
        composition_graph_min_size=int(graph_min_size),
        composition_graph_random_state=int(graph_random_state),
        composition_graph_min_nonzero_samples_per_level=int(graph_min_nonzero_samples_per_level),
        composition_graph_n_permutations=int(graph_n_permutations),
        composition_graph_effect_shrink_k=float(graph_effect_shrink_k),
    )


def _build_cfg_enrichment_cluster(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    regenerate_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    condition_key: Optional[str],
    gene_filter: Optional[List[str]],
    decoupler_pseudobulk_agg: str,
    decoupler_use_raw: bool,
    decoupler_method: str,
    decoupler_consensus_methods: Optional[List[str]],
    decoupler_min_n_targets: int,
    decoupler_bar_split_signed: bool,
    decoupler_bar_top_n_up: Optional[int],
    decoupler_bar_top_n_down: Optional[int],
    msigdb_gene_sets: Optional[List[str]],
    msigdb_method: str,
    msigdb_min_n_targets: int,
    run_gsea: bool,
    gsea_min_size: int,
    gsea_max_size: int,
    gsea_eps: float,
    gsea_rank_col: Optional[str],
    joint_enrichment_alpha: float,
    joint_enrichment_top_n: int,
    joint_enrichment_require_concordant: bool,
    joint_enrichment_require_gsea_sig: bool,
    joint_enrichment_leading_edge_top_n: int,
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
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.enrichment.log"
    init_logging(log_path)

    msigdb_sets = list(msigdb_gene_sets) if msigdb_gene_sets else None
    doro_conf = list(dorothea_confidence) if dorothea_confidence else None
    gene_filters = tuple(gene_filter or ())

    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=regenerate_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        condition_key=condition_key,
        gene_filter=gene_filters,
        run_decoupler=True,
        decoupler_pseudobulk_agg=str(decoupler_pseudobulk_agg),
        decoupler_use_raw=bool(decoupler_use_raw),
        decoupler_method=str(decoupler_method),
        decoupler_consensus_methods=decoupler_consensus_methods,
        decoupler_min_n_targets=int(decoupler_min_n_targets),
        decoupler_bar_split_signed=bool(decoupler_bar_split_signed),
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
        msigdb_gene_sets=msigdb_sets if msigdb_sets is not None else ["HALLMARK", "REACTOME"],
        msigdb_method=str(msigdb_method),
        msigdb_min_n_targets=int(msigdb_min_n_targets),
        run_gsea=bool(run_gsea),
        gsea_min_size=int(gsea_min_size),
        gsea_max_size=int(gsea_max_size),
        gsea_eps=float(gsea_eps),
        gsea_rank_col=None if gsea_rank_col is None else str(gsea_rank_col),
        joint_enrichment_alpha=float(joint_enrichment_alpha),
        joint_enrichment_top_n=int(joint_enrichment_top_n),
        joint_enrichment_require_concordant=bool(joint_enrichment_require_concordant),
        joint_enrichment_require_gsea_sig=bool(joint_enrichment_require_gsea_sig),
        joint_enrichment_leading_edge_top_n=int(joint_enrichment_leading_edge_top_n),
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
    )


def _build_cfg_enrichment_de(
    *,
    input_dir: Path,
    output_dir: Optional[Path],
    output_name: str,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    gene_filter: Optional[List[str]],
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
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input_dir(input_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.enrichment-de.log"
    init_logging(log_path)

    msigdb_sets = list(msigdb_gene_sets) if msigdb_gene_sets else None
    doro_conf = list(dorothea_confidence) if dorothea_confidence else None
    gene_filters = tuple(gene_filter or ())

    return MarkersAndDEConfig(
        input_path=input_dir,
        input_dir=input_dir,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=False,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        gene_filter=gene_filters,
        run_decoupler=True,
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
    )


def _build_cfg_module_score(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    condition_key: Optional[str],
    module_files: Sequence[Path],
    module_set_name: Optional[str],
    module_score_method: str,
    module_score_use_raw: bool,
    module_score_layer: Optional[str],
    module_score_ctrl_size: int,
    module_score_n_bins: int,
    module_score_random_state: int,
    module_score_max_umaps: int,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.module-score.log"
    init_logging(log_path)

    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        condition_key=condition_key,
        module_files=tuple(str(Path(p)) for p in module_files),
        module_set_name=(str(module_set_name).strip() if module_set_name else None),
        module_score_method=str(module_score_method),
        module_score_use_raw=bool(module_score_use_raw),
        module_score_layer=(str(module_score_layer) if module_score_layer else None),
        module_score_ctrl_size=int(module_score_ctrl_size),
        module_score_n_bins=int(module_score_n_bins),
        module_score_random_state=int(module_score_random_state),
        module_score_max_umaps=int(module_score_max_umaps),
    )


def _build_cfg_ccc_liana(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    group_key: Optional[str],
    label_source: str,
    condition_keys: Optional[List[str]],
    condition_values: Optional[List[str]],
    compare_levels: Optional[List[str]],
    dataset_key: Optional[str],
    source_levels: Optional[List[str]],
    target_levels: Optional[List[str]],
    signal_scope: str,
    liana_methods: Sequence[str],
    liana_resource: str,
    liana_expr_prop: float,
    liana_input_mode: str,
    liana_lognorm_target_sum: float,
    liana_use_raw: bool,
    liana_layer: Optional[str],
    liana_n_perms: Optional[int],
    liana_seed: int,
    liana_return_all_lrs: bool,
    liana_top_n: int,
    liana_plot_top_n: int,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.ccc.liana.log"
    init_logging(log_path)

    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        ccc_backend="liana",
        ccc_condition_key=(str(cond_keys[0]).strip() if cond_keys else None),
        ccc_condition_keys=cond_keys,
        ccc_condition_values=tuple(_parse_csv_repeat(condition_values) or ()),
        ccc_compare_levels=tuple(_parse_csv_repeat(compare_levels) or ()),
        ccc_dataset_key=(str(dataset_key).strip() if dataset_key else None),
        ccc_source_levels=tuple(_parse_csv_repeat(source_levels) or ()),
        ccc_target_levels=tuple(_parse_csv_repeat(target_levels) or ()),
        ccc_signal_scope=str(signal_scope).strip().lower(),
        liana_resource=str(liana_resource).strip().lower(),
        liana_methods=tuple(str(x).strip().lower() for x in liana_methods if str(x).strip()),
        liana_expr_prop=float(liana_expr_prop),
        liana_input_mode=str(liana_input_mode).strip().lower(),
        liana_lognorm_target_sum=float(liana_lognorm_target_sum),
        liana_use_raw=bool(liana_use_raw),
        liana_layer=(str(liana_layer) if liana_layer else None),
        liana_n_perms=(None if liana_n_perms is None else int(liana_n_perms)),
        liana_seed=int(liana_seed),
        liana_return_all_lrs=bool(liana_return_all_lrs),
        liana_top_n=int(liana_top_n),
        liana_plot_top_n=int(liana_plot_top_n),
    )


def _build_cfg_ccc_nichenet(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    group_key: Optional[str],
    label_source: str,
    condition_keys: Optional[List[str]],
    condition_values: Optional[List[str]],
    compare_levels: Optional[List[str]],
    dataset_key: Optional[str],
    source_levels: Optional[List[str]],
    target_levels: Optional[List[str]],
    signal_scope: str,
    receiver_cluster: str,
    sender_clusters: Optional[List[str]],
    gene_list_file: Optional[Path],
    expression_pct: float,
    input_mode: str,
    lognorm_target_sum: float,
    top_n_ligands: int,
    top_n_targets: int,
    min_logfc: float,
    padj_threshold: float,
    organism: str,
    install_missing_r_deps: bool,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.ccc.nichenet.log"
    init_logging(log_path)

    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        ccc_backend="nichenet",
        ccc_condition_key=(str(cond_keys[0]).strip() if cond_keys else None),
        ccc_condition_keys=cond_keys,
        ccc_condition_values=tuple(_parse_csv_repeat(condition_values) or ()),
        ccc_compare_levels=tuple(_parse_csv_repeat(compare_levels) or ()),
        ccc_dataset_key=(str(dataset_key).strip() if dataset_key else None),
        ccc_source_levels=tuple(_parse_csv_repeat(source_levels) or ()),
        ccc_target_levels=tuple(_parse_csv_repeat(target_levels) or ()),
        ccc_signal_scope=str(signal_scope).strip().lower(),
        nichenet_receiver_cluster=str(receiver_cluster).strip(),
        nichenet_sender_clusters=tuple(_parse_csv_repeat(sender_clusters) or ()),
        nichenet_gene_list_file=(str(gene_list_file) if gene_list_file else None),
        nichenet_expression_pct=float(expression_pct),
        nichenet_input_mode=str(input_mode).strip().lower(),
        nichenet_lognorm_target_sum=float(lognorm_target_sum),
        nichenet_top_n_ligands=int(top_n_ligands),
        nichenet_top_n_targets=int(top_n_targets),
        nichenet_min_logfc=float(min_logfc),
        nichenet_padj_threshold=float(padj_threshold),
        nichenet_organism=str(organism).strip().lower(),
        nichenet_install_missing_r_deps=bool(install_missing_r_deps),
    )


def _build_cfg_ccc_mebocost(
    *,
    input_path: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    group_key: Optional[str],
    label_source: str,
    condition_keys: Optional[List[str]],
    condition_values: Optional[List[str]],
    compare_levels: Optional[List[str]],
    dataset_key: Optional[str],
    source_levels: Optional[List[str]],
    target_levels: Optional[List[str]],
    organism: str,
    input_mode: str,
    lognorm_target_sum: float,
    n_shuffle: int,
    seed: int,
    min_cell_number: int,
    pval_cutoff: float,
    plot_top_n: int,
    install_missing_python_deps: bool,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.ccc.mebocost.log"
    init_logging(log_path)

    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        ccc_backend="mebocost",
        ccc_condition_key=(str(cond_keys[0]).strip() if cond_keys else None),
        ccc_condition_keys=cond_keys,
        ccc_condition_values=tuple(_parse_csv_repeat(condition_values) or ()),
        ccc_compare_levels=tuple(_parse_csv_repeat(compare_levels) or ()),
        ccc_dataset_key=(str(dataset_key).strip() if dataset_key else None),
        ccc_source_levels=tuple(_parse_csv_repeat(source_levels) or ()),
        ccc_target_levels=tuple(_parse_csv_repeat(target_levels) or ()),
        mebocost_organism=str(organism).strip().lower(),
        mebocost_input_mode=str(input_mode).strip().lower(),
        mebocost_lognorm_target_sum=float(lognorm_target_sum),
        mebocost_n_shuffle=int(n_shuffle),
        mebocost_seed=int(seed),
        mebocost_min_cell_number=int(min_cell_number),
        mebocost_pval_cutoff=float(pval_cutoff),
        mebocost_plot_top_n=int(plot_top_n),
        mebocost_install_missing_python_deps=bool(install_missing_python_deps),
    )


def _build_cfg_ccc_mebocost_paired(
    *,
    input_path: Path,
    candidate_events: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    group_key: Optional[str],
    label_source: str,
    condition_keys: Optional[List[str]],
    condition_values: Optional[List[str]],
    compare_levels: Optional[List[str]],
    dataset_key: Optional[str],
    source_levels: Optional[List[str]],
    target_levels: Optional[List[str]],
    pairing_key: str,
    organism: str,
    input_mode: str,
    lognorm_target_sum: float,
    source_filter: Optional[List[str]],
    target_filter: Optional[List[str]],
    metabolite_filter: Optional[List[str]],
    sensor_filter: Optional[List[str]],
    superclass_filter: Optional[List[str]],
    class_filter: Optional[List[str]],
    subclass_filter: Optional[List[str]],
    max_events: int,
    score_method: str,
    min_sender_cells: int,
    min_receiver_cells: int,
    min_scored_donors_per_group: int,
    install_missing_python_deps: bool,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.ccc.mebocost.paired.log"
    init_logging(log_path)

    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        ccc_backend="mebocost_paired_rescore",
        ccc_condition_key=(str(cond_keys[0]).strip() if cond_keys else None),
        ccc_condition_keys=cond_keys,
        ccc_condition_values=tuple(_parse_csv_repeat(condition_values) or ()),
        ccc_compare_levels=tuple(_parse_csv_repeat(compare_levels) or ()),
        ccc_dataset_key=(str(dataset_key).strip() if dataset_key else None),
        ccc_source_levels=tuple(_parse_csv_repeat(source_levels) or ()),
        ccc_target_levels=tuple(_parse_csv_repeat(target_levels) or ()),
        mebocost_candidate_events=str(candidate_events),
        mebocost_pairing_key=str(pairing_key).strip(),
        mebocost_organism=str(organism).strip().lower(),
        mebocost_input_mode=str(input_mode).strip().lower(),
        mebocost_lognorm_target_sum=float(lognorm_target_sum),
        mebocost_source_filter=tuple(_parse_csv_repeat(source_filter) or ()),
        mebocost_target_filter=tuple(_parse_csv_repeat(target_filter) or ()),
        mebocost_metabolite_filter=tuple(_parse_csv_repeat(metabolite_filter) or ()),
        mebocost_sensor_filter=tuple(_parse_csv_repeat(sensor_filter) or ()),
        mebocost_superclass_filter=tuple(_parse_csv_repeat(superclass_filter) or ()),
        mebocost_class_filter=tuple(_parse_csv_repeat(class_filter) or ()),
        mebocost_subclass_filter=tuple(_parse_csv_repeat(subclass_filter) or ()),
        mebocost_max_events=int(max_events),
        mebocost_score_method=str(score_method).strip().lower(),
        mebocost_min_sender_cells=int(min_sender_cells),
        mebocost_min_receiver_cells=int(min_receiver_cells),
        mebocost_min_scored_donors_per_group=int(min_scored_donors_per_group),
        mebocost_install_missing_python_deps=bool(install_missing_python_deps),
    )


def _build_cfg_ccc_liana_paired(
    *,
    input_path: Path,
    candidate_events: Path,
    output_dir: Optional[Path],
    output_name: str,
    save_h5ad: bool,
    n_jobs: int,
    make_figures: bool,
    figdir_name: str,
    figure_formats: Sequence[str],
    round_id: Optional[str],
    group_key: Optional[str],
    label_source: str,
    condition_keys: Optional[List[str]],
    condition_values: Optional[List[str]],
    compare_levels: Optional[List[str]],
    dataset_key: Optional[str],
    source_levels: Optional[List[str]],
    target_levels: Optional[List[str]],
    pairing_key: str,
    input_mode: str,
    lognorm_target_sum: float,
    source_filter: Optional[List[str]],
    target_filter: Optional[List[str]],
    ligand_filter: Optional[List[str]],
    receptor_filter: Optional[List[str]],
    route_family_filter: Optional[List[str]],
    max_edges: int,
    min_sender_cells: int,
    min_receiver_cells: int,
    min_scored_donors_per_group: int,
) -> MarkersAndDEConfig:
    out_dir = output_dir or _default_results_dir_for_input(input_path)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "markers-and-de.ccc.liana.paired.log"
    init_logging(log_path)

    cond_keys = tuple(_parse_csv_repeat(condition_keys) or ())
    return MarkersAndDEConfig(
        input_path=input_path,
        input_dir=None,
        output_dir=out_dir,
        output_name=output_name,
        save_h5ad=save_h5ad,
        run="pseudobulk",
        n_jobs=n_jobs,
        logfile=log_path,
        make_figures=make_figures,
        regenerate_figures=False,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        groupby=group_key,
        label_source=label_source,
        round_id=round_id,
        ccc_backend="liana_paired_rescore",
        ccc_condition_key=(str(cond_keys[0]).strip() if cond_keys else None),
        ccc_condition_keys=cond_keys,
        ccc_condition_values=tuple(_parse_csv_repeat(condition_values) or ()),
        ccc_compare_levels=tuple(_parse_csv_repeat(compare_levels) or ()),
        ccc_dataset_key=(str(dataset_key).strip() if dataset_key else None),
        ccc_source_levels=tuple(_parse_csv_repeat(source_levels) or ()),
        ccc_target_levels=tuple(_parse_csv_repeat(target_levels) or ()),
        liana_candidate_events=str(candidate_events),
        liana_pairing_key=str(pairing_key).strip(),
        liana_input_mode=str(input_mode).strip().lower(),
        liana_lognorm_target_sum=float(lognorm_target_sum),
        liana_source_filter=tuple(_parse_csv_repeat(source_filter) or ()),
        liana_target_filter=tuple(_parse_csv_repeat(target_filter) or ()),
        liana_ligand_filter=tuple(_parse_csv_repeat(ligand_filter) or ()),
        liana_receptor_filter=tuple(_parse_csv_repeat(receptor_filter) or ()),
        liana_route_family_filter=tuple(_parse_csv_repeat(route_family_filter) or ()),
        liana_max_edges=int(max_edges),
        liana_min_sender_cells=int(min_sender_cells),
        liana_min_receiver_cells=int(min_receiver_cells),
        liana_min_scored_donors_per_group=int(min_scored_donors_per_group),
    )


def _default_output_name(input_path: Path, suffix: str, round_id: Optional[str] = None) -> str:
    name = input_path.name
    for ext in (".zarr.tar.zst", ".zarr", ".h5ad", ".h5", ".hdf5"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    if round_id:
        round_token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(round_id).strip())
        suffix = f"{suffix}_{round_token}"
    return f"{name}.{suffix}"


@ccc_app.command(
    "liana",
    help="Run LIANA cell-cell communication inference on cluster labels.",
)
def ccc_liana(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_keys: List[str] = typer.Option(
        [],
        "--condition-key",
        help="Optional condition spec(s); repeatable/comma-separated. Supports A, A:B, and A@B.",
    ),
    condition_values: List[str] = typer.Option(
        [],
        "--condition-value",
        help="Restrict selected condition levels. For A@B this filters the B-side subset levels.",
    ),
    compare_levels: List[str] = typer.Option(
        [],
        "--compare-level",
        help="Restrict levels of the comparison variable within each condition spec (repeatable/comma-separated). For A@B this filters the A-side levels.",
    ),
    dataset_key: Optional[str] = typer.Option(
        None,
        "--dataset-key",
        help="Optional obs key defining tissue/dataset origin for cross-tissue LIANA mode.",
    ),
    source_levels: List[str] = typer.Option(
        [],
        "--source-level",
        help="Allowed sender dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    target_levels: List[str] = typer.Option(
        [],
        "--target-level",
        help="Allowed receiver dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    signal_scope: str = typer.Option(
        "all",
        "--signal-scope",
        help="Resource/output filter for LIANA interactions: all or secreted.",
    ),
    liana_method: List[str] = typer.Option(
        ["rank_aggregate"],
        "--liana-method",
        callback=validate_liana_methods,
        help="LIANA method(s) to run. Use rank_aggregate for consensus. By default rank_aggregate uses a sparse-safer subset: cellphonedb,natmi,sca,logfc.",
    ),
    liana_resource: str = typer.Option("consensus", "--resource"),
    liana_expr_prop: float = typer.Option(0.1, "--expr-prop"),
    liana_input_mode: str = typer.Option(
        "counts",
        "--input-mode",
        help="LIANA expression input mode. 'counts' uses counts_cb/counts_raw directly; 'lognorm' builds and reuses a log-normalized layer from counts_cb or counts_raw.",
    ),
    liana_lognorm_target_sum: float = typer.Option(
        1e4,
        "--lognorm-target-sum",
        help="Target library size used when --input-mode lognorm builds a log-normalized LIANA layer.",
    ),
    liana_use_raw: bool = typer.Option(
        False,
        "--use-raw/--no-use-raw",
        help="Use adata.raw explicitly. Default behavior prefers counts_cb, then counts_raw, then adata.X.",
    ),
    liana_layer: Optional[str] = typer.Option(
        None,
        "--layer",
        help="Explicit layer override. By default LIANA prefers counts_cb, then counts_raw, then adata.X.",
    ),
    liana_n_perms: Optional[int] = typer.Option(
        1000,
        "--n-perms",
        help="Permutation count for permutation-based LIANA methods. Use 0 to disable.",
    ),
    liana_seed: int = typer.Option(42, "--seed"),
    liana_return_all_lrs: bool = typer.Option(
        False,
        "--return-all-lrs/--no-return-all-lrs",
    ),
    liana_top_n: int = typer.Option(250, "--top-n"),
    liana_plot_top_n: int = typer.Option(60, "--plot-top-n"),
):
    if str(liana_input_mode).strip().lower() not in {"counts", "lognorm"}:
        raise typer.BadParameter("--input-mode must be one of: counts, lognorm.")
    if liana_use_raw and liana_layer:
        raise typer.BadParameter("Cannot use both --use-raw and --layer.")
    if dataset_key and (not source_levels or not target_levels):
        raise typer.BadParameter("--dataset-key requires at least one --source-level and one --target-level.")
    if output_name is None:
        output_name = _default_output_name(input_path, "ccc_liana", round_id=round_id)

    cfg = _build_cfg_ccc_liana(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        group_key=group_key,
        label_source=label_source,
        condition_keys=condition_keys,
        condition_values=condition_values,
        compare_levels=compare_levels,
        dataset_key=dataset_key,
        source_levels=source_levels,
        target_levels=target_levels,
        signal_scope=signal_scope,
        liana_methods=liana_method,
        liana_resource=liana_resource,
        liana_expr_prop=liana_expr_prop,
        liana_input_mode=liana_input_mode,
        liana_lognorm_target_sum=liana_lognorm_target_sum,
        liana_use_raw=liana_use_raw,
        liana_layer=liana_layer,
        liana_n_perms=None if liana_n_perms in (None, 0) else liana_n_perms,
        liana_seed=liana_seed,
        liana_return_all_lrs=liana_return_all_lrs,
        liana_top_n=liana_top_n,
        liana_plot_top_n=liana_plot_top_n,
    )

    run_liana_ccc(cfg)


@ccc_app.command(
    "liana-paired",
    help="Run donor/sample-level focused LIANA rescoring on a candidate LR table.",
)
def ccc_liana_paired(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    candidate_events: Path = typer.Option(..., "--candidate-events"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_keys: List[str] = typer.Option(
        [],
        "--condition-key",
        help="Optional subset spec(s); repeatable/comma-separated. Supports A and A@B.",
    ),
    condition_values: List[str] = typer.Option(
        [],
        "--condition-value",
        help="Restrict subset levels for A@B by filtering the B-side context levels.",
    ),
    compare_levels: List[str] = typer.Option(
        [],
        "--compare-level",
        help="Optional levels of the A-side condition variable to keep when using A@B specs.",
    ),
    dataset_key: Optional[str] = typer.Option(
        None,
        "--dataset-key",
        help="Optional obs key defining tissue/dataset origin for cross-tissue sender/receiver restriction.",
    ),
    source_levels: List[str] = typer.Option(
        [],
        "--source-level",
        help="Allowed sender dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    target_levels: List[str] = typer.Option(
        [],
        "--target-level",
        help="Allowed receiver dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    pairing_key: str = typer.Option("sample_id", "--pairing-key"),
    input_mode: str = typer.Option(
        "counts",
        "--input-mode",
        help="LIANA paired expression input mode. 'counts' uses counts_cb/counts_raw directly; 'lognorm' builds and reuses a log-normalized layer from counts_cb or counts_raw.",
    ),
    lognorm_target_sum: float = typer.Option(
        1e4,
        "--lognorm-target-sum",
        help="Target library size used when --input-mode lognorm builds a log-normalized LIANA layer.",
    ),
    source_filter: List[str] = typer.Option([], "--source-filter"),
    target_filter: List[str] = typer.Option([], "--target-filter"),
    ligand_filter: List[str] = typer.Option([], "--ligand-filter"),
    receptor_filter: List[str] = typer.Option([], "--receptor-filter"),
    route_family_filter: List[str] = typer.Option([], "--route-family-filter"),
    max_edges: int = typer.Option(200, "--max-edges"),
    min_sender_cells: int = typer.Option(5, "--min-sender-cells"),
    min_receiver_cells: int = typer.Option(5, "--min-receiver-cells"),
    min_scored_donors_per_group: int = typer.Option(3, "--min-scored-donors-per-group"),
):
    if str(input_mode).strip().lower() not in {"counts", "lognorm"}:
        raise typer.BadParameter("--input-mode must be one of: counts, lognorm.")
    if dataset_key and (not source_levels or not target_levels):
        raise typer.BadParameter("--dataset-key requires at least one --source-level and one --target-level.")
    if output_name is None:
        output_name = _default_output_name(input_path, "ccc_liana_paired", round_id=round_id)

    cfg = _build_cfg_ccc_liana_paired(
        input_path=input_path,
        candidate_events=candidate_events,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        group_key=group_key,
        label_source=label_source,
        condition_keys=condition_keys,
        condition_values=condition_values,
        compare_levels=compare_levels,
        dataset_key=dataset_key,
        source_levels=source_levels,
        target_levels=target_levels,
        pairing_key=pairing_key,
        input_mode=input_mode,
        lognorm_target_sum=lognorm_target_sum,
        source_filter=source_filter,
        target_filter=target_filter,
        ligand_filter=ligand_filter,
        receptor_filter=receptor_filter,
        route_family_filter=route_family_filter,
        max_edges=max_edges,
        min_sender_cells=min_sender_cells,
        min_receiver_cells=min_receiver_cells,
        min_scored_donors_per_group=min_scored_donors_per_group,
    )

    run_liana_paired_rescore(cfg)


@ccc_app.command(
    "mebocost",
    help="Run MEBOCOST metabolite-mediated cell-cell communication inference on cluster labels.",
)
def ccc_mebocost(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_keys: List[str] = typer.Option(
        [],
        "--condition-key",
        help="Optional subset spec(s); repeatable/comma-separated. Supports A and A@B.",
    ),
    condition_values: List[str] = typer.Option(
        [],
        "--condition-value",
        help="Restrict subset levels for A@B by filtering the B-side context levels.",
    ),
    compare_levels: List[str] = typer.Option(
        [],
        "--compare-level",
        help="Optional levels of the A-side condition variable to keep when using A@B specs.",
    ),
    dataset_key: Optional[str] = typer.Option(
        None,
        "--dataset-key",
        help="Optional obs key defining tissue/dataset origin for cross-tissue sender/receiver restriction.",
    ),
    source_levels: List[str] = typer.Option(
        [],
        "--source-level",
        help="Allowed sender dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    target_levels: List[str] = typer.Option(
        [],
        "--target-level",
        help="Allowed receiver dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    organism: str = typer.Option("human", "--organism"),
    input_mode: str = typer.Option(
        "counts",
        "--input-mode",
        help="MEBOCOST expression input mode. 'counts' uses counts_cb/counts_raw directly; 'lognorm' builds and reuses a log-normalized layer from counts_cb or counts_raw.",
    ),
    lognorm_target_sum: float = typer.Option(
        1e4,
        "--lognorm-target-sum",
        help="Target library size used when --input-mode lognorm builds a log-normalized MEBOCOST layer.",
    ),
    n_shuffle: int = typer.Option(1000, "--n-shuffle"),
    seed: int = typer.Option(42, "--seed"),
    min_cell_number: int = typer.Option(10, "--min-cell-number"),
    pval_cutoff: float = typer.Option(0.05, "--pval-cutoff"),
    plot_top_n: int = typer.Option(40, "--plot-top-n"),
    install_missing_python_deps: bool = typer.Option(
        False,
        "--install-missing-python-deps/--no-install-missing-python-deps",
        help="Install missing MEBOCOST Python dependencies into the active environment before running.",
    ),
):
    if str(input_mode).strip().lower() not in {"counts", "lognorm"}:
        raise typer.BadParameter("--input-mode must be one of: counts, lognorm.")
    if dataset_key and (not source_levels or not target_levels):
        raise typer.BadParameter("--dataset-key requires at least one --source-level and one --target-level.")
    if output_name is None:
        output_name = _default_output_name(input_path, "ccc_mebocost", round_id=round_id)

    cfg = _build_cfg_ccc_mebocost(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        group_key=group_key,
        label_source=label_source,
        condition_keys=condition_keys,
        condition_values=condition_values,
        compare_levels=compare_levels,
        dataset_key=dataset_key,
        source_levels=source_levels,
        target_levels=target_levels,
        organism=organism,
        input_mode=input_mode,
        lognorm_target_sum=lognorm_target_sum,
        n_shuffle=n_shuffle,
        seed=seed,
        min_cell_number=min_cell_number,
        pval_cutoff=pval_cutoff,
        plot_top_n=plot_top_n,
        install_missing_python_deps=install_missing_python_deps,
    )

    run_mebocost_ccc(cfg)


@ccc_app.command(
    "mebocost-paired",
    help="Run donor/sample-level focused MEBOCOST rescoring on a candidate event table.",
)
def ccc_mebocost_paired(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    candidate_events: Path = typer.Option(..., "--candidate-events"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(False, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_keys: List[str] = typer.Option(
        [],
        "--condition-key",
        help="Optional subset spec(s); repeatable/comma-separated. Supports A and A@B.",
    ),
    condition_values: List[str] = typer.Option(
        [],
        "--condition-value",
        help="Restrict subset levels for A@B by filtering the B-side context levels.",
    ),
    compare_levels: List[str] = typer.Option(
        [],
        "--compare-level",
        help="Optional levels of the A-side condition variable to keep when using A@B specs.",
    ),
    dataset_key: Optional[str] = typer.Option(
        None,
        "--dataset-key",
        help="Optional obs key defining tissue/dataset origin for cross-tissue sender/receiver restriction.",
    ),
    source_levels: List[str] = typer.Option(
        [],
        "--source-level",
        help="Allowed sender dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    target_levels: List[str] = typer.Option(
        [],
        "--target-level",
        help="Allowed receiver dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    pairing_key: str = typer.Option("sample_id", "--pairing-key"),
    organism: str = typer.Option("human", "--organism"),
    input_mode: str = typer.Option(
        "counts",
        "--input-mode",
        help="MEBOCOST expression input mode. 'counts' uses counts_cb/counts_raw directly; 'lognorm' builds and reuses a log-normalized layer from counts_cb or counts_raw.",
    ),
    lognorm_target_sum: float = typer.Option(
        1e4,
        "--lognorm-target-sum",
        help="Target library size used when --input-mode lognorm builds a log-normalized MEBOCOST layer.",
    ),
    source_filter: List[str] = typer.Option([], "--source-filter"),
    target_filter: List[str] = typer.Option([], "--target-filter"),
    metabolite_filter: List[str] = typer.Option([], "--metabolite-filter"),
    sensor_filter: List[str] = typer.Option([], "--sensor-filter"),
    superclass_filter: List[str] = typer.Option([], "--superclass-filter"),
    class_filter: List[str] = typer.Option([], "--class-filter"),
    subclass_filter: List[str] = typer.Option([], "--subclass-filter"),
    max_events: int = typer.Option(200, "--max-events"),
    score_method: str = typer.Option(
        "mebocost-metabolite-sensor",
        "--score-method",
        help="Focused donor-level score mode: mebocost-metabolite-sensor or associated-gene-proxy.",
    ),
    min_sender_cells: int = typer.Option(5, "--min-sender-cells"),
    min_receiver_cells: int = typer.Option(5, "--min-receiver-cells"),
    min_scored_donors_per_group: int = typer.Option(3, "--min-scored-donors-per-group"),
    install_missing_python_deps: bool = typer.Option(
        False,
        "--install-missing-python-deps/--no-install-missing-python-deps",
        help="Install missing MEBOCOST Python dependencies into the active environment before running.",
    ),
):
    if str(input_mode).strip().lower() not in {"counts", "lognorm"}:
        raise typer.BadParameter("--input-mode must be one of: counts, lognorm.")
    if str(score_method).strip().lower() not in {"mebocost-metabolite-sensor", "associated-gene-proxy"}:
        raise typer.BadParameter(
            "--score-method must be one of: mebocost-metabolite-sensor, associated-gene-proxy."
        )
    if dataset_key and (not source_levels or not target_levels):
        raise typer.BadParameter("--dataset-key requires at least one --source-level and one --target-level.")
    if output_name is None:
        output_name = _default_output_name(input_path, "ccc_mebocost_paired", round_id=round_id)

    cfg = _build_cfg_ccc_mebocost_paired(
        input_path=input_path,
        candidate_events=candidate_events,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        group_key=group_key,
        label_source=label_source,
        condition_keys=condition_keys,
        condition_values=condition_values,
        compare_levels=compare_levels,
        dataset_key=dataset_key,
        source_levels=source_levels,
        target_levels=target_levels,
        pairing_key=pairing_key,
        organism=organism,
        input_mode=input_mode,
        lognorm_target_sum=lognorm_target_sum,
        source_filter=source_filter,
        target_filter=target_filter,
        metabolite_filter=metabolite_filter,
        sensor_filter=sensor_filter,
        superclass_filter=superclass_filter,
        class_filter=class_filter,
        subclass_filter=subclass_filter,
        max_events=max_events,
        score_method=score_method,
        min_sender_cells=min_sender_cells,
        min_receiver_cells=min_receiver_cells,
        min_scored_donors_per_group=min_scored_donors_per_group,
        install_missing_python_deps=install_missing_python_deps,
    )

    run_mebocost_paired_rescore(cfg)


@ccc_app.command(
    "nichenet",
    help="Run sender-focused NicheNet ligand activity analysis for one receiver cluster or all clusters.",
)
def ccc_nichenet(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    group_key: Optional[str] = typer.Option(None, "--group-key"),
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_keys: List[str] = typer.Option(
        [],
        "--condition-key",
        help="Optional comparison spec(s); repeatable/comma-separated. Supports A and A@B.",
    ),
    condition_values: List[str] = typer.Option(
        [],
        "--condition-value",
        help="Restrict subset levels for A@B by filtering the B-side context levels.",
    ),
    compare_levels: List[str] = typer.Option(
        [],
        "--compare-level",
        help="Exactly two comparison levels for the A-side condition variable.",
    ),
    dataset_key: Optional[str] = typer.Option(
        None,
        "--dataset-key",
        help="Optional obs key defining tissue/dataset origin for cross-tissue sender/receiver restriction.",
    ),
    source_levels: List[str] = typer.Option(
        [],
        "--source-level",
        help="Allowed sender dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    target_levels: List[str] = typer.Option(
        [],
        "--target-level",
        help="Allowed receiver dataset/tissue levels in cross-tissue mode (repeatable/comma-separated).",
    ),
    signal_scope: str = typer.Option(
        "all",
        "--signal-scope",
        help="Resource/output filter for downstream LR interpretation: all or secreted.",
    ),
    receiver_cluster: str = typer.Option(
        "all",
        "--receiver-cluster",
        help="Receiver cluster to explain, using either raw ids or pretty labels. Use 'all' to batch over every cluster.",
    ),
    sender_clusters: List[str] = typer.Option(
        [],
        "--sender-cluster",
        help="Optional sender cluster restriction (repeatable/comma-separated). If omitted, all allowed sender clusters are considered.",
    ),
    gene_list_file: Optional[Path] = typer.Option(
        None,
        "--gene-list-file",
        help="Optional one-gene-per-line receiver gene set. If omitted, NicheNet derives the gene set from receiver-cluster DE.",
    ),
    expression_pct: float = typer.Option(0.10, "--expression-pct"),
    input_mode: str = typer.Option(
        "counts",
        "--input-mode",
        help="NicheNet expression input mode for sender/receiver expressed-gene filtering. 'counts' uses counts_cb/counts_raw directly; 'lognorm' builds and reuses a log-normalized layer from counts_cb or counts_raw.",
    ),
    lognorm_target_sum: float = typer.Option(
        1e4,
        "--lognorm-target-sum",
        help="Target library size used when --input-mode lognorm builds a log-normalized NicheNet expression layer.",
    ),
    top_n_ligands: int = typer.Option(30, "--top-n-ligands"),
    top_n_targets: int = typer.Option(200, "--top-n-targets"),
    min_logfc: float = typer.Option(0.25, "--min-logfc"),
    padj_threshold: float = typer.Option(0.05, "--padj-threshold"),
    organism: str = typer.Option("human", "--organism"),
    install_missing_r_deps: bool = typer.Option(
        False,
        "--install-missing-r-deps/--no-install-missing-r-deps",
        help="Install missing NicheNet R dependencies into scOmnom's project-local R library before running.",
    ),
):
    if str(input_mode).strip().lower() not in {"counts", "lognorm"}:
        raise typer.BadParameter("--input-mode must be one of: counts, lognorm.")
    if dataset_key and (not source_levels or not target_levels):
        raise typer.BadParameter("--dataset-key requires at least one --source-level and one --target-level.")
    if organism.strip().lower() != "human":
        raise typer.BadParameter("NicheNet v1 support in scOmnom currently expects --organism human.")
    if gene_list_file is None and not condition_keys:
        raise typer.BadParameter("Provide either --gene-list-file or at least one --condition-key.")
    if gene_list_file is None and len(_parse_csv_repeat(compare_levels) or ()) != 2:
        raise typer.BadParameter("Receiver-DE NicheNet requires exactly two --compare-level values.")
    if output_name is None:
        output_name = _default_output_name(input_path, "ccc_nichenet", round_id=round_id)

    cfg = _build_cfg_ccc_nichenet(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        group_key=group_key,
        label_source=label_source,
        condition_keys=condition_keys,
        condition_values=condition_values,
        compare_levels=compare_levels,
        dataset_key=dataset_key,
        source_levels=source_levels,
        target_levels=target_levels,
        signal_scope=signal_scope,
        receiver_cluster=receiver_cluster,
        sender_clusters=sender_clusters,
        gene_list_file=gene_list_file,
        expression_pct=expression_pct,
        input_mode=input_mode,
        lognorm_target_sum=lognorm_target_sum,
        top_n_ligands=top_n_ligands,
        top_n_targets=top_n_targets,
        min_logfc=min_logfc,
        padj_threshold=padj_threshold,
        organism=organism,
        install_missing_r_deps=install_missing_r_deps,
    )

    run_nichenet_ccc(cfg)


@markers_and_de_app.command(
    "markers",
    help="Markers: Define marker genes for each cluster (vs all others.)",
)
def cluster_vs_rest(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: sibling results/ directory, or input parent if already inside results/).",
    ),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),

    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    regenerate_figures: bool = typer.Option(
        False,
        "--regenerate-figures",
        help="[Plots] Regenerate figures from stored results only (no recomputation).",
    ),
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
    max_workers: int = typer.Option(
        8,
        "--max-workers",
        "--pb-max-workers",
        help="Maximum parallel DE worker tasks for both pseudobulk and cell-level phases.",
    ),
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
    run_gsea: bool = typer.Option(True, "--run-gsea/--no-run-gsea"),
    gsea_min_size: int = typer.Option(10, "--gsea-min-size"),
    gsea_max_size: int = typer.Option(500, "--gsea-max-size"),
    gsea_eps: float = typer.Option(1e-10, "--gsea-eps"),
    gsea_rank_col: Optional[str] = typer.Option(
        None,
        "--gsea-rank-col",
        help="Preferred DE statistic column for MSigDB GSEA. Defaults to the DE decoupler statistic column.",
    ),
    joint_enrichment_alpha: float = typer.Option(0.05, "--joint-enrichment-alpha"),
    joint_enrichment_top_n: int = typer.Option(20, "--joint-enrichment-top-n"),
    joint_enrichment_require_concordant: bool = typer.Option(
        True,
        "--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant",
    ),
    joint_enrichment_require_gsea_sig: bool = typer.Option(
        True,
        "--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig",
    ),
    joint_enrichment_leading_edge_top_n: int = typer.Option(8, "--joint-enrichment-leading-edge-top-n"),

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
    plot_top_n_per_cluster: int = typer.Option(12, "--plot-top-n-per-cluster"),
    plot_dotplot_top_n_per_cluster: int = typer.Option(16, "--plot-dotplot-top-n-per-cluster"),
    plot_max_genes_total: int = typer.Option(80, "--plot-max-genes-total"),
    plot_use_raw: bool = typer.Option(False, "--plot-use-raw/--no-plot-use-raw"),
    plot_layer: Optional[str] = typer.Option(None, "--plot-layer"),
    plot_umap_ncols: int = typer.Option(3, "--plot-umap-ncols"),
    gene_filter: List[str] = typer.Option(
        [],
        "--gene-filter",
        help="Filter genes before DE computation (repeatable).",
    ),
    plot_gene_filter: List[str] = typer.Option(
        [],
        "--plot-gene-filter",
        help="Filter expression for plot gene selection (e.g. \"gene_chrom not in ['X','Y']\").",
    ),
    plot_sample_annotation_keys: List[str] = typer.Option(
        [],
        "--plot-sample-annotation-keys",
        help="Obs keys for sample heatmap category bars (repeatable). Defaults to condition_keys.",
    ),
):
    if output_name is None:
        output_name = _default_output_name(input_path, "markers", round_id=round_id)
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
        regenerate_figures=regenerate_figures,
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
        max_workers=max_workers,
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=tuple(_parse_csv_repeat(pb_covariates) or ()),
        prune_uns_de=prune_uns_de,
        condition_keys=[],
        target_groups=None,
        de_decoupler_source="none",
        de_decoupler_stat_col="stat",
        decoupler_method="consensus",
        decoupler_consensus_methods=None,
        decoupler_min_n_targets=5,
        decoupler_bar_split_signed=decoupler_bar_split_signed,
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
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
        plot_gene_filter=plot_gene_filter,
        plot_sample_annotation_keys=plot_sample_annotation_keys,
    )

    # enforce mode semantics
    cfg.condition_key = None
    cfg.condition_contrasts = ()
    cfg.contrast_key = None
    cfg.contrast_contrasts = ()
    cfg.contrast_conditional_de = False

    run_cluster_vs_rest(cfg)


@enrichment_app.command(
    "cluster",
    help="Run round-native enrichment scoring (MSigDB, PROGENy, DoRothEA) on an existing AnnData.",
)
def enrichment_cluster(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: sibling results/ directory, or nearest results/ ancestor when the input already lives inside results/).",
    ),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    regenerate_figures: bool = typer.Option(
        False,
        "--regenerate-figures",
        help="[Plots] Regenerate enrichment figures from stored round payloads only (no recomputation).",
    ),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_key: Optional[str] = typer.Option(
        None,
        "--condition-key",
        help="[Enrichment] Aggregate cluster-by-condition instead of cluster-only, e.g. 'sex' or 'timepoint:masld_status'.",
    ),
    gene_filter: List[str] = typer.Option(
        [],
        "--gene-filter",
        help="[Enrichment] Filter genes before decoupler using pandas-query expressions against adata.var, e.g. \"not gene.str.startswith('MT-')\".",
    ),
    decoupler_pseudobulk_agg: str = typer.Option("mean", "--decoupler-pseudobulk-agg"),
    decoupler_use_raw: bool = typer.Option(
        True,
        "--decoupler-use-raw/--no-decoupler-use-raw",
    ),
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
    run_gsea: bool = typer.Option(True, "--run-gsea/--no-run-gsea"),
    gsea_min_size: int = typer.Option(10, "--gsea-min-size"),
    gsea_max_size: int = typer.Option(500, "--gsea-max-size"),
    gsea_eps: float = typer.Option(1e-10, "--gsea-eps"),
    gsea_rank_col: Optional[str] = typer.Option(
        None,
        "--gsea-rank-col",
        help="Preferred statistic column for MSigDB GSEA. Defaults to the enrichment decoupler statistic column.",
    ),
    joint_enrichment_alpha: float = typer.Option(0.05, "--joint-enrichment-alpha"),
    joint_enrichment_top_n: int = typer.Option(20, "--joint-enrichment-top-n"),
    joint_enrichment_require_concordant: bool = typer.Option(
        True,
        "--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant",
    ),
    joint_enrichment_require_gsea_sig: bool = typer.Option(
        True,
        "--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig",
    ),
    joint_enrichment_leading_edge_top_n: int = typer.Option(8, "--joint-enrichment-leading-edge-top-n"),
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
):
    if output_name is None:
        output_name = _default_output_name(input_path, "enrichment", round_id=round_id)

    if msigdb_gene_sets_cli is None:
        msigdb_gene_sets = None
    else:
        msigdb_gene_sets = [x.strip() for x in msigdb_gene_sets_cli.split(",") if x.strip()]

    cfg = _build_cfg_enrichment_cluster(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        regenerate_figures=regenerate_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        condition_key=condition_key,
        gene_filter=gene_filter,
        decoupler_pseudobulk_agg=decoupler_pseudobulk_agg,
        decoupler_use_raw=decoupler_use_raw,
        decoupler_method=decoupler_method,
        decoupler_consensus_methods=decoupler_consensus_methods,
        decoupler_min_n_targets=decoupler_min_n_targets,
        decoupler_bar_split_signed=decoupler_bar_split_signed,
        decoupler_bar_top_n_up=decoupler_bar_top_n_up,
        decoupler_bar_top_n_down=decoupler_bar_top_n_down,
        msigdb_gene_sets=msigdb_gene_sets,
        msigdb_method=msigdb_method,
        msigdb_min_n_targets=msigdb_min_n_targets,
        run_gsea=run_gsea,
        gsea_min_size=gsea_min_size,
        gsea_max_size=gsea_max_size,
        gsea_eps=gsea_eps,
        gsea_rank_col=gsea_rank_col,
        joint_enrichment_alpha=joint_enrichment_alpha,
        joint_enrichment_top_n=joint_enrichment_top_n,
        joint_enrichment_require_concordant=joint_enrichment_require_concordant,
        joint_enrichment_require_gsea_sig=joint_enrichment_require_gsea_sig,
        joint_enrichment_leading_edge_top_n=joint_enrichment_leading_edge_top_n,
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
    )

    run_enrichment_cluster(cfg)


@enrichment_app.command(
    "de",
    help="Run pathway and TF enrichment from exported DE result tables without loading AnnData.",
)
def enrichment_de(
    input_dir: Path = typer.Option(..., "--input-dir", "-i"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: sibling results/ directory, or nearest results/ ancestor when the input already lives inside results/).",
    ),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    gene_filter: List[str] = typer.Option(
        [],
        "--gene-filter",
        help="[Enrichment] Filter genes before decoupler using pandas-query expressions against the DE gene table metadata, e.g. \"not gene.str.startswith('MT-')\".",
    ),
    de_decoupler_source: str = typer.Option("auto", "--de-decoupler-source"),
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
):
    if output_name is None:
        output_name = f"enrichment_de_{re.sub(r'[^A-Za-z0-9._-]+', '_', input_dir.name.strip())}"

    if msigdb_gene_sets_cli is None:
        msigdb_gene_sets = None
    else:
        msigdb_gene_sets = [x.strip() for x in msigdb_gene_sets_cli.split(",") if x.strip()]

    cfg = _build_cfg_enrichment_de(
        input_dir=input_dir,
        output_dir=output_dir,
        output_name=str(output_name),
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        gene_filter=gene_filter,
        de_decoupler_source=de_decoupler_source,
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
    )

    run_enrichment_de_from_tables(cfg)


@enrichment_app.command(
    "module-score",
    help="Score user-defined gene modules per cell, then summarize by cluster or cluster-condition.",
)
def enrichment_module_score(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    condition_key: Optional[str] = typer.Option(
        None,
        "--condition-key",
        help="[Module score] Aggregate cluster-by-condition instead of cluster-only, e.g. 'sex' or 'sex:MASLD'.",
    ),
    module_files: List[Path] = typer.Option(
        ...,
        "--module-file",
        help="[Module score] Module definition file(s). Supports .gmt, .tsv, .csv, and single-module .txt.",
    ),
    module_set_name: Optional[str] = typer.Option(
        None,
        "--module-set-name",
        help="[Module score] Optional stable name for this module collection. Defaults to the first module file stem.",
    ),
    module_score_method: str = typer.Option(
        "scanpy",
        "--module-score-method",
        help="[Module score] Backend. Use 'scanpy' for score_genes or 'aucell' for rank-based AUCell scoring.",
    ),
    module_score_use_raw: bool = typer.Option(
        False,
        "--module-score-use-raw/--no-module-score-use-raw",
    ),
    module_score_layer: Optional[str] = typer.Option(None, "--module-score-layer"),
    module_score_ctrl_size: int = typer.Option(50, "--module-score-ctrl-size"),
    module_score_n_bins: int = typer.Option(25, "--module-score-n-bins"),
    module_score_random_state: int = typer.Option(0, "--module-score-random-state"),
    module_score_max_umaps: int = typer.Option(12, "--module-score-max-umaps"),
):
    if not module_files:
        raise typer.BadParameter("Provide at least one --module-file.")
    method = str(module_score_method or "scanpy").strip().lower()
    if method not in {"scanpy", "aucell"}:
        raise typer.BadParameter("Invalid --module-score-method. Use: scanpy, aucell.")
    if module_score_use_raw and module_score_layer:
        raise typer.BadParameter("Cannot use both --module-score-use-raw and --module-score-layer.")

    inferred_set_name = str(module_set_name).strip() if module_set_name else Path(module_files[0]).stem
    if output_name is None:
        output_name = _default_output_name(
            input_path,
            f"module_score_{re.sub(r'[^A-Za-z0-9._-]+', '_', inferred_set_name)}",
            round_id=round_id,
        )

    cfg = _build_cfg_module_score(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        condition_key=condition_key,
        module_files=module_files,
        module_set_name=inferred_set_name,
        module_score_method=method,
        module_score_use_raw=module_score_use_raw,
        module_score_layer=module_score_layer,
        module_score_ctrl_size=module_score_ctrl_size,
        module_score_n_bins=module_score_n_bins,
        module_score_random_state=module_score_random_state,
        module_score_max_umaps=module_score_max_umaps,
    )

    run_module_score(cfg)


@markers_and_de_app.command(
    "da",
    help="DA: differential abundance vs condition (compositional models).",
)
def composition(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),

    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    regenerate_figures: bool = typer.Option(
        False,
        "--regenerate-figures",
        help="[Plots] Regenerate figures from stored results only (no recomputation).",
    ),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),

    round_id: Optional[str] = typer.Option(None, "--round-id"),
    replicate_key: Optional[str] = typer.Option(None, "--replicate-key"),

    condition_keys: List[str] = typer.Option(
        [],
        "--condition-keys",
        help="One or more condition keys. Use ':' to build composite keys (e.g., MASLD:sex).",
    ),
    covariates: List[str] = typer.Option([], "--covariates"),
    method: List[str] = typer.Option(
        ["sccoda", "glm", "clr", "graph"],
        "--method",
        help="Composition methods to run (default: all).",
    ),
    reference: str = typer.Option("most_stable", "--reference"),
    min_mean_prop: float = typer.Option(0.01, "--min-mean-prop"),
    min_cells_per_sample_cluster: int = typer.Option(20, "--min-cells-per-sample-cluster"),
    alpha: float = typer.Option(0.05, "--alpha"),

    graph_n_seeds: int = typer.Option(2000, "--graph-n-seeds"),
    graph_k_ref: int = typer.Option(30, "--graph-k-ref"),
    graph_max_k: int = typer.Option(200, "--graph-max-k"),
    graph_min_size: int = typer.Option(20, "--graph-min-size"),
    graph_random_state: int = typer.Option(42, "--graph-random-state"),
    graph_min_nonzero_samples_per_level: int = typer.Option(3, "--graph-min-nonzero-samples-per-level"),
    graph_n_permutations: int = typer.Option(
        0,
        "--graph-n-permutations",
        help="[Deprecated, ignored] GraphDA uses NB-GLM + spatial weighted-BH FDR.",
    ),
    graph_effect_shrink_k: float = typer.Option(10.0, "--graph-effect-shrink-k"),
):
    if output_name is None:
        output_name = _default_output_name(input_path, "da", round_id=round_id)
    if not condition_keys:
        raise typer.BadParameter("--condition-keys is required.")

    cfg = _build_cfg_composition(
        input_path=input_path,
        output_dir=output_dir,
        output_name=str(output_name),
        save_h5ad=save_h5ad,
        n_jobs=n_jobs,
        make_figures=make_figures,
        regenerate_figures=regenerate_figures,
        figdir_name=figdir_name,
        figure_formats=figure_formats,
        round_id=round_id,
        replicate_key=replicate_key,
        condition_keys=condition_keys,
        covariates=covariates,
        methods=method,
        reference=reference,
        min_mean_prop=min_mean_prop,
        min_cells_per_sample_cluster=min_cells_per_sample_cluster,
        alpha=alpha,
        graph_n_seeds=graph_n_seeds,
        graph_k_ref=graph_k_ref,
        graph_max_k=graph_max_k,
        graph_min_size=graph_min_size,
        graph_random_state=graph_random_state,
        graph_min_nonzero_samples_per_level=graph_min_nonzero_samples_per_level,
        graph_n_permutations=graph_n_permutations,
        graph_effect_shrink_k=graph_effect_shrink_k,
    )

    run_composition(cfg)


@markers_and_de_app.command(
    "de",
    help="Within-cluster contrasts: compare condition levels within each group.",
)
def within_cluster(
    input_path: Path = typer.Option(..., "--input-path", "-i"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="[I/O] Output directory (default: sibling results/ directory, or input parent if already inside results/).",
    ),
    output_name: Optional[str] = typer.Option(None, "--output-name"),
    save_h5ad: bool = typer.Option(False, "--save-h5ad/--no-save-h5ad"),
    n_jobs: int = typer.Option(1, "--n-jobs"),

    make_figures: bool = typer.Option(True, "--make-figures/--no-make-figures"),
    regenerate_figures: bool = typer.Option(
        False,
        "--regenerate-figures",
        help="[Plots] Regenerate figures from stored results only (no recomputation).",
    ),
    figdir_name: str = typer.Option("figures", "--figdir-name"),
    figure_formats: List[str] = typer.Option(["png", "pdf"], "--figure-formats", "-F"),

    run: RunWhich = typer.Option(RunWhich.both, "--run", case_sensitive=False),

    group_key: Optional[Path] = typer.Option(None, "--group-key"),  # NOTE: fixed below to Optional[str]
    label_source: str = typer.Option("pretty", "--label-source"),
    round_id: Optional[str] = typer.Option(None, "--round-id"),
    replicate_key: Optional[str] = typer.Option(None, "--replicate-key"),

    condition_key: Optional[str] = typer.Option(None, "--condition-key"),
    condition_keys: List[str] = typer.Option([], "--condition-keys"),
    target_groups: List[str] = typer.Option(
        [],
        "--target-groups",
        help="Restrict within-cluster DE to selected group labels (repeatable/comma-separated).",
    ),
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
    max_workers: int = typer.Option(
        8,
        "--max-workers",
        "--pb-max-workers",
        help="Maximum parallel DE worker tasks for both pseudobulk and cell-level phases.",
    ),
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
    run_gsea: bool = typer.Option(True, "--run-gsea/--no-run-gsea"),
    gsea_min_size: int = typer.Option(10, "--gsea-min-size"),
    gsea_max_size: int = typer.Option(500, "--gsea-max-size"),
    gsea_eps: float = typer.Option(1e-10, "--gsea-eps"),
    gsea_rank_col: Optional[str] = typer.Option(
        None,
        "--gsea-rank-col",
        help="Preferred DE statistic column for MSigDB GSEA. Defaults to the DE decoupler statistic column.",
    ),
    joint_enrichment_alpha: float = typer.Option(0.05, "--joint-enrichment-alpha"),
    joint_enrichment_top_n: int = typer.Option(20, "--joint-enrichment-top-n"),
    joint_enrichment_require_concordant: bool = typer.Option(
        True,
        "--joint-enrichment-require-concordant/--no-joint-enrichment-require-concordant",
    ),
    joint_enrichment_require_gsea_sig: bool = typer.Option(
        True,
        "--joint-enrichment-require-gsea-sig/--no-joint-enrichment-require-gsea-sig",
    ),
    joint_enrichment_leading_edge_top_n: int = typer.Option(8, "--joint-enrichment-leading-edge-top-n"),

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
    plot_top_n_per_cluster: int = typer.Option(12, "--plot-top-n-per-cluster"),
    plot_dotplot_top_n_per_cluster: int = typer.Option(16, "--plot-dotplot-top-n-per-cluster"),
    plot_max_genes_total: int = typer.Option(80, "--plot-max-genes-total"),
    plot_use_raw: bool = typer.Option(False, "--plot-use-raw/--no-plot-use-raw"),
    plot_layer: Optional[str] = typer.Option(None, "--plot-layer"),
    plot_umap_ncols: int = typer.Option(3, "--plot-umap-ncols"),
    gene_filter: List[str] = typer.Option(
        [],
        "--gene-filter",
        help="Filter genes before DE computation (repeatable).",
    ),
    plot_gene_filter: List[str] = typer.Option(
        [],
        "--plot-gene-filter",
        help="Filter expression for plot gene selection (e.g. \"gene_chrom not in ['X','Y']\").",
    ),
    plot_sample_annotation_keys: List[str] = typer.Option(
        [],
        "--plot-sample-annotation-keys",
        help="Obs keys for sample heatmap category bars (repeatable). Defaults to condition_keys.",
    ),
):
    # fix typo from signature (keep CLI option stable)
    group_key = None if group_key is None else str(group_key)
    if output_name is None:
        output_name = _default_output_name(input_path, "de", round_id=round_id)

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
        max_workers=max_workers,
        pb_min_counts_per_lib=pb_min_counts_per_lib,
        pb_min_lib_pct=pb_min_lib_pct,
        pb_covariates=pb_covariates,
        prune_uns_de=prune_uns_de,
        condition_keys=condition_keys,
        target_groups=target_groups,
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
        run_gsea=run_gsea,
        gsea_min_size=gsea_min_size,
        gsea_max_size=gsea_max_size,
        gsea_eps=gsea_eps,
        gsea_rank_col=gsea_rank_col,
        joint_enrichment_alpha=joint_enrichment_alpha,
        joint_enrichment_top_n=joint_enrichment_top_n,
        joint_enrichment_require_concordant=joint_enrichment_require_concordant,
        joint_enrichment_require_gsea_sig=joint_enrichment_require_gsea_sig,
        joint_enrichment_leading_edge_top_n=joint_enrichment_leading_edge_top_n,
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
        gene_filter=gene_filter,
        plot_gene_filter=plot_gene_filter,
        plot_sample_annotation_keys=plot_sample_annotation_keys,
        regenerate_figures=regenerate_figures,
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
