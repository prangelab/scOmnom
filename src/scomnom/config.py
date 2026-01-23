from __future__ import annotations

from numba.core.types import Boolean
from pydantic import BaseModel, Field, validator, model_validator, field_validator
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from matplotlib.figure import Figure
import multiprocessing


class LoadAndFilterConfig(BaseModel):

    # ---- Input ----
    raw_sample_dir: Optional[Path] = None
    filtered_sample_dir: Optional[Path] = None
    cellbender_dir: Optional[Path] = None

    metadata_tsv: Optional[Path] = None
    batch_key: Optional[str] = None

    # ---- Output ----
    output_dir: Path
    output_name: str = "adata.filtered"
    save_h5ad: bool = False

    # ---- Compute ----
    n_jobs: int = 4

    # ---- QC ----
    min_cells: int = 3
    min_genes: int = 500
    min_cells_per_sample: int = 20
    max_pct_mt: float = 5.0
    n_top_genes: int = 2000

    # ---- QC: upper-cut filters (MAD + quantile) ----
    max_genes_mad: float = Field(
        5.0,
        description="Upper cutoff for n_genes_by_counts as median + k*MAD (default: 5).",
    )
    max_genes_quantile: float = Field(
        0.999,
        description="Upper quantile cutoff for n_genes_by_counts (default: 0.999).",
    )
    max_counts_mad: float = Field(
        5.0,
        description="Upper cutoff for total_counts as median + k*MAD (default: 5).",
    )
    max_counts_quantile: float = Field(
        0.999,
        description="Upper quantile cutoff for total_counts (default: 0.999).",
    )

    # ---- Doublets (SOLO) ----
    expected_doublet_rate: float = 0.1
    apply_doublet_score: Optional[bool] = None
    apply_doublet_score_path: Optional[Path] = "results/adata.merged.zarr"

    # ---- Patterns ----
    raw_pattern: str = "*.raw_feature_bc_matrix"
    filtered_pattern: str = "*.filtered_feature_bc_matrix"
    cellbender_pattern: str = "*.cellbender_filtered.output"
    cellbender_h5_suffix: str = ".cellbender_out_filtered.h5"
    cellbender_barcode_suffix: str = Field(
        ".cellbender_out_cell_barcodes.csv",
        description="Suffix for CellBender barcode file "
                    "(e.g. '_cellbender_out_cell_barcodes.csv')."
    )

    # ---- Figures ----
    make_figures: bool = True
    figdir_name: str = "figures"
    figure_formats: List[str] = Field(default_factory=lambda: ["png", "pdf"])

    # ---- Logging ----
    logfile: Optional[Path] = None

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    # ---- Validators ----
    @model_validator(mode="after")
    def check_inputs(self):
        # --apply-doublet-score mode
        if self.apply_doublet_score is not None:
            # metadata not required
            return self

        # Normal modes require metadata
        if self.metadata_tsv is None:
            raise ValueError(
                "metadata_tsv is required unless --apply-doublet-score is used"
            )

        if self.filtered_sample_dir is not None:
            if self.raw_sample_dir or self.cellbender_dir:
                raise ValueError("filtered cannot be combined with raw or cellbender")

        if self.raw_sample_dir is None and self.filtered_sample_dir is None:
            raise ValueError("Must provide raw_sample_dir or filtered_sample_dir")

        return self


from pathlib import Path
from typing import List, Optional, Sequence
from pydantic import BaseModel, Field, field_validator


class IntegrateConfig(BaseModel):
    # ------------------------------------------------------------------
    # IO
    # ------------------------------------------------------------------
    input_path: Path
    output_dir: Path
    output_name: str = "adata.integrated"
    save_h5ad: bool = False

    logfile: Optional[Path] = None

    # ------------------------------------------------------------------
    # Core keys
    # ------------------------------------------------------------------
    batch_key: Optional[str] = None
    label_key: str = "leiden"  # used for scIB + downstream reporting

    # ------------------------------------------------------------------
    # Integration methods
    # ------------------------------------------------------------------
    methods: Optional[List[str]] = None
    benchmark_n_jobs: int = 16
    benchmark_threshold: int = 100_000
    benchmark_n_cells: int = 100_000
    benchmark_random_state: int = 42

    # ------------------------------------------------------------------
    # scANVI supervision (NEW)
    # ------------------------------------------------------------------
    scanvi_label_source: Literal["leiden", "bisc_light"] = "bisc_light"

    scanvi_max_prelabel_clusters: int = 25  # ↑ increase for atlas-scale data
    scanvi_preflight_resolutions: Optional[list[float]] = None

    scanvi_preflight_min_stability: float = 0.60
    scanvi_preflight_parsimony_eps: float = 0.03

    scanvi_w_stability: float = 0.50
    scanvi_w_silhouette: float = 0.35
    scanvi_w_tiny: float = 0.15

    scanvi_prelabels_key: str = "scanvi_prelabels"
    scanvi_labels_key: str = "leiden"  # only used when scanvi_label_source != "bisc_light"

    scanvi_batch_trap_threshold: float = 0.90
    scanvi_batch_trap_min_cells: int = 200
    scanvi_tiny_cluster_min_cells: int = 30

    # ------------------------------------------------------------------
    # Annotated secondary integration (SECOND PASS; explicit + guarded)
    # ------------------------------------------------------------------
    annotated_run: bool = False
    scib_truth_label_key: str = "leiden"

    # Which cluster round to source "final" labels from. If None -> use active_cluster_round.
    annotated_run_cluster_round: Optional[str] = None

    # Optional override for which obs column to treat as "final labels".
    # If None -> use rounds[round_id]["annotation"]["pretty_cluster_key"], else fall back to adata.obs["cluster_label"].
    annotated_run_final_label_key: Optional[str] = None

    # Where to store the derived boolean mask in adata.obs
    annotated_run_confidence_mask_key: str = "celltypist_confident_entropy_margin"

    # Where to store the scANVI supervision labels (final label where confident, else "Unknown")
    annotated_run_scanvi_labels_key: str = "scanvi_labels__annotated"

    # Entropy-margin mask thresholds (must match cluster_and_annotate policy)
    bio_entropy_abs_limit: float = 0.5
    bio_entropy_quantile: float = 0.7
    bio_margin_min: float = 0.10

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figdir_name: str = "figures"
    figure_formats: List[str] = Field(default_factory=lambda: ["png", "pdf"])

    # ------------------------------------------------------------------
    # Derived paths
    # ------------------------------------------------------------------
    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("methods")
    @classmethod
    def normalize_methods(cls, v):
        if v is None:
            return None
        return [m.lower() for m in v]

    @field_validator("scanvi_label_source")
    @classmethod
    def validate_scanvi_label_source(cls, v):
        allowed = {"leiden", "bisc_light"}
        if v not in allowed:
            raise ValueError(
                f"scanvi_label_source must be one of {sorted(allowed)}, got {v!r}"
            )
        return v


# ---------------------------------------------------------------------
# CLUSTER AND ANNOTATE CONFIG
# ---------------------------------------------------------------------
class ClusterAnnotateConfig(BaseModel):
    # ------------------------------------------------------------------
    # I/O (mirror IntegrateConfig)
    # ------------------------------------------------------------------
    input_path: Path = Field(
        ...,
        description="Integrated dataset from the integration step (.zarr or .h5ad)",
    )

    output_dir: Optional[Path] = Field(
        None,
        description="Directory for outputs. Defaults to input_path.parent",
    )
    output_name: str = Field(
        "adata.clustered.annotated",
        description="Base name for outputs (no extension)",
    )
    save_h5ad: bool = Field(
        False,
        description="If True, also write an .h5ad copy in addition to the .zarr output",
    )

    # ------------------------------------------------------------------
    # Embedding / batch / labels
    # ------------------------------------------------------------------
    embedding_key: str = Field(
        "X_integrated",
        description="Key in .obsm to use for neighbors / silhouette",
    )
    batch_key: Optional[str] = Field(
        None,
        description="Batch/sample key in adata.obs (default: auto-detect)",
    )
    label_key: str = Field(
        "leiden",
        description="Final cluster label key in adata.obs",
    )
    final_auto_idents_key: str = "final_auto_idents"

    # ------------------------------------------------------------------
    # bio-guided clustering weights
    # ------------------------------------------------------------------
    bio_guided_clustering: bool = True
    w_hom: float = 0.15
    w_frag: float = 0.10
    w_bioari: float = 0.15

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    figdir_name: str = "figures"
    make_figures: bool = True
    figure_formats: List[str] = Field(
        default_factory=lambda: ["png", "pdf"],
        description="Figure formats to save",
    )

    # ------------------------------------------------------------------
    # Resolution sweep / stability
    # ------------------------------------------------------------------
    res_min: float = Field(0.1, ge=0.0)
    res_max: float = Field(2.5, ge=0.0)
    n_resolutions: int = Field(25, ge=2)
    penalty_alpha: float = Field(0.02, ge=0.0)

    stability_repeats: int = Field(5, ge=1)
    subsample_frac: float = Field(0.8, gt=0.0, le=1.0)
    random_state: int = 42

    tiny_cluster_size: int = 20
    min_cluster_size: int = 20
    min_plateau_len: int = 3
    max_cluster_jump_frac: float = 0.4
    stability_threshold: float = 0.85

    w_stab: float = 0.50
    w_sil: float = 0.35
    w_tiny: float = 0.15

    # ------------------------------------------------------------------
    # CellTypist
    # ------------------------------------------------------------------
    celltypist_model: Optional[str] = Field(
        "Immune_All_Low.pkl",
        description="Path or name of CellTypist model (.pkl). If None, skip annotation",
    )
    celltypist_majority_voting: bool = True
    celltypist_label_key: str = "celltypist_label"
    celltypist_cluster_label_key: str = "celltypist_cluster_label"

    annotation_csv: Optional[Path] = None
    available_models: Optional[List[Dict[str, str]]] = None

    # ------------------------------------------------------------------
    # Bio mask (CellTypist confidence gate)
    # ------------------------------------------------------------------
    bio_mask_mode: Literal["entropy_margin", "none"] = Field(
        "entropy_margin",
        description="Bio mask mode. 'entropy_margin' uses entropy+margin on CellTypist probabilities; 'none' disables bio mask.",
    )

    bio_entropy_abs_limit: float = Field(
        0.5,
        description="Absolute entropy ceiling for CellTypist proba mask (cells with H <= this pass).",
    )

    bio_entropy_quantile: float = Field(
        0.7,
        description="Entropy quantile threshold for mask. Final entropy cut is max(abs_limit, quantile value).",
    )

    bio_margin_min: float = Field(
        0.10,
        description="Minimum CellTypist margin (top1 - top2) to pass mask.",
    )

    bio_mask_min_cells: int = Field(
        500,
        ge=0,
        description="Safety gate: disable bio mask if fewer than this many cells pass.",
    )

    bio_mask_min_frac: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Safety gate: disable bio mask if fewer than this fraction of cells pass.",
    )

    pretty_label_min_masked_cells: int = Field(
        25,
        ge=0,
        description="Minimum number of masked (high-confidence) cells required in a cluster to assign a CellTypist cluster label.",
    )

    pretty_label_min_masked_frac: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of masked cells within a cluster required to assign a CellTypist cluster label.",
    )

    # ------------------------------------------------------------------
    # Decoupler (cluster-level pseudobulk + nets)
    # ------------------------------------------------------------------
    run_decoupler: bool = True

    # Pseudobulk settings (shared by msigdb / dorothea / progeny)
    decoupler_pseudobulk_agg: str = Field(
        "mean",
        description="Aggregation for cluster-level pseudobulk. One of: 'mean', 'median'.",
    )
    decoupler_use_raw: bool = True
    decoupler_min_n_targets: int = Field(
        5,
        description="Minimum targets per source for decoupler methods.",
    )
    decoupler_method: str = Field(
        "consensus",
        description="Decoupler method (default: consensus).",
    )
    decoupler_consensus_methods: Optional[List[str]] = ["ulm", "mlm", "wsum"]

    # MSigDB (GMT-driven pathway nets)
    msigdb_gene_sets: List[str] = Field(
        default_factory=lambda: ["HALLMARK", "REACTOME"],
        description="MSigDB collections/keywords and/or paths to .gmt files.",
    )
    msigdb_method: str = Field(
        "consensus",
        description="Decoupler method to use for MSigDB nets (default: consensus).",
    )
    msigdb_min_n_targets: int = Field(
        5,
        description="Minimum targets per pathway for MSigDB decoupler run.",
    )

    # PROGENy
    run_progeny: bool = True
    progeny_method: str = Field("consensus")
    progeny_min_n_targets: int = Field(5)
    progeny_top_n: int = Field(100)
    progeny_organism: str = Field("human")

    # DoRothEA
    run_dorothea: bool = True
    dorothea_method: str = Field("consensus")
    dorothea_min_n_targets: int = Field(5)
    dorothea_confidence: List[str] = Field(default_factory=lambda: ["A", "B", "C"])
    dorothea_organism: str = Field("human")

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------
    enable_compacting: bool = Field(
        True,
        description="If True, create an additional 'compacted' clustering round using multiview agreement (progeny/dorothea/msigdb).",
    )

    compact_min_cells: int = Field(
        0,
        ge=0,
        description="Exclude clusters smaller than this from compaction decisions (0 disables).",
    )

    compact_zscore_scope: Literal["within_celltypist_label", "global"] = Field(
        "global",
        description="Z-score scope for similarity comparisons during compaction.",
    )

    compact_grouping: Literal["connected_components", "clique"] = Field(
        "connected_components",
        description="How to form compaction groups from pairwise-pass edges.",
    )

    compact_skip_unknown_celltypist_groups: bool = Field(
        False,
        description="If True, do not compact clusters whose round-scoped CellTypist cluster label is Unknown/UNKNOWN.",
    )

    # Thresholds used by compaction decision engine
    thr_progeny: float = Field(
        0.98,
        ge=-1.0,
        le=1.0,
        description="Similarity threshold for PROGENy activities when deciding compaction edges.",
    )

    thr_dorothea: float = Field(
        0.98,
        ge=-1.0,
        le=1.0,
        description="Similarity threshold for DoRothEA activities when deciding compaction edges.",
    )

    thr_msigdb_default: float = Field(
        0.98,
        ge=-1.0,
        le=1.0,
        description="Default similarity threshold for each MSigDB GMT block when deciding compaction edges.",
    )

    thr_msigdb_by_gmt: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-GMT similarity thresholds for MSigDB compaction (overrides thr_msigdb_default per GMT key).",
    )

    msigdb_required: bool = Field(
        True,
        description="If True, require MSigDB activity_by_gmt for compaction (otherwise compaction can run with progeny+doro only).",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logfile: Optional[Path] = None

    # ------------------------------------------------------------------
    # Derived paths (same pattern as IntegrateConfig)
    # ------------------------------------------------------------------
    @property
    def resolved_output_dir(self) -> Path:
        return (self.output_dir or self.input_path.parent).resolve()

    @property
    def figdir(self) -> Path:
        return self.resolved_output_dir / self.figdir_name

    @property
    def output_zarr_path(self) -> Path:
        return self.resolved_output_dir / f"{self.output_name}.zarr"

    @property
    def output_h5ad_path(self) -> Path:
        return self.resolved_output_dir / f"{self.output_name}.h5ad"

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_resolution_range(self):
        if self.res_min >= self.res_max:
            raise ValueError("res_min must be < res_max")
        return self

    @field_validator("figure_formats")
    @classmethod
    def validate_formats(cls, fmts: list[str]) -> list[str]:
        supported = Figure().canvas.get_supported_filetypes()
        out: list[str] = []
        for fmt in fmts:
            fmt_l = str(fmt).lower()
            if fmt_l not in supported:
                raise ValueError(
                    f"Unsupported figure format '{fmt}'. "
                    f"Supported formats include: {', '.join(sorted(supported))}"
                )
            out.append(fmt_l)
        return out

    @field_validator("decoupler_pseudobulk_agg")
    @classmethod
    def _validate_decoupler_pseudobulk_agg(cls, v: str) -> str:
        v = str(v).lower().strip()
        if v not in {"mean", "median"}:
            raise ValueError("decoupler_pseudobulk_agg must be one of: 'mean', 'median'")
        return v

    @field_validator("bio_entropy_abs_limit")
    @classmethod
    def _validate_bio_entropy_abs_limit(cls, v: float) -> float:
        v = float(v)
        if v < 0.0:
            raise ValueError("bio_entropy_abs_limit must be >= 0")
        return v

    @field_validator("bio_entropy_quantile")
    @classmethod
    def _validate_bio_entropy_quantile(cls, v: float) -> float:
        v = float(v)
        if not (0.0 < v <= 1.0):
            raise ValueError("bio_entropy_quantile must be in (0, 1]")
        return v

    @field_validator("bio_margin_min")
    @classmethod
    def _validate_bio_margin_min(cls, v: float) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError("bio_margin_min must be in [0, 1]")
        return v

    @field_validator("pretty_label_min_masked_frac", "bio_mask_min_frac")
    @classmethod
    def _validate_frac_01(cls, v: float) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError("value must be in [0, 1]")
        return v

    @field_validator("thr_progeny", "thr_dorothea", "thr_msigdb_default")
    @classmethod
    def _validate_similarity_thresholds(cls, v: float) -> float:
        v = float(v)
        # cosine similarities live in [-1, 1]
        if v < -1.0 or v > 1.0:
            raise ValueError("similarity threshold must be in [-1, 1]")
        return v

    @field_validator("thr_msigdb_by_gmt")
    @classmethod
    def _validate_thr_msigdb_by_gmt(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return None
        out: Dict[str, float] = {}
        for k, val in dict(v).items():
            vv = float(val)
            if vv < -1.0 or vv > 1.0:
                raise ValueError(f"thr_msigdb_by_gmt[{k!r}] must be in [-1, 1]")
            out[str(k)] = vv
        return out


# ---------------------------------------------------------------------
# MARKERS AND DE CONFIG
# ---------------------------------------------------------------------
class MarkersAndDEConfig(BaseModel):
    # ------------------------------------------------------------------
    # IO / run scaffolding
    # ------------------------------------------------------------------
    input_path: Path
    output_dir: Path
    output_name: str = "adata.markers_and_de"
    logfile: Optional[Path] = None

    # figures
    figdir_name: str = "figures"
    figure_formats: Sequence[str] = Field(default_factory=lambda: ["png", "pdf"])
    make_figures: bool = True

    # outputs
    save_h5ad: bool = False

    # ------------------------------------------------------------------
    # Grouping / round awareness
    # ------------------------------------------------------------------
    # If groupby is None, resolve from round/annotation (pretty labels by default).
    groupby: Optional[str] = None
    label_source: str = "pretty"  # forwarded to resolve_groupby_from_round
    round_id: Optional[str] = None

    # replicate key for pseudobulk (donor/patient/sample)
    # orchestrator uses: sample_key or batch_key or adata.uns["batch_key"]
    sample_key: Optional[str] = None
    batch_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Cell-level marker calling (scanpy rank_genes_groups)
    # ------------------------------------------------------------------
    markers_key: str = "cluster_markers_wilcoxon"
    markers_method: str = "wilcoxon"  # "wilcoxon" | "t-test" | "logreg" (scanpy)
    markers_n_genes: int = 300
    markers_rankby_abs: bool = True
    markers_use_raw: bool = False

    # OOM guards for cell-level marker calling
    markers_downsample_threshold: int = 500_000
    markers_downsample_max_per_group: int = 2_000
    random_state: int = 42

    # ------------------------------------------------------------------
    # Counts selection for pseudobulk DE
    # ------------------------------------------------------------------
    # preference order in AnnData.layers; fall back to .X only if allow_X_counts=True
    counts_layers: Tuple[str, ...] = ("counts_cb", "counts_raw")
    allow_X_counts: bool = True

    # ------------------------------------------------------------------
    # Pseudobulk DE: cluster vs rest
    # ------------------------------------------------------------------
    min_cells_target: int = 20
    alpha: float = 0.05
    store_key: str = "scomnom_de"

    # ------------------------------------------------------------------
    # Optional condition-within-cluster DE
    # ------------------------------------------------------------------
    condition_key: Optional[str] = None
    condition_contrasts: Tuple[str, ...] = ()
    min_cells_condition: int = 20

    # ------------------------------------------------------------------
    # Optional contrast-conditional mode
    # ------------------------------------------------------------------
    contrast_conditional_de: bool = False
    contrast_key: Optional[str] = None  # defaults to sample_key
    contrast_methods: Tuple[str, ...] = ("wilcoxon", "logreg")
    contrast_contrasts: Tuple[str, ...] = ()
    contrast_min_cells_per_level: int = 50
    contrast_max_cells_per_level: int = 2000
    contrast_min_total_counts: int = 10
    contrast_pseudocount: float = 1.0

    contrast_cl_alpha: float = 0.05
    contrast_cl_min_abs_logfc: float = 0.25
    contrast_lr_min_abs_coef: float = 0.25
    contrast_pb_min_abs_log2fc: float = 0.5

    # ------------------------------------------------------------------
    # Plot controls (used only by the orchestrator)
    # ------------------------------------------------------------------
    plot_lfc_thresh: float = 1.0
    plot_volcano_top_label_n: int = 15

    # gene selection for expression plots (dotplot/heatmap/umap/violin)
    plot_top_n_per_cluster: int = 10
    plot_max_genes_total: int = 80

    # scanpy expression plotting source
    plot_use_raw: bool = False
    plot_layer: Optional[str] = None

    # umap features grid layout
    plot_umap_ncols: int = 3
