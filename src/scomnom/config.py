from __future__ import annotations

from numba.core.types import Boolean
from pydantic import BaseModel, Field, validator, model_validator, field_validator
from pathlib import Path
from typing import Optional, Dict, List, Literal
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


class IntegrateConfig(BaseModel):

    input_path: Path
    output_dir: Path
    output_name: str = "adata.integrated"
    save_h5ad: bool = False

    batch_key: Optional[str] = None
    label_key: str = "leiden"

    methods: Optional[List[str]] = None
    benchmark_n_jobs: int = 16
    benchmark_threshold: int = 100000,
    benchmark_n_cells: int = 100000,
    benchmark_random_state: int = 42,

    figdir_name: str = "figures"
    figure_formats: List[str] = Field(default_factory=lambda: ["png", "pdf"])

    logfile: Optional[Path] = None

    @property
    def figdir(self) -> Path:
        return self.output_dir / self.figdir_name

    @field_validator("methods")
    @classmethod
    def normalize_methods(cls, v):
        if v is None:
            return None
        return [m.lower() for m in v]


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
    # Bio-metrics masking (CellTypist confidence gate)
    # Applies ONLY to biological metrics (bio_hom/bio_frag/bio_ari),
    # not to clustering itself.
    # ------------------------------------------------------------------
    bio_mask_enabled: bool = True

    # "Hybrid Entropy Gate" parameters
    bio_mask_entropy_abs_limit: float = Field(
        0.5,
        description="Absolute entropy ceiling. Cells with entropy <= this always pass.",
    )
    bio_mask_entropy_quantile: float = Field(
        0.7,
        description="Fallback entropy quantile (relative). Used as max(abs_limit, q-threshold).",
    )

    # Margin gate (p1 - p2)
    bio_mask_margin_min: float = Field(
        0.1,
        description="Minimum margin (top1 - top2) required to pass biomask.",
    )

    # Safety: if mask is too strict, relax to keep at least this fraction (0 disables)
    bio_mask_min_frac: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Ensure at least this fraction of cells remain for bio metrics; if fewer, relax entropy cutoff.",
    )


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

    @field_validator("bio_mask_entropy_abs_limit")
    @classmethod
    def _validate_bio_mask_entropy_abs(cls, v: float) -> float:
        v = float(v)
        if v < 0.0:
            raise ValueError("bio_mask_entropy_abs_limit must be >= 0")
        return v

    @field_validator("bio_mask_entropy_quantile")
    @classmethod
    def _validate_bio_mask_entropy_quantile(cls, v: float) -> float:
        v = float(v)
        if not (0.0 < v <= 1.0):
            raise ValueError("bio_mask_entropy_quantile must be in (0, 1]")
        return v

    @field_validator("bio_mask_margin_min")
    @classmethod
    def _validate_bio_mask_margin_min(cls, v: float) -> float:
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise ValueError("bio_mask_margin_min must be in [0, 1]")
        return v
