# scANVI Supervision

When `scANVI` is enabled, `scOmnom` needs a supervision label for each cell. scOmnom supports two main supervision patterns:

* **Initial integration:** generate labels automatically with BISC on the scVI latent space, or use an existing `adata.obs` column.
* **Annotated secondary integration:** after `cluster-and-annotate`, reuse final round labels with low-confidence cells set to `Unknown`.

## Initial Integration Supervision

By default, scOmnom uses BISC-generated labels for the first scANVI integration run:

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --methods scVI scANVI \
  --scanvi-label-source bisc \
  --output-dir results/integrate/
```

The BISC supervision step uses the **scVI latent space** with a structural-only sweep:

1. **Latent-space resolution sweep**
   A coarse, limited-bandwidth Leiden sweep is performed on the scVI latent space.

2. **Structural selection**
   Candidate resolutions are scored using stability, centroid-based silhouette, and tiny-cluster penalties.

3. **Parsimony**
   The lowest resolution near the top score is selected.

This produces labels used **only for scANVI supervision**, which will be replaced by downstream clustering of the final integrated embedding.

### Standard scANVI knobs

| Option | Default | Meaning |
| --- | --- | --- |
| `--methods` | all methods | Include `scANVI` to run supervised scANVI. The default method set already includes it. |
| `--scanvi-label-source` | `bisc` | Supervision source for the first scANVI run. Use `bisc` for automatic structural labels or `leiden` for an existing obs column. |
| `--scanvi-labels-key` | `leiden` | Obs column used when `--scanvi-label-source leiden`. Ignored when source is `bisc`. |
| `--batch-key` | inferred when possible | Batch/sample column used by scVI/scANVI. |
| `--multi-gpu` / `--single-gpu` | `--single-gpu` | Use all visible CUDA devices for scVI/scANVI training via distributed training. |

Example using an existing label column:

```bash
scomnom integrate \
  --input-path results/load_and_filter/adata.filtered.zarr \
  --batch-key sample_id \
  --methods scVI scANVI \
  --scanvi-label-source leiden \
  --scanvi-labels-key coarse_cell_state \
  --output-dir results/integrate/
```

Notes:

* scVI is trained first and scANVI reuses that trained scVI model.
* scOmnom prefers `counts_cb`, then `counts_raw`, for scVI/scANVI. It refuses to train these models on `adata.X` when no counts layer is available.
* scANVI uses `Unknown` as the unlabeled category.
* scANVI training epochs and `n_samples_per_label` are currently chosen internally rather than exposed as CLI options.

## CellTypist And Benchmarking Labels

CellTypist does not directly provide the default initial scANVI supervision labels when `--scanvi-label-source bisc`. It is still important during `integrate` because scIB benchmarking defaults to CellTypist confident cell-level labels.

Related options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--celltypist-model` | `Immune_All_Low.pkl` | CellTypist model used to generate/reuse cell-level labels for benchmarking and later annotation reuse. Use `None`/`none` to disable. |
| `--celltypist-majority-voting` / `--no-celltypist-majority-voting` | majority voting on | CellTypist prediction mode. |
| `--celltypist-label-key` | `celltypist_label` | Obs key for cell-level CellTypist labels. |
| `--celltypist-cluster-label-key` | `celltypist_cluster_label` | Base obs key for cluster-level CellTypist labels used later by `cluster-and-annotate`. |
| `--scib-truth-label-key` | `celltypist` | Truth label choice for scIB benchmarking. Valid values include `celltypist`, `leiden`, and `final`. |
| `--label-key` | `leiden` | scIB/reporting label key when benchmarking against a plain obs column. This does not control scANVI supervision labels. |
| `--bio-entropy-abs-limit` | `0.5` | Absolute entropy ceiling used when building confident CellTypist masks. |
| `--bio-entropy-quantile` | `0.7` | Entropy quantile used with the absolute ceiling; the entropy cutoff is `max(abs_limit, quantile_value)`. |
| `--bio-margin-min` | `0.10` | Minimum top1-top2 CellTypist probability margin required for confidence. |

Shared options that are mostly consumed by `cluster-and-annotate`, but are accepted on `integrate` for consistency:

| Option | Default | Meaning |
| --- | --- | --- |
| `--bio-mask-mode` | `entropy_margin` | Bio-mask mode used by downstream BISC/annotation logic. Use `none` to disable that mask downstream. |
| `--bio-mask-min-cells` | `500` | Downstream safety gate: disable the bio mask if too few cells pass. |
| `--bio-mask-min-frac` | `0.05` | Downstream safety gate: disable the bio mask if too small a fraction of cells pass. |
| `--pretty-label-min-masked-cells` | `25` | Minimum confident cells in a cluster needed to assign a CellTypist-backed pretty label. |
| `--pretty-label-min-masked-frac` | `0.10` | Minimum confident fraction in a cluster needed to assign a CellTypist-backed pretty label. |

See also [scIB benchmarking and truth labels](benchmarking.md).

## Annotated Secondary scANVI Supervision

After `cluster-and-annotate`, `integrate --annotated-run` performs a second supervised scANVI pass using final annotation labels. This is where the supervision is most customizable.

In this mode, scOmnom:

1. Resolves a source clustering round.
2. Resolves a final label column from that round.
3. Reconstructs a CellTypist entropy/margin confidence mask.
4. Writes a new scANVI supervision column where confident cells keep their final label and low-confidence cells become `Unknown`.
5. Trains a new scVI/scANVI model and writes `adata.obsm["scANVI__annotated"]`.
6. Creates a derived projection round so the previous clustering state is preserved.

Annotated-run knobs:

| Option | Default | Meaning |
| --- | --- | --- |
| `--annotated-run` / `--no-annotated-run` | off | Enable the secondary annotated scANVI mode. When enabled, scOmnom forces a scANVI-only run. |
| `--annotated-run-cluster-round` | active round | Source cluster round for final labels. |
| `--annotated-run-final-label-key` | round `pretty_cluster_key`, then `cluster_label` | Override the obs column used as final labels. |
| `--annotated-run-confidence-mask-key` | `celltypist_confident_entropy_margin` | Obs key where the reconstructed boolean confidence mask is stored. |
| `--annotated-run-scanvi-labels-key` | `scanvi_labels__annotated` | Obs key where final-label-or-`Unknown` supervision labels are stored. |
| `--bio-entropy-abs-limit` | `0.5` | Absolute entropy ceiling for confident CellTypist predictions. |
| `--bio-entropy-quantile` | `0.7` | Entropy quantile used with the absolute ceiling; the entropy cutoff is `max(abs_limit, quantile_value)`. |
| `--bio-margin-min` | `0.10` | Minimum top1-top2 CellTypist probability margin required for confidence. |
| `--multi-gpu` / `--single-gpu` | `--single-gpu` | Use all visible CUDA devices for the annotated scVI/scANVI training. |

Example:

```bash
scomnom integrate \
  --input-path results/cluster_and_annotate/adata.clustered.annotated.zarr \
  --annotated-run \
  --annotated-run-cluster-round r2_manual_rename \
  --annotated-run-final-label-key cluster_label__r2_manual_rename \
  --annotated-run-scanvi-labels-key scanvi_labels__manual_r2 \
  --bio-entropy-abs-limit 0.45 \
  --bio-entropy-quantile 0.65 \
  --bio-margin-min 0.15 \
  --output-dir results/integrate_annotated/ \
  --output-name adata.clustered.annotated.projected
```

See also [Annotated refinement](annotated-run.md).

---
