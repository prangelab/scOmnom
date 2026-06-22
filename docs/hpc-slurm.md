# Running on HPC / SLURM clusters

Example SLURM job scripts are provided:

```text
slurm/
├── scomnom_template.job
├── scomnom_1_load_and_filter.job
├── scomnom_2_integrate.job
├── scomnom_3_cluster_and_annotate.job
├── scomnom_4_markers.job
├── scomnom_5_integrate_annotated_run.job
├── scomnom_5a_subset.job
├── scomnom_5b_annotation_merge.job
├── scomnom_6_rename.job
├── scomnom_6a_rename_archetypes.job
├── scomnom_7_de.job
├── scomnom_8_da.job
└── scomnom_9_enrichment_cluster.job
```

These scripts are configured for our local HPC ([SURF's Snellius](https://www.surf.nl/diensten/rekenen/snellius-de-nationale-supercomputer)). Users on other systems must adapt module names, CUDA/driver versions, and conda initialization paths.

For large datasets, running on a **GPU partition is strongly recommended**, particularly for:

* `load-and-filter` doublet detection (scVI-SOLO)
* `integrate` (scVI/scANVI)

#### Performance reference (Snellius)

Benchmarks obtained using **1× NVIDIA A100 GPU** and **18 CPU cores**.

On Snellius, this CPU allocation implies a **minimum memory slice of 120 GB**, even though the example job scripts do not explicitly request that amount.

- **~40k cells**
  - `load-and-filter`: < 20 minutes
  - `integrate`: < 30 minutes

- **~400k cells**
  - `load-and-filter`: < 1.5 hours
  - `integrate`: < 7 hours

In practice, the pipeline is likely to run with **substantially less memory (≈ ≤ 60 GB)**, depending on e.g., selected integration methods.

When submitting jobs, ensure `sbatch` is called with appropriate wall time, CPU core reservations, and GPU resources.

The provided SLURM scripts are intended as starting points, not drop-in solutions.
