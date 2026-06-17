# Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/prangelab/scOmnom.git
cd scOmnom
```

`scOmnom` is installed into a dedicated environment defined in a platform-specific YAML file.

## 1) Create The Environment

Use the YAML that matches your platform:

- Linux/HPC: `environment_linux.yml`
- macOS: `environment_macos.yml`

On Linux or HPC systems, create the environment with `micromamba`:

```bash
micromamba create -f environment_linux.yml
micromamba activate scOmnom_env
```

On macOS, create the environment with Conda:

```bash
conda env create -f environment_macos.yml
conda activate scOmnom_env
```

This installs all required dependencies from `conda-forge`, `bioconda`, and related channels, including the Scanpy and PyTorch ecosystems.

## 2) Install `scOmnom`

For normal use, install the package into the active environment:

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Both commands register the `scomnom` command-line interface.

## Optional: Run With A Memory Guard On macOS

For large local runs on macOS, you can wrap `scomnom` with [`scripts/run_with_mem_limit.py`](https://github.com/prangelab/scOmnom/blob/main/scripts/run_with_mem_limit.py) to kill the run if either the combined process-tree RSS or macOS system pressure indicators grow too far.

```bash
python scripts/run_with_mem_limit.py \
  --rss-limit-gb 28 \
  --compressed-limit-gb 12 \
  --compressed-delta-limit-gb 4 \
  --swap-used-limit-gb 8 \
  --swap-used-delta-limit-gb 3 \
  --pressure-consecutive-breaches 2 \
  --poll-seconds 5 \
  -- scomnom load-and-filter -c data/cellbender -r data/raw -o results -m metadata.tsv
```

The RSS threshold is summed across the parent process plus worker subprocesses. On macOS, the wrapper also samples system compressed memory and swap usage, records a baseline at startup, and prints both current values and their deltas. Swap is treated as the primary kill signal; compressed-memory thresholds are only allowed to trigger when swap is also in use.
