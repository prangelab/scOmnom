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
- macOS (Apple Silicon): `environment_macos.yml`

Create and activate the environment:

```bash
# Linux / HPC
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment_linux.yml
conda activate scOmnom_env

# macOS
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment_macos.yml
conda activate scOmnom_env
```

This installs the declared Conda and pip dependencies, including the Scanpy and
PyTorch ecosystems. NicheNet additionally requires the R backend described below.

The Linux manifest keeps the HPC channel order and exact PyTorch 2.5.1 / CUDA
12.4 builds. Keep `constraints_linux.txt` beside `environment_linux.yml`: the pip
stage uses it to preserve the reviewed Python dependency versions, including
the indirect dependencies of Lightning, Numba, scVI and JAX. With micromamba, use
`micromamba env create --channel-priority flexible -f environment_linux.yml`
and `micromamba activate scOmnom_env`. Constraints are platform-specific and
are not a complete lock of Conda, R and all transitive dependencies.

On Apple Silicon, the manifest selects PyTorch/libtorch builds from conda-forge,
torchvision/torchaudio from the PyTorch channel, and SciPy from defaults. Strict
channel priority can exclude these builds even when their channels are explicitly
specified. The command-local setting above allows this combination without
changing your global Conda configuration. Keep the Linux/HPC installation
separate; do not transfer these macOS build selections to Linux.

Keep `constraints_macos.txt` beside `environment_macos.yml`. Conda's pip step
reads this file to retain the tested Python dependency versions, including the
indirect scVI, JAX and NumPyro dependencies. A constraint does not install an
unused package. These constraints are specific to the Apple Silicon environment;
do not apply them to Linux/HPC. Update the YAML and constraints together when
validating a dependency upgrade.

The `cpu_generic` PyTorch build name does not exclude Apple MPS acceleration.
After installation, check availability in a normal terminal before training:

```bash
python -c 'import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())'
```

Both MPS values should be `True` on a supported Apple Silicon setup. A sandboxed
process may not see the Metal device.

## 2) Install `scOmnom`

For normal use, install the package into the active environment:

```bash
# Linux / HPC
pip install -c constraints_linux.txt .

# macOS
pip install -c constraints_macos.txt .
```

This registers the `scomnom` command-line interface and checks the package's own
dependencies. Keep the platform's constraints enabled for this step too: the Conda
environment's pip step does not necessarily revisit dependencies of packages
that Conda already supplied.

## NicheNet R Backend

NicheNet is a separate CCC backend, not a requirement for LIANA or the core RNA
workflow. The macOS YAML supplies R and available helper packages. The remaining
CRAN packages (`fdrtool`, `mlrMBO`, `emoa`, and, if absent, `DiceKriging`) and the
`saeyslab/nichenetr` R package use scOmnom's existing cache library:

- macOS: `~/Library/Caches/scOmnom/r-libs/nichenet/`
- Linux: `~/.cache/scOmnom/r-libs/nichenet/`

`scomnom ccc nichenet` checks this backend and stops with installation
instructions when it is missing or fails to load. Package installation during
analysis is disabled. The legacy `--install-missing-r-deps` flag remains accepted
but does not install anything. Existing libraries remain usable.

Provision the R packages separately, before submitting analysis jobs. The
release acceptance used `saeyslab/nichenetr` version 2.2.1.1 at commit
`af7e2564c316ad2abf187ef612260f442d92fb33` and CRAN DiceKriging 1.6.1.
Install their required dependencies into the same environment or cache library
first. On macOS these include `fdrtool`, `mlrMBO`, and `emoa`. For reviewed local
source archives, use the active environment's R and an explicit library:

```bash
# Linux cache; use the macOS path above on macOS.
export R_LIBS_USER="$HOME/.cache/scOmnom/r-libs/nichenet"
mkdir -p "$R_LIBS_USER"
R CMD INSTALL -l "$R_LIBS_USER" DiceKriging_1.6.1.tar.gz
R CMD INSTALL -l "$R_LIBS_USER" nichenetr-reviewed.tar.gz
Rscript -e 'stopifnot(requireNamespace("nichenetr", quietly=TRUE))'
```

`nichenetr-reviewed.tar.gz` must contain the pinned revision above, not the
current default branch. Verify source provenance and checksums before running
an installation. These commands do not resolve missing dependencies.

A new Conda environment can still see an existing home-directory R cache.
Consequently, success with that cache does not demonstrate a fresh NicheNet
installation. See [NicheNet runtime requirements](markers-and-de/nichenet.md#runtime-requirements)
for the backend options.

## MEBOCOST Backend

The Linux manifest pins `kaifuchenlab/MEBOCOST` to commit
`3e0be6f638b71c0f5bd08cef8b1448ae5230560e` (MEBOCOST 1.2.2). Provision this
package and its dependencies as an environment-management step. Neither
`scomnom ccc mebocost` nor `scomnom ccc mebocost-paired` installs code or
downloads a resource repository during analysis. The legacy
`--install-missing-python-deps` flag remains accepted but cannot install anything.

Stage the `data/` directory from that same reviewed revision under:

- macOS: `~/Library/Caches/scOmnom/mebocost/MEBOCOST/data/`
- Linux: `~/.cache/scOmnom/mebocost/MEBOCOST/data/`

Preserve its directory structure and verify the staged files against your source
manifest. scOmnom checks for the required human/mouse enzyme and sensor tables,
HMDB annotation, scFEA information, and Compass reaction/metabolite annotations.
It can generate `mebocost.conf` from an already complete local bundle. Missing
files stop the run and are listed in the error. On HPC, the cache may be a link
to persistent project storage accessible on compute nodes.

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
