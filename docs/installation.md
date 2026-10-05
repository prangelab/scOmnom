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
conda env create -f environment_linux.yml
conda activate scOmnom_env

# macOS
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment_macos.yml
conda activate scOmnom_env
```

This installs the declared Conda and pip dependencies, including the Scanpy and
PyTorch ecosystems. NicheNet additionally requires the R backend described below.

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
pip install .

# macOS
pip install -c constraints_macos.txt .
```

This registers the `scomnom` command-line interface and checks the package's own
dependencies. Keep constraints enabled for this macOS step too: the Conda
environment's pip step does not necessarily revisit dependencies of packages
that Conda already supplied.

## NicheNet R Backend

NicheNet is a separate CCC backend, not a requirement for LIANA or the core RNA
workflow. The macOS YAML supplies R and available helper packages. The remaining
CRAN packages (`fdrtool`, `mlrMBO`, `emoa`, and, if absent, `DiceKriging`) and the
`saeyslab/nichenetr` R package use scOmnom's existing cache library:

- macOS: `~/Library/Caches/scOmnom/r-libs/nichenet/`
- Linux: `~/.cache/scOmnom/r-libs/nichenet/`

The default `scomnom ccc nichenet` command checks this backend and stops with
installation instructions when it is missing. Its explicit
`--install-missing-r-deps` option enables CRAN installation and a source build
from `saeyslab/nichenetr`. This bootstrap currently follows upstream versions;
it is not a version-locked installation. Review it before use, and provision a
reviewed, revision-pinned R library separately for reproducible analyses. Keep
package installation separate from HPC analysis jobs.

A new Conda environment can still see an existing home-directory R cache.
Consequently, success with that cache does not demonstrate a fresh NicheNet
installation. See [NicheNet runtime requirements](markers-and-de/nichenet.md#runtime-requirements)
for the backend options.

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
