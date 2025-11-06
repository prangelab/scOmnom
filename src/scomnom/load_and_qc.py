from .config import LoadAndQCConfig
from . import io_utils, plot_utils


def merge_samples(raw_list):


    pass


def add_metadata(adata, metadata_file):


    pass


def compute_qc_metrics(adata, config: LoadAndQCConfig):


    pass


def filter_cells(adata, config: LoadAndQCConfig):


    pass


def detect_doublets(adata, config: LoadAndQCConfig):


    pass


def normalize_and_hvg(adata, config: LoadAndQCConfig):


    pass


def run_load_and_qc(config: LoadAndQCConfig):


    pass


# full orchestrator for CLI
raw = io_utils.load_raw_data(config.input_dir)
if config.cellbender:
    cb = io_utils.load_cellbender_data(config.input_dir)
else:
cb = None
adata = merge_samples([raw, cb]) if cb is not None else raw
adata = add_metadata(adata, config.metadata_file)
adata = compute_qc_metrics(adata, config)
adata = filter_cells(adata, config)
adata = detect_doublets(adata, config)
adata = normalize_and_hvg(adata, config)
if config.figures:
    plot_utils.plot_qc_metrics(adata, config.output_dir)
io_utils.save_adata(adata, config.output_dir / "adata_qc.h5ad")
return adata
