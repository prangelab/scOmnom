import pytest
import anndata as ad
import numpy as np
import pandas as pd
from scomnom.load_and_qc import run_load_and_qc
from scomnom.config import LoadAndQCConfig

def make_fake_anndata(n_cells=50, n_genes=100):
    X = np.random.poisson(1.0, (n_cells, n_genes))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)

def test_fake_qc(tmp_path):
    cfg = LoadAndQCConfig(input_dir=tmp_path, output_dir=tmp_path)
    adata = make_fake_anndata()
    assert adata.n_obs == 50