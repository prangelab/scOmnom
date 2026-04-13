__version__ = "0.4.1"

from types import SimpleNamespace

from .io_utils import load_dataset, save_dataset  # noqa: F401
from . import plotting  # noqa: F401
from . import adata_public as adata_ops  # noqa: F401

markers_and_de = SimpleNamespace(
    enrichment_cluster=adata_ops.enrichment_cluster,
    enrichment_de_from_tables=adata_ops.enrichment_de_from_tables,
    module_score=adata_ops.module_score,
)
