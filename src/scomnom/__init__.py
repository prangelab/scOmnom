__version__ = "0.2.0"

from .io_utils import load_dataset, save_dataset  # noqa: F401
from . import plotting  # noqa: F401
from . import adata_public as adata_ops  # noqa: F401


def api_canary() -> None:
    print("yep updated!")
