import logging
from pathlib import Path
from typing import Optional


def init_logging(logfile: Optional[Path] = None, level: int = logging.INFO) -> None:
    """
    Initialize logging with a stream handler + optional file handler.
    All existing handlers are removed to avoid duplicates.
    """

    # Remove any pre-configured handlers (important for Typer)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    handlers = [logging.StreamHandler()]

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="w"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
