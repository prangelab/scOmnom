# tests/test_logging_utils.py

import logging
import pytest
from pathlib import Path

from scomnom.logging_utils import init_logging


def _get_handler_types():
    """Helper: return a list of handler class types currently installed."""
    return tuple(type(h) for h in logging.root.handlers)


@pytest.fixture
def reset_logging():
    """Ensure clean logging handlers before/after each test."""
    # save original handlers
    orig = logging.root.handlers[:]
    # clear
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    yield
    # restore original
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    for h in orig:
        logging.root.addHandler(h)


def test_init_logging_stream_only(tmp_path, reset_logging):
    init_logging(logfile=None, level=logging.DEBUG)

    types = _get_handler_types()
    # Only a StreamHandler should exist
    assert types == (logging.StreamHandler,)

    # Logging format should be active
    logger = logging.getLogger("test")
    logger.debug("hello")


def test_init_logging_stream_and_file(tmp_path, reset_logging):
    log_path = tmp_path / "log" / "test.log"
    init_logging(logfile=log_path, level=logging.INFO)

    types = _get_handler_types()
    # Order: StreamHandler then FileHandler
    assert types == (logging.StreamHandler, logging.FileHandler)

    # Verify file exists after logging something
    logger = logging.getLogger("test")
    logger.info("This is a test message.")

    assert log_path.exists()
    txt = log_path.read_text()
    assert "This is a test message." in txt


def test_init_logging_overwrites_previous_handlers(tmp_path, reset_logging):
    # Install a dummy handler first
    dummy = logging.StreamHandler()
    logging.root.addHandler(dummy)

    # Now init logging (should remove all prior handlers)
    init_logging(None)

    types = _get_handler_types()
    assert types == (logging.StreamHandler,)  # dummy removed


def test_init_logging_respects_level(tmp_path, reset_logging):
    log_path = tmp_path / "test.log"
    init_logging(logfile=log_path, level=logging.WARNING)

    logger = logging.getLogger("x")

    logger.info("info msg")     # Should not be written
    logger.warning("warn msg")  # Should be written

    txt = log_path.read_text()
    assert "warn msg" in txt
    assert "info msg" not in txt


def test_init_logging_creates_parent_dir(tmp_path, reset_logging):
    # Directory doesn't exist
    log_path = tmp_path / "nested" / "deeper" / "log.txt"

    init_logging(logfile=log_path)

    assert log_path.exists()
