"""
Logging configuration for ConformalLab.

Provides a single, centrally-configured logging setup used across every
component of the research framework. `configure_logging` sets up console
and file handlers exactly once per run; `get_logger` retrieves named
child loggers that route into that same configuration, so an entire run
(across many modules) writes to one log file.

"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler

_ROOT_LOGGER_NAME = "conformallab"
_LOG_DIR = Path("logs")
_LOG_FORMAT_FILE = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False
_log_filepath: Optional[Path] = None


def _create_log_filepath() -> Path:
    """
    Build a unique, timestamped log file path for this run.

    Returns
    -------
    pathlib.Path
        Path of the form ``logs/YYYY-MM-DD_HH-MM-SS.log``. Including
        seconds avoids filename collisions if two runs start within
        the same minute.
    """
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _LOG_DIR / f"{timestamp}.log"


def configure_logging(
    log_file: Optional[Path] = None, level: int = logging.INFO
) -> Path:
    """
    Configure logging for the entire application. Call this once, at the
    start of a run.

    If called more than once, subsequent calls are ignored and the
    original configuration is kept — this makes it safe to call
    defensively from ``get_logger`` without risking duplicate handlers
    or multiple log files per run.

    Parameters
    ----------
    log_file
        Path to write logs to. If ``None``, a timestamped path under
        ``logs/`` is generated automatically.
    level
        Minimum severity level to log. Defaults to ``logging.INFO``.

    Returns
    -------
    pathlib.Path
        The log file path actually in use for this run.
    """
    global _configured, _log_filepath

    if _configured:
        return _log_filepath  # type: ignore[return-value]

    if log_file is None:
        log_file = _create_log_filepath()

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    root_logger.setLevel(level)
    root_logger.propagate = False

    console_handler = RichHandler(
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT_FILE))
        root_logger.addHandler(file_handler)
    except OSError as error:
        root_logger.warning(
            "Could not create log file (%s). Logging to console only.", error
        )
        log_file = None  # type: ignore[assignment]

    _configured = True
    _log_filepath = log_file
    return log_file  # type: ignore[return-value]


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger that routes into the application's shared
    logging configuration.

    Automatically calls `configure_logging` with defaults if it has not
    been called yet, so individual modules/tests can safely call this
    without needing to know whether `run.py` has already configured
    logging.

    Parameters
    ----------
    name
        Name of the logger, conventionally ``__name__`` of the calling
        module.

    Returns
    -------
    logging.Logger
        A logger under the shared ``"conformallab"`` namespace, sharing
        console and file handlers with every other logger in this run.

    """
    if not _configured:
        configure_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")