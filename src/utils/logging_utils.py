"""Project-wide logging helpers.

Single entry point (:func:`get_logger`) so every script logs in the same format
to both the console and a rotating file under the configured ``logs/`` dir.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
BACKUP_COUNT = 3


def get_logger(
    name: str = "nids",
    *,
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    logs_dir: Optional[Union[str, Path]] = None,
    to_console: bool = True,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, e.g. ``"inspect_dataset"``.
    level:
        Level name or numeric level.
    log_file:
        File name (resolved inside ``logs_dir``) or an absolute path. When
        ``None`` and ``logs_dir`` is given, ``<name>.log`` is used.
    logs_dir:
        Directory for log files, normally ``cfg.paths.logs_dir``. When ``None``
        no file handler is attached.
    to_console:
        Also stream records to stdout.

    Calling this twice with the same ``name`` does not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    existing = {type(h) for h in logger.handlers}

    if to_console and logging.StreamHandler not in existing:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

    if (log_file or logs_dir) and RotatingFileHandler not in existing:
        path = _resolve_log_path(name, log_file, logs_dir)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def _resolve_log_path(
    name: str,
    log_file: Optional[Union[str, Path]],
    logs_dir: Optional[Union[str, Path]],
) -> Optional[Path]:
    if log_file is not None:
        candidate = Path(log_file)
        if candidate.is_absolute():
            return candidate
        return Path(logs_dir) / candidate if logs_dir else candidate
    if logs_dir is not None:
        return Path(logs_dir) / f"{name}.log"
    return None


def log_section(logger: logging.Logger, title: str, width: int = 72) -> None:
    """Log a visually separated section header."""
    logger.info("-" * width)
    logger.info(title)
    logger.info("-" * width)
