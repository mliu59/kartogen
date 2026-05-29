"""Logging + progress-bar plumbing for the worldgen pipeline.

Library default is silent (level WARNING). The CLI calls
``configure_logging(logging.INFO)`` on startup so end-users see per-layer
timing and progress bars. Tests don't call it, so they stay quiet.

``progress(iterable, ...)`` is a thin wrapper over ``tqdm.auto.tqdm`` that
self-disables when the worldgen logger isn't at INFO or below, so library
users who haven't configured logging don't get stray bars on stderr.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TypeVar

from tqdm.auto import tqdm

_ROOT_LOGGER_NAME = "worldgen"
T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``worldgen`` namespace logger."""
    if name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent setup of the ``worldgen`` logger.

    Adds a single StreamHandler (stderr) the first time it's called; on
    subsequent calls just updates the level. CLI uses this on startup;
    library users can call it too if they want per-layer logs in their
    own apps.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    # Avoid double-handlers if configure_logging is called twice.
    if not any(getattr(h, "_worldgen_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s %(levelname)s] %(message)s")
        )
        handler._worldgen_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    # Don't propagate up to root — keeps test output clean.
    logger.propagate = False


def _progress_enabled() -> bool:
    """tqdm bars only show when the worldgen logger is at INFO or below."""
    return get_logger(_ROOT_LOGGER_NAME).isEnabledFor(logging.INFO)


def progress(
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap an iterable in tqdm, but only when verbose logging is active.

    ``leave=False`` (the default) clears each bar when it finishes so the
    pipeline output isn't a stack of completed bars.
    """
    if not _progress_enabled():
        return iterable
    return tqdm(
        iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True,
    )


@contextmanager
def timed_layer(name: str):
    """Context manager: log ``starting <name>`` on entry, ``<name> done in Ns``
    on exit. Used by ``pipeline.generate`` to surface per-layer timing."""
    log = get_logger("pipeline")
    log.info("starting %s", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info("%s done in %.2fs", name, dt)
