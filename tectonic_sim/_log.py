"""Logging + progress-bar plumbing for ``tectonic_sim``.

Mirrors ``worldgen._log``'s shape so the two modules can be configured
independently. Library default is silent (level WARNING). Callers (a CLI,
worldgen, a test harness) opt in via ``configure_logging(level)``.

``progress(iterable, ...)`` is a thin wrapper over ``tqdm.auto.tqdm`` that
self-disables when the ``tectonic_sim`` logger isn't at INFO or below, so
library users who haven't configured logging don't get stray bars on stderr.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from contextlib import contextmanager
from typing import TypeVar

from tqdm.auto import tqdm

_ROOT_LOGGER_NAME = "tectonic_sim"
T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``tectonic_sim`` namespace logger."""
    if name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent setup of the ``tectonic_sim`` logger."""
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(h, "_tectonic_sim_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(name)s %(levelname)s] %(message)s")
        )
        handler._tectonic_sim_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False


def _progress_enabled() -> bool:
    return get_logger(_ROOT_LOGGER_NAME).isEnabledFor(logging.INFO)


def progress(
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap an iterable in tqdm only when verbose logging is active."""
    if not _progress_enabled():
        return iterable
    return tqdm(
        iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True,
    )


@contextmanager
def timed_phase(name: str):
    """Context manager: log ``starting <name>`` on entry, ``<name> done in Ns``
    on exit. Used by ``simulate`` to surface per-phase timing."""
    log = get_logger("simulate")
    log.info("starting %s", name)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info("%s done in %.2fs", name, dt)
