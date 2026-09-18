"""Phase, wall-time and memory instrumentation for batch runs.

A run on HTCondor is a black box: `.out` and `.err` only arrive when the job
ends, the sandbox is deleted with it, and the periodic `MemoryUsage` samples in
the Condor log are too coarse to say *which* phase allocated what. Peak RSS is
the binding constraint on which slots a job can match, so knowing that a run
peaks at 30 GB is much less useful than knowing whether it peaks during fit or
during evaluation.

These helpers emit one line per phase boundary carrying elapsed wall time,
current RSS and the high-water mark, so a finished job's stdout answers "where
did the memory go" without re-running anything.
"""

from __future__ import annotations

import gc
import os
import resource
import time
from contextlib import contextmanager
from typing import Iterator

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)

_BYTES_PER_MIB = 1024.0 * 1024.0


def peak_rss_mib() -> float:
    """Return the process high-water RSS in MiB.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. This is a
    high-water mark and never decreases, which is exactly what is wanted for
    sizing ``request_memory``: it survives the garbage collection that makes
    instantaneous readings misleading.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1024.0 if os.uname().sysname == "Linux" else raw / _BYTES_PER_MIB


def current_rss_mib() -> float:
    """Return the process's current RSS in MiB, or NaN if unavailable.

    Read from /proc rather than a third-party dependency so this works inside
    the batch container without adding a package. Falls back quietly: the
    instrumentation must never be the reason a run fails.
    """
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError):
        return float("nan")

    return pages * os.sysconf("SC_PAGE_SIZE") / _BYTES_PER_MIB


def log_memory(label: str) -> None:
    """Emit a single memory reading under ``label``."""
    log.info(
        "[mem] %-38s current=%8.0f MiB  peak=%8.0f MiB",
        label,
        current_rss_mib(),
        peak_rss_mib(),
    )


@contextmanager
def log_phase(name: str, *, collect: bool = False) -> Iterator[None]:
    """Log entry to and exit from a named phase with time and memory.

    :param name: Human-readable phase name, e.g. "fit" or "run validation".
    :param collect: Run ``gc.collect()`` before the closing reading. Use this
        after a phase that is expected to release large tensors, so the exit
        line reflects what was actually freed rather than what is merely
        unreachable. It costs a full collection, so it is off by default.
    """
    log.info("[phase] ---- BEGIN %s ----", name)
    log_memory(f"{name}: begin")
    started = time.monotonic()
    failed = False

    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        elapsed = time.monotonic() - started

        if collect:
            gc.collect()

        log_memory(f"{name}: end")
        log.info(
            "[phase] ---- END %s (%s) in %.1f s (%.1f min) ----",
            name,
            "FAILED" if failed else "ok",
            elapsed,
            elapsed / 60.0,
        )
