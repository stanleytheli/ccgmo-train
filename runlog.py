"""Verbose, time-based run logging for long remote (tinker) jobs.

Completion-based progress bars go silent when a remote call hangs, which is exactly
when you most need output. Everything here is wall-clock driven:

  * `log()`        — every line carries an absolute timestamp AND elapsed run time,
                     and is flushed immediately so a tailed logfile stays current.
  * `Heartbeat`    — a daemon thread that keeps printing while a phase is in flight,
                     so a stalled remote call shows up as a growing "still running"
                     age rather than as silence.
  * `Phase`        — context manager: logs entry/exit + duration, runs a heartbeat
                     for the duration, and reports the traceback on failure.
  * `StallWatch`   — for iterative loops: flags any iteration slower than a multiple
                     of the running median, which is how crash-retry loops and
                     degrading throughput first show up.

All output goes to stdout and, if `attach_file()` is called, is mirrored to a log
file for later inspection.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

_START = time.monotonic()
_FILE: TextIO | None = None
_LOCK = threading.Lock()


def attach_file(path: str | Path) -> Path:
    """Mirror all subsequent log output to `path` (line-buffered, appended)."""
    global _FILE
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _FILE = p.open("a", encoding="utf-8", buffering=1)
    log(f"log file: {p.resolve()}")
    return p


def elapsed() -> float:
    return time.monotonic() - _START


def _hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def log(message: str, tag: str = "") -> None:
    """Timestamped, elapsed-stamped, immediately flushed."""
    prefix = f"[{time.strftime('%H:%M:%S')} +{_hms(elapsed())}]"
    line = f"{prefix} {f'[{tag}] ' if tag else ''}{message}"
    with _LOCK:
        print(line, flush=True)
        if _FILE is not None:
            _FILE.write(line + "\n")


class Heartbeat:
    """Prints '<label> still running (Ns)' every `every` seconds until stopped.

    This is the crash-loop / hang detector: if a remote call wedges, the heartbeat
    keeps ticking with a growing age instead of the job going quiet."""

    def __init__(self, label: str, every: float = 30.0) -> None:
        self.label = label
        self.every = every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0
        self.note: str = ""

    def set_note(self, note: str) -> None:
        """Extra context shown on each tick (e.g. '43/1000 done')."""
        self.note = note

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            age = time.monotonic() - self._started
            suffix = f" | {self.note}" if self.note else ""
            log(f"{self.label} still running ({_hms(age)}){suffix}", tag="hb")

    def __enter__(self) -> "Heartbeat":
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@contextmanager
def Phase(name: str, heartbeat_every: float = 30.0):
    """Log entry/exit + duration for a named phase, with a heartbeat throughout."""
    log(f"=== BEGIN {name} ===")
    t0 = time.monotonic()
    try:
        with Heartbeat(name, heartbeat_every) as hb:
            yield hb
    except BaseException as exc:
        log(f"=== FAILED {name} after {_hms(time.monotonic() - t0)}: "
            f"{type(exc).__name__}: {exc} ===")
        log(traceback.format_exc())
        raise
    log(f"=== END {name} ({_hms(time.monotonic() - t0)}) ===")


class StallWatch:
    """Detect iterations that are anomalously slow relative to the running median.

    A crash-retry loop, a throttled endpoint, or a degrading sampler all show up as
    iteration times drifting away from the median long before the job actually dies.
    `tick()` returns the duration of the iteration just completed."""

    def __init__(self, label: str, factor: float = 4.0, min_samples: int = 5,
                 absolute_secs: float | None = None) -> None:
        self.label = label
        self.factor = factor
        self.min_samples = min_samples
        self.absolute_secs = absolute_secs
        self.times: list[float] = []
        self._last = time.monotonic()
        self.n_stalls = 0

    def reset(self) -> None:
        """Restart the interval clock without recording it.

        Call after work that legitimately happens between ticks (an eval, a
        checkpoint save) — otherwise its duration is charged to the next
        iteration and reported as a stall, and false alarms teach you to ignore
        the real ones."""
        self._last = time.monotonic()

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self._last
        self._last = now
        median = statistics.median(self.times) if len(self.times) >= self.min_samples else None
        slow_relative = median is not None and dt > self.factor * max(median, 1e-6)
        slow_absolute = self.absolute_secs is not None and dt > self.absolute_secs
        if slow_relative or slow_absolute:
            self.n_stalls += 1
            reason = []
            if slow_relative:
                reason.append(f"{dt / median:.1f}x median {median:.1f}s")
            if slow_absolute:
                reason.append(f">{self.absolute_secs:.0f}s absolute")
            log(f"SLOW {self.label}: {dt:.1f}s ({', '.join(reason)}) "
                f"[stall #{self.n_stalls}]", tag="warn")
        self.times.append(dt)
        return dt

    def summary(self) -> dict[str, Any]:
        if not self.times:
            return {"n": 0, "stalls": self.n_stalls}
        return {
            "n": len(self.times),
            "stalls": self.n_stalls,
            "median_secs": statistics.median(self.times),
            "max_secs": max(self.times),
            "total_secs": sum(self.times),
        }


def die(message: str, code: int = 1) -> None:
    log(f"FATAL: {message}", tag="fatal")
    sys.exit(code)
