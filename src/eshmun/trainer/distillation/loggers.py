"""Logging utilities for knowledge distillation training."""

import logging
from collections import defaultdict
from typing import Any


logger = logging.getLogger(__name__)


class DistillationLogger:
    """
    Accumulates and emits distillation training metrics.

    Tracks a running window of per-step metric values and emits a summary at
    the configured ``log_every`` cadence.  Supports arbitrary metric names so
    it stays decoupled from specific loss combinations.

    Args:
        log_every: Emit a log line every this many optimizer steps.
        level: Python logging level used for the emitted messages.
    """

    def __init__(self, log_every: int = 10, level: int = logging.INFO) -> None:
        self.log_every = log_every
        self.level = level
        self._step = 0
        self._accum: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)

    def log(self, step: int, epoch: int, metrics: dict[str, Any]) -> None:
        """
        Accumulate metrics for one step and emit a summary when due.

        Args:
            step: Current global optimizer step.
            epoch: Current training epoch (1-indexed).
            metrics: Dict of metric name → scalar value for this step.
        """
        self._step = step
        for key, value in metrics.items():
            if isinstance(value, float | int):
                self._accum[key] += float(value)
                self._counts[key] += 1

        if step % self.log_every == 0:
            self._emit(epoch)

    def _emit(self, epoch: int) -> None:
        parts = [f"epoch={epoch} step={self._step}"]
        for key in sorted(self._accum):
            avg = self._accum[key] / max(self._counts[key], 1)
            parts.append(f"{key}={avg:.4f}")

        logger.log(self.level, "  ".join(parts))
        self._accum.clear()
        self._counts.clear()

    def flush(self, epoch: int) -> None:
        """Emit any remaining accumulated metrics (call at end of epoch)."""
        if any(c > 0 for c in self._counts.values()):
            self._emit(epoch)
