"""JSONL log of every LLM call: which model, what effort, and what it cost.

With temperature emulation the model and the reasoning effort are *sampled* per
call (see :mod:`skydiscover.llm.temperature`), so neither the config nor the
ordinary log says what a given iteration actually ran on. This records the
resolved choice, plus retries and quota pauses, one JSON object per line at
``<output_dir>/logs/llm_calls.jsonl``.

Call context (iteration, phase) travels through a :class:`~contextvars.ContextVar`
rather than through call signatures: the sampling happens deep inside
``LLMPool.generate`` and threading an iteration number down to it would touch
every controller and backend. ContextVars follow ``await``, so the value set at
the top of an iteration is still visible there.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("skydiscover.llm")

LOG_FILENAME = "llm_calls.jsonl"

# Set per iteration / paradigm generation; merged into every record.
_call_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "skydiscover_llm_call_context", default={}
)


def set_call_context(**fields: Any) -> None:
    """Attach ``fields`` (e.g. ``iteration=7, phase="mutate"``) to later records."""
    _call_context.set(dict(fields))


def get_call_context() -> Dict[str, Any]:
    return dict(_call_context.get())


class LLMCallLog:
    """Append-only JSONL sink. Disabled until :meth:`configure` is called."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path: Optional[str] = None

    def configure(self, log_dir: Optional[str]) -> None:
        if not log_dir:
            self._path = None
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            self._path = os.path.join(log_dir, LOG_FILENAME)
        except OSError as exc:
            # Never let logging break a run.
            logger.warning("Could not open the LLM call log in %s: %s", log_dir, exc)
            self._path = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    def record(self, event: str, **fields: Any) -> None:
        if self._path is None:
            return
        row = {"ts": round(time.time(), 3), "event": event}
        row.update(get_call_context())
        row.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(row, default=str)
        try:
            with self._lock:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            logger.warning("Could not write to the LLM call log: %s", exc)
            self._path = None


GLOBAL_CALL_LOG = LLMCallLog()


def describe_backend(model) -> Dict[str, Any]:
    """Identify the backend instance a pool sampled."""
    return {
        "backend": type(model).__name__,
        "model": getattr(model, "model", None),
    }
