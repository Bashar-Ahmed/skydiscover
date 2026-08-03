"""LLM-as-a-diagnoser: explains why a solution under-performed its parent.

Sibling of :mod:`skydiscover.evaluation.llm_judge`.  The judge answers "how
good is this?" and emits metrics; the diagnoser answers "why did this get
worse?" and emits prose, which is stored as a ``diagnosis`` artifact and
resurfaces in the prompt the next time that program is mutated.

Prompts are built inline rather than through the ``TemplateManager`` on
purpose: a template would have to be added to every builder's template
directory (default, adaevolve, gepa_native, evox) to be reachable, and the
diagnosis text is strategy-independent.

This never raises into the discovery loop.  A failed diagnosis returns
``None`` and the iteration proceeds -- an explanation is a nice-to-have, and
must never cost a successfully evaluated program.
"""

import difflib
import logging
from typing import Any, Dict, Optional

from skydiscover.llm.llm_pool import LLMPool
from skydiscover.utils.diagnostics import describe_metric_deltas

logger = logging.getLogger(__name__)

#: The prompt carries a unified diff rather than both full solutions.  Sending
#: two truncated sources buries the change: on a long program the modified
#: region falls outside the window and the model can only report that it cannot
#: see what changed.  A diff is change-first, so truncation costs context
#: rather than the subject itself.
MAX_DIFF_CHARS = 4000
MAX_ARTIFACT_CHARS = 1000
#: Lines of unchanged code kept around each hunk.
DIFF_CONTEXT_LINES = 4

SYSTEM_MESSAGE = """You are diagnosing why a candidate program scored worse than the one it was derived from.

You are given the parent program, the child program, what changed, and both sets of metrics.

Rules:
- State the most likely cause of the regression, concretely. Name the construct, algorithm, or parameter responsible.
- Ground every claim in the code or the metrics you were given. If the cause is not determinable from them, say so plainly rather than guessing.
- Then give one specific, actionable suggestion for the next attempt.
- Under 80 words total. No headers, no preamble, no restating the metrics."""


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _unified_diff(parent_solution: str, child_solution: str) -> str:
    """Parent -> child as a unified diff. Empty string when they are identical."""
    return "\n".join(
        difflib.unified_diff(
            (parent_solution or "").splitlines(),
            (child_solution or "").splitlines(),
            fromfile="parent",
            tofile="child",
            lineterm="",
            n=DIFF_CONTEXT_LINES,
        )
    )


class SolutionDiagnoser:
    """Asks an LLM why a child regressed against its parent."""

    def __init__(self, llm_pool: LLMPool, database: Optional[Any] = None):
        """
        Args:
            llm_pool: Pool to generate with.  Callers should pass the
                controller's ``evaluator_llms`` so diagnosis can run on a
                cheaper model than generation.  Note that ``evaluator_models``
                defaults to a shallow copy of ``models``, so without explicit
                config this is the *same* model the run generates with.
            database: Optional, for prompt logging only.
        """
        self.llm_pool = llm_pool
        self.database = database

    async def diagnose(
        self,
        *,
        child_solution: str,
        parent_solution: str,
        child_metrics: Dict[str, Any],
        parent_metrics: Dict[str, Any],
        changes_summary: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        program_id: str = "",
    ) -> Optional[str]:
        """Return a short diagnosis, or ``None`` if one could not be produced."""
        try:
            user_msg = self._build_user_message(
                child_solution=child_solution,
                parent_solution=parent_solution,
                child_metrics=child_metrics,
                parent_metrics=parent_metrics,
                changes_summary=changes_summary,
                artifacts=artifacts,
            )

            response = await self.llm_pool.generate(
                SYSTEM_MESSAGE, [{"role": "user", "content": user_msg}]
            )
            diagnosis = (response.text or "").strip()
            if not diagnosis:
                return None

            if self.database and program_id:
                try:
                    self.database.log_prompt(
                        program_id=program_id,
                        template_key="diagnosis",
                        prompt={"system": SYSTEM_MESSAGE, "user": user_msg},
                        responses=[diagnosis],
                    )
                except Exception:
                    logger.debug("Failed to log diagnosis prompt", exc_info=True)

            return diagnosis
        except Exception as e:
            logger.warning(f"Solution diagnosis failed: {e}")
            return None

    def _build_user_message(
        self,
        *,
        child_solution: str,
        parent_solution: str,
        child_metrics: Dict[str, Any],
        parent_metrics: Dict[str, Any],
        changes_summary: Optional[str],
        artifacts: Optional[Dict[str, Any]],
    ) -> str:
        diff = _unified_diff(parent_solution, child_solution)
        if diff:
            sections = [
                "# Change from parent to child (unified diff)",
                f"```diff\n{_truncate(diff, MAX_DIFF_CHARS)}\n```",
            ]
        else:
            sections = [
                "# Change from parent to child",
                "The two programs are textually identical.",
            ]

        if changes_summary:
            sections += ["# Stated change", _truncate(changes_summary, MAX_ARTIFACT_CHARS)]

        deltas = describe_metric_deltas(child_metrics, parent_metrics, limit=8)
        if deltas:
            sections += ["# Metric movement (parent -> child)", deltas]

        evaluator_feedback = (artifacts or {}).get("feedback")
        if evaluator_feedback:
            sections += [
                "# Evaluator feedback",
                _truncate(str(evaluator_feedback), MAX_ARTIFACT_CHARS),
            ]

        sections.append("Why did the child score worse, and what should the next attempt do?")
        return "\n\n".join(sections)
