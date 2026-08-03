"""Tests for the LLM diagnoser and the controller's diagnostics gating."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List

from skydiscover.evaluation.diagnoser import SolutionDiagnoser
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import DiscoveryController


@dataclass
class FakeResponse:
    text: str


class FakeLLMPool:
    """Records the prompts it is asked to generate from."""

    def __init__(self, text: str = "The new sort is O(n^2). Restore the heap."):
        self.text = text
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, system_message, messages, **kwargs):
        self.calls.append({"system": system_message, "messages": messages})
        return FakeResponse(self.text)


class ExplodingLLMPool:
    async def generate(self, *a, **kw):
        raise RuntimeError("backend down")


@dataclass
class FakeEvaluatorConfig:
    diagnose_regressions: bool = True
    llm_diagnosis: bool = False
    llm_diagnosis_min_drop: float = 0.0


@dataclass
class FakeConfig:
    evaluator: FakeEvaluatorConfig = field(default_factory=FakeEvaluatorConfig)


def _parent(score=0.99):
    return Program(
        id="p", solution="def solve():\n    return heap_sort()", metrics={"combined_score": score}
    )


def _controller(evaluator_config, pool):
    """Build a controller shell without touching LLMs, Docker, or the database."""
    controller = object.__new__(DiscoveryController)
    controller.config = FakeConfig(evaluator=evaluator_config)
    controller.evaluator_llms = pool
    controller.database = None
    return controller


def _attach(controller, child_metrics, parent, **kw):
    return asyncio.run(
        controller._attach_diagnostics(
            child_solution="def solve():\n    return bubble_sort()",
            child_metrics=child_metrics,
            parent=parent,
            changes_summary="Replaced heap sort with bubble sort",
            artifacts={"feedback": "3 cases timed out"},
            **kw,
        )
    )


# ── SolutionDiagnoser ────────────────────────────────────────────────


def test_diagnoser_returns_text_and_includes_context():
    pool = FakeLLMPool()
    diagnoser = SolutionDiagnoser(pool)
    result = asyncio.run(
        diagnoser.diagnose(
            child_solution="bubble_sort()",
            parent_solution="heap_sort()",
            child_metrics={"combined_score": 0.5, "runtime_s": 4.0},
            parent_metrics={"combined_score": 0.9, "runtime_s": 1.0},
            changes_summary="swapped the sort",
            artifacts={"feedback": "3 cases timed out"},
        )
    )
    assert result == pool.text

    sent = pool.calls[0]["messages"][0]["content"]
    assert "bubble_sort()" in sent and "heap_sort()" in sent
    assert "swapped the sort" in sent
    assert "3 cases timed out" in sent
    assert "runtime_s" in sent  # metric movement is included


def test_diagnoser_swallows_backend_failure():
    diagnoser = SolutionDiagnoser(ExplodingLLMPool())
    result = asyncio.run(
        diagnoser.diagnose(
            child_solution="a",
            parent_solution="b",
            child_metrics={"combined_score": 0.1},
            parent_metrics={"combined_score": 0.9},
        )
    )
    assert result is None


def test_diagnoser_treats_empty_response_as_no_diagnosis():
    diagnoser = SolutionDiagnoser(FakeLLMPool(text="   "))
    result = asyncio.run(
        diagnoser.diagnose(
            child_solution="a",
            parent_solution="b",
            child_metrics={"combined_score": 0.1},
            parent_metrics={"combined_score": 0.9},
        )
    )
    assert result is None


def test_prompt_carries_a_diff_not_two_full_sources():
    # Sending both sources truncated buries the change on a long program; the
    # diff keeps the modified region in view.
    pool = FakeLLMPool()
    parent = "\n".join(f"line {i}" for i in range(300)) + "\nreturn heap_sort()"
    child = parent.replace("heap_sort()", "bubble_sort()")
    asyncio.run(
        SolutionDiagnoser(pool).diagnose(
            child_solution=child,
            parent_solution=parent,
            child_metrics={"combined_score": 0.1},
            parent_metrics={"combined_score": 0.9},
        )
    )
    sent = pool.calls[0]["messages"][0]["content"]
    assert "-return heap_sort()" in sent
    assert "+return bubble_sort()" in sent
    assert "line 150" not in sent  # untouched bulk is not shipped


def test_identical_solutions_are_reported_as_such():
    pool = FakeLLMPool()
    asyncio.run(
        SolutionDiagnoser(pool).diagnose(
            child_solution="same()",
            parent_solution="same()",
            child_metrics={"combined_score": 0.1},
            parent_metrics={"combined_score": 0.9},
        )
    )
    assert "textually identical" in pool.calls[0]["messages"][0]["content"]


def test_long_solutions_are_truncated():
    pool = FakeLLMPool()
    asyncio.run(
        SolutionDiagnoser(pool).diagnose(
            child_solution="x" * 10_000,
            parent_solution="y" * 10_000,
            child_metrics={"combined_score": 0.1},
            parent_metrics={"combined_score": 0.9},
        )
    )
    assert "(truncated)" in pool.calls[0]["messages"][0]["content"]


# ── Controller gating ────────────────────────────────────────────────


def test_disabled_entirely_leaves_artifacts_untouched():
    pool = FakeLLMPool()
    controller = _controller(FakeEvaluatorConfig(diagnose_regressions=False), pool)
    out = _attach(controller, {"combined_score": 0.5}, _parent())
    assert out == {"feedback": "3 cases timed out"}
    assert pool.calls == []


def test_regression_adds_report_but_not_llm_by_default():
    pool = FakeLLMPool()
    controller = _controller(FakeEvaluatorConfig(), pool)
    out = _attach(controller, {"combined_score": 0.5}, _parent())
    assert "regression_report" in out
    assert "diagnosis" not in out
    assert pool.calls == []  # LLM is opt-in


def test_improvement_produces_no_report():
    pool = FakeLLMPool()
    controller = _controller(FakeEvaluatorConfig(llm_diagnosis=True), pool)
    out = _attach(controller, {"combined_score": 1.0}, _parent(0.5))
    assert "regression_report" not in out
    assert "diagnosis" not in out
    assert pool.calls == []


def test_llm_diagnosis_runs_when_enabled():
    pool = FakeLLMPool()
    controller = _controller(FakeEvaluatorConfig(llm_diagnosis=True), pool)
    out = _attach(controller, {"combined_score": 0.5}, _parent())
    assert out["diagnosis"] == pool.text
    assert "regression_report" in out
    assert len(pool.calls) == 1


def test_min_drop_suppresses_trivial_regressions():
    pool = FakeLLMPool()
    controller = _controller(
        FakeEvaluatorConfig(llm_diagnosis=True, llm_diagnosis_min_drop=0.1), pool
    )
    # 0.99 -> 0.98 is a 0.01 drop, below the 0.1 threshold.
    out = _attach(controller, {"combined_score": 0.98}, _parent(0.99))
    assert "regression_report" in out  # deterministic half still runs
    assert "diagnosis" not in out
    assert pool.calls == []


def test_plateau_gets_a_report_but_costs_no_generation():
    # Equal scores: "why did this regress?" would just get "it did not".
    pool = FakeLLMPool()
    controller = _controller(FakeEvaluatorConfig(llm_diagnosis=True), pool)
    out = _attach(controller, {"combined_score": 0.99}, _parent(0.99))
    assert "matched its parent" in out["regression_report"]
    assert "diagnosis" not in out
    assert pool.calls == []


def test_llm_failure_does_not_lose_the_report():
    controller = _controller(FakeEvaluatorConfig(llm_diagnosis=True), ExplodingLLMPool())
    out = _attach(controller, {"combined_score": 0.5}, _parent())
    assert "regression_report" in out
    assert "diagnosis" not in out


def test_evaluator_artifacts_are_preserved_and_not_mutated():
    original = {"feedback": "3 cases timed out"}
    controller = _controller(FakeEvaluatorConfig(), FakeLLMPool())
    out = asyncio.run(
        controller._attach_diagnostics(
            child_solution="x",
            child_metrics={"combined_score": 0.5},
            parent=_parent(),
            changes_summary="c",
            artifacts=original,
        )
    )
    assert out["feedback"] == "3 cases timed out"
    assert original == {"feedback": "3 cases timed out"}  # caller's dict untouched
