"""`mode` reaches Python evaluators that ask for it, and nothing else changes.

Without this, the post-loop "test evaluation" was a re-run of the training
evaluation for every Python benchmark, so a run could not tell whether it had
overfitted its own evaluator.
"""

import asyncio
import os
import textwrap

import pytest

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluator import Evaluator


def write_evaluator(tmp_path, name, body):
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body))
    return str(path)


def make_evaluator(path, cascade=False):
    return Evaluator(EvaluatorConfig(evaluation_file=path, cascade_evaluation=cascade, timeout=30))


@pytest.mark.asyncio
async def test_legacy_single_argument_evaluator_is_untouched(tmp_path):
    """The ~49 evaluators already in the tree take one positional arg."""
    path = write_evaluator(
        tmp_path,
        "legacy",
        """
        def evaluate(program_path):
            return {"combined_score": 1.0}
        """,
    )
    evaluator = make_evaluator(path)

    train = await evaluator.evaluate_program("x")
    test = await evaluator.evaluate_program("x", mode="test")

    assert train.metrics["combined_score"] == 1.0
    assert test.metrics["combined_score"] == 1.0


@pytest.mark.asyncio
async def test_mode_aware_evaluator_receives_the_mode(tmp_path):
    path = write_evaluator(
        tmp_path,
        "aware",
        """
        def evaluate(program_path, mode="train"):
            return {"combined_score": 1.0 if mode == "train" else 0.5}
        """,
    )
    evaluator = make_evaluator(path)

    assert (await evaluator.evaluate_program("x")).metrics["combined_score"] == 1.0
    assert (await evaluator.evaluate_program("x", mode="test")).metrics["combined_score"] == 0.5


@pytest.mark.asyncio
async def test_var_keyword_does_not_count_as_opting_in(tmp_path):
    """A catch-all signature must keep its current behaviour.

    `benchmarks/frontier-cs-eval/evaluator.py` has `**kwargs`; injecting `mode`
    there would silently change what it sees.
    """
    path = write_evaluator(
        tmp_path,
        "catchall",
        """
        def evaluate(program_path, **kwargs):
            return {"combined_score": 0.0 if "mode" in kwargs else 1.0}
        """,
    )
    evaluator = make_evaluator(path)

    result = await evaluator.evaluate_program("x", mode="test")
    assert result.metrics["combined_score"] == 1.0


@pytest.mark.asyncio
async def test_test_mode_bypasses_the_cascade(tmp_path):
    """cascade_evaluation defaults to True, so without this the authoritative
    test score could come from a stage1-gated screen rather than the full
    evaluator."""
    path = write_evaluator(
        tmp_path,
        "cascade",
        """
        CALLS = []

        def evaluate(program_path, mode="train"):
            CALLS.append(("full", mode))
            return {"combined_score": 0.5}

        def evaluate_stage1(program_path, mode="train"):
            CALLS.append(("stage1", mode))
            return {"combined_score": 0.9}

        def evaluate_stage2(program_path, mode="train"):
            CALLS.append(("stage2", mode))
            return {"combined_score": 0.8}
        """,
    )
    evaluator = make_evaluator(path, cascade=True)

    await evaluator.evaluate_program("x")
    await evaluator.evaluate_program("x", mode="test")

    assert evaluator._eval_module.CALLS == [
        ("stage1", "train"),
        ("stage2", "train"),
        ("full", "test"),
    ]


@pytest.mark.asyncio
async def test_mode_is_exported_to_the_environment(tmp_path, monkeypatch):
    """Subprocess-style evaluators that shell out need the mode too."""
    monkeypatch.delenv("SKYDISCOVER_EVAL_MODE", raising=False)
    path = write_evaluator(
        tmp_path,
        "envmode",
        """
        import os

        def evaluate(program_path):
            return {"combined_score": 1.0,
                    "saw_test": float(os.environ.get("SKYDISCOVER_EVAL_MODE") == "test"),
                    "saw_train": float(os.environ.get("SKYDISCOVER_EVAL_MODE") == "train")}
        """,
    )
    evaluator = make_evaluator(path)

    assert (await evaluator.evaluate_program("x")).metrics["saw_train"] == 1.0
    assert (await evaluator.evaluate_program("x", mode="test")).metrics["saw_test"] == 1.0


@pytest.mark.asyncio
async def test_user_env_vars_are_still_scoped_and_restored(tmp_path, monkeypatch):
    """The env_vars contract is unchanged: applied for the call, then restored."""
    monkeypatch.delenv("TASK_DATA_DIR", raising=False)
    path = write_evaluator(
        tmp_path,
        "envscope",
        """
        import os

        def evaluate(program_path):
            return {"combined_score": float(os.environ.get("TASK_DATA_DIR") == "/data")}
        """,
    )
    evaluator = Evaluator(
        EvaluatorConfig(evaluation_file=path, cascade_evaluation=False, timeout=30),
        env_vars={"TASK_DATA_DIR": "/data"},
    )

    assert (await evaluator.evaluate_program("x")).metrics["combined_score"] == 1.0
    assert "TASK_DATA_DIR" not in os.environ


@pytest.mark.asyncio
async def test_evaluations_are_not_serialized_when_no_env_vars_are_set(tmp_path):
    """Regression: exporting the mode must not put every evaluation behind the
    process-wide env lock, which is held across the whole user evaluation."""
    path = write_evaluator(
        tmp_path,
        "concurrent",
        """
        import threading, time

        BARRIER = threading.Barrier(3, timeout=5)

        def evaluate(program_path):
            # Deadlocks (and times out) if evaluations are serialized.
            BARRIER.wait()
            return {"combined_score": 1.0}
        """,
    )
    evaluator = make_evaluator(path)

    results = await asyncio.gather(*(evaluator.evaluate_program("x") for _ in range(3)))
    assert [r.metrics["combined_score"] for r in results] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_uninspectable_callable_falls_back_to_the_legacy_contract(tmp_path):
    """We load arbitrary user files; a callable with no signature must not crash."""
    path = write_evaluator(
        tmp_path,
        "builtin_like",
        """
        import functools

        def _impl(program_path):
            return {"combined_score": 1.0}

        # A partial over a builtin has no introspectable signature in some
        # Python builds; guard the general case.
        evaluate = functools.partial(_impl)
        """,
    )
    evaluator = make_evaluator(path)

    result = await evaluator.evaluate_program("x", mode="test")
    assert result.metrics["combined_score"] == 1.0
