"""No-repeat-model-per-prompt sampling.

Near-deterministic backends (the Codex/Claude CLIs) replay themselves on an
identical prompt, so redrawing the same model spends an evaluation on a
duplicate program.
"""

import random

import pytest

from skydiscover.config import LLMModelConfig
from skydiscover.llm.llm_pool import LLMPool


@pytest.fixture(autouse=True)
def clean_history():
    LLMPool._prompt_history.clear()
    yield
    LLMPool._prompt_history.clear()


def pool(names):
    p = object.__new__(LLMPool)
    p.models_cfg = [LLMModelConfig(name=n) for n in names]
    p.models = [f"backend:{n}" for n in names]
    p.weights = [1.0] * len(names)
    p.effective_weights = [1.0] * len(names)
    p.temperature_emulator = None
    p.temperature = None
    p.random_state = random.Random(0)
    return p


def key(p, text="same"):
    return p._prompt_key("sys", [{"role": "user", "content": text}])


class TestNoRepeat:
    def test_identical_prompt_cycles_through_every_model(self):
        p = pool(["A", "B", "C"])
        k = key(p)
        assert len({p._sample_model(k) for _ in range(3)}) == 3

    def test_the_cycle_restarts_once_all_models_are_used(self):
        p = pool(["A", "B"])
        k = key(p)
        first = {p._sample_model(k) for _ in range(2)}
        assert len(first) == 2
        assert p._sample_model(k) in first  # reset, not starved

    def test_a_different_prompt_is_tracked_independently(self):
        p = pool(["A", "B"])
        p._sample_model(key(p, "one"))
        # A fresh prompt must be free to draw anything, including the same model.
        drawn = {p._sample_model(key(p, "two")) for _ in range(2)}
        assert len(drawn) == 2

    def test_omitting_the_prompt_key_keeps_plain_weighted_sampling(self):
        """Back-compat: pool stand-ins call _sample_model() with no arguments."""
        p = pool(["A", "B"])
        assert p._sample_model() in p.models
        assert not LLMPool._prompt_history

    def test_single_model_pool_is_not_tracked(self):
        p = pool(["only"])
        p._sample_model(key(p))
        assert not LLMPool._prompt_history


class TestPoolIsolation:
    """The history stores model *indices*, so it is only meaningful within one
    roster — and a run builds llms / evaluator_llms / guide_llms separately."""

    def test_another_pool_does_not_consume_this_pools_cycle(self):
        main, guide = pool(["A", "B", "C"]), pool(["X", "Y"])
        k = key(main)
        first, second = main._sample_model(k), main._sample_model(k)
        guide._sample_model(k)  # would previously clear main's entry
        assert len({first, second, main._sample_model(k)}) == 3

    def test_pools_with_different_rosters_get_separate_entries(self):
        main, guide = pool(["A", "B", "C"]), pool(["X", "Y"])
        k = key(main)
        main._sample_model(k)
        guide._sample_model(k)
        assert len(LLMPool._prompt_history) == 2

    def test_pools_with_the_same_roster_share_one_entry(self):
        """evaluator_models defaults to a copy of models; those two genuinely
        are the same roster and should keep sharing the cycle."""
        a, b = pool(["A", "B"]), pool(["A", "B"])
        k = key(a)
        assert {a._sample_model(k), b._sample_model(k)} == set(a.models)
        assert len(LLMPool._prompt_history) == 1


class TestHistoryBounds:
    def test_history_is_capped(self, monkeypatch):
        monkeypatch.setattr(LLMPool, "_PROMPT_HISTORY_MAX", 8)
        p = pool(["A", "B"])
        for i in range(30):
            p._sample_model(key(p, f"prompt-{i}"))
        assert len(LLMPool._prompt_history) <= 8
