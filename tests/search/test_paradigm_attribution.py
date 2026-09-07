"""Per-paradigm outcome attribution in ParadigmTracker.

A paradigm batch used to share one SUCCESS/FAILED verdict computed from
the batch-level score improvement, so an idea whose children never
improved anything was credited whenever a batch-mate's child did. These
tests pin the attributed semantics: gains credit exactly the paradigm
whose use ticket the improving child carried.
"""

from skydiscover.search.adaevolve.paradigm.tracker import ParadigmTracker


def _paradigms(n):
    return [{"idea": f"idea-{i}", "approach_type": "test"} for i in range(n)]


def _make_tracker(**kwargs):
    defaults = dict(window_size=5, max_paradigm_uses=2, num_paradigms_to_generate=3)
    defaults.update(kwargs)
    return ParadigmTracker(**defaults)


def test_use_paradigm_returns_batch_and_index_round_robin():
    t = _make_tracker()
    t.set_paradigms(_paradigms(3), current_best_score=0.0)
    batch = t.batch_id

    # Callers use the production contract: check get_current_paradigm()
    # (has_active guard) before recording a use.
    tickets = []
    while t.get_current_paradigm() is not None:
        tickets.append(t.use_paradigm())
    assert tickets == [(batch, 0), (batch, 1), (batch, 2), (batch, 0), (batch, 1), (batch, 2)]
    # Exhausted after max_uses * n: no further guided generations
    assert t.get_current_paradigm() is None


def test_only_the_guiding_paradigm_is_credited():
    t = _make_tracker()
    t.set_paradigms(_paradigms(3), current_best_score=1.0)

    ticket0 = t.use_paradigm()  # paradigm 0
    ticket1 = t.use_paradigm()  # paradigm 1
    t.use_paradigm()  # paradigm 2 (its child never improves)

    # Paradigm 1's child improves the global best; 0's and 2's do not.
    t.record_improvement(False, 1.0, paradigm_use=ticket0)
    t.record_improvement(True, 1.5, paradigm_use=ticket1)

    t.set_paradigms(_paradigms(3), current_best_score=1.5)  # archives previous batch

    outcomes = {p["idea"]: p["outcome"] for p in t.tried_paradigms}
    assert outcomes["idea-1"] == "SUCCESS"
    assert outcomes["idea-0"] == "FAILED"
    assert outcomes["idea-2"] == "FAILED"

    by_idea = {p["idea"]: p for p in t.tried_paradigms}
    assert abs(by_idea["idea-1"]["attributed_improvement"] - 0.5) < 1e-12
    assert by_idea["idea-0"]["attributed_improvement"] == 0.0
    # Batch-level improvement is still reported alongside
    assert abs(by_idea["idea-0"]["score_improvement"] - 0.5) < 1e-12


def test_unguided_improvement_credits_no_paradigm():
    t = _make_tracker()
    t.set_paradigms(_paradigms(2), current_best_score=0.0)
    t.use_paradigm()
    t.use_paradigm()

    # Improvement from a child generated without paradigm guidance
    t.record_improvement(True, 0.7, paradigm_use=None)

    t.clear_paradigms()
    assert all(p["outcome"] == "FAILED" for p in t.tried_paradigms)
    assert all(p["attributed_improvement"] == 0.0 for p in t.tried_paradigms)
    # The batch-level improvement still reflects what happened on its watch
    assert all(abs(p["score_improvement"] - 0.7) < 1e-12 for p in t.tried_paradigms)


def test_stale_ticket_from_replaced_batch_is_ignored():
    t = _make_tracker()
    t.set_paradigms(_paradigms(2), current_best_score=0.0)
    stale = t.use_paradigm()

    # Batch replaced while the old child was still being evaluated
    t.set_paradigms(_paradigms(2), current_best_score=0.0)
    t.use_paradigm()

    t.record_improvement(True, 0.9, paradigm_use=stale)

    t.clear_paradigms()
    # The new batch's paradigm at the stale index must not be credited
    current = [p for p in t.tried_paradigms if p.get("uses", 0) > 0]
    assert all(p["attributed_improvement"] == 0.0 for p in current)
    assert all(p["outcome"] == "FAILED" for p in current)


def test_negative_scores_attribute_correctly():
    t = _make_tracker()
    t.set_paradigms(_paradigms(1), current_best_score=-0.5)
    ticket = t.use_paradigm()
    t.record_improvement(True, -0.2, paradigm_use=ticket)

    t.clear_paradigms()
    (archived,) = t.tried_paradigms
    assert abs(archived["attributed_improvement"] - 0.3) < 1e-12
    assert archived["outcome"] == "SUCCESS"


def test_tiny_attributed_gain_is_failed():
    t = _make_tracker()
    t.set_paradigms(_paradigms(1), current_best_score=0.0)
    ticket = t.use_paradigm()
    t.record_improvement(True, 0.0005, paradigm_use=ticket)  # below 0.001

    t.clear_paradigms()
    (archived,) = t.tried_paradigms
    assert archived["outcome"] == "FAILED"


def test_checkpoint_round_trip_preserves_attribution():
    t = _make_tracker()
    t.set_paradigms(_paradigms(2), current_best_score=0.0)
    ticket = t.use_paradigm()
    t.record_improvement(True, 0.4, paradigm_use=ticket)

    restored = ParadigmTracker.from_dict(t.to_dict())
    assert restored.batch_id == t.batch_id
    assert restored.paradigm_attributed_gain == {0: 0.4}

    # Ticket from before the checkpoint still attributes after restore,
    # including when it arrives as a JSON-style list instead of a tuple
    t2 = restored
    t2.record_improvement(True, 0.6, paradigm_use=list(ticket))
    assert abs(t2.paradigm_attributed_gain[0] - 0.6) < 1e-12


def test_old_checkpoint_without_attribution_fields_loads():
    t = _make_tracker()
    t.set_paradigms(_paradigms(1), current_best_score=0.0)
    data = t.to_dict()
    del data["batch_id"]
    del data["paradigm_attributed_gain"]

    restored = ParadigmTracker.from_dict(data)
    assert restored.batch_id == 0
    assert restored.paradigm_attributed_gain == {}
    # get_previously_tried_ideas tolerates archives without the new field
    restored.tried_paradigms = [
        {"idea": "x", "approach_type": "t", "outcome": "FAILED", "score_improvement": 0.0}
    ]
    assert restored.get_previously_tried_ideas() == [
        "FAILED: t - x (improvement: +0.0000)"
    ]


def test_exhaustion_notice_logged_once_per_batch(caplog):
    import logging

    t = _make_tracker()
    with caplog.at_level(logging.INFO):
        t.set_paradigms(_paradigms(2), current_best_score=0.0)
        while t.get_current_paradigm() is not None:
            t.use_paradigm()
        for _ in range(10):  # repeated polling must not repeat the notice
            assert not t.has_active_paradigm()
        notices = [r for r in caplog.records if "paradigms exhausted" in r.message]
        assert len(notices) == 1

        # A fresh batch re-arms the notice
        caplog.clear()
        t.set_paradigms(_paradigms(2), current_best_score=0.0)
        while t.get_current_paradigm() is not None:
            t.use_paradigm()
        for _ in range(10):
            assert not t.has_active_paradigm()
        notices = [r for r in caplog.records if "paradigms exhausted" in r.message]
        assert len(notices) == 1
