"""Tests for deterministic regression diagnostics."""

import pytest

from skydiscover.utils.diagnostics import (
    build_regression_report,
    describe_metric_deltas,
    metric_deltas,
)


def test_fitness_metric_is_reported_first():
    deltas = metric_deltas(
        {"runtime_s": 2.0, "combined_score": 0.5, "accuracy": 0.9},
        {"runtime_s": 1.0, "combined_score": 0.6, "accuracy": 0.9},
    )
    assert deltas[0].name == "combined_score"


def test_remaining_metrics_ordered_by_relative_change():
    deltas = metric_deltas(
        {"small": 1.01, "large": 2.0},
        {"small": 1.00, "large": 1.0},
    )
    assert [d.name for d in deltas] == ["large", "small"]


def test_metrics_missing_from_parent_are_skipped():
    deltas = metric_deltas({"a": 1.0, "brand_new": 5.0}, {"a": 2.0})
    assert [d.name for d in deltas] == ["a"]


def test_bools_are_not_treated_as_numeric():
    # bool is a subclass of int; timeout: True must not become a delta.
    deltas = metric_deltas({"timeout": True, "score": 1.0}, {"timeout": False, "score": 2.0})
    assert [d.name for d in deltas] == ["score"]


def test_status_metrics_are_excluded():
    deltas = metric_deltas({"validity": 1, "score": 1.0}, {"validity": 0, "score": 2.0})
    assert [d.name for d in deltas] == ["score"]


def test_zero_parent_value_yields_no_percentage():
    (delta,) = metric_deltas({"x": 1.0}, {"x": 0.0})
    assert delta.pct is None
    assert "%" not in describe_metric_deltas({"x": 1.0}, {"x": 0.0})


def test_percentage_is_relative_to_parent():
    (delta,) = metric_deltas({"x": 1.5}, {"x": 1.0})
    assert delta.pct == pytest.approx(50.0)


def test_no_report_when_child_improved():
    assert build_regression_report({"combined_score": 0.9}, {"combined_score": 0.5}) is None


def test_no_report_without_comparable_metrics():
    assert build_regression_report({"combined_score": 0.1}, {}) is None


def test_report_names_the_metric_that_moved():
    report = build_regression_report(
        {"combined_score": 0.9877, "runtime_s": 0.0426, "accuracy": 1.0},
        {"combined_score": 0.9918, "runtime_s": 0.0305, "accuracy": 1.0},
    )
    assert report is not None
    assert "scored below its parent" in report
    assert "runtime_s" in report
    assert "accuracy unchanged" in report


def test_report_flags_a_change_that_bought_nothing():
    report = build_regression_report(
        {"combined_score": 0.5, "steps": 10.0},
        {"combined_score": 0.5, "steps": 20.0},
    )
    assert report is not None
    assert "matched its parent" in report


def test_falls_back_to_mean_when_no_combined_score():
    # get_score() averages numeric metrics when combined_score is absent, so
    # the improve/regress decision still works for such evaluators.
    assert build_regression_report({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}) is None
    assert build_regression_report({"a": 0.0, "b": 0.0}, {"a": 1.0, "b": 1.0}) is not None


def test_empty_metrics_are_safe():
    assert build_regression_report({}, {}) is None
    assert build_regression_report(None, None) is None
    assert describe_metric_deltas(None, None) == ""
