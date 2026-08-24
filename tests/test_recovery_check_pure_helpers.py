"""Direct unit tests for the pure comparison/mapping helpers behind Phase
6's Action Executor + Recovery Check.

No DB, no LLM, no graph -- these are the two functions the earlier code
review flagged as only exercised indirectly through the full end-to-end
scenarios in tests/test_action_executor_recovery_check.py. That e2e
coverage is real, but a targeted table test over the tolerance arithmetic
itself catches a regression there faster and more precisely, especially
for edge cases (a single-point or empty baseline sample) the e2e tests
don't happen to construct.
"""

from __future__ import annotations

import pytest

from backend.agents.action_executor_node import (
    ON_CORRECT_METRIC_MAP,
    resolve_on_correct_targets,
)
from backend.agents.recovery_check_node import (
    ABSOLUTE_FLOOR,
    BASELINE_PERCENT_TOLERANCE,
    BASELINE_STDDEV_MULTIPLE,
    _compare_to_baseline,
    _mean_stddev,
)

# --- _mean_stddev -----------------------------------------------------------


def test_mean_stddev_empty_list_returns_zero_zero():
    assert _mean_stddev([]) == (0.0, 0.0)


def test_mean_stddev_single_point_has_zero_stddev():
    mean, stddev = _mean_stddev([42.0])
    assert mean == 42.0
    assert stddev == 0.0


def test_mean_stddev_multiple_points_computes_sample_stddev():
    mean, stddev = _mean_stddev([8.0, 9.0, 7.0, 8.0])
    assert mean == pytest.approx(8.0)
    assert stddev == pytest.approx(0.816496580927726)


# --- _compare_to_baseline: the tolerance formula itself ---------------------


def test_compare_to_baseline_recovered_within_stddev_tolerance():
    # baseline mean 8.0, stddev 1.5 -> tolerance = max(2*1.5, 0.20*8.0, 0.01) = 3.0
    baseline = [7.0, 8.0, 9.0, 8.0, 8.0]  # mean 8.0
    post = [9.0, 10.0, 9.0, 10.0, 9.5]  # mean ~9.5, within 3.0 of 8.0
    result = _compare_to_baseline(baseline, post)
    assert result["recovered"] is True
    assert result["tolerance"] == pytest.approx(max(2.0 * _mean_stddev(baseline)[1], 1.6, 0.01))


def test_compare_to_baseline_still_degraded_far_outside_tolerance():
    baseline = [8.0, 8.0, 8.0, 8.0]  # mean 8.0, stddev 0.0
    post = [44.0, 45.0, 43.0, 44.0]  # anomalous level, matches injector.py's
    # db_connection_exhaustion ramp target (~5.5x baseline mean)
    result = _compare_to_baseline(baseline, post)
    assert result["recovered"] is False
    assert result["post_action_mean"] == pytest.approx(44.0)


def test_compare_to_baseline_low_variance_metric_uses_percent_floor():
    # error_rate-style baseline: mean 0.005, stddev 0.0015 -> stddev term is
    # tiny, so the 20%-of-mean term or the absolute floor should dominate.
    baseline = [0.005, 0.006, 0.004, 0.005]
    stddev_term = BASELINE_STDDEV_MULTIPLE * _mean_stddev(baseline)[1]
    percent_term = BASELINE_PERCENT_TOLERANCE * abs(_mean_stddev(baseline)[0])
    expected_tolerance = max(stddev_term, percent_term, ABSOLUTE_FLOOR)
    assert expected_tolerance in (pytest.approx(ABSOLUTE_FLOOR), pytest.approx(percent_term))

    # A real degraded value (30-40% error rate, per the injector's error
    # bursts) must clear that (necessarily small) tolerance as "not recovered".
    result = _compare_to_baseline(baseline, [0.30, 0.35, 0.32, 0.31])
    assert result["recovered"] is False


def test_compare_to_baseline_empty_post_action_sample_is_not_recovered():
    """No post-action data at all must never be silently treated as
    'recovered' -- an empty sample means (0.0, 0.0) via _mean_stddev, which
    only reads as recovered if baseline itself is ~0; for a real nonzero
    baseline this correctly falls outside tolerance."""
    baseline = [8.0, 8.0, 8.0, 8.0]
    result = _compare_to_baseline(baseline, [])
    assert result["post_action_sample_size"] == 0
    assert result["recovered"] is False


# --- resolve_on_correct_targets ---------------------------------------------


def test_resolve_on_correct_targets_skips_incident_status_hint():
    targets = resolve_on_correct_targets(
        {"error_rate": "recovers_to_baseline", "incident_status": "resolved"},
        affected_service="checkout-service",
    )
    assert targets == [("error_rate", "checkout-service")]


def test_resolve_on_correct_targets_uses_service_override_for_dependency_keys():
    targets = resolve_on_correct_targets(
        {"payment_latency": "recovers_to_baseline"}, affected_service="checkout-service"
    )
    assert targets == [("latency_p99_ms", "payment-service")]


def test_resolve_on_correct_targets_covers_every_key_in_the_map():
    """Every key ON_CORRECT_METRIC_MAP declares must actually resolve
    without raising -- catches a stale/typo'd map entry."""
    for key in ON_CORRECT_METRIC_MAP:
        targets = resolve_on_correct_targets({key: "recovers_to_baseline"}, "checkout-service")
        assert len(targets) == 1


def test_resolve_on_correct_targets_raises_on_unknown_key():
    with pytest.raises(ValueError, match="no known metric mapping"):
        resolve_on_correct_targets(
            {"totally_unmapped_key": "recovers_to_baseline"}, affected_service="checkout-service"
        )
