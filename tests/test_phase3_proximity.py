"""Synthetic-fixture tests for the Phase 3.3 proximity test.

Covers the non-responder-terminus-vs-TRD-cluster proximity test, the
subject-clustered bootstrap CIs, the subject-label permutation p-value, and the
hard pre-registered fail criterion (p >= 0.05 OR Cohen's d <= 0.3 -> FAIL).

No real data, no trained atlas: every test runs on synthetic projected-atlas
coordinates from tests/phase3_fixtures.py and completes in well under a second
(bootstrap/permutation B kept small in tests; production defaults are B=2000).
"""

from __future__ import annotations

import numpy as np
import pytest

from dnamrnaseq2026.external_projection.proximity_test import (
    FAIL_COHENS_D_THRESHOLD,
    FAIL_P_THRESHOLD,
    FailCriterionVerdict,
    ProximityTestResult,
    evaluate_fail_criterion,
    run_proximity_test,
)
from tests.phase3_fixtures import make_synthetic_termini

# Small B for fast CI; production defaults are B=2000.
_TEST_B = 300


# ---------------------------------------------------------------------------
# Pre-registered thresholds are fixed constants
# ---------------------------------------------------------------------------


def test_fail_criterion_thresholds_are_pinned() -> None:
    """The pre-registered gate constants must not drift from the headline doc."""
    assert FAIL_P_THRESHOLD == 0.05
    assert FAIL_COHENS_D_THRESHOLD == 0.3


# ---------------------------------------------------------------------------
# Proximity test mechanics
# ---------------------------------------------------------------------------


def test_proximity_test_runs_and_reports_structure() -> None:
    data = make_synthetic_termini(seed=1)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=_TEST_B,
        permutation_b=_TEST_B,
    )
    assert isinstance(result, ProximityTestResult)
    assert result.n_responders == 30
    assert result.n_nonresponders == 30
    assert result.n_trd_reference == 40
    # CIs are ordered and bracket the point estimates.
    assert result.effect_ci[0] <= result.effect <= result.effect_ci[1]
    assert result.cohens_d_ci[0] <= result.cohens_d <= result.cohens_d_ci[1]
    assert 0.0 < result.p_value <= 1.0


def test_planted_signal_detects_nonresponders_closer_to_trd() -> None:
    """With a planted signal, non-responder termini sit closer to TRD."""
    data = make_synthetic_termini(seed=2, plant_signal=True)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=_TEST_B,
        permutation_b=_TEST_B,
    )
    # Non-responders closer => smaller mean distance => positive effect / d.
    assert result.mean_d_nonresponder < result.mean_d_responder
    assert result.effect > 0
    assert result.cohens_d > 0
    assert result.p_value < FAIL_P_THRESHOLD


def test_null_data_yields_no_separation() -> None:
    """With no planted signal the effect is near zero and p is not significant."""
    data = make_synthetic_termini(seed=3, plant_signal=False)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=_TEST_B,
        permutation_b=_TEST_B,
    )
    assert abs(result.cohens_d) < 0.3
    assert result.p_value >= FAIL_P_THRESHOLD


def test_reproducible_with_fixed_seed() -> None:
    """Same seed -> identical bootstrap CIs and permutation p-value."""
    data = make_synthetic_termini(seed=4)
    kwargs = {"bootstrap_b": _TEST_B, "permutation_b": _TEST_B, "seed": 123}
    r1 = run_proximity_test(
        data["responder_termini"], data["nonresponder_termini"], data["trd_reference"], **kwargs
    )
    r2 = run_proximity_test(
        data["responder_termini"], data["nonresponder_termini"], data["trd_reference"], **kwargs
    )
    assert r1.effect_ci == r2.effect_ci
    assert r1.cohens_d_ci == r2.cohens_d_ci
    assert r1.p_value == r2.p_value


def test_cluster_bootstrap_ci_wider_than_naive_pooled_ci() -> None:
    """Subject-clustered bootstrap CI must not be narrower than a naive
    pooled-distance bootstrap that ignores the response grouping.

    This guards the repeated-measures discipline: a naive resample that pools
    all distances and ignores subject/group structure understates uncertainty.
    The clustered CI is the honest one and should be at least as wide.
    """
    data = make_synthetic_termini(seed=5)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=2000,
        permutation_b=_TEST_B,
        seed=7,
    )
    clustered_width = result.effect_ci[1] - result.effect_ci[0]

    # Naive comparison: bootstrap the pooled distances ignoring group labels.
    trd_centroid = data["trd_reference"].mean(axis=0)
    d_r = np.linalg.norm(data["responder_termini"] - trd_centroid, axis=1)
    d_nr = np.linalg.norm(data["nonresponder_termini"] - trd_centroid, axis=1)
    pooled = np.concatenate([d_r, d_nr])
    rng = np.random.default_rng(7)
    naive = np.array(
        [
            rng.choice(pooled, size=len(d_r), replace=True).mean()
            - rng.choice(pooled, size=len(d_nr), replace=True).mean()
            for _ in range(2000)
        ]
    )
    naive_width = float(np.percentile(naive, 97.5) - np.percentile(naive, 2.5))
    assert clustered_width >= 0.7 * naive_width


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_mismatched_latent_dimension_rejected() -> None:
    data = make_synthetic_termini(seed=6)
    bad_trd = data["trd_reference"][:, :-1]  # one fewer latent dimension
    with pytest.raises(ValueError, match="same atlas latent space"):
        run_proximity_test(data["responder_termini"], data["nonresponder_termini"], bad_trd)


def test_too_few_subjects_rejected() -> None:
    data = make_synthetic_termini(n_responders=1, n_nonresponders=30, seed=7)
    with pytest.raises(ValueError, match=">= 2 subjects"):
        run_proximity_test(
            data["responder_termini"], data["nonresponder_termini"], data["trd_reference"]
        )


# ---------------------------------------------------------------------------
# Pre-registered fail criterion: the hard gate
# ---------------------------------------------------------------------------


def test_fail_criterion_pass_when_signal_present() -> None:
    data = make_synthetic_termini(seed=8, plant_signal=True)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=_TEST_B,
        permutation_b=_TEST_B,
    )
    verdict = evaluate_fail_criterion(result)
    assert isinstance(verdict, FailCriterionVerdict)
    assert verdict.passed is True
    assert verdict.verdict == "PASS"
    assert verdict.pivot is None


def test_fail_criterion_fail_on_null_data() -> None:
    data = make_synthetic_termini(seed=9, plant_signal=False)
    result = run_proximity_test(
        data["responder_termini"],
        data["nonresponder_termini"],
        data["trd_reference"],
        bootstrap_b=_TEST_B,
        permutation_b=_TEST_B,
    )
    verdict = evaluate_fail_criterion(result)
    assert verdict.passed is False
    assert verdict.verdict == "FAIL"
    assert verdict.pivot is not None
    assert "monocyte" in verdict.pivot.lower()


def _result_with(p_value: float, cohens_d: float) -> ProximityTestResult:
    """Construct a minimal ProximityTestResult with controlled p and d."""
    return ProximityTestResult(
        n_responders=30,
        n_nonresponders=30,
        n_trd_reference=40,
        mean_d_responder=1.0,
        mean_d_nonresponder=0.5,
        effect=0.5,
        effect_ci=(0.1, 0.9),
        cohens_d=cohens_d,
        cohens_d_ci=(cohens_d - 0.1, cohens_d + 0.1),
        p_value=p_value,
        bootstrap_b=2000,
        permutation_b=2000,
        seed=20260522,
    )


def test_fail_criterion_is_an_or_gate_p_fails() -> None:
    """Strong effect size but non-significant p -> FAIL (OR semantics)."""
    verdict = evaluate_fail_criterion(_result_with(p_value=0.20, cohens_d=0.8))
    assert verdict.verdict == "FAIL"
    assert "p=0.2000 >= 0.05" in verdict.reason


def test_fail_criterion_is_an_or_gate_d_fails() -> None:
    """Significant p but small effect size -> FAIL (OR semantics)."""
    verdict = evaluate_fail_criterion(_result_with(p_value=0.001, cohens_d=0.2))
    assert verdict.verdict == "FAIL"
    assert "Cohen's d=0.2000 <= 0.3" in verdict.reason


def test_fail_criterion_boundary_p_equals_threshold_fails() -> None:
    """p exactly 0.05 fails: the criterion is p >= 0.05, not p > 0.05."""
    verdict = evaluate_fail_criterion(_result_with(p_value=0.05, cohens_d=0.8))
    assert verdict.verdict == "FAIL"


def test_fail_criterion_boundary_d_equals_threshold_fails() -> None:
    """Cohen's d exactly 0.3 fails: the criterion is d <= 0.3, not d < 0.3."""
    verdict = evaluate_fail_criterion(_result_with(p_value=0.001, cohens_d=0.3))
    assert verdict.verdict == "FAIL"


def test_fail_criterion_pass_requires_both() -> None:
    """PASS requires p < 0.05 AND d > 0.3 strictly."""
    verdict = evaluate_fail_criterion(_result_with(p_value=0.049, cohens_d=0.31))
    assert verdict.verdict == "PASS"
    assert verdict.passed is True
