"""Weighted Mondrian conformal on trajectory direction (design Section 3 iv).

Owner: Dr. Helen Zhao (Track B conformal). This module scaffolds the primary
leaderboard metric (iv): empirical coverage of per-subject trajectory-direction
conformal prediction sets.

Geometry. The prediction object is the unit direction of the latent trajectory
``direction = delta_z / ||delta_z||`` on the sphere S^{d-1}. The conformity
score is the arc-length (geodesic) distance between predicted and observed
directions, ``s = arccos(<pred, obs>)``. The prediction set at level alpha is a
spherical cap of angular radius ``q_hat`` centred on the predicted direction.

Conformal flavour: WEIGHTED Mondrian (Tibshirani et al. NeurIPS 2019). Plain
Mondrian assumes calibration/test exchangeability; Emory -> BEST is a different
cohort so the protocol uses importance weights from the Gate 0-S source-domain
classifier, truncated at the 99th percentile (Lei & Candes JRSSB 2021).
Mondrian strata are Response codes (binary for Emory, 3-class for BEST).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.10
WEIGHT_TRUNCATION_PERCENTILE = 99.0


def unit_directions(delta_z: np.ndarray) -> np.ndarray:
    """Normalise (n, d) latent deltas to unit directions on S^{d-1}."""
    norms = np.linalg.norm(delta_z, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return np.asarray(delta_z / norms, dtype=np.float64)


def arc_length_score(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Geodesic conformity score s = arccos(<pred, obs>) per row.

    Inputs are (n, d) unit-direction matrices. Output is (n,) in [0, pi].
    """
    dots = np.clip(np.sum(predicted * observed, axis=-1), -1.0, 1.0)
    return np.asarray(np.arccos(dots), dtype=np.float64)


def importance_weights(
    p_best: np.ndarray,
    *,
    truncation_percentile: float = WEIGHT_TRUNCATION_PERCENTILE,
) -> np.ndarray:
    """Covariate-shift importance weights w = P(BEST|x) / P(Emory|x).

    Parameters
    ----------
    p_best:
        (n,) calibration-subject probabilities P(BEST | x) from the Gate 0-S
        source-domain classifier.
    truncation_percentile:
        Percentile at which to cap the weights (variance control,
        Lei & Candes JRSSB 2021).
    """
    p_best = np.clip(p_best, 1e-6, 1 - 1e-6)
    w = p_best / (1.0 - p_best)
    cap = np.percentile(w, truncation_percentile)
    return np.minimum(w, cap)


def weighted_conformal_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Weighted conformal quantile q_hat (design Section 3 iv step 4).

    q_hat = inf { q : sum_i w_tilde_i * 1[s_i <= q] >= 1 - alpha }, where the
    normalised weights include the test-point mass term w_test (taken as the
    mean calibration weight, the standard plug-in).
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights[order]
    w_test = float(weights.mean())
    w_tilde = w_sorted / (w_sorted.sum() + w_test)
    cumulative = np.cumsum(w_tilde)
    idx = np.searchsorted(cumulative, 1.0 - alpha, side="left")
    if idx >= len(s_sorted):
        return float(s_sorted[-1])
    return float(s_sorted[idx])


@dataclass
class ConformalResult:
    """Per-stratum + marginal weighted-Mondrian conformal result."""

    alpha: float
    marginal_coverage: float
    marginal_radius: float
    per_stratum_coverage: dict[str, float]
    per_stratum_radius: dict[str, float]
    q_hat_per_stratum: dict[str, float]


def weighted_mondrian_conformal(
    cal_predicted: np.ndarray,
    cal_observed: np.ndarray,
    cal_strata: np.ndarray,
    cal_p_best: np.ndarray,
    test_predicted: np.ndarray,
    test_observed: np.ndarray,
    test_strata: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> ConformalResult:
    """Run weighted-Mondrian directional conformal (design Section 3 iv).

    Calibrates a per-stratum spherical-cap radius on the calibration fold, then
    measures empirical coverage and mean angular radius on the test fold.

    Parameters
    ----------
    cal_predicted, cal_observed:
        (n_cal, d) predicted vs observed unit directions on the calibration set.
    cal_strata:
        (n_cal,) Mondrian stratum label per calibration subject (Response code).
    cal_p_best:
        (n_cal,) Gate 0-S P(BEST | x) per calibration subject.
    test_predicted, test_observed, test_strata:
        Test-fold counterparts.
    alpha:
        Miscoverage level (0.10 -> 90% target coverage).

    Returns
    -------
    ConformalResult with marginal and per-stratum coverage + angular radius.
    """
    cal_scores = arc_length_score(cal_predicted, cal_observed)
    cal_weights = importance_weights(cal_p_best)
    test_scores = arc_length_score(test_predicted, test_observed)

    q_hat: dict[str, float] = {}
    for stratum in np.unique(cal_strata):
        key = str(stratum)
        mask = cal_strata == stratum
        if mask.sum() < 2:
            logger.warning("Stratum %s has < 2 calibration subjects; widening cap", key)
            q_hat[key] = float(np.pi)
            continue
        q_hat[key] = weighted_conformal_quantile(cal_scores[mask], cal_weights[mask], alpha=alpha)

    # Fallback radius for test strata absent from calibration.
    fallback_q = float(np.max(list(q_hat.values()))) if q_hat else float(np.pi)

    covered: list[bool] = []
    radii: list[float] = []
    per_cov: dict[str, list[bool]] = {}
    per_rad: dict[str, list[float]] = {}
    for i, stratum in enumerate(test_strata):
        key = str(stratum)
        q = q_hat.get(key, fallback_q)
        is_covered = bool(test_scores[i] <= q)
        covered.append(is_covered)
        radii.append(q)
        per_cov.setdefault(key, []).append(is_covered)
        per_rad.setdefault(key, []).append(q)

    return ConformalResult(
        alpha=alpha,
        marginal_coverage=float(np.mean(covered)) if covered else 0.0,
        marginal_radius=float(np.mean(radii)) if radii else float(np.pi),
        per_stratum_coverage={k: float(np.mean(v)) for k, v in per_cov.items()},
        per_stratum_radius={k: float(np.mean(v)) for k, v in per_rad.items()},
        q_hat_per_stratum=q_hat,
    )
