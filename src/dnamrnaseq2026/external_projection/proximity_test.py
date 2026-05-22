"""Phase 3.3 proximity test: do PTSD non-responder termini land closer to the
GSE98793 TRD-inflammatory cluster than responder termini?

This is the statistical core of Phase 3 (ANALYSIS_PLAN.md Step 3.3, the
headline figure). It operates on *projected atlas latent coordinates*: every
(subject, visit) trajectory terminus and a GSE98793 TRD-inflammatory reference
cloud, all embedded in the same trained-atlas latent space. The projection step
itself is a separate module (Phase 3 projection pipeline); this module consumes
its output and never produces coordinates of its own.

The test
--------
For each PTSD subject the *terminus* is the POST-IOP (12-week) latent
coordinate. We measure the Euclidean distance from each terminus to the
GSE98793 TRD-inflammatory centroid. The contrast of interest is:

    effect = mean d(responder terminus, TRD) - mean d(non-responder terminus, TRD)

A positive effect means non-responder termini sit *closer* to the
TRD-inflammatory cloud (smaller distance), i.e. the expected direction.

Pre-registered fail criterion (headline-framing-rationale.md, 2026-05-19)
------------------------------------------------------------------------
    p >= 0.05  OR  Cohen's d <= 0.3  ->  FAIL

On FAIL the transdiagnostic immunometabolic frame is not supported at this N
and the translational paper pivots to the monocyte-WNT narrative without the
cross-disorder comparison. The criterion is encoded here as a hard, pre-committed
gate (:func:`evaluate_fail_criterion`) with the thresholds fixed as module-level
constants *before any real projected coordinates exist*. The verdict function
takes no tuning parameters: the gate cannot be moved after seeing the data.

Confidence intervals
--------------------
Trajectory termini are repeated-measures: each PTSD subject contributes a
within-subject PRE/POST pair, and the terminus is one end of that pair. The CI
on the effect and on Cohen's d therefore uses a *subject-clustered* nonparametric
bootstrap (resample whole subjects with replacement, B=2000, documented seed),
not a naive epoch-level bootstrap. Naive epoch-level resampling treats the two
visits of one subject as independent observations and produces intervals
5-7x too narrow; reviewers catch this. See the programme-wide cluster-bootstrap
discipline note.

The p-value is a subject-label permutation test: under the null the Response
label is exchangeable across subjects, so permuting it B times and recomputing
the effect gives an exact (Monte-Carlo) one-sided p-value. Permutation is at the
subject level for the same clustering reason.

All inference here is associational. A smaller terminus-to-TRD distance for
non-responders is an association in projected latent space; it is not evidence
that the TRD-inflammatory state causes non-response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# --- Pre-registered fail-criterion thresholds (FIXED 2026-05-19, headline doc) -
# Do not edit these after real projected coordinates exist. They are the
# pre-committed gate; moving them retrospectively voids the pre-registration.
FAIL_P_THRESHOLD = 0.05
FAIL_COHENS_D_THRESHOLD = 0.3

# --- Subject-clustered bootstrap / permutation defaults -----------------------
BOOTSTRAP_B = 2000
PERMUTATION_B = 2000
BOOTSTRAP_SEED = 20260522  # documented seed; programme cluster-bootstrap discipline
CI_ALPHA = 0.05  # 95% percentile interval


@dataclass(frozen=True)
class ProximityTestResult:
    """Result of the Phase 3.3 non-responder-terminus-vs-TRD proximity test.

    Attributes
    ----------
    n_responders, n_nonresponders, n_trd_reference:
        Group sizes actually used.
    mean_d_responder, mean_d_nonresponder:
        Mean Euclidean distance from terminus to the TRD-inflammatory centroid.
    effect:
        mean_d_responder - mean_d_nonresponder. Positive => non-responder termini
        closer to TRD (expected direction).
    effect_ci:
        (low, high) subject-clustered bootstrap percentile CI on ``effect``.
    cohens_d:
        Cohen's d for the responder-vs-non-responder distance contrast
        (pooled-SD standardised; positive => non-responders closer to TRD).
    cohens_d_ci:
        (low, high) subject-clustered bootstrap percentile CI on Cohen's d.
    p_value:
        One-sided subject-label permutation p-value for H1: effect > 0.
    bootstrap_b, permutation_b, seed:
        Reproducibility parameters actually used.
    """

    n_responders: int
    n_nonresponders: int
    n_trd_reference: int
    mean_d_responder: float
    mean_d_nonresponder: float
    effect: float
    effect_ci: tuple[float, float]
    cohens_d: float
    cohens_d_ci: tuple[float, float]
    p_value: float
    bootstrap_b: int
    permutation_b: int
    seed: int


@dataclass(frozen=True)
class FailCriterionVerdict:
    """Hard, pre-committed Phase 3.3 gate outcome.

    ``passed`` is True only when BOTH p < 0.05 AND Cohen's d > 0.3. Any other
    outcome is FAIL and triggers the monocyte-WNT pivot.
    """

    passed: bool
    verdict: str  # "PASS" or "FAIL"
    p_value: float
    cohens_d: float
    p_threshold: float
    cohens_d_threshold: float
    reason: str
    pivot: str | None  # the mandated fallback framing on FAIL, else None


def _terminus_distances_to_centroid(
    termini: np.ndarray,
    trd_centroid: np.ndarray,
) -> np.ndarray:
    """Euclidean distance from each terminus row to the TRD centroid."""
    diff = termini - trd_centroid[None, :]
    return np.asarray(np.linalg.norm(diff, axis=1), dtype=np.float64)


def _cohens_d(d_responder: np.ndarray, d_nonresponder: np.ndarray) -> float:
    """Cohen's d for the responder-vs-non-responder distance contrast.

    Sign convention: positive when non-responders are *closer* to TRD
    (smaller distance), matching ``effect``. Pooled SD, n-1 denominator.
    """
    n_r, n_nr = d_responder.size, d_nonresponder.size
    if n_r < 2 or n_nr < 2:
        return 0.0
    var_r = float(np.var(d_responder, ddof=1))
    var_nr = float(np.var(d_nonresponder, ddof=1))
    pooled = ((n_r - 1) * var_r + (n_nr - 1) * var_nr) / (n_r + n_nr - 2)
    if pooled <= 0.0:
        return 0.0
    pooled_sd = float(np.sqrt(pooled))
    return (float(d_responder.mean()) - float(d_nonresponder.mean())) / pooled_sd


def run_proximity_test(
    responder_termini: np.ndarray,
    nonresponder_termini: np.ndarray,
    trd_reference: np.ndarray,
    *,
    bootstrap_b: int = BOOTSTRAP_B,
    permutation_b: int = PERMUTATION_B,
    seed: int = BOOTSTRAP_SEED,
) -> ProximityTestResult:
    """Run the Phase 3.3 proximity test on projected atlas latent coordinates.

    Parameters
    ----------
    responder_termini:
        (n_responders, d_latent) POST-IOP latent coordinates of responder
        subjects. One row per subject (the within-subject terminus).
    nonresponder_termini:
        (n_nonresponders, d_latent) POST-IOP latent coordinates of
        non-responder subjects.
    trd_reference:
        (n_trd, d_latent) projected GSE98793 TRD-inflammatory reference cloud.
        Its centroid (column mean) is the proximity anchor.
    bootstrap_b:
        Number of subject-clustered bootstrap resamples for the CIs.
    permutation_b:
        Number of subject-label permutations for the p-value.
    seed:
        Random seed (documented for reproducibility).

    Returns
    -------
    ProximityTestResult

    Notes
    -----
    The bootstrap resamples *whole subjects* with replacement within each
    response group, preserving the repeated-measures structure: a subject's
    terminus is never split from itself. The permutation shuffles the Response
    label across the pooled subject set, which is the exact exchangeability
    null for "response group does not affect terminus-to-TRD distance".
    """
    responder_termini = np.asarray(responder_termini, dtype=np.float64)
    nonresponder_termini = np.asarray(nonresponder_termini, dtype=np.float64)
    trd_reference = np.asarray(trd_reference, dtype=np.float64)

    if responder_termini.ndim != 2 or nonresponder_termini.ndim != 2:
        raise ValueError("Termini arrays must be 2-D (n_subjects, d_latent).")
    if trd_reference.ndim != 2:
        raise ValueError("TRD reference must be 2-D (n_trd, d_latent).")
    d_latent = trd_reference.shape[1]
    if responder_termini.shape[1] != d_latent or nonresponder_termini.shape[1] != d_latent:
        raise ValueError(
            "Termini and TRD reference must share the latent dimension "
            f"(got responder={responder_termini.shape[1]}, "
            f"nonresponder={nonresponder_termini.shape[1]}, trd={d_latent}). "
            "All inputs must be projected into the same atlas latent space."
        )
    n_r, n_nr = responder_termini.shape[0], nonresponder_termini.shape[0]
    if n_r < 2 or n_nr < 2:
        raise ValueError("Need >= 2 subjects in each response group.")

    trd_centroid = trd_reference.mean(axis=0)
    d_responder = _terminus_distances_to_centroid(responder_termini, trd_centroid)
    d_nonresponder = _terminus_distances_to_centroid(nonresponder_termini, trd_centroid)

    mean_d_r = float(d_responder.mean())
    mean_d_nr = float(d_nonresponder.mean())
    effect = mean_d_r - mean_d_nr  # positive => non-responders closer to TRD
    cohens_d = _cohens_d(d_responder, d_nonresponder)

    # --- Subject-clustered bootstrap CIs --------------------------------------
    rng = np.random.default_rng(seed)
    boot_effects = np.empty(bootstrap_b, dtype=np.float64)
    boot_ds = np.empty(bootstrap_b, dtype=np.float64)
    for b in range(bootstrap_b):
        idx_r = rng.integers(0, n_r, size=n_r)  # resample whole responder subjects
        idx_nr = rng.integers(0, n_nr, size=n_nr)  # resample whole non-responders
        bd_r = d_responder[idx_r]
        bd_nr = d_nonresponder[idx_nr]
        boot_effects[b] = float(bd_r.mean()) - float(bd_nr.mean())
        boot_ds[b] = _cohens_d(bd_r, bd_nr)
    lo, hi = 100.0 * CI_ALPHA / 2.0, 100.0 * (1.0 - CI_ALPHA / 2.0)
    effect_ci = (
        float(np.percentile(boot_effects, lo)),
        float(np.percentile(boot_effects, hi)),
    )
    cohens_d_ci = (
        float(np.percentile(boot_ds, lo)),
        float(np.percentile(boot_ds, hi)),
    )

    # --- Subject-label permutation p-value ------------------------------------
    pooled = np.concatenate([d_responder, d_nonresponder])
    perm_rng = np.random.default_rng(seed + 1)
    perm_effects = np.empty(permutation_b, dtype=np.float64)
    for b in range(permutation_b):
        shuffled = perm_rng.permutation(pooled)
        perm_effects[b] = float(shuffled[:n_r].mean()) - float(shuffled[n_r:].mean())
    # One-sided: H1 is effect > 0 (non-responders closer to TRD). +1 in the
    # numerator and denominator is the standard Monte-Carlo permutation
    # correction so the p-value is never exactly zero.
    p_value = float((np.sum(perm_effects >= effect) + 1) / (permutation_b + 1))

    logger.info(
        "Phase 3.3 proximity test: effect=%.4f (95%% CI %.4f, %.4f), "
        "Cohen's d=%.4f (95%% CI %.4f, %.4f), permutation p=%.4f",
        effect,
        effect_ci[0],
        effect_ci[1],
        cohens_d,
        cohens_d_ci[0],
        cohens_d_ci[1],
        p_value,
    )

    return ProximityTestResult(
        n_responders=n_r,
        n_nonresponders=n_nr,
        n_trd_reference=trd_reference.shape[0],
        mean_d_responder=mean_d_r,
        mean_d_nonresponder=mean_d_nr,
        effect=effect,
        effect_ci=effect_ci,
        cohens_d=cohens_d,
        cohens_d_ci=cohens_d_ci,
        p_value=p_value,
        bootstrap_b=bootstrap_b,
        permutation_b=permutation_b,
        seed=seed,
    )


def evaluate_fail_criterion(result: ProximityTestResult) -> FailCriterionVerdict:
    """Apply the hard, pre-committed Phase 3.3 fail criterion.

    Pre-registered gate (headline-framing-rationale.md, 2026-05-19):

        p >= 0.05  OR  Cohen's d <= 0.3  ->  FAIL

    PASS requires BOTH p < 0.05 AND Cohen's d > 0.3. The thresholds are fixed
    module constants and this function accepts no tuning parameters: the gate
    cannot be moved after seeing the data.

    Parameters
    ----------
    result:
        Output of :func:`run_proximity_test`.

    Returns
    -------
    FailCriterionVerdict
        On FAIL, ``pivot`` names the mandated fallback framing.
    """
    p = result.p_value
    d = result.cohens_d
    p_ok = p < FAIL_P_THRESHOLD
    d_ok = d > FAIL_COHENS_D_THRESHOLD
    passed = p_ok and d_ok

    if passed:
        reason = (
            f"PASS: permutation p={p:.4f} < {FAIL_P_THRESHOLD} "
            f"AND Cohen's d={d:.4f} > {FAIL_COHENS_D_THRESHOLD}. "
            "The transdiagnostic immunometabolic frame is supported; "
            "the two-anchor recovery-axis headline ships."
        )
        pivot: str | None = None
        verdict = "PASS"
    else:
        failed_on = []
        if not p_ok:
            failed_on.append(f"p={p:.4f} >= {FAIL_P_THRESHOLD}")
        if not d_ok:
            failed_on.append(f"Cohen's d={d:.4f} <= {FAIL_COHENS_D_THRESHOLD}")
        reason = (
            "FAIL (" + " AND ".join(failed_on) + "). "
            "The transdiagnostic immunometabolic frame is not supported at this N."
        )
        pivot = (
            "Pivot: the translational paper reframes around the monocyte-specific "
            "WNT narrative without the cross-disorder TRD comparison."
        )
        verdict = "FAIL"

    logger.info("Phase 3.3 fail-criterion verdict: %s -- %s", verdict, reason)
    return FailCriterionVerdict(
        passed=passed,
        verdict=verdict,
        p_value=p,
        cohens_d=d,
        p_threshold=FAIL_P_THRESHOLD,
        cohens_d_threshold=FAIL_COHENS_D_THRESHOLD,
        reason=reason,
        pivot=pivot,
    )
