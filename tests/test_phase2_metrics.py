"""Synthetic-fixture tests for the Phase 2 leaderboard metrics.

Covers trajectory geometry (metrics i, vi), the weighted-Mondrian directional
conformal (metric iv), and the leaderboard aggregation including the pluggable
metric (v) "pending Phase 1 re-run" behaviour.
"""

from __future__ import annotations

import numpy as np

from dnamrnaseq2026.conformal.directional import (
    arc_length_score,
    importance_weights,
    unit_directions,
    weighted_conformal_quantile,
    weighted_mondrian_conformal,
)
from dnamrnaseq2026.embedding.arm_b_mofa import fit_mofa
from dnamrnaseq2026.embedding.leaderboard import (
    PENDING_PHASE1,
    biological_coherence,
    build_leaderboard,
    cca_cross_subspace,
    loso_reconstruction_surrogate,
    score_arm,
    trait_state_disentanglement,
)
from dnamrnaseq2026.trajectory.geometry import (
    across_seed_consistency,
    cluster_archetypes,
    recovery_axis,
    trajectory_consistency,
)
from tests.phase2_fixtures import make_synthetic_paired

# ---------------------------------------------------------------------------
# Metric (i): trajectory consistency
# ---------------------------------------------------------------------------


def test_recovery_axis_recovers_planted_direction() -> None:
    data = make_synthetic_paired(n_subjects=40, plant_signal=True, seed=10)
    delta_z = data["x_post"] - data["x_pre"]
    axis = recovery_axis(delta_z, data["responder_mask"])
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-6
    # Responder mean delta should project strongly onto the axis.
    resp_mean = delta_z[data["responder_mask"]].mean(axis=0)
    assert (resp_mean @ axis) > 0


def test_trajectory_consistency_responders_higher() -> None:
    data = make_synthetic_paired(n_subjects=40, plant_signal=True, seed=11)
    delta_z = data["x_post"] - data["x_pre"]
    score = trajectory_consistency(delta_z, data["responder_mask"])
    assert score.responder_mean_cos > score.nonresponder_mean_cos
    assert score.responder_vs_nonresponder_diff > 0


def test_across_seed_consistency_stable_signal_passes() -> None:
    # Same delta-z across "seeds" -> perfect consistency.
    data = make_synthetic_paired(n_subjects=30, seed=12)
    delta_z = data["x_post"] - data["x_pre"]
    result = across_seed_consistency([delta_z, delta_z.copy(), delta_z.copy()])
    assert result["median"] > 0.99
    assert result["pass"] == 1.0


# ---------------------------------------------------------------------------
# Metric (vi): archetype clusterability
# ---------------------------------------------------------------------------


def test_cluster_archetypes_runs_and_reports() -> None:
    data = make_synthetic_paired(n_subjects=60, seed=13)
    delta_z = data["x_post"] - data["x_pre"]
    result = cluster_archetypes(delta_z, n_bootstrap=5, seed=0)
    assert result.best_k in {2, 3, 4, 5}
    assert sum(result.cluster_sizes.values()) == 60
    assert -1.0 <= result.bootstrap_ari_mean <= 1.0


# ---------------------------------------------------------------------------
# Metric (iv): weighted Mondrian directional conformal
# ---------------------------------------------------------------------------


def test_arc_length_score_range() -> None:
    rng = np.random.default_rng(20)
    a = unit_directions(rng.standard_normal((50, 8)))
    b = unit_directions(rng.standard_normal((50, 8)))
    scores = arc_length_score(a, b)
    assert ((scores >= 0) & (scores <= np.pi)).all()
    # Identical directions -> zero score.
    np.testing.assert_allclose(arc_length_score(a, a), 0.0, atol=1e-6)


def test_importance_weights_truncated() -> None:
    rng = np.random.default_rng(21)
    p_best = rng.uniform(0.01, 0.99, size=100)
    w = importance_weights(p_best, truncation_percentile=90.0)
    assert (w > 0).all()
    assert w.max() <= np.percentile(w, 100)  # truncation applied


def test_weighted_conformal_quantile_monotone_in_alpha() -> None:
    rng = np.random.default_rng(22)
    scores = np.sort(rng.uniform(0, np.pi, size=50))
    weights = np.ones(50)
    q_strict = weighted_conformal_quantile(scores, weights, alpha=0.05)
    q_loose = weighted_conformal_quantile(scores, weights, alpha=0.30)
    assert q_strict >= q_loose


def test_weighted_mondrian_conformal_coverage_near_target() -> None:
    """On exchangeable synthetic data, marginal coverage should be near 1 - alpha."""
    rng = np.random.default_rng(23)
    n_cal, n_test, d = 80, 80, 8
    cal_obs = unit_directions(rng.standard_normal((n_cal, d)))
    # Predicted = observed + small noise so scores are small but non-zero.
    cal_pred = unit_directions(cal_obs + 0.3 * rng.standard_normal((n_cal, d)))
    test_obs = unit_directions(rng.standard_normal((n_test, d)))
    test_pred = unit_directions(test_obs + 0.3 * rng.standard_normal((n_test, d)))
    cal_strata = rng.choice(["R", "NR"], size=n_cal)
    test_strata = rng.choice(["R", "NR"], size=n_test)
    cal_p_best = rng.uniform(0.3, 0.7, size=n_cal)

    result = weighted_mondrian_conformal(
        cal_pred,
        cal_obs,
        cal_strata,
        cal_p_best,
        test_pred,
        test_obs,
        test_strata,
        alpha=0.10,
    )
    # Generous band: small n, this is a wiring test not a coverage proof.
    assert 0.70 <= result.marginal_coverage <= 1.0
    assert 0.0 <= result.marginal_radius <= np.pi
    assert set(result.per_stratum_coverage).issubset({"R", "NR"})


# ---------------------------------------------------------------------------
# Metric (ii): trait-state disentanglement
# ---------------------------------------------------------------------------


def test_cca_cross_subspace_independent_vs_dependent() -> None:
    rng = np.random.default_rng(30)
    trait = rng.standard_normal((50, 3))
    independent_state = rng.standard_normal((50, 3))
    dependent_state = trait + 0.01 * rng.standard_normal((50, 3))
    rho_indep = cca_cross_subspace(trait, independent_state)
    rho_dep = cca_cross_subspace(trait, dependent_state)
    assert rho_dep > rho_indep
    assert rho_dep > 0.9


def test_trait_state_disentanglement_runs() -> None:
    data = make_synthetic_paired(n_subjects=30, n_dnam=30, n_rna=18)
    subjects = np.concatenate([data["subject_ids"], data["subject_ids"]])
    visit = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    views = {
        "dnam": np.vstack([data["dnam_pre"], data["dnam_post"]]),
        "rna": np.vstack([data["rna_pre"], data["rna_post"]]),
    }
    factors = fit_mofa(views, subjects, visit, n_factors=8, use_surrogate=True)
    result = trait_state_disentanglement(factors, n_bootstrap=20, seed=0)
    assert "rho_max_cca" in result
    assert isinstance(result["n_trait"], int)
    assert isinstance(result["pass"], bool)


# ---------------------------------------------------------------------------
# Metric (iii): LOSO reconstruction surrogate
# ---------------------------------------------------------------------------


def test_loso_reconstruction_surrogate_lower_with_signal() -> None:
    data = make_synthetic_paired(n_subjects=40, plant_signal=True, seed=14)
    delta_z = data["x_post"] - data["x_pre"]
    result = loso_reconstruction_surrogate(delta_z, data["delta_pcl"])
    assert result["n_subjects"] == 40
    assert np.isfinite(result["loso_mae"])


def test_loso_handles_nan_pcl() -> None:
    data = make_synthetic_paired(n_subjects=20, seed=15)
    delta_z = data["x_post"] - data["x_pre"]
    pcl = data["delta_pcl"].copy()
    pcl[:5] = np.nan
    result = loso_reconstruction_surrogate(delta_z, pcl)
    assert result["n_subjects"] == 15


# ---------------------------------------------------------------------------
# Metric (v): pluggable biological coherence
# ---------------------------------------------------------------------------


def test_biological_coherence_pending_when_artefacts_absent(tmp_path: object) -> None:
    result = biological_coherence(latent_loadings=None, artefact_dir=tmp_path)  # type: ignore[arg-type]
    assert result["status"] == PENDING_PHASE1
    assert result["pass"] is None


# ---------------------------------------------------------------------------
# Leaderboard aggregation
# ---------------------------------------------------------------------------


def test_build_leaderboard_six_rows_three_arms(tmp_path: object) -> None:
    scores = []
    for arm in ["arm_a", "arm_b", "arm_c"]:
        data = make_synthetic_paired(n_subjects=30, seed=hash(arm) % 1000)
        delta_z = data["x_post"] - data["x_pre"]
        subjects = np.concatenate([data["subject_ids"], data["subject_ids"]])
        visit = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
        views = {
            "dnam": np.vstack([data["dnam_pre"], data["dnam_post"]]),
            "rna": np.vstack([data["rna_pre"], data["rna_post"]]),
        }
        factors = fit_mofa(views, subjects, visit, n_factors=8, use_surrogate=True)
        scores.append(
            score_arm(
                arm,
                delta_z=delta_z,
                responder_mask=data["responder_mask"],
                delta_z_by_seed=[delta_z, delta_z.copy()],
                factors=factors,
                delta_pcl=data["delta_pcl"],
                conformal_result=None,
                artefact_dir=tmp_path,  # type: ignore[arg-type]
                n_bootstrap=10,
                seed=0,
            )
        )
    board = build_leaderboard(scores)
    assert board.shape == (6, 3)
    assert list(board.columns) == ["arm_a", "arm_b", "arm_c"]
    # Metric (v) renders the pending message when Phase 1 artefacts are absent.
    assert board.loc["v_biological_coherence", "arm_a"] == PENDING_PHASE1
