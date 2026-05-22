"""Synthetic-fixture tests for the Arm B real-data run helpers.

Covers the covariate-residualisation, JAK-STAT outlier flagging, and the MOFA+
expectations-extraction fix, without needing the real Phase 1 artefacts. The
end-to-end real-data path is exercised by ``scripts/23_phase2_arm_b_run.py`` and
is not unit-tested here (it depends on ``analysis/latest/``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dnamrnaseq2026.embedding.arm_b_mofa import classify_factors, fit_mofa
from dnamrnaseq2026.embedding.arm_b_run import (
    N_PROGENY_PATHWAYS,
    TOP_TF_BY_VARIANCE,
    ArmBData,
    _build_covariate_matrix,
    _flag_jakstat_outliers,
    _residualise,
)
from dnamrnaseq2026.embedding.feature_selection import select_top_tf_by_variance


def test_residualise_removes_covariate_signal() -> None:
    """A covariate-driven view has its covariate variance removed by residualising."""
    rng = np.random.default_rng(0)
    n = 80
    cov = rng.normal(size=(n, 2))
    # view column 0 is pure covariate signal; column 1 is noise.
    view = np.column_stack([3.0 * cov[:, 0] - 2.0 * cov[:, 1], rng.normal(size=n)])
    resid = _residualise(view, cov)
    # Residual of the covariate-driven column is near-zero variance.
    assert resid[:, 0].std() < 1e-8
    # The noise column survives.
    assert resid[:, 1].std() > 0.5


def test_residualise_preserves_shape() -> None:
    """Residualising returns a matrix of the same shape as the input view."""
    rng = np.random.default_rng(1)
    view = rng.normal(size=(40, 12))
    cov = rng.normal(size=(40, 3))
    assert _residualise(view, cov).shape == view.shape


def test_flag_jakstat_outliers_picks_high_tail() -> None:
    """The JAK-STAT flag marks the upper-tail (z > 2) sample-visits only."""
    values = np.concatenate([np.zeros(95), np.full(5, 10.0)])
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "TNFa": rng.normal(size=100),
            "JAK-STAT": values,
        }
    )
    mask = _flag_jakstat_outliers(df)
    assert mask.sum() == 5
    assert mask[-5:].all()
    assert not mask[:95].any()


def test_flag_jakstat_outliers_absent_column_is_all_false() -> None:
    """When no JAK-STAT column exists, the flag is all-False (matrix is a superset)."""
    df = pd.DataFrame({"TNFa": [0.1, 0.2, 0.3], "NFkB": [0.4, 0.5, 0.6]})
    assert not _flag_jakstat_outliers(df).any()


def test_build_covariate_matrix_mean_imputes() -> None:
    """Sparse missing covariate cells are mean-imputed; sex is required."""
    pdata = pd.DataFrame(
        {
            "sex": [1, 2, 1, 2],
            "Age": [30.0, np.nan, 40.0, 50.0],
            "ancestry_pca_PCA1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    cov = _build_covariate_matrix(pdata)
    assert cov.shape == (4, 3)
    assert np.isfinite(cov).all()
    # The imputed Age cell takes the column mean of the observed values.
    assert np.isclose(cov[1, 1], np.mean([30.0, 40.0, 50.0]))


def test_select_top_tf_by_variance_keeps_progeny_and_top_tfs() -> None:
    """TF selection keeps all PROGENy columns and the highest-variance TFs.

    This is the Tier 1 RNA leakage fix (Helen Zhao 2026-05-22): the TF panel is
    a variance rank, so it must be selectable per fold rather than baked
    cohort-wide. PROGENy's fixed columns are always kept.
    """
    rng = np.random.default_rng(7)
    n = 50
    # 3 PROGENy columns (low variance) + 6 TF columns with increasing variance.
    cols = {f"progeny_{i}": rng.normal(scale=0.01, size=n) for i in range(3)}
    for j in range(6):
        cols[f"TF_{j}"] = rng.normal(scale=float(j + 1), size=n)
    df = pd.DataFrame(cols)
    keep = select_top_tf_by_variance(df, n_pathway=3, top_tf_by_variance=2)
    # All 3 PROGENy columns kept, plus the 2 highest-variance TFs (TF_4, TF_5).
    assert keep[:3] == ["progeny_0", "progeny_1", "progeny_2"]
    assert set(keep[3:]) == {"TF_4", "TF_5"}


def test_select_tf_panel_is_fold_dependent() -> None:
    """ArmBData.select_tf_panel ranks TFs on the masked rows, so it is fold-aware.

    Two disjoint row subsets with deliberately different TF variance structure
    select different TF panels: this is exactly the per-fold behaviour the
    leakage fix requires (a cohort-wide rank would be fold-invariant).
    """
    rng = np.random.default_rng(11)
    n = 60
    n_tf = TOP_TF_BY_VARIANCE + 10
    rna = np.zeros((n, N_PROGENY_PATHWAYS + n_tf))
    rna[:, :N_PROGENY_PATHWAYS] = rng.normal(scale=0.01, size=(n, N_PROGENY_PATHWAYS))
    rna[:, N_PROGENY_PATHWAYS:] = rng.normal(size=(n, n_tf))
    # First half: inflate variance of the last 10 TFs. Second half: inflate the first 10.
    first = np.arange(n) < n // 2
    rna[np.ix_(first, np.arange(N_PROGENY_PATHWAYS + n_tf - 10, N_PROGENY_PATHWAYS + n_tf))] *= 50
    rna[np.ix_(~first, np.arange(N_PROGENY_PATHWAYS, N_PROGENY_PATHWAYS + 10))] *= 50
    cols = np.array(
        [f"P{i}" for i in range(N_PROGENY_PATHWAYS)] + [f"TF{j}" for j in range(n_tf)],
        dtype=object,
    )
    data = ArmBData(
        dnam=rng.normal(size=(n, 5)),
        rna=rna,
        rna_columns=cols,
        subject_ids=np.repeat(np.arange(n // 2), 2),
        visit=np.tile([0, 1], n // 2),
        sentrix_ids=np.arange(n).astype(object),
        jakstat_outlier=np.zeros(n, dtype=bool),
    )
    panel_first = data.select_tf_panel(row_mask=first)
    panel_second = data.select_tf_panel(row_mask=~first)
    # Both panels keep the 14 PROGENy + 150 TF columns.
    assert panel_first.shape[1] == N_PROGENY_PATHWAYS + TOP_TF_BY_VARIANCE
    # The two folds select genuinely different TF panels (not bit-identical).
    assert not np.array_equal(panel_first, panel_second)


def test_mofa_fit_extracts_factor_scores() -> None:
    """The mofapy2 expectations fix yields a finite (n_obs, K) score matrix.

    Uses data with genuine shared latent structure (a low-rank signal plus
    subject-level random intercepts) so the MOFA+ ELBO and the downstream LMM
    fits are well-conditioned, mirroring the real multi-omics regime.
    """
    rng = np.random.default_rng(3)
    n_subj = 40
    n_obs = n_subj * 2
    subject_ids = np.repeat(np.arange(n_subj), 2)
    visit = np.tile([0, 1], n_subj)
    # Three shared latent factors with a subject-level component.
    k_true = 3
    subj_factor = rng.normal(size=(n_subj, k_true))
    z = subj_factor[subject_ids] + 0.3 * rng.normal(size=(n_obs, k_true))
    w_dnam = rng.normal(size=(k_true, 40))
    w_rna = rng.normal(size=(k_true, 25))
    views = {
        "dnam": z @ w_dnam + 0.1 * rng.normal(size=(n_obs, 40)),
        "rna": z @ w_rna + 0.1 * rng.normal(size=(n_obs, 25)),
    }
    factors = fit_mofa(views, subject_ids, visit, n_factors=5, seed=1, use_surrogate=False)
    assert factors.scores.shape[0] == n_obs
    assert factors.n_factors >= 1
    assert np.isfinite(factors.scores).all()
    classified = classify_factors(factors, n_bootstrap=50, seed=1)
    assert len(classified) == factors.n_factors
    assert set(classified["classification"]).issubset({"trait", "state", "mixed"})
