"""Tests for the Arm A/C end-to-end runner (training + leaderboard scoring).

The GPU training path itself is exercised by the real-data run on the 5090;
these tests cover the pure, CPU-fast pieces: the sample-visit key construction
for leakage-safe per-fold TF selection, the latent-embedding -> MOFAFactors
adapter that feeds metric (ii), and the leakage-safe ``tf_rank_keys`` argument
on ``build_rna_activity_matrix``. No real data, no GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnamrnaseq2026.embedding.arm_ac_run import (
    _sample_visit_keys,
    _training_tf_rank_keys,
    embedding_to_factors,
)
from dnamrnaseq2026.embedding.arm_b_mofa import MOFAFactors
from dnamrnaseq2026.embedding.leaderboard import build_leaderboard, score_arm


def test_sample_visit_keys_use_iop_suffix() -> None:
    """Keys are the {Subcode}-PRE-IOP / -POST-IOP activity-matrix form."""
    pre, post = _sample_visit_keys("AMC-123")
    assert pre == "AMC-123-PRE-IOP"
    assert post == "AMC-123-POST-IOP"


def test_training_tf_rank_keys_excludes_held_out_subject() -> None:
    """The TF-rank key list drops both visits of the held-out subject."""
    subjects = np.array(["S1", "S2", "S3"], dtype=object)
    keys = _training_tf_rank_keys(subjects, held="S2")
    assert "S2-PRE-IOP" not in keys and "S2-POST-IOP" not in keys
    assert set(keys) == {"S1-PRE-IOP", "S1-POST-IOP", "S3-PRE-IOP", "S3-POST-IOP"}


def test_training_tf_rank_keys_none_keeps_every_subject() -> None:
    """held=None ranks over every subject (the cohort-wide descriptive path)."""
    subjects = np.array(["S1", "S2"], dtype=object)
    keys = _training_tf_rank_keys(subjects, held=None)
    assert len(keys) == 4


def test_embedding_to_factors_stacks_pre_then_post() -> None:
    """The adapter stacks PRE rows then POST rows, observation-major."""
    rng = np.random.default_rng(0)
    n, d = 5, 8
    z_pre = rng.standard_normal((n, d))
    z_post = rng.standard_normal((n, d))
    subjects = np.array([f"S{i}" for i in range(n)], dtype=object)
    factors = embedding_to_factors(z_pre, z_post, subjects)
    assert isinstance(factors, MOFAFactors)
    assert factors.scores.shape == (2 * n, d)
    assert np.array_equal(factors.scores[:n], z_pre)
    assert np.array_equal(factors.scores[n:], z_post)
    assert list(factors.visit[:n]) == [0] * n
    assert list(factors.visit[n:]) == [1] * n
    assert factors.n_factors == d


def test_embedding_to_factors_feeds_score_arm() -> None:
    """A neural embedding wrapped as MOFAFactors scores on the leaderboard."""
    rng = np.random.default_rng(1)
    n, d = 30, 6
    z_pre = rng.standard_normal((n, d))
    z_post = z_pre + 0.3 * rng.standard_normal((n, d))
    subjects = np.array([f"S{i}" for i in range(n)], dtype=object)
    responder = rng.random(n) < 0.5
    delta_pcl = rng.standard_normal(n) * 10
    factors = embedding_to_factors(z_pre, z_post, subjects)
    delta_z = z_post - z_pre
    score = score_arm(
        "arm_a_fm",
        delta_z=delta_z,
        responder_mask=responder,
        delta_z_by_seed=[delta_z, delta_z + 0.01],
        factors=factors,
        delta_pcl=delta_pcl,
        n_bootstrap=50,
        seed=1,
    )
    board = build_leaderboard([score])
    assert board.shape == (6, 1)
    assert "arm_a_fm" in board.columns
    # Metric (ii) must classify against the ICC continuum without erroring.
    assert "ii_trait_state_disentanglement" in score.metrics


def test_build_rna_activity_matrix_tf_rank_keys_is_leakage_safe() -> None:
    """tf_rank_keys ranks the TF panel on training rows only.

    A synthetic TF matrix is built so one TF is high-variance ONLY on the
    held-out rows. With cohort-wide ranking that TF would be selected; with a
    training-row key list it is correctly dropped.
    """
    from dnamrnaseq2026.embedding import real_data

    keys = [f"S{i}-PRE-IOP" for i in range(10)] + [f"S{i}-POST-IOP" for i in range(10)]
    progeny = pd.DataFrame(np.zeros((20, 2)), index=keys, columns=["p0", "p1"])
    # tf_quiet is flat on training rows; tf_spike spikes only on the last 4 rows.
    tf = pd.DataFrame(index=keys)
    tf["tf_quiet"] = np.linspace(0, 1, 20)
    spike = np.zeros(20)
    spike[-4:] = [100.0, -100.0, 100.0, -100.0]
    tf["tf_spike"] = spike

    captured: dict[str, pd.DataFrame] = {}

    def _fake_read(path: object, what: str) -> pd.DataFrame:
        return progeny.copy() if "PROGENy" in what else tf.copy()

    orig = real_data._read_activity
    real_data._read_activity = _fake_read  # type: ignore[assignment]
    try:
        train_keys = keys[:-4]  # exclude the 4 spike rows
        out = real_data.build_rna_activity_matrix(
            pd.DataFrame(), top_tf_by_variance=1, tf_rank_keys=train_keys
        )
        captured["train"] = out
        out_cohort = real_data.build_rna_activity_matrix(
            pd.DataFrame(), top_tf_by_variance=1, tf_rank_keys=None
        )
        captured["cohort"] = out_cohort
    finally:
        real_data._read_activity = orig  # type: ignore[assignment]

    # On training rows tf_quiet has more variance than the (flat-on-train)
    # tf_spike, so the leakage-safe path keeps tf_quiet.
    assert "tf_quiet" in captured["train"].columns
    assert "tf_spike" not in captured["train"].columns
    # Cohort-wide ranking is dominated by the spike rows and would pick tf_spike,
    # which is exactly the leak the keyed path avoids.
    assert "tf_spike" in captured["cohort"].columns


def test_build_rna_activity_matrix_rejects_unmatched_keys() -> None:
    """A key list that matches <2 TF rows raises a precise error."""
    from dnamrnaseq2026.embedding import real_data
    from dnamrnaseq2026.embedding.real_data import Phase1ArtefactError

    keys = [f"S{i}-PRE-IOP" for i in range(5)]
    progeny = pd.DataFrame(np.zeros((5, 1)), index=keys, columns=["p0"])
    tf = pd.DataFrame({"tf0": np.arange(5.0)}, index=keys)

    def _fake_read(path: object, what: str) -> pd.DataFrame:
        return progeny.copy() if "PROGENy" in what else tf.copy()

    orig = real_data._read_activity
    real_data._read_activity = _fake_read  # type: ignore[assignment]
    try:
        with pytest.raises(Phase1ArtefactError, match="TF variance rank needs"):
            real_data.build_rna_activity_matrix(pd.DataFrame(), tf_rank_keys=["NO-MATCH-PRE-IOP"])
    finally:
        real_data._read_activity = orig  # type: ignore[assignment]
