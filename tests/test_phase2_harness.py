"""Synthetic-fixture tests for the Phase 2 data harness + feature selection.

No OneDrive, no GPU, no R. Validates subject-level pairing, the GroupKFold
no-leakage guarantee, the inner calibration split, and the two-tier feature
subsetting fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnamrnaseq2026.embedding.data_harness import (
    PairedPreprocessor,
    build_paired_dataset,
    inner_calibration_split,
    normalise_visit,
    subject_level_folds,
)
from dnamrnaseq2026.embedding.feature_selection import (
    resolve_feature_tier,
    variance_filter_dnam,
    variance_filter_rna,
)
from tests.phase2_fixtures import make_synthetic_sample_frame


def test_normalise_visit_aliases() -> None:
    assert normalise_visit("PRE_IOP") == "PRE"
    assert normalise_visit("post_iop") == "POST"
    assert normalise_visit("BL") == "PRE"
    assert normalise_visit("12W") == "POST"
    with pytest.raises(ValueError, match="Unrecognised visit"):
        normalise_visit("FOLLOWUP")


def test_build_paired_dataset_shapes() -> None:
    features, pdata = make_synthetic_sample_frame(n_subjects=20, n_features=50)
    ds = build_paired_dataset(features, pdata)
    assert ds.n_subjects == 20
    assert ds.n_features == 50
    assert ds.x_pre.shape == ds.x_post.shape == (20, 50)
    assert ds.delta_x.shape == (20, 50)
    np.testing.assert_allclose(ds.delta_x, ds.x_post - ds.x_pre)


def test_build_paired_dataset_drops_unpaired() -> None:
    features, pdata = make_synthetic_sample_frame(n_subjects=10, n_features=20)
    # Drop one POST sample -> that subject becomes unpaired.
    drop_id = "SUBJ005_POST_IOP"
    features = features.drop(index=drop_id)
    pdata = pdata.drop(index=drop_id)
    ds = build_paired_dataset(features, pdata)
    assert ds.n_subjects == 9
    assert "SUBJ005" not in ds.subject_ids


def test_subject_level_folds_no_leakage() -> None:
    features, pdata = make_synthetic_sample_frame(n_subjects=25, n_features=30)
    ds = build_paired_dataset(features, pdata)
    folds = subject_level_folds(ds, n_splits=5, seed=7)
    assert len(folds) == 5
    all_test: list[int] = []
    for train_idx, test_idx in folds:
        # No subject index appears in both train and test of the same fold.
        assert set(train_idx).isdisjoint(set(test_idx))
        all_test.extend(test_idx.tolist())
    # Every subject is held out exactly once across the 5 folds.
    assert sorted(all_test) == list(range(ds.n_subjects))


def test_inner_calibration_split_disjoint() -> None:
    train_idx = np.arange(40)
    fit_idx, calib_idx = inner_calibration_split(train_idx, calib_fraction=0.2, seed=1)
    assert set(fit_idx).isdisjoint(set(calib_idx))
    assert len(fit_idx) + len(calib_idx) == 40
    assert len(calib_idx) == 8  # 20% of 40


def test_stacked_observation_layout() -> None:
    features, pdata = make_synthetic_sample_frame(n_subjects=12, n_features=15)
    ds = build_paired_dataset(features, pdata)
    x, subj, visit = ds.stacked()
    assert x.shape == (24, 15)
    # Row i and row i + n_subjects are the same subject (PRE then POST).
    assert subj[0] == subj[ds.n_subjects]
    assert visit[0] == 0 and visit[ds.n_subjects] == 1


def test_variance_filter_dnam_respects_blacklist_and_range() -> None:
    rng = np.random.default_rng(0)
    bvals = pd.DataFrame(
        rng.uniform(0, 1, size=(100, 10)),
        index=[f"cg{i:08d}" for i in range(100)],
    )
    # Make 10 probes near-constant -> should be dropped by the range floor.
    bvals.iloc[:10] = 0.5
    blacklist = {bvals.index[20], bvals.index[21]}
    selected = variance_filter_dnam(bvals, top_n=30, cross_reactive=blacklist)
    assert len(selected) == 30
    assert not (blacklist & set(selected))
    assert not (set(bvals.index[:10]) & set(selected))


def test_variance_filter_rna_top_n() -> None:
    rng = np.random.default_rng(1)
    expr = pd.DataFrame(
        rng.standard_normal((200, 8)),
        index=[f"GENE{i}" for i in range(200)],
    )
    selected = variance_filter_rna(expr, top_n=50)
    assert len(selected) == 50


def test_paired_preprocessor_selection_uses_training_fold_only() -> None:
    """Tier 2 variance/HVG ranking is fit on the training fold, not the full set.

    Design Section 4.2 hard rule: ranking on the full cohort lets held-out test
    rows decide which features exist. This builds data where the top-variance
    features DIFFER between the training subset and the full set, then asserts
    the preprocessor's selection follows the training subset.
    """
    rng = np.random.default_rng(11)
    n_samples, n_dnam, n_rna = 40, 12, 12
    dnam = pd.DataFrame(
        rng.standard_normal((n_samples, n_dnam)),
        columns=[f"cg{i:02d}" for i in range(n_dnam)],
    )
    rna = pd.DataFrame(
        rng.standard_normal((n_samples, n_rna)),
        columns=[f"GENE{i:02d}" for i in range(n_rna)],
    )
    train_idx = np.arange(20)
    test_idx = np.arange(20, 40)
    # Inflate variance of two features ONLY in the test-fold rows. If selection
    # leaked, these would rank top; fit on the training fold, they must not.
    dnam.iloc[test_idx, 0] *= 50.0
    rna.iloc[test_idx, 0] *= 50.0

    prep = PairedPreprocessor(tier2_dnam_top=4, tier2_rna_top=4)
    prep.fit(dnam.iloc[train_idx], rna.iloc[train_idx])

    assert len(prep.dnam_features) == 4
    assert len(prep.rna_features) == 4
    # The test-fold-inflated feature must NOT be selected -- that would be a leak.
    assert "cg00" not in prep.dnam_features
    assert "GENE00" not in prep.rna_features

    # transform applies the frozen training selection to the held-out fold.
    dnam_te, rna_te = prep.transform(dnam.iloc[test_idx], rna.iloc[test_idx])
    assert list(dnam_te.columns) == prep.dnam_features
    assert list(rna_te.columns) == prep.rna_features
    assert dnam_te.shape == (20, 4)


def test_paired_preprocessor_transform_before_fit_raises() -> None:
    """transform before fit is a programming error and must fail loud."""
    prep = PairedPreprocessor()
    empty = pd.DataFrame()
    with pytest.raises(RuntimeError, match="before fit"):
        prep.transform(empty, empty)


def test_resolve_feature_tier_falls_back_to_tier2(tmp_path: object) -> None:
    """With no Phase 1 CellDMC artefact present, resolution falls to Tier 2."""
    rng = np.random.default_rng(2)
    bvals = pd.DataFrame(
        rng.uniform(0, 1, size=(80, 10)),
        index=[f"cg{i:08d}" for i in range(80)],
    )
    expr = pd.DataFrame(
        rng.standard_normal((120, 10)),
        index=[f"GENE{i}" for i in range(120)],
    )
    tier = resolve_feature_tier(bvals, expr, artefact_dir=tmp_path)  # type: ignore[arg-type]
    assert tier.tier == 2
    assert "Tier 2 fallback" in tier.rationale
    assert len(tier.dnam_cpgs) > 0
    assert len(tier.rna_genes) > 0
