"""Synthetic-fixture tests for Phase 3 external-cohort projection pipeline.

Tests the projection of GTEx (healthy) and GSE98793 TRD-inflammatory cohorts
into the atlas latent space, and the construction of the two-anchor recovery
axis. All tests run against a synthetic atlas encoder fixture that mimics the
interface the Phase 2 winning arm will satisfy.

No real data, no OneDrive, no GPU. All tests run in CI.

Output interface for Helen's proximity test
--------------------------------------------
These tests confirm that ProjectionResult has the correct structure and
field types/shapes that test_phase3_proximity.py will consume. If this test
suite stays green, Helen's proximity test can import ProjectionResult and
rely on the documented contract.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dnamrnaseq2026.trajectory.external_projection import (
    ExternalCohortData,
    ProjectionResult,
    PtsdAtlasData,
    build_two_anchor_recovery_axis,
    load_projection_result,
    project_external_cohorts,
    project_onto_recovery_axis,
    save_projection_result,
)

# ---------------------------------------------------------------------------
# Synthetic atlas encoder fixture
# ---------------------------------------------------------------------------


class _SyntheticAtlasEncoder:
    """Synthetic encoder that mimics the Phase 2 winning arm's RNA encoder.

    Maps (n_samples, n_rna_features) -> (n_samples, d_latent) via a fixed
    random linear projection. The projection is seeded so the GTEx and TRD
    clouds map to structurally distinct regions of the latent space, making
    the recovery-axis test non-trivial.

    The planted structure:
    - GTEx inputs have feature means near +1 -> project to latent coords
      with a positive bias in the first few dimensions.
    - GSE98793 TRD inputs have feature means near -1 -> project to latent
      coords with a negative bias, ensuring gtex_centroid != gse_trd_centroid
      so the recovery axis is non-degenerate.
    """

    def __init__(self, d_rna_in: int = 50, d_latent: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._W = rng.standard_normal((d_rna_in, d_latent)).astype(np.float64)
        self._W /= np.linalg.norm(self._W, axis=0, keepdims=True)
        self.d_latent = d_latent

    def encode(self, rna_matrix: np.ndarray) -> np.ndarray:
        return (rna_matrix @ self._W).astype(np.float64)


# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_external_data(
    n_gtex: int = 30,
    n_gse_trd: int = 20,
    n_rna: int = 50,
    *,
    seed: int = 0,
) -> ExternalCohortData:
    """Build synthetic external-cohort data with planted GTEx/TRD separation.

    GTEx samples have positive feature means (healthy); TRD samples have
    negative feature means (pathological). This ensures a non-degenerate
    recovery axis after projection.
    """
    rng = np.random.default_rng(seed)
    # Positive bias -> healthy
    gtex_rna = rng.standard_normal((n_gtex, n_rna)) + 2.0
    gtex_ids = np.array([f"GTEX_{i:04d}" for i in range(n_gtex)], dtype=object)
    # Negative bias -> TRD-inflammatory
    gse_trd_rna = rng.standard_normal((n_gse_trd, n_rna)) - 2.0
    gse_trd_ids = np.array([f"GSE_{i:04d}" for i in range(n_gse_trd)], dtype=object)
    gse_trd_centroid_rna = gse_trd_rna.mean(axis=0)
    feature_names = np.array([f"GENE_{j}" for j in range(n_rna)], dtype=object)
    return ExternalCohortData(
        gtex_rna=gtex_rna.astype(np.float64),
        gtex_sample_ids=gtex_ids,
        gse_trd_rna=gse_trd_rna.astype(np.float64),
        gse_trd_sample_ids=gse_trd_ids,
        gse_trd_centroid_rna=gse_trd_centroid_rna.astype(np.float64),
        feature_names=feature_names,
    )


def _make_ptsd_data(
    n_subjects: int = 40,
    d_latent: int = 16,
    *,
    seed: int = 7,
    n_responders: int = 20,
    recovery_axis_vec: np.ndarray | None = None,
) -> PtsdAtlasData:
    """Build synthetic PTSD trajectory-terminus data with planted signal.

    Responders' termini are biased along ``recovery_axis_vec`` (toward health);
    non-responders' termini are biased against it (toward TRD). If
    ``recovery_axis_vec`` is None a canonical axis (first standard basis vector)
    is used; the planted signal tests always pass an axis computed from the
    encoder and reference clouds so the bias is guaranteed to align.
    """
    rng = np.random.default_rng(seed)
    terminus_latent = rng.standard_normal((n_subjects, d_latent)).astype(np.float64)
    response = np.array(["R"] * n_responders + ["NR"] * (n_subjects - n_responders), dtype=object)

    if recovery_axis_vec is not None:
        # Plant signal along the actual recovery axis direction
        axis = recovery_axis_vec / (np.linalg.norm(recovery_axis_vec) + 1e-12)
        terminus_latent[:n_responders] += 3.0 * axis[None, :]
        terminus_latent[n_responders:] -= 3.0 * axis[None, :]
    else:
        # Fallback: bias along latent dim 0 (used for shape/type tests only)
        terminus_latent[:n_responders, 0] += 3.0
        terminus_latent[n_responders:, 0] -= 3.0

    subject_ids = np.array([f"SUBJ{i:04d}" for i in range(n_subjects)], dtype=object)
    cohort = np.array(["Emory"] * n_subjects, dtype=object)
    return PtsdAtlasData(
        terminus_latent=terminus_latent,
        subject_ids=subject_ids,
        response=response,
        cohort=cohort,
    )


# ---------------------------------------------------------------------------
# Tests: two-anchor recovery axis
# ---------------------------------------------------------------------------


def test_recovery_axis_unit_norm() -> None:
    """Axis vector must be unit-length."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    gtex_lat = enc.encode(ext.gtex_rna)
    gse_lat = enc.encode(ext.gse_trd_rna)
    axis, _, _ = build_two_anchor_recovery_axis(gtex_lat, gse_lat)
    assert abs(float(np.linalg.norm(axis)) - 1.0) < 1e-9


def test_recovery_axis_direction() -> None:
    """Axis should point FROM TRD centroid TOWARD GTEx centroid.

    With planted GTEx/TRD separation, projecting gtex_centroid onto the axis
    should give a positive score and gse_trd_centroid a negative score.
    """
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    gtex_lat = enc.encode(ext.gtex_rna)
    gse_lat = enc.encode(ext.gse_trd_rna)
    axis, gtex_c, gse_c = build_two_anchor_recovery_axis(gtex_lat, gse_lat)
    assert float(gtex_c @ axis) > float(gse_c @ axis), (
        "GTEx centroid should project higher than TRD centroid onto the recovery axis"
    )


def test_recovery_axis_degenerate_raises() -> None:
    """Identical cloud centroids must raise ValueError, not produce NaN."""
    d = 8
    cloud = np.ones((10, d))
    with pytest.raises(ValueError, match="degenerate"):
        build_two_anchor_recovery_axis(cloud, cloud)


def test_recovery_axis_too_few_gtex_raises() -> None:
    """GTEx cloud below minimum sample count raises ValueError."""
    rng = np.random.default_rng(1)
    small = rng.standard_normal((3, 8))
    large = rng.standard_normal((20, 8))
    with pytest.raises(ValueError, match="GTEx cloud"):
        build_two_anchor_recovery_axis(small, large)


def test_recovery_axis_too_few_trd_raises() -> None:
    """TRD cloud below minimum sample count raises ValueError."""
    rng = np.random.default_rng(2)
    large = rng.standard_normal((20, 8))
    small = rng.standard_normal((2, 8))
    with pytest.raises(ValueError, match="GSE98793 TRD cloud"):
        build_two_anchor_recovery_axis(large, small)


# ---------------------------------------------------------------------------
# Tests: recovery score projection
# ---------------------------------------------------------------------------


def test_recovery_scores_shape() -> None:
    """Scores must have shape (n_subjects,)."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    gtex_lat = enc.encode(ext.gtex_rna)
    gse_lat = enc.encode(ext.gse_trd_rna)
    axis, _, _ = build_two_anchor_recovery_axis(gtex_lat, gse_lat)
    scores = project_onto_recovery_axis(ptsd.terminus_latent, axis)
    assert scores.shape == (ptsd.terminus_latent.shape[0],)


def test_recovery_scores_planted_signal() -> None:
    """Responders should have higher mean recovery score than non-responders.

    The PTSD terminus fixture plants a bias *along* the computed recovery axis
    (R toward healthy, NR toward TRD) so the test is guaranteed to be
    non-trivially satisfied regardless of the random encoder projection.
    """
    ext = _make_external_data(n_gtex=50, n_gse_trd=30, seed=0)
    enc = _SyntheticAtlasEncoder()
    gtex_lat = enc.encode(ext.gtex_rna)
    gse_lat = enc.encode(ext.gse_trd_rna)
    axis, _, _ = build_two_anchor_recovery_axis(gtex_lat, gse_lat)
    # Pass the recovery axis so the planted bias is axis-aligned
    ptsd = _make_ptsd_data(n_subjects=40, d_latent=enc.d_latent, seed=7, recovery_axis_vec=axis)
    scores = project_onto_recovery_axis(ptsd.terminus_latent, axis)
    r_mean = float(scores[ptsd.response == "R"].mean())
    nr_mean = float(scores[ptsd.response == "NR"].mean())
    assert r_mean > nr_mean, (
        f"Expected responders (mean={r_mean:.3f}) to score higher than "
        f"non-responders (mean={nr_mean:.3f}) on the recovery axis"
    )


def test_recovery_scores_dimension_mismatch_raises() -> None:
    """Mismatched latent dimension between scores and axis must raise."""
    rng = np.random.default_rng(3)
    latent = rng.standard_normal((10, 8))
    axis = rng.standard_normal(16)
    axis /= np.linalg.norm(axis)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        project_onto_recovery_axis(latent, axis)


# ---------------------------------------------------------------------------
# Tests: full projection pipeline
# ---------------------------------------------------------------------------


def test_project_external_cohorts_result_types() -> None:
    """ProjectionResult fields must have correct types and shapes."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="synthetic_test")
    d = enc.d_latent
    n_subj = ptsd.terminus_latent.shape[0]

    assert isinstance(result, ProjectionResult)
    assert result.terminus_latent.shape == (n_subj, d)
    assert result.gtex_latent.shape == (ext.gtex_rna.shape[0], d)
    assert result.gse_trd_latent.shape == (ext.gse_trd_rna.shape[0], d)
    assert result.recovery_axis.shape == (d,)
    assert result.terminus_recovery_score.shape == (n_subj,)
    assert result.subject_ids.shape == (n_subj,)
    assert result.response.shape == (n_subj,)
    assert result.cohort.shape == (n_subj,)
    assert result.gtex_centroid_latent.shape == (d,)
    assert result.gse_trd_centroid_latent.shape == (d,)
    assert result.n_gtex == ext.gtex_rna.shape[0]
    assert result.n_gse_trd == ext.gse_trd_rna.shape[0]


def test_project_external_cohorts_recovery_axis_unit() -> None:
    """The recovery axis in ProjectionResult must be unit-length."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="synthetic_test")
    assert abs(float(np.linalg.norm(result.recovery_axis)) - 1.0) < 1e-9


def test_project_external_cohorts_provenance_keys() -> None:
    """Provenance dict must contain load-bearing metadata keys."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    result = project_external_cohorts(
        enc,
        ext,
        ptsd,
        encoder_name="synthetic_test",
        gtex_source="test_gtex",
        gse_source="test_gse",
    )
    for key in (
        "encoder",
        "n_rna_features",
        "d_latent",
        "gtex_source",
        "gse_source",
        "n_gtex",
        "n_gse_trd",
        "axis_computation_method",
    ):
        assert key in result.provenance, f"Missing provenance key: {key}"


def test_project_external_cohorts_dimension_mismatch_raises() -> None:
    """PTSD latent dim != encoder latent dim must raise ValueError."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder(d_latent=16)
    # PTSD data has wrong latent dim
    ptsd_bad = _make_ptsd_data(d_latent=8)
    with pytest.raises(ValueError, match="does not match"):
        project_external_cohorts(enc, ext, ptsd_bad, encoder_name="synthetic_test")


def test_project_external_cohorts_planted_signal_preserved() -> None:
    """R subjects should have higher recovery scores than NR with planted signal.

    The PTSD termini are biased *along* the recovery axis (computed first from the
    projected reference clouds) so the test is encoder-agnostic.
    """
    ext = _make_external_data(n_gtex=50, n_gse_trd=30, seed=0)
    enc = _SyntheticAtlasEncoder()
    # Compute the recovery axis from the projected reference clouds
    gtex_lat = enc.encode(ext.gtex_rna)
    gse_lat = enc.encode(ext.gse_trd_rna)
    axis, _, _ = build_two_anchor_recovery_axis(gtex_lat, gse_lat)
    # Plant terminus bias along that axis
    ptsd = _make_ptsd_data(
        n_subjects=40,
        d_latent=enc.d_latent,
        seed=7,
        n_responders=20,
        recovery_axis_vec=axis,
    )
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="synthetic_planted")
    r_scores = result.terminus_recovery_score[result.response == "R"]
    nr_scores = result.terminus_recovery_score[result.response == "NR"]
    assert r_scores.mean() > nr_scores.mean(), (
        f"Planted signal not captured: R mean={r_scores.mean():.3f}, NR mean={nr_scores.mean():.3f}"
    )


# ---------------------------------------------------------------------------
# Tests: serialise / deserialise round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip() -> None:
    """ProjectionResult serialised then loaded must reproduce all arrays.

    Requires pyarrow (pandas parquet engine). Skipped in minimal CI
    environments that do not have pyarrow installed; runs locally and in
    the full integration suite.
    """
    pytest.importorskip("pyarrow")
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="round_trip_test")

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        save_projection_result(result, out)

        # Check expected files are written
        assert (out / "terminus_coords.parquet").exists()
        assert (out / "reference_clouds.parquet").exists()
        assert (out / "recovery_axis.npy").exists()
        assert (out / "provenance.json").exists()

        loaded = load_projection_result(out)

    # Shape checks
    assert loaded.terminus_latent.shape == result.terminus_latent.shape
    assert loaded.gtex_latent.shape == result.gtex_latent.shape
    assert loaded.gse_trd_latent.shape == result.gse_trd_latent.shape
    assert loaded.recovery_axis.shape == result.recovery_axis.shape
    assert loaded.terminus_recovery_score.shape == result.terminus_recovery_score.shape

    # Numerical fidelity
    np.testing.assert_allclose(loaded.recovery_axis, result.recovery_axis, rtol=1e-6)
    np.testing.assert_allclose(
        loaded.terminus_recovery_score, result.terminus_recovery_score, rtol=1e-6
    )
    np.testing.assert_allclose(loaded.gtex_centroid_latent, result.gtex_centroid_latent, rtol=1e-6)
    np.testing.assert_allclose(
        loaded.gse_trd_centroid_latent, result.gse_trd_centroid_latent, rtol=1e-6
    )

    # Provenance preserved
    assert loaded.provenance["encoder"] == "round_trip_test"
    assert loaded.n_gtex == result.n_gtex
    assert loaded.n_gse_trd == result.n_gse_trd


def test_provenance_json_serialisable() -> None:
    """Provenance dict must be JSON-serialisable (no raw numpy scalars)."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    ptsd = _make_ptsd_data(d_latent=enc.d_latent)
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="json_test")
    # Should not raise
    _ = json.dumps(result.provenance)


# ---------------------------------------------------------------------------
# Tests: output interface completeness for Helen's proximity test
# ---------------------------------------------------------------------------


def test_projection_result_output_interface() -> None:
    """Confirm all fields documented in the output interface contract are present
    and have the correct dtype.

    This is the contract test that Helen's proximity test relies on. If these
    assertions pass, test_phase3_proximity.py can import ProjectionResult and
    use terminus_recovery_score, subject_ids, and response directly.
    """
    ext = _make_external_data(n_gtex=30, n_gse_trd=20)
    enc = _SyntheticAtlasEncoder(d_latent=12)
    ptsd = _make_ptsd_data(n_subjects=30, d_latent=12)
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="contract_test")

    # Required fields for proximity test
    assert result.terminus_latent.dtype == np.float64
    assert result.recovery_axis.dtype == np.float64
    assert result.terminus_recovery_score.dtype == np.float64
    assert result.gtex_latent.dtype == np.float64
    assert result.gse_trd_latent.dtype == np.float64

    # Subject alignment: recovery_score[i] corresponds to subject_ids[i] / response[i]
    assert len(result.terminus_recovery_score) == len(result.subject_ids)
    assert len(result.terminus_recovery_score) == len(result.response)
    assert len(result.terminus_recovery_score) == len(result.cohort)

    # Proximity test needs to group by response
    r_mask = result.response == "R"
    nr_mask = result.response == "NR"
    assert r_mask.sum() + nr_mask.sum() == len(result.response), (
        "All response labels must be 'R' or 'NR' in this fixture"
    )


def test_projection_result_with_mixed_cohort() -> None:
    """ProjectionResult must handle mixed Emory + BEST cohorts gracefully."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    rng = np.random.default_rng(99)
    d = enc.d_latent
    n = 30
    terminus = rng.standard_normal((n, d)).astype(np.float64)
    terminus[:15, 0] += 3.0  # R
    terminus[15:, 0] -= 3.0  # NR
    cohort = np.array(["Emory"] * 20 + ["BEST"] * 10, dtype=object)
    response = np.array(["R"] * 15 + ["NR"] * 15, dtype=object)
    subject_ids = np.array([f"MIX{i:03d}" for i in range(n)], dtype=object)
    ptsd = PtsdAtlasData(
        terminus_latent=terminus,
        subject_ids=subject_ids,
        response=response,
        cohort=cohort,
    )
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="mixed_cohort_test")
    assert result.cohort.tolist() == cohort.tolist()
    assert result.n_gtex == ext.gtex_rna.shape[0]


def test_projection_result_partial_response_labels() -> None:
    """Recovery scores must work with BEST 3-class integer response labels."""
    ext = _make_external_data()
    enc = _SyntheticAtlasEncoder()
    rng = np.random.default_rng(55)
    d = enc.d_latent
    n = 30
    terminus = rng.standard_normal((n, d)).astype(np.float64)
    response = np.array([1, 2, 3] * 10, dtype=object)  # BEST partial-response categories
    subject_ids = np.array([f"BEST{i:03d}" for i in range(n)], dtype=object)
    cohort = np.array(["BEST"] * n, dtype=object)
    ptsd = PtsdAtlasData(
        terminus_latent=terminus,
        subject_ids=subject_ids,
        response=response,
        cohort=cohort,
    )
    result = project_external_cohorts(enc, ext, ptsd, encoder_name="partial_response_test")
    assert result.terminus_recovery_score.shape == (n,)
    # All values finite
    assert np.all(np.isfinite(result.terminus_recovery_score))
