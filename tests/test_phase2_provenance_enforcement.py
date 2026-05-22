"""Consumer-side enforcement tests for the Tier 2 ``cv_loop_safe`` stamp.

These run UNCONDITIONALLY in default CI -- no OneDrive, no ``analysis/latest/``,
and no parquet engine: the provenance check reads only the ``.provenance.json``
sidecar (JSON, no engine needed). The one test that exercises the parquet read
path of ``load_feature_matrix_for_cv`` monkeypatches ``pandas.read_parquet`` so
the CI smoke-test job (pandas/numpy/scipy only, no pyarrow) still covers it.

Motivation: ``scripts/22_phase2_build_feature_matrices.py`` writes a
``cv_loop_safe = False`` provenance marker beside the Tier 2 *candidate*
matrices, because the data-driven variance / HVG top-N ranking has not been
applied and must be fit per training fold by ``PairedPreprocessor`` (design doc
Section 4.2). The stamp is theatre unless a consumer enforces it. These tests
prove the enforcement fires: a ``cv_loop_safe=False`` matrix entering the CV /
training path raises, and a ``cv_loop_safe=True`` matrix does not. A missing
sidecar fails closed (refuse), per the documented fail-closed policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dnamrnaseq2026.embedding.feature_selection import (
    assert_cv_loop_safe,
    load_feature_matrix_for_cv,
)
from dnamrnaseq2026.embedding.real_data import Phase1ArtefactError

# Mirrors scripts/22_phase2_build_feature_matrices.py TIER2_PROVENANCE: the
# writer stamps cv_loop_safe as a JSON *string* "False".
_TIER2_CANDIDATE_PROVENANCE = {
    "selection_stage": "EDA_ONLY",
    "cv_loop_safe": "False",
    "note": "Biology-filtered Tier 2 candidate set; per-fold ranking pending.",
}
_CV_SAFE_PROVENANCE = {
    "selection_stage": "PER_FOLD_PAIRED_PREPROCESSOR",
    "cv_loop_safe": "True",
    "note": "Frozen per-fold selection; safe to admit to a CV path.",
}


def _write_matrix_with_provenance(tmp_path: Path, provenance: dict[str, object] | None) -> Path:
    """Write a feature-matrix parquet plus its provenance sidecar to ``tmp_path``.

    ``provenance=None`` writes no sidecar (the missing-sidecar fail-closed case).
    The parquet itself is a tiny placeholder; the enforcement path under test
    reads the sidecar, not the parquet.
    """
    matrix_path = tmp_path / "feature_matrix_tier2_dnam.parquet"
    matrix_path.write_bytes(b"PAR1-placeholder-bytes")  # non-empty; not a real parquet
    if provenance is not None:
        sidecar = matrix_path.with_suffix(matrix_path.suffix + ".provenance.json")
        sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    return matrix_path


def test_cv_loop_safe_false_matrix_is_refused(tmp_path: Path) -> None:
    """A Tier 2 candidate matrix stamped cv_loop_safe=False must fail loud.

    This is the exact footgun the provenance marker was introduced to prevent:
    a cohort-wide-ranked candidate matrix being fed straight into a CV path.
    """
    matrix_path = _write_matrix_with_provenance(tmp_path, _TIER2_CANDIDATE_PROVENANCE)
    with pytest.raises(Phase1ArtefactError, match="cv_loop_safe"):
        assert_cv_loop_safe(matrix_path)
    # And via the canonical loader entry point.
    with pytest.raises(Phase1ArtefactError, match="cv_loop_safe"):
        load_feature_matrix_for_cv(matrix_path)


def test_missing_sidecar_fails_closed(tmp_path: Path) -> None:
    """A matrix with no provenance sidecar is refused, not admitted (fail-closed)."""
    matrix_path = _write_matrix_with_provenance(tmp_path, None)
    with pytest.raises(Phase1ArtefactError, match="no provenance sidecar"):
        assert_cv_loop_safe(matrix_path)


def test_cv_loop_safe_true_matrix_is_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cv_loop_safe=True matrix passes the guard and loads without raising.

    ``read_parquet`` is monkeypatched because the CI smoke-test job has no
    parquet engine; the guard under test (the sidecar provenance check) is
    untouched by the patch.
    """
    matrix_path = _write_matrix_with_provenance(tmp_path, _CV_SAFE_PROVENANCE)
    # assert_cv_loop_safe returns the parsed provenance and does not raise.
    provenance = assert_cv_loop_safe(matrix_path)
    assert provenance["cv_loop_safe"] == "True"

    expected = pd.DataFrame({"cg00": [0.1, 0.2], "cg01": [0.3, 0.4]})
    monkeypatch.setattr(pd, "read_parquet", lambda _p: expected)
    loaded = load_feature_matrix_for_cv(matrix_path)
    pd.testing.assert_frame_equal(loaded, expected)


def test_missing_matrix_raises(tmp_path: Path) -> None:
    """A wholly absent matrix file raises before any provenance check."""
    with pytest.raises(Phase1ArtefactError, match="missing or a zero-byte stub"):
        load_feature_matrix_for_cv(tmp_path / "does_not_exist.parquet")
