"""Partial-NaN contamination guard tests for the Phase 2 real-data loader.

These run UNCONDITIONALLY in default CI -- they build a complete, minimal set
of Phase 1 artefacts in ``tmp_path`` (no OneDrive, no ``analysis/latest/``),
then corrupt one of them with *partial* NaN contamination and assert the loader
fails loud.

Motivation: the upstream ``_read_activity`` stub guard uses
``np.isfinite(...).any()``, which rejects only an *entirely* non-finite
artefact. A parquet with a single finite cell and the rest NaN passes that
guard, and the NaNs then flow into ``x_pre`` / ``x_post`` and the encoders. The
in-progress Phase 1 step 1.4/1.5 re-run is exactly the regime where a partially
populated activity parquet is plausible (some TFs converge, some do not).

The ``build_arm_inputs`` finite assertion (``np.isfinite(...).all()`` on the
assembled paired blocks) is the guard under test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dnamrnaseq2026.embedding.real_data import (
    Phase1ArtefactError,
    Phase1Paths,
    build_arm_inputs,
)

# Three synthetic paired subjects -> six samples (PRE-IOP / POST-IOP each).
_SUBCODES = ("AMC-000001", "AMC-000002", "AMC-000003")
_RAW_VISITS = ("PRE-IOP", "POST-IOP")


def _activity_keys() -> list[str]:
    """The {Subcode}-{raw visit} sample keys the activity parquets are indexed by."""
    return [f"{s}-{v}" for s in _SUBCODES for v in _RAW_VISITS]


def _write_minimal_phase1(root, *, nan_mode: str) -> Phase1Paths:  # type: ignore[no-untyped-def]
    """Write a complete minimal Phase 1 artefact set into ``root``.

    ``nan_mode``:
      - ``"clean"``      -- all activity values finite (loader should succeed).
      - ``"partial"``    -- one TF-activity cell finite, the rest NaN. This is
                            the contamination the ``.any()`` guard misses.
      - ``"all_nan"``    -- the entire TF-activity block NaN (the case the
                            upstream guard already catches).
    """
    keys = _activity_keys()
    rng = np.random.default_rng(0)

    # --- corrected pData (six samples, three paired subjects) ---
    pdata = pd.DataFrame(
        {
            "Subcode": [s for s in _SUBCODES for _ in _RAW_VISITS],
            "Visit": [v for _ in _SUBCODES for v in _RAW_VISITS],
            "Response": ["R", "R", "NR", "NR", "R", "R"],
            "PCL_total": [50.0, 20.0, 48.0, 45.0, 55.0, 25.0],
            "Age": rng.integers(25, 65, size=6),
            "smokingScore": rng.standard_normal(6),
            "epidish_B": rng.uniform(0.02, 0.1, size=6),
            "epidish_NK": rng.uniform(0.02, 0.1, size=6),
            "epidish_CD4T": rng.uniform(0.1, 0.3, size=6),
            "epidish_CD8T": rng.uniform(0.05, 0.2, size=6),
            "epidish_Mono": rng.uniform(0.05, 0.15, size=6),
            "epidish_Neutro": rng.uniform(0.4, 0.6, size=6),
            "epidish_Eosino": rng.uniform(0.01, 0.05, size=6),
        }
    )
    pdata.to_csv(root / "pdata_emory_with_epidish.csv", index=False)

    # --- PROGENy activity: always clean (3 pathways) ---
    progeny = pd.DataFrame(
        rng.standard_normal((len(keys), 3)),
        index=keys,
        columns=["PI3K", "TGFb", "TNFa"],
    )
    progeny.to_parquet(root / "progeny_activity_emory.parquet")

    # --- TF activity: clean / partial-NaN / all-NaN per nan_mode ---
    tf = pd.DataFrame(
        rng.standard_normal((len(keys), 4)),
        index=keys,
        columns=["STAT1", "NFKB1", "JUN", "FOS"],
    )
    if nan_mode == "partial":
        # One finite cell, everything else NaN -> passes np.isfinite(...).any()
        # but is unusable. This is the case the all-NaN guard misses.
        finite_cell = tf.iloc[0, 0]
        tf.loc[:, :] = np.nan
        tf.iloc[0, 0] = finite_cell
    elif nan_mode == "all_nan":
        tf.loc[:, :] = np.nan
    elif nan_mode != "clean":
        raise ValueError(f"unknown nan_mode {nan_mode!r}")
    tf.to_parquet(root / "tf_activity_emory.parquet")

    return Phase1Paths(root=root)


def test_partial_nan_activity_raises_named_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A partially-NaN TF-activity parquet must fail loud, naming the artefact.

    This is the gap Helen flagged: the all-NaN guard (``.any()``) lets a single
    finite cell through. The ``build_arm_inputs`` finite assertion must catch
    it. Runs in default CI -- no ``analysis/latest/`` required.
    """
    paths = _write_minimal_phase1(tmp_path, nan_mode="partial")
    with pytest.raises(Phase1ArtefactError) as excinfo:
        build_arm_inputs(paths)
    msg = str(excinfo.value).lower()
    assert "non-finite" in msg
    # The error must name the offending block so the failure is diagnosable.
    assert "pre" in msg or "post" in msg


def test_all_nan_activity_still_caught(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The pre-existing all-NaN stub case stays covered (no regression)."""
    paths = _write_minimal_phase1(tmp_path, nan_mode="all_nan")
    with pytest.raises(Phase1ArtefactError):
        build_arm_inputs(paths)


def test_clean_activity_assembles_finite_inputs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """With clean artefacts the loader assembles finite paired inputs.

    Confirms the new finite assertion does not false-positive on good data.
    """
    paths = _write_minimal_phase1(tmp_path, nan_mode="clean")
    inputs = build_arm_inputs(paths)
    assert inputs.paired.n_subjects == 3
    assert np.isfinite(inputs.paired.x_pre).all()
    assert np.isfinite(inputs.paired.x_post).all()
    # Response coding R/NR -> a non-degenerate responder fraction.
    frac = float(inputs.responder_mask.mean())
    assert 0.0 < frac < 1.0
