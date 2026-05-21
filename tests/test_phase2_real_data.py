"""Real-data smoke tests for the Phase 2 loader wiring.

These exercise ``dnamrnaseq2026.embedding.real_data`` against the genuine
Phase 1 outputs in ``analysis/latest/``. They are SKIPPED automatically when
those artefacts are absent, so default CI (no OneDrive, no analysis/latest/)
stays green; they run on the 5090 box and any machine with the Phase 1 outputs.

Scope: small and fast, no GPU, no training. They confirm the real artefacts
load, the paired-subject construction is sane, the per-arm batches have the
shapes the arm encoders expect, and a single CPU forward pass per GPU-bound arm
produces finite embeddings. They do NOT run training.

The synthetic fixtures and their tests (tests/test_phase2_arms.py,
tests/test_phase2_harness.py) are unaffected and remain the CI coverage.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnamrnaseq2026.embedding.arm_a_fm import ArmAConfig, ArmAEncoder
from dnamrnaseq2026.embedding.arm_c_contrastive import ArmCConfig, ArmCEncoder
from dnamrnaseq2026.embedding.data_harness import subject_level_folds
from dnamrnaseq2026.embedding.feature_selection import read_celldmc_interactions
from dnamrnaseq2026.embedding.real_data import (
    Phase1ArtefactError,
    Phase1Paths,
    build_arm_inputs,
    build_rna_activity_matrix,
    load_emory_pdata,
)

# Skip the whole module unless the real Phase 1 artefact directory is present.
_PATHS = Phase1Paths()
_HAVE_REAL_DATA = _PATHS.exists()
pytestmark = pytest.mark.skipif(
    not _HAVE_REAL_DATA,
    reason=f"Phase 1 artefacts absent at {_PATHS.root} (expected in default CI)",
)


@pytest.fixture(scope="module")
def arm_inputs():  # type: ignore[no-untyped-def]
    """Real per-arm inputs; skips the module if an artefact is a stub."""
    try:
        return build_arm_inputs()
    except Phase1ArtefactError as err:
        pytest.skip(f"Phase 1 artefact not ready: {err}")


def test_corrected_pdata_loads_with_canonical_visits() -> None:
    pdata = load_emory_pdata(_PATHS)
    assert {"Subcode", "Visit", "Response"}.issubset(pdata.columns)
    assert set(pdata["Visit"].dropna().unique()).issubset({"PRE", "POST"})


def test_rna_activity_matrix_progeny_and_tf_or_stub_detected() -> None:
    """Real RNA activity loads with PROGENy + TF columns, OR the stub is caught.

    The in-progress Phase 1 step 1.4/1.5 re-run leaves an all-NaN activity
    parquet. The loader must either return a usable matrix or raise a precise
    Phase1ArtefactError naming the stub -- never silently pass NaNs downstream.
    """
    pdata = load_emory_pdata(_PATHS)
    try:
        rna = build_rna_activity_matrix(pdata, _PATHS)
    except Phase1ArtefactError as err:
        assert "stub" in str(err).lower() or "non-finite" in str(err).lower()
        pytest.skip(f"RNA activity artefact is a stub (pending Phase 1 re-run): {err}")
    # PROGENy contributes ~14 pathway columns; the matrix must be non-trivial.
    assert rna.shape[1] >= 2
    assert rna.shape[0] > 0
    assert all("-" in str(k) for k in rna.index[:5])


def test_build_arm_inputs_shapes_consistent(arm_inputs) -> None:  # type: ignore[no-untyped-def]
    inp = arm_inputs
    n = inp.paired.n_subjects
    assert n > 0
    assert inp.rna_pre.shape == inp.rna_post.shape == (n, inp.d_rna_in)
    assert inp.dnam_pre.shape == inp.dnam_post.shape == (n, inp.d_dnam_in)
    assert inp.clin_pre.shape == inp.clin_post.shape == (n, inp.d_clinical_in)
    assert inp.responder_mask.shape == (n,)
    # The concatenated PairedDataset width is the sum of the three blocks.
    assert inp.paired.n_features == inp.d_dnam_in + inp.d_rna_in + inp.d_clinical_in


def test_arm_inputs_are_finite(arm_inputs) -> None:  # type: ignore[no-untyped-def]
    inp = arm_inputs
    for name, arr in [
        ("rna_pre", inp.rna_pre),
        ("dnam_pre", inp.dnam_pre),
        ("clin_pre", inp.clin_pre),
        ("x_pre", inp.paired.x_pre),
    ]:
        assert np.isfinite(arr).all(), f"{name} has non-finite values"


def test_subject_level_folds_on_real_data(arm_inputs) -> None:  # type: ignore[no-untyped-def]
    """The real PairedDataset must split cleanly with no subject leakage."""
    folds = subject_level_folds(arm_inputs.paired, n_splits=5, seed=42)
    assert len(folds) == 5
    seen: list[int] = []
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(arm_inputs.paired.n_subjects))


def test_arm_a_forward_pass_on_real_inputs(arm_inputs) -> None:  # type: ignore[no-untyped-def]
    """Arm A encoder, sized to real dims, produces finite d_latent=32 embeddings."""
    inp = arm_inputs
    cfg = ArmAConfig(
        d_rna_in=inp.d_rna_in,
        d_dnam_in=inp.d_dnam_in,
        d_clinical_in=inp.d_clinical_in,
    )
    encoder = ArmAEncoder(cfg)
    to_t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(
            to_t(inp.rna_pre),
            to_t(inp.dnam_pre),
            to_t(inp.clin_pre),
            to_t(inp.rna_post),
            to_t(inp.dnam_post),
            to_t(inp.clin_post),
        )
    assert z_pre.shape == (inp.paired.n_subjects, 32)
    assert torch.isfinite(z_pre).all() and torch.isfinite(z_post).all()


def test_arm_c_forward_pass_on_real_inputs(arm_inputs) -> None:  # type: ignore[no-untyped-def]
    """Arm C encoder, sized to the real concatenated width, embeds finitely."""
    inp = arm_inputs
    cfg = ArmCConfig(d_in=inp.paired.n_features)
    encoder = ArmCEncoder(cfg)
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(
            torch.tensor(inp.paired.x_pre, dtype=torch.float32),
            torch.tensor(inp.paired.x_post, dtype=torch.float32),
        )
    assert z_pre.shape == (inp.paired.n_subjects, 32)
    assert torch.isfinite(z_pre).all() and torch.isfinite(z_post).all()


def test_celldmc_artefact_reads_when_present() -> None:
    """The CellDMC reader accepts the real v5 TSV artefact (or returns None)."""
    celldmc = read_celldmc_interactions(_PATHS.root)
    if celldmc is None:
        pytest.skip("CellDMC interaction artefact not present in analysis/latest/")
    # Real v5 schema: cpg, cell_type, coef, se, t_stat, p_val, fdr, sig.
    assert {"cpg", "cell_type", "fdr"}.issubset(celldmc.columns)


def test_missing_artefact_raises_named_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An empty artefact dir yields a Phase1ArtefactError naming the file."""
    paths = Phase1Paths(root=tmp_path)
    with pytest.raises(Phase1ArtefactError, match="pData"):
        load_emory_pdata(paths)
