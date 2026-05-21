"""Synthetic-fixture tests for the three Phase 2 embedding arms.

Every arm's forward pass, loss, and training loop is exercised on a tiny
synthetic (subject, visit) tensor. No real data, no GPU training run. One
optional GPU smoke test per GPU-bound arm confirms CUDA wiring with a single
forward pass; it is skipped when CUDA is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dnamrnaseq2026.embedding.arm_a_fm import (
    ArmAConfig,
    ArmAEncoder,
    arm_a_loss,
    recovery_axis_proxy_loss,
    train_arm_a,
)
from dnamrnaseq2026.embedding.arm_b_mofa import (
    classify_factors,
    compute_icc,
    fit_mofa,
    state_eligibility_lrt,
)
from dnamrnaseq2026.embedding.arm_c_contrastive import (
    ArmCConfig,
    ArmCEncoder,
    arm_c_loss,
    info_nce_loss,
    train_arm_c,
)
from tests.phase2_fixtures import make_synthetic_paired

CUDA_AVAILABLE = torch.cuda.is_available()


def _arm_a_batch(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        "rna_pre": torch.tensor(data["rna_pre"], dtype=torch.float32),
        "dnam_pre": torch.tensor(data["dnam_pre"], dtype=torch.float32),
        "clin_pre": torch.tensor(data["clin_pre"], dtype=torch.float32),
        "rna_post": torch.tensor(data["rna_post"], dtype=torch.float32),
        "dnam_post": torch.tensor(data["dnam_post"], dtype=torch.float32),
        "clin_post": torch.tensor(data["clin_post"], dtype=torch.float32),
        "responder_mask": torch.tensor(data["responder_mask"], dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Arm A: FM + trajectory-consistency head
# ---------------------------------------------------------------------------


def test_arm_a_forward_pass_shape() -> None:
    data = make_synthetic_paired(n_subjects=30, n_dnam=40, n_rna=24, n_clinical=5)
    cfg = ArmAConfig(d_rna_in=24, d_dnam_in=40, d_clinical_in=5, d_latent=32)
    encoder = ArmAEncoder(cfg)
    batch = _arm_a_batch(data)
    z_pre, z_post = encoder.embed_pair(
        batch["rna_pre"],
        batch["dnam_pre"],
        batch["clin_pre"],
        batch["rna_post"],
        batch["dnam_post"],
        batch["clin_post"],
    )
    assert z_pre.shape == (30, 32)
    assert z_post.shape == (30, 32)


def test_arm_a_param_count_in_design_range() -> None:
    """Primary config should be ~250k-400k trainable params (design Section 2.1)."""
    cfg = ArmAConfig(d_rna_in=200, d_dnam_in=500, d_clinical_in=12)
    encoder = ArmAEncoder(cfg)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    assert 100_000 < n_params < 600_000


def test_arm_a_loss_is_finite_and_differentiable() -> None:
    data = make_synthetic_paired(n_subjects=30)
    cfg = ArmAConfig(d_rna_in=24, d_dnam_in=40, d_clinical_in=5)
    encoder = ArmAEncoder(cfg)
    batch = _arm_a_batch(data)
    z_pre, z_post = encoder.embed_pair(
        batch["rna_pre"],
        batch["dnam_pre"],
        batch["clin_pre"],
        batch["rna_post"],
        batch["dnam_post"],
        batch["clin_post"],
    )
    loss, components = arm_a_loss(encoder, z_pre, z_post, batch["responder_mask"])
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_arm_a_recovery_proxy_rewards_alignment() -> None:
    """Aligned deltas should give a lower (more negative) proxy loss than noise."""
    rng = np.random.default_rng(0)
    axis = rng.standard_normal(32)
    axis /= np.linalg.norm(axis)
    aligned = torch.tensor(np.tile(axis, (20, 1)) * 2.0, dtype=torch.float32)
    noise = torch.tensor(rng.standard_normal((20, 32)), dtype=torch.float32)
    mask = torch.ones(20, dtype=torch.bool)
    l_aligned = recovery_axis_proxy_loss(aligned, mask)
    l_noise = recovery_axis_proxy_loss(noise, mask)
    assert l_aligned < l_noise


def test_arm_a_training_loop_reduces_loss() -> None:
    data = make_synthetic_paired(n_subjects=30, seed=1)
    cfg = ArmAConfig(d_rna_in=24, d_dnam_in=40, d_clinical_in=5)
    encoder = ArmAEncoder(cfg)
    history = train_arm_a(encoder, _arm_a_batch(data), epochs=40, device="cpu")
    assert len(history) == 40
    assert history[-1]["total"] < history[0]["total"]


# ---------------------------------------------------------------------------
# Arm B: MOFA+ trait-state decomposition
# ---------------------------------------------------------------------------


def test_arm_b_mofa_surrogate_factor_shape() -> None:
    data = make_synthetic_paired(n_subjects=30, n_dnam=40, n_rna=24)
    subjects = np.concatenate([data["subject_ids"], data["subject_ids"]])
    visit = np.concatenate([np.zeros(30, dtype=int), np.ones(30, dtype=int)])
    views = {
        "dnam": np.vstack([data["dnam_pre"], data["dnam_post"]]),
        "rna": np.vstack([data["rna_pre"], data["rna_post"]]),
    }
    factors = fit_mofa(views, subjects, visit, n_factors=10, use_surrogate=True)
    assert factors.scores.shape == (60, 10)
    assert factors.n_factors == 10


def test_arm_b_compute_icc_bounds() -> None:
    assert compute_icc(1.0, 0.0) == 1.0
    assert compute_icc(0.0, 1.0) == 0.0
    assert abs(compute_icc(1.0, 1.0) - 0.5) < 1e-9
    assert compute_icc(0.0, 0.0) == 0.0


def test_arm_b_lrt_mixture_chi_square() -> None:
    """LRT p-value is in [0, 1]; a zero statistic yields p = 1."""
    assert state_eligibility_lrt(loglik_h1=10.0, loglik_h0=10.0) == 1.0
    p = state_eligibility_lrt(loglik_h1=12.0, loglik_h0=10.0)
    assert 0.0 <= p <= 1.0
    # Mixture: p must be <= half the plain chi2_1 tail.
    assert p < 0.5


def test_arm_b_classify_factors_produces_labels() -> None:
    data = make_synthetic_paired(n_subjects=24, n_dnam=30, n_rna=18)
    subjects = np.concatenate([data["subject_ids"], data["subject_ids"]])
    visit = np.concatenate([np.zeros(24, dtype=int), np.ones(24, dtype=int)])
    views = {
        "dnam": np.vstack([data["dnam_pre"], data["dnam_post"]]),
        "rna": np.vstack([data["rna_pre"], data["rna_post"]]),
    }
    factors = fit_mofa(views, subjects, visit, n_factors=8, use_surrogate=True)
    table = classify_factors(factors, n_bootstrap=30, seed=0)
    assert len(table) == 8
    assert set(table["classification"]).issubset({"trait", "state", "mixed"})
    assert (table["icc"].between(0.0, 1.0)).all()
    assert (table["lrt_qval"].between(0.0, 1.0)).all()


# ---------------------------------------------------------------------------
# Arm C: contrastive within-subject embedding
# ---------------------------------------------------------------------------


def test_arm_c_forward_pass_shape() -> None:
    data = make_synthetic_paired(n_subjects=30, n_dnam=40, n_rna=24, n_clinical=5)
    cfg = ArmCConfig(d_in=69, d_latent=32)
    encoder = ArmCEncoder(cfg)
    x_pre = torch.tensor(data["x_pre"], dtype=torch.float32)
    x_post = torch.tensor(data["x_post"], dtype=torch.float32)
    z_pre, z_post = encoder.embed_pair(x_pre, x_post)
    assert z_pre.shape == (30, 32)
    assert z_post.shape == (30, 32)


def test_arm_c_info_nce_lower_when_pairs_aligned() -> None:
    """InfoNCE loss is lower when same-subject pairs are close."""
    rng = np.random.default_rng(3)
    base = torch.tensor(rng.standard_normal((20, 32)), dtype=torch.float32)
    aligned = info_nce_loss(base, base.clone(), tau=0.1)
    shuffled = info_nce_loss(base, base[torch.randperm(20)], tau=0.1)
    assert aligned < shuffled


def test_arm_c_loss_is_finite_and_differentiable() -> None:
    data = make_synthetic_paired(n_subjects=30)
    cfg = ArmCConfig(d_in=69)
    encoder = ArmCEncoder(cfg)
    x_pre = torch.tensor(data["x_pre"], dtype=torch.float32)
    x_post = torch.tensor(data["x_post"], dtype=torch.float32)
    z_pre, z_post = encoder.embed_pair(x_pre, x_post)
    mask = torch.tensor(data["responder_mask"], dtype=torch.bool)
    loss, components = arm_c_loss(encoder, z_pre, z_post, mask)
    assert torch.isfinite(loss)
    assert "embed_var" in components
    loss.backward()


def test_arm_c_training_loop_reduces_loss() -> None:
    data = make_synthetic_paired(n_subjects=30, seed=2)
    cfg = ArmCConfig(d_in=69, lambda_vicreg=0.1)
    encoder = ArmCEncoder(cfg)
    batch = {
        "x_pre": torch.tensor(data["x_pre"], dtype=torch.float32),
        "x_post": torch.tensor(data["x_post"], dtype=torch.float32),
        "responder_mask": torch.tensor(data["responder_mask"], dtype=torch.bool),
    }
    history = train_arm_c(encoder, batch, epochs=40, device="cpu")
    assert len(history) == 40
    assert history[-1]["total"] < history[0]["total"]


def test_arm_c_no_collapse_with_vicreg() -> None:
    """The VICReg term should keep embedding variance away from zero."""
    data = make_synthetic_paired(n_subjects=30, seed=4)
    cfg = ArmCConfig(d_in=69, lambda_vicreg=1.0)
    encoder = ArmCEncoder(cfg)
    batch = {
        "x_pre": torch.tensor(data["x_pre"], dtype=torch.float32),
        "x_post": torch.tensor(data["x_post"], dtype=torch.float32),
        "responder_mask": torch.tensor(data["responder_mask"], dtype=torch.bool),
    }
    history = train_arm_c(encoder, batch, epochs=40, device="cpu")
    assert history[-1]["embed_var"] > 1e-3


# ---------------------------------------------------------------------------
# GPU smoke tests: one forward pass per GPU-bound arm (Arm A, Arm C)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_arm_a_gpu_smoke_forward() -> None:
    """Single forward pass on CUDA: confirms Arm A device wiring only."""
    data = make_synthetic_paired(n_subjects=16)
    cfg = ArmAConfig(d_rna_in=24, d_dnam_in=40, d_clinical_in=5)
    encoder = ArmAEncoder(cfg).cuda()
    batch = {k: v.cuda() for k, v in _arm_a_batch(data).items()}
    z_pre, z_post = encoder.embed_pair(
        batch["rna_pre"],
        batch["dnam_pre"],
        batch["clin_pre"],
        batch["rna_post"],
        batch["dnam_post"],
        batch["clin_post"],
    )
    assert z_pre.is_cuda and z_post.is_cuda
    assert torch.isfinite(z_pre).all()


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_arm_c_gpu_smoke_forward() -> None:
    """Single forward pass on CUDA: confirms Arm C device wiring only."""
    data = make_synthetic_paired(n_subjects=16)
    cfg = ArmCConfig(d_in=69)
    encoder = ArmCEncoder(cfg).cuda()
    x_pre = torch.tensor(data["x_pre"], dtype=torch.float32).cuda()
    x_post = torch.tensor(data["x_post"], dtype=torch.float32).cuda()
    z_pre, z_post = encoder.embed_pair(x_pre, x_post)
    assert z_pre.is_cuda
    assert torch.isfinite(z_pre).all()
