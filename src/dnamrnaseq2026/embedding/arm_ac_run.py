"""Phase 2 Arms A and C: end-to-end training + six-metric leaderboard scoring.

Arm A is the pathway-activity foundation-model arm (PROGENy/decoupleR RNA side,
learned-linear DNAm projection, trajectory-consistency head). Arm C is the
from-scratch contrastive within-subject embedding. Both are scored on the same
six-metric leaderboard as Arm B (design Section 3) so the three arms are
directly comparable.

This module is the Arm A/C analogue of ``arm_b_run.py``: it owns the
training-then-scoring pipeline that ``scripts/20_phase2_train_embedding.py``
drives. It runs the deep-learning encoders on the GPU; ``scripts/20`` supplies
the CUDA-device guard.

Leakage discipline (design Section 4.2)
---------------------------------------
The Arm A RNA-side input is PROGENy + a top-N TF panel selected by variance.
Ranking that panel cohort-wide lets held-out sample-visits influence which TFs
exist, the exact train/test leak PRs #14-#16 closed for Arm B. The only
CV-evaluated leaderboard metric is (iii) LOSO Delta-PCL reconstruction; that
metric is therefore run through :func:`leakage_clean_loso_mae_ac`, which for
each held-out subject (a) re-selects the TF panel on the training subjects only
via ``build_arm_inputs(tf_rank_keys=...)``, (b) retrains the encoder on the
training rows, (c) embeds the held-out subject with the trained weights, and
(d) fits the linear probe on the training fold. The descriptive metrics
(i, ii, vi) are computed on a single cohort-wide embedding: those metrics are
not CV-evaluated (they characterise the embedding's geometry, they do not score
held-out prediction), so cohort-wide TF selection there is sound, exactly the
rationale Arm B documents for its cohort-wide covariate residualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from dnamrnaseq2026.embedding.arm_a_fm import ArmAConfig, ArmAEncoder, train_arm_a
from dnamrnaseq2026.embedding.arm_b_mofa import MOFAFactors
from dnamrnaseq2026.embedding.arm_c_contrastive import ArmCConfig, ArmCEncoder, train_arm_c
from dnamrnaseq2026.embedding.leaderboard import ArmScore, score_arm
from dnamrnaseq2026.embedding.real_data import ArmInputs, build_arm_inputs
from dnamrnaseq2026.trajectory.geometry import recovery_axis

logger = logging.getLogger(__name__)

EPOCHS = 50
SEEDS = (42, 43, 44, 45, 46)
N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Sample-visit key helpers (the TF-rank training-fold key list)
# ---------------------------------------------------------------------------


def _sample_visit_keys(subject_id: str) -> tuple[str, str]:
    """Return the (PRE, POST) ``{Subcode}-{Visit}`` activity-matrix keys."""
    return f"{subject_id}-PRE-IOP", f"{subject_id}-POST-IOP"


def _training_tf_rank_keys(all_subjects: np.ndarray, held: str | None) -> list[str]:
    """Build the TF-variance-rank key list for the training fold.

    ``held`` is the held-out subject (None for the cohort-wide descriptive
    embedding, which legitimately ranks over every subject). The returned list
    is the PRE+POST sample-visit keys of every NON-held-out subject, so the TF
    panel is ranked on training rows only.
    """
    keys: list[str] = []
    for subj in all_subjects:
        if held is not None and subj == held:
            continue
        pre, post = _sample_visit_keys(str(subj))
        keys.extend([pre, post])
    return keys


# ---------------------------------------------------------------------------
# Encoder training + embedding
# ---------------------------------------------------------------------------


def _f32(array: np.ndarray) -> torch.Tensor:
    """Convert an array-like to a float32 tensor."""
    return torch.tensor(np.asarray(array), dtype=torch.float32)


def _arm_a_batch(inputs: ArmInputs) -> dict[str, torch.Tensor]:
    """Build the Arm A tensor batch from per-arm real-data inputs."""
    return {
        "rna_pre": _f32(inputs.rna_pre),
        "dnam_pre": _f32(inputs.dnam_pre),
        "clin_pre": _f32(inputs.clin_pre),
        "rna_post": _f32(inputs.rna_post),
        "dnam_post": _f32(inputs.dnam_post),
        "clin_post": _f32(inputs.clin_post),
        "responder_mask": torch.tensor(inputs.responder_mask, dtype=torch.bool),
    }


def _arm_c_batch(inputs: ArmInputs) -> dict[str, torch.Tensor]:
    """Build the Arm C tensor batch from per-arm real-data inputs."""
    return {
        "x_pre": torch.tensor(inputs.paired.x_pre, dtype=torch.float32),
        "x_post": torch.tensor(inputs.paired.x_post, dtype=torch.float32),
        "responder_mask": torch.tensor(inputs.responder_mask, dtype=torch.bool),
    }


def train_embed_arm_a(
    inputs: ArmInputs,
    *,
    device: str,
    epochs: int = EPOCHS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an Arm A encoder and return its (z_pre, z_post) embedding."""
    cfg = ArmAConfig(
        d_rna_in=inputs.d_rna_in,
        d_dnam_in=inputs.d_dnam_in,
        d_clinical_in=inputs.d_clinical_in,
    )
    encoder = ArmAEncoder(cfg)
    batch = _arm_a_batch(inputs)
    train_arm_a(encoder, batch, epochs=epochs, device=device, seed=seed)
    encoder.eval()
    moved = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(
            moved["rna_pre"],
            moved["dnam_pre"],
            moved["clin_pre"],
            moved["rna_post"],
            moved["dnam_post"],
            moved["clin_post"],
        )
    return z_pre.cpu().numpy(), z_post.cpu().numpy()


def train_embed_arm_c(
    inputs: ArmInputs,
    *,
    device: str,
    epochs: int = EPOCHS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an Arm C encoder and return its (z_pre, z_post) embedding."""
    cfg = ArmCConfig(d_in=inputs.paired.n_features, lambda_vicreg=0.1)
    encoder = ArmCEncoder(cfg)
    batch = _arm_c_batch(inputs)
    train_arm_c(encoder, batch, epochs=epochs, device=device, seed=seed)
    encoder.eval()
    moved = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(moved["x_pre"], moved["x_post"])
    return z_pre.cpu().numpy(), z_post.cpu().numpy()


# ---------------------------------------------------------------------------
# Latent embedding -> MOFAFactors adapter (metric ii)
# ---------------------------------------------------------------------------


def embedding_to_factors(
    z_pre: np.ndarray, z_post: np.ndarray, subjects: np.ndarray
) -> MOFAFactors:
    """Wrap a neural (z_pre, z_post) embedding as :class:`MOFAFactors`.

    Metric (ii) (trait-state disentanglement) classifies "factors" via the
    shared ICC-continuum machinery in ``arm_b_mofa.classify_factors``. For a
    neural arm the d_latent embedding dimensions ARE the factors: the PRE row
    and POST row of each subject are stacked observation-major (all PRE rows,
    then all POST rows), exactly the layout ``MOFAFactors`` expects.
    """
    scores = np.vstack([z_pre, z_post])
    subject_ids = np.concatenate([subjects, subjects])
    visit = np.concatenate([np.zeros(len(subjects), dtype=int), np.ones(len(subjects), dtype=int)])
    return MOFAFactors(scores=scores, subject_ids=subject_ids, visit=visit, loadings={})


# ---------------------------------------------------------------------------
# Metric (iii): leakage-clean LOSO Delta-PCL reconstruction surrogate
# ---------------------------------------------------------------------------


@dataclass
class CleanLosoResult:
    """Leakage-clean LOSO MAE for an Arm A/C embedding."""

    loso_mae: float
    n_subjects: int
    selection: str = "leakage_clean_per_fold_tf_refit"


def leakage_clean_loso_mae_ac(
    arm: str,
    *,
    device: str,
    epochs: int = EPOCHS,
    seed: int = 42,
) -> CleanLosoResult:
    """Leakage-clean LOSO Delta-PCL MAE for Arm A or Arm C (metric iii).

    For each held-out subject: re-select the TF panel on the training subjects
    only, retrain the encoder on the training rows, embed the held-out subject
    with the trained weights, fit ``Delta_PCL ~ linear(delta_z)`` on the
    training fold, and score the held-out subject. This is the Arm A/C analogue
    of ``arm_b_run.leakage_clean_loso_mae``: it is the ONLY leaderboard metric
    materially exposed to the Tier 1 RNA TF-selection leak, so it is the one
    metric that must refit per fold.

    ``arm`` is ``"a"`` or ``"c"``.
    """
    from sklearn.linear_model import LinearRegression

    # Cohort-wide pass once, only to enumerate the paired subjects in a stable
    # order; the TF panel from this pass is NOT used for any held-out scoring.
    base = build_arm_inputs()
    subjects = base.paired.subject_ids
    delta_pcl = base.paired.delta_pcl

    errors: list[float] = []
    n_scored = 0
    for held in subjects:
        # TF panel + encoder refit on training subjects only.
        train_keys = _training_tf_rank_keys(subjects, held)
        fold_inputs = build_arm_inputs(tf_rank_keys=train_keys)
        fold_subjects = fold_inputs.paired.subject_ids
        train_mask = fold_subjects != held
        held_mask = fold_subjects == held
        if held_mask.sum() != 1 or train_mask.sum() < 3:
            continue

        train_inputs = _subset_inputs(fold_inputs, train_mask)
        if arm == "a":
            z_pre, z_post = train_embed_arm_a(train_inputs, device=device, epochs=epochs, seed=seed)
            full_pre, full_post = _embed_with_trained_arm_a(
                fold_inputs, train_inputs, device=device, epochs=epochs, seed=seed
            )
        else:
            z_pre, z_post = train_embed_arm_c(train_inputs, device=device, epochs=epochs, seed=seed)
            full_pre, full_post = _embed_with_trained_arm_c(
                fold_inputs, train_inputs, device=device, epochs=epochs, seed=seed
            )

        train_delta = z_post - z_pre
        train_pcl = delta_pcl[np.isin(subjects, fold_subjects[train_mask])]
        valid = ~np.isnan(train_pcl)
        if valid.sum() < 3:
            continue
        probe = LinearRegression().fit(train_delta[valid], train_pcl[valid])

        held_delta = (full_post - full_pre)[held_mask]
        held_pcl = delta_pcl[subjects == held]
        if np.isnan(held_pcl).any():
            continue
        pred = probe.predict(held_delta)
        errors.append(float(abs(held_pcl[0] - pred[0])))
        n_scored += 1

    mae = float(np.mean(errors)) if errors else float("nan")
    logger.info("Arm %s leakage-clean LOSO: MAE %.3f over %d subjects", arm.upper(), mae, n_scored)
    return CleanLosoResult(loso_mae=mae, n_subjects=n_scored)


def _subset_inputs(inputs: ArmInputs, mask: np.ndarray) -> ArmInputs:
    """Return an :class:`ArmInputs` restricted to the subjects in ``mask``."""
    from dnamrnaseq2026.embedding.data_harness import PairedDataset

    paired = inputs.paired
    sub_paired = PairedDataset(
        x_pre=paired.x_pre[mask],
        x_post=paired.x_post[mask],
        subject_ids=paired.subject_ids[mask],
        response=paired.response[mask],
        feature_names=paired.feature_names,
        delta_pcl=paired.delta_pcl[mask],
        cohort=paired.cohort[mask],
    )
    return ArmInputs(
        paired=sub_paired,
        rna_pre=inputs.rna_pre[mask],
        rna_post=inputs.rna_post[mask],
        dnam_pre=inputs.dnam_pre[mask],
        dnam_post=inputs.dnam_post[mask],
        clin_pre=inputs.clin_pre[mask],
        clin_post=inputs.clin_post[mask],
        responder_mask=inputs.responder_mask[mask],
    )


def _embed_with_trained_arm_a(
    full_inputs: ArmInputs,
    train_inputs: ArmInputs,
    *,
    device: str,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train Arm A on the training fold, embed the FULL fold (incl. held-out).

    The held-out subject is embedded with weights trained only on the training
    rows; its TF panel is the training-fold panel (``full_inputs`` was built
    with the training-fold ``tf_rank_keys``). No held-out row touched training.
    """
    cfg = ArmAConfig(
        d_rna_in=train_inputs.d_rna_in,
        d_dnam_in=train_inputs.d_dnam_in,
        d_clinical_in=train_inputs.d_clinical_in,
    )
    encoder = ArmAEncoder(cfg)
    train_arm_a(encoder, _arm_a_batch(train_inputs), epochs=epochs, device=device, seed=seed)
    encoder.eval()
    batch = {k: v.to(device) for k, v in _arm_a_batch(full_inputs).items()}
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(
            batch["rna_pre"],
            batch["dnam_pre"],
            batch["clin_pre"],
            batch["rna_post"],
            batch["dnam_post"],
            batch["clin_post"],
        )
    return z_pre.cpu().numpy(), z_post.cpu().numpy()


def _embed_with_trained_arm_c(
    full_inputs: ArmInputs,
    train_inputs: ArmInputs,
    *,
    device: str,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train Arm C on the training fold, embed the FULL fold (incl. held-out)."""
    cfg = ArmCConfig(d_in=train_inputs.paired.n_features, lambda_vicreg=0.1)
    encoder = ArmCEncoder(cfg)
    train_arm_c(encoder, _arm_c_batch(train_inputs), epochs=epochs, device=device, seed=seed)
    encoder.eval()
    batch = {k: v.to(device) for k, v in _arm_c_batch(full_inputs).items()}
    with torch.no_grad():
        z_pre, z_post = encoder.embed_pair(batch["x_pre"], batch["x_post"])
    return z_pre.cpu().numpy(), z_post.cpu().numpy()


# ---------------------------------------------------------------------------
# End-to-end: train, score six metrics, return ArmScore
# ---------------------------------------------------------------------------


def run_arm_ac(arm: str, *, device: str, epochs: int = EPOCHS) -> ArmScore:
    """Train Arm A or Arm C end to end and return its six-metric :class:`ArmScore`.

    ``arm`` is ``"a"`` or ``"c"``. Metrics i/ii/vi are scored on a cohort-wide
    embedding (multi-seed for metric i); metric iii uses the leakage-clean LOSO.
    """
    if arm not in {"a", "c"}:
        raise ValueError(f"arm must be 'a' or 'c', got {arm!r}")

    inputs = build_arm_inputs()
    subjects = inputs.paired.subject_ids
    responder = inputs.responder_mask
    delta_pcl = inputs.paired.delta_pcl
    train_fn = train_embed_arm_a if arm == "a" else train_embed_arm_c

    # Multi-seed embeddings: seed 42 is the primary, all five feed metric (i)
    # across-seed consistency.
    delta_z_by_seed: list[np.ndarray] = []
    z_pre_primary = z_post_primary = None
    for seed in SEEDS:
        z_pre, z_post = train_fn(inputs, device=device, epochs=epochs, seed=seed)
        delta_z_by_seed.append(z_post - z_pre)
        if seed == SEEDS[0]:
            z_pre_primary, z_post_primary = z_pre, z_post
    assert z_pre_primary is not None and z_post_primary is not None
    delta_z = z_post_primary - z_pre_primary

    # Embedding-collapse guard: a degenerate constant embedding makes every
    # downstream metric meaningless. Fail loud rather than reporting a spurious
    # leaderboard row.
    embed_var = float(np.var(np.vstack([z_pre_primary, z_post_primary]), axis=0).mean())
    if embed_var < 1e-6:
        raise RuntimeError(
            f"Arm {arm.upper()} embedding collapsed (mean latent variance "
            f"{embed_var:.2e}); the leaderboard row would be spurious. Inspect "
            "the loss curve before reporting."
        )

    factors = embedding_to_factors(z_pre_primary, z_post_primary, subjects)
    clean_loso = leakage_clean_loso_mae_ac(arm, device=device, epochs=epochs)

    arm_name = f"arm_{arm}_fm" if arm == "a" else f"arm_{arm}_contrastive"
    arm_score = score_arm(
        arm_name,
        delta_z=delta_z,
        responder_mask=responder,
        delta_z_by_seed=delta_z_by_seed,
        factors=factors,
        delta_pcl=delta_pcl,
        conformal_result=None,  # metric (iv): downstream calibration step
        latent_loadings=None,  # metric (v): gated on Phase 1 enrichment artefacts
        n_bootstrap=N_BOOTSTRAP,
        seed=SEEDS[0],
    )

    # Metric (iii) override: replace score_arm's in-fit LOSO (which trained on a
    # cohort-wide TF panel) with the leakage-clean per-fold refit.
    contaminated = arm_score.metrics["iii_loso_reconstruction"].get("loso_mae")
    arm_score.metrics["iii_loso_reconstruction"] = {
        "loso_mae": clean_loso.loso_mae,
        "n_subjects": clean_loso.n_subjects,
        "selection": clean_loso.selection,
    }
    logger.info(
        "Arm %s metric (iii) leakage-clean LOSO MAE %.3f (cohort-wide-TF LOSO was %s)",
        arm.upper(),
        clean_loso.loso_mae,
        f"{contaminated:.3f}" if isinstance(contaminated, float) else "n/a",
    )
    # Recovery-axis diagnostic for the log (not a leaderboard cell).
    axis = recovery_axis(delta_z, responder)
    logger.info("Arm %s recovery axis norm %.4f", arm.upper(), float(np.linalg.norm(axis)))
    return arm_score
