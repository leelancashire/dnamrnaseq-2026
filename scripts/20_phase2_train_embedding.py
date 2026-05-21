"""Phase 2: train the three embedding arms (A: FM, B: MOFA+, C: contrastive).

Wired into the Snakemake rule ``train_embedding`` (workflow/rules/embedding.smk).

Design reference: 04-projects/dnamrnaseq/2026-05-19-phase-2-design.md (v1.1),
Section 2 (arms) and Section 5 (compute budget).

This script is the real-data entry point. As of the real-data wiring it loads
the genuine Phase 1 outputs from ``analysis/latest/`` via
``dnamrnaseq2026.embedding.real_data`` and assembles per-arm batches; the
synthetic fixtures are retained only for the unit tests.

Modes
-----
``--dry-run``
    CPU-only. Loads the real Phase 1 inputs, constructs each arm's model, and
    runs ONE forward pass per arm. No optimiser steps, no GPU. This is the
    staging check: it proves the real-data wiring end to end without consuming
    the GPU reserved for the MedFict LLM panel.
``--arm {a,c,all}`` (default ``all``)
    Selects which deep-learning arm(s) to train. Arm B (MOFA+, CPU) is run by
    ``Helen`` via the Arm B path and is not launched here.
(no flag)
    Real training. GPU deep-learning run for Arms A and C -- do NOT launch while
    the MedFict LLM panel holds the GPU. The script asserts an idle CUDA device
    is visible and refuses to start otherwise.

Outputs (per the embedding.smk rule):
  - analysis/latest/embedding_fm.pt
  - analysis/latest/embedding_contrastive.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from dnamrnaseq2026.embedding.arm_a_fm import ArmAConfig, ArmAEncoder, train_arm_a
from dnamrnaseq2026.embedding.arm_c_contrastive import ArmCConfig, ArmCEncoder, train_arm_c
from dnamrnaseq2026.embedding.real_data import ArmInputs, Phase1ArtefactError, build_arm_inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LATEST_DIR = Path("analysis/latest")


def _arm_a_batch(inputs: ArmInputs) -> dict[str, torch.Tensor]:
    """Build the Arm A tensor batch from the real per-arm inputs."""
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {
        "rna_pre": t(inputs.rna_pre),
        "dnam_pre": t(inputs.dnam_pre),
        "clin_pre": t(inputs.clin_pre),
        "rna_post": t(inputs.rna_post),
        "dnam_post": t(inputs.dnam_post),
        "clin_post": t(inputs.clin_post),
        "responder_mask": torch.tensor(inputs.responder_mask, dtype=torch.bool),
    }


def _arm_c_batch(inputs: ArmInputs) -> dict[str, torch.Tensor]:
    """Build the Arm C tensor batch from the real per-arm inputs."""
    return {
        "x_pre": torch.tensor(inputs.paired.x_pre, dtype=torch.float32),
        "x_post": torch.tensor(inputs.paired.x_post, dtype=torch.float32),
        "responder_mask": torch.tensor(inputs.responder_mask, dtype=torch.bool),
    }


def _build_arm_a(inputs: ArmInputs) -> ArmAEncoder:
    """Construct the Arm A encoder sized to the real feature dimensions."""
    cfg = ArmAConfig(
        d_rna_in=inputs.d_rna_in,
        d_dnam_in=inputs.d_dnam_in,
        d_clinical_in=inputs.d_clinical_in,
    )
    return ArmAEncoder(cfg)


def _build_arm_c(inputs: ArmInputs) -> ArmCEncoder:
    """Construct the Arm C encoder sized to the real feature dimensions."""
    cfg = ArmCConfig(d_in=inputs.paired.n_features, lambda_vicreg=0.1)
    return ArmCEncoder(cfg)


def dry_run(arm: str) -> None:
    """Load real Phase 1 inputs, build each selected arm, run one CPU forward pass.

    No optimiser steps, no GPU. Proves the real-data wiring without training.
    """
    inputs = build_arm_inputs()
    logger.info(
        "Dry-run inputs: %d paired subjects | d_rna=%d d_dnam=%d d_clinical=%d",
        inputs.paired.n_subjects,
        inputs.d_rna_in,
        inputs.d_dnam_in,
        inputs.d_clinical_in,
    )
    if arm in {"a", "all"}:
        encoder = _build_arm_a(inputs)
        batch = _arm_a_batch(inputs)
        with torch.no_grad():
            z_pre, z_post = encoder.embed_pair(
                batch["rna_pre"],
                batch["dnam_pre"],
                batch["clin_pre"],
                batch["rna_post"],
                batch["dnam_post"],
                batch["clin_post"],
            )
        assert torch.isfinite(z_pre).all() and torch.isfinite(z_post).all()
        logger.info(
            "Arm A dry-run OK: z_pre %s, z_post %s", tuple(z_pre.shape), tuple(z_post.shape)
        )
    if arm in {"c", "all"}:
        encoder_c = _build_arm_c(inputs)
        batch_c = _arm_c_batch(inputs)
        with torch.no_grad():
            zc_pre, zc_post = encoder_c.embed_pair(batch_c["x_pre"], batch_c["x_post"])
        assert torch.isfinite(zc_pre).all() and torch.isfinite(zc_post).all()
        logger.info(
            "Arm C dry-run OK: z_pre %s, z_post %s", tuple(zc_pre.shape), tuple(zc_post.shape)
        )
    logger.info("Dry-run complete. Real-data wiring is sound; GPU training is staged.")


def _require_idle_gpu() -> str:
    """Return the CUDA device string, refusing if CUDA is unavailable.

    The MedFict LLM panel holds the GPU. This guard does not poll utilisation;
    it asserts a CUDA device is visible. The operator confirms the GPU is idle
    via ``nvidia-smi`` before invoking real training (see the task ledger).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Real Arm A/C training needs a CUDA device. None visible. "
            "Confirm the MedFict panel has released the GPU, then re-run."
        )
    return "cuda"


def train(arm: str) -> None:
    """Real GPU training for Arms A and/or C (design Section 5.1 / 5.3).

    GUARDED: requires a visible CUDA device. Do NOT invoke while the MedFict
    LLM panel holds the GPU.
    """
    device = _require_idle_gpu()
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    inputs = build_arm_inputs()
    if arm in {"a", "all"}:
        encoder = _build_arm_a(inputs)
        train_arm_a(encoder, _arm_a_batch(inputs), epochs=50, device=device)
        torch.save(encoder.state_dict(), LATEST_DIR / "embedding_fm.pt")
        logger.info("Arm A trained -> %s", LATEST_DIR / "embedding_fm.pt")
    if arm in {"c", "all"}:
        encoder_c = _build_arm_c(inputs)
        train_arm_c(encoder_c, _arm_c_batch(inputs), epochs=50, device=device)
        torch.save(encoder_c.state_dict(), LATEST_DIR / "embedding_contrastive.pt")
        logger.info("Arm C trained -> %s", LATEST_DIR / "embedding_contrastive.pt")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See module docstring for the mode matrix."""
    parser = argparse.ArgumentParser(description="Phase 2 embedding-arm training")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only: build models from real inputs, one forward pass, no training.",
    )
    parser.add_argument(
        "--arm",
        choices=["a", "c", "all"],
        default="all",
        help="Which deep-learning arm(s) to run (Arm B MOFA+ is run separately).",
    )
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            dry_run(args.arm)
        else:
            train(args.arm)
    except Phase1ArtefactError as err:
        logger.error("Phase 1 artefact not ready: %s", err)
        return 2
    except RuntimeError as err:
        logger.error("%s", err)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
