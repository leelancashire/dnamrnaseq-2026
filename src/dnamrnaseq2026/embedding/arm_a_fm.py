"""Arm A: pathway-activity encoder + trajectory-consistency head (design Section 2.1).

The 2026-05-19 Kai override defines the PRIMARY Arm A configuration:

- RNA side: PROGENy pathway activities + decoupleR/CollecTRI TF activity scores
  (~120-220 features), learned-linear projection to ``d_rna = 128``.
- DNAm side: learned-linear projection over EpiDISH-corrected M-values on the
  Tier 1 CellDMC-prioritised CpG set, ``d_dnam = 256``.
- Concatenate ``[z_rna, z_dnam, z_clinical]`` -> 2-layer MLP head -> ``d_latent = 32``.

Geneformer / scGPT / Nucleotide-Transformer frozen encoders are a SUPPLEMENTARY
ablation only and are not implemented in this scaffold; the architecture below
is the primary pathway-activity configuration, which carries no frozen FM
weights and has ~250-400k trainable parameters.

Training objective (design Section 2.1, "Practical training note"): the
tractable proxy ``L_traj_proxy = -cos(delta_z_i, recovery_axis)`` for responder
subjects, plus an optional reconstruction term and a down-weighted response
discriminator. The multi-seed consistency target is the *validation* metric
(Section 3 i), not the training loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class ArmAConfig:
    """Hyperparameters for the Arm A primary configuration."""

    d_rna_in: int  # PROGENy + TF activity feature count (~120-220)
    d_dnam_in: int  # Tier 1 CellDMC-prioritised CpG count
    d_clinical_in: int  # clinical covariate count
    d_rna: int = 128
    d_dnam: int = 256
    d_latent: int = 32
    hidden: int = 256
    dropout: float = 0.3
    lambda_recon: float = 0.0  # optional reconstruction term; 0 disables
    lambda_disc: float = 0.1  # down-weighted response discriminator
    tau_recovery: float = 1.0  # scaling on the recovery-axis proxy term


class ArmAEncoder(nn.Module):
    """Two-path pathway-activity encoder with a shallow projection head.

    Forward pass embeds a single (subject, visit) observation. The trajectory
    unit ``(z_PRE, z_POST)`` is formed by calling the encoder twice with shared
    weights; see :meth:`embed_pair`.
    """

    def __init__(self, config: ArmAConfig) -> None:
        super().__init__()
        self.config = config

        # Learned-linear projections (no frozen FM weights on the critical path).
        self.rna_proj = nn.Linear(config.d_rna_in, config.d_rna)
        self.dnam_proj = nn.Linear(config.d_dnam_in, config.d_dnam)

        head_in = config.d_rna + config.d_dnam + config.d_clinical_in
        self.head = nn.Sequential(
            nn.Linear(head_in, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, config.d_latent),
        )

        # Down-weighted auxiliary response discriminator on z_PRE only.
        self.response_disc = nn.Linear(config.d_latent, 1)

        # Optional linear decoder for the reconstruction term.
        self.decoder = nn.Linear(config.d_latent, head_in) if config.lambda_recon > 0 else None

    def forward(self, rna: Tensor, dnam: Tensor, clinical: Tensor) -> Tensor:
        """Embed a batch of single (subject, visit) observations.

        Parameters
        ----------
        rna:
            (batch, d_rna_in) pathway/TF activity features.
        dnam:
            (batch, d_dnam_in) EpiDISH-corrected M-values on the Tier 1 CpG set.
        clinical:
            (batch, d_clinical_in) clinical covariates.

        Returns
        -------
        (batch, d_latent) latent embedding.
        """
        z_rna = torch.relu(self.rna_proj(rna))
        z_dnam = torch.relu(self.dnam_proj(dnam))
        fused = torch.cat([z_rna, z_dnam, clinical], dim=-1)
        return cast(Tensor, self.head(fused))

    def embed_pair(
        self,
        rna_pre: Tensor,
        dnam_pre: Tensor,
        clin_pre: Tensor,
        rna_post: Tensor,
        dnam_post: Tensor,
        clin_post: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Embed a paired PRE/POST batch with shared weights -> (z_pre, z_post)."""
        z_pre = self.forward(rna_pre, dnam_pre, clin_pre)
        z_post = self.forward(rna_post, dnam_post, clin_post)
        return z_pre, z_post


def recovery_axis_proxy_loss(
    delta_z: Tensor,
    responder_mask: Tensor,
    tau: float = 1.0,
) -> Tensor:
    """Within-batch trajectory-consistency proxy (design Section 2.1).

    Estimates the recovery axis as the mean responder delta-z direction within
    the batch and rewards responder delta-z vectors that align with it:
    ``L = -tau * mean_i cos(delta_z_i, recovery_axis)`` over responders.

    Differentiable; the recovery-axis estimate is detached so the gradient
    pulls each responder toward the consensus rather than collapsing the
    consensus toward each point.

    Parameters
    ----------
    delta_z:
        (batch, d_latent) per-subject latent deltas.
    responder_mask:
        (batch,) boolean tensor; True for responder subjects.
    tau:
        Scaling on the proxy term.
    """
    if responder_mask.sum() < 2:
        # Too few responders in the batch to estimate an axis; no-op.
        return delta_z.new_zeros(())
    resp = delta_z[responder_mask]
    axis = resp.mean(dim=0)
    axis = axis / (axis.norm() + 1e-8)
    axis = axis.detach()
    resp_unit = resp / (resp.norm(dim=-1, keepdim=True) + 1e-8)
    cos = resp_unit @ axis
    return cast(Tensor, -tau * cos.mean())


def arm_a_loss(
    encoder: ArmAEncoder,
    z_pre: Tensor,
    z_post: Tensor,
    responder_mask: Tensor,
    *,
    recon_target: Tensor | None = None,
    recon_input: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Composite Arm A loss ``L_A = L_traj + lambda_recon * L_recon + lambda_disc * L_disc``.

    Returns the scalar loss and a dict of component values for logging.
    """
    cfg = encoder.config
    delta_z = z_post - z_pre

    l_traj = recovery_axis_proxy_loss(delta_z, responder_mask, tau=cfg.tau_recovery)

    # Down-weighted response discriminator on baseline z_PRE only.
    disc_logits = encoder.response_disc(z_pre).squeeze(-1)
    disc_target = responder_mask.to(disc_logits.dtype)
    l_disc = nn.functional.binary_cross_entropy_with_logits(disc_logits, disc_target)

    l_recon = z_pre.new_zeros(())
    if cfg.lambda_recon > 0 and encoder.decoder is not None and recon_target is not None:
        recon = encoder.decoder(z_pre)
        l_recon = nn.functional.mse_loss(recon, recon_target)

    total = l_traj + cfg.lambda_recon * l_recon + cfg.lambda_disc * l_disc
    components = {
        "l_traj": float(l_traj.detach()),
        "l_recon": float(l_recon.detach()),
        "l_disc": float(l_disc.detach()),
        "total": float(total.detach()),
    }
    return total, components


def train_arm_a(
    encoder: ArmAEncoder,
    batch: dict[str, Tensor],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> list[dict[str, float]]:
    """Full-batch training loop for Arm A (design Section 5.1).

    ``batch`` provides the keys ``rna_pre, dnam_pre, clin_pre, rna_post,
    dnam_post, clin_post, responder_mask``. Full-batch is appropriate at
    n ~ 131 training subjects per outer fold.

    Returns the per-epoch loss-component history.
    """
    torch.manual_seed(seed)
    encoder = encoder.to(device)
    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    moved = {k: v.to(device) for k, v in batch.items()}
    recon_target = torch.cat([moved["rna_pre"], moved["dnam_pre"], moved["clin_pre"]], dim=-1)
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        opt.zero_grad()
        z_pre, z_post = encoder.embed_pair(
            moved["rna_pre"],
            moved["dnam_pre"],
            moved["clin_pre"],
            moved["rna_post"],
            moved["dnam_post"],
            moved["clin_post"],
        )
        loss, components = arm_a_loss(
            encoder,
            z_pre,
            z_post,
            moved["responder_mask"].bool(),
            recon_target=recon_target,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        history.append(components)
    return history
