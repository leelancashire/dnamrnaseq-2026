"""Arm C: contrastive within-subject embedding (design Section 2.3).

A two-tower siamese encoder trained from scratch (NO pretrained FM weights;
this is the explicit differentiator vs Arm A). The training objective is the
contrastive InfoNCE loss with same-subject visit pairs as positives and
cross-subject pairs as negatives.

The hypothesis test: does a trajectory-first contrastive objective itself
encode a useful inductive bias when there is no pretrained body?

Collapse-prevention (design Section 6.4): an optional VICReg-style covariance
regularisation term guards against the degenerate constant-embedding solution
that InfoNCE with a same-subject anchor can fall into.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


@dataclass
class ArmCConfig:
    """Hyperparameters for the Arm C contrastive encoder."""

    d_in: int  # concatenated DNAm + RNA + covariate feature count
    d_latent: int = 32
    hidden: int = 256
    n_layers: int = 4
    dropout: float = 0.4  # aggressive: from-scratch overfit risk (Section 6.4)
    tau: float = 0.1  # InfoNCE temperature
    lambda_archetype: float = 0.2  # supplementary response-aware term
    lambda_vicreg: float = 0.0  # covariance regulariser; >0 enables collapse guard


class ArmCEncoder(nn.Module):
    """Shallow MLP-Mixer-style encoder trained from scratch.

    PRE and POST visits pass through the same weights (siamese).
    """

    def __init__(self, config: ArmCConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = [nn.Linear(config.d_in, config.hidden)]
        for _ in range(config.n_layers - 1):
            layers += [
                nn.LayerNorm(config.hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden, config.hidden),
            ]
        layers += [
            nn.LayerNorm(config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.d_latent),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of single (subject, visit) observations -> (batch, d_latent)."""
        return cast(Tensor, self.body(x))

    def embed_pair(self, x_pre: Tensor, x_post: Tensor) -> tuple[Tensor, Tensor]:
        """Embed a paired PRE/POST batch with shared weights."""
        return self.forward(x_pre), self.forward(x_post)


def info_nce_loss(z_pre: Tensor, z_post: Tensor, tau: float = 0.1) -> Tensor:
    """Same-subject InfoNCE loss (design Section 2.3).

    Anchor: subject i at PRE. Positive: subject i at POST. Negatives: all
    subjects j != i at POST in the batch. Symmetrised over PRE<->POST.

    Parameters
    ----------
    z_pre, z_post:
        (batch, d_latent) latent embeddings; row i is the same subject.
    tau:
        Temperature.
    """
    z_pre_n = nn.functional.normalize(z_pre, dim=-1)
    z_post_n = nn.functional.normalize(z_post, dim=-1)
    logits = (z_pre_n @ z_post_n.t()) / tau  # (batch, batch)
    targets = torch.arange(z_pre.shape[0], device=z_pre.device)
    loss_a = nn.functional.cross_entropy(logits, targets)
    loss_b = nn.functional.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)


def archetype_loss(
    z_pre: Tensor,
    z_post: Tensor,
    responder_mask: Tensor,
    tau: float = 0.1,
) -> Tensor:
    """Supplementary response-aware term: pull same-archetype subjects together.

    Operates on the delta-z direction; same-response-class pairs are positives.
    Returns a zero scalar if a class is too small to form pairs.
    """
    delta = nn.functional.normalize(z_post - z_pre, dim=-1)
    mask = responder_mask.bool()
    if mask.sum() < 2 or (~mask).sum() < 2:
        return delta.new_zeros(())
    sim = (delta @ delta.t()) / tau
    same = mask.unsqueeze(0) == mask.unsqueeze(1)
    eye = torch.eye(same.shape[0], dtype=torch.bool, device=same.device)
    same = same & ~eye
    # Soft supervised-contrastive: maximise log-prob mass on same-class pairs.
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=-1, keepdim=True)
    pos_count = same.sum(dim=-1).clamp(min=1)
    per_anchor = (log_prob * same).sum(dim=-1) / pos_count
    return -per_anchor.mean()


def vicreg_covariance_loss(z: Tensor) -> Tensor:
    """VICReg-style covariance regulariser (Bardes et al. ICLR 2022).

    Penalises off-diagonal covariance among latent dimensions; guards against
    the collapsed constant-embedding solution (design Section 6.4).
    """
    z_centred = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0] - 1, 1)
    cov = (z_centred.t() @ z_centred) / n
    d = cov.shape[0]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag.pow(2).sum()) / d


def arm_c_loss(
    encoder: ArmCEncoder,
    z_pre: Tensor,
    z_post: Tensor,
    responder_mask: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Composite Arm C loss: InfoNCE + archetype aux + optional VICReg guard."""
    cfg = encoder.config
    l_nce = info_nce_loss(z_pre, z_post, tau=cfg.tau)
    l_arch = archetype_loss(z_pre, z_post, responder_mask, tau=cfg.tau)
    l_vic = (
        vicreg_covariance_loss(torch.cat([z_pre, z_post], dim=0))
        if cfg.lambda_vicreg > 0
        else z_pre.new_zeros(())
    )
    total = l_nce + cfg.lambda_archetype * l_arch + cfg.lambda_vicreg * l_vic
    components = {
        "l_nce": float(l_nce.detach()),
        "l_archetype": float(l_arch.detach()),
        "l_vicreg": float(l_vic.detach()),
        "total": float(total.detach()),
        "embed_var": float(torch.cat([z_pre, z_post], dim=0).var(dim=0).mean().detach()),
    }
    return total, components


def train_arm_c(
    encoder: ArmCEncoder,
    batch: dict[str, Tensor],
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> list[dict[str, float]]:
    """Full-batch training loop for Arm C (design Section 5.3).

    ``batch`` provides ``x_pre``, ``x_post``, ``responder_mask``. Full-batch
    contrastive is appropriate at this N (design Section 6.4: small negative
    pool favours full-batch over mini-batch).

    Returns the per-epoch loss-component history. The ``embed_var`` entry is the
    collapse monitor (design Section 6.4): a value trending to zero flags the
    degenerate solution.
    """
    torch.manual_seed(seed)
    encoder = encoder.to(device)
    encoder.train()
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)

    moved = {k: v.to(device) for k, v in batch.items()}
    history: list[dict[str, float]] = []
    for _ in range(epochs):
        opt.zero_grad()
        z_pre, z_post = encoder.embed_pair(moved["x_pre"], moved["x_post"])
        loss, components = arm_c_loss(encoder, z_pre, z_post, moved["responder_mask"].bool())
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        history.append(components)
    return history
