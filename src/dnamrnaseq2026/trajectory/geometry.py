"""Trajectory geometry: recovery axis + archetype clustering (design Section 3 i / vi).

Shared geometry helpers consumed by the leaderboard metrics:

- :func:`recovery_axis` — first principal direction of the responder-only
  delta-z cloud (design Section 3 i step 3; v2.2 Step 3.0).
- :func:`trajectory_consistency` — within-seed cosine of each subject's
  delta-z direction with the recovery axis, plus the responder vs non-responder
  contrast (metric i).
- :func:`across_seed_consistency` — pairwise cosine of the same subject's
  delta-z direction across training seeds (metric i, the load-bearing
  stability number).
- :func:`cluster_archetypes` — GMM on delta-z vectors with BIC model selection
  and bootstrap ARI stability (metric vi).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

ACROSS_SEED_PASS = 0.50  # design Section 3 i pass threshold
ARCHETYPE_ARI_PASS = 0.50  # design Section 3 vi pass threshold
ARCHETYPE_MIN_SUBJECTS = 10  # design Section 3 vi: each archetype >= 10 subjects


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe at zero norm."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.asarray(v / np.where(n < 1e-12, 1.0, n), dtype=np.float64)


def recovery_axis(delta_z: np.ndarray, responder_mask: np.ndarray) -> np.ndarray:
    """First principal direction of the responder-only delta-z cloud.

    Parameters
    ----------
    delta_z:
        (n_subjects, d_latent) per-subject latent deltas.
    responder_mask:
        (n_subjects,) boolean; True for responders.

    Returns
    -------
    (d_latent,) unit vector. Sign is oriented so the mean responder delta-z
    projects positively onto it.
    """
    resp = delta_z[responder_mask]
    if resp.shape[0] < 2:
        raise ValueError("Need at least 2 responders to estimate the recovery axis")
    centred = resp - resp.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    if resp.mean(axis=0) @ axis < 0:
        axis = -axis
    return np.asarray(axis / (np.linalg.norm(axis) + 1e-12), dtype=np.float64)


@dataclass
class ConsistencyScore:
    """Within-seed trajectory-consistency result (design Section 3 i step 4)."""

    responder_mean_cos: float
    responder_sd_cos: float
    nonresponder_mean_cos: float
    nonresponder_sd_cos: float
    responder_vs_nonresponder_diff: float


def trajectory_consistency(
    delta_z: np.ndarray,
    responder_mask: np.ndarray,
    axis: np.ndarray | None = None,
) -> ConsistencyScore:
    """Within-seed cosine of each delta-z direction with the recovery axis.

    If ``axis`` is None it is estimated from the responder cloud.
    """
    if axis is None:
        axis = recovery_axis(delta_z, responder_mask)
    directions = _unit(delta_z)
    cos = directions @ axis
    resp = cos[responder_mask]
    nonresp = cos[~responder_mask]
    return ConsistencyScore(
        responder_mean_cos=float(resp.mean()) if resp.size else 0.0,
        responder_sd_cos=float(resp.std()) if resp.size else 0.0,
        nonresponder_mean_cos=float(nonresp.mean()) if nonresp.size else 0.0,
        nonresponder_sd_cos=float(nonresp.std()) if nonresp.size else 0.0,
        responder_vs_nonresponder_diff=(
            float(resp.mean() - nonresp.mean()) if resp.size and nonresp.size else 0.0
        ),
    )


def across_seed_consistency(delta_z_by_seed: list[np.ndarray]) -> dict[str, float]:
    """Pairwise cosine of the same subject's delta-z direction across seeds.

    Parameters
    ----------
    delta_z_by_seed:
        List of (n_subjects, d_latent) delta-z matrices, one per training seed.
        Row i is the same subject across every matrix.

    Returns
    -------
    dict with ``median`` and ``p05`` (5th percentile) of all subject-seed-pair
    cosines, plus a ``pass`` flag against the 0.50 threshold (design Section 3 i).
    """
    if len(delta_z_by_seed) < 2:
        raise ValueError("Need >= 2 seeds for across-seed consistency")
    dirs = [_unit(d) for d in delta_z_by_seed]
    cosines: list[float] = []
    for a, b in combinations(range(len(dirs)), 2):
        per_subject = np.sum(dirs[a] * dirs[b], axis=-1)
        cosines.extend(per_subject.tolist())
    arr = np.asarray(cosines)
    median = float(np.median(arr))
    return {
        "median": median,
        "p05": float(np.percentile(arr, 5)),
        "pass": float(median >= ACROSS_SEED_PASS),
    }


@dataclass
class ArchetypeResult:
    """Trajectory archetype clustering result (design Section 3 vi)."""

    best_k: int
    labels: np.ndarray
    bootstrap_ari_mean: float
    bootstrap_ari_sd: float
    cluster_sizes: dict[int, int]
    passes: bool


def cluster_archetypes(
    delta_z: np.ndarray,
    *,
    k_grid: tuple[int, ...] = (2, 3, 4, 5),
    n_bootstrap: int = 10,
    seed: int = 42,
) -> ArchetypeResult:
    """GMM archetype clustering on delta-z with BIC selection + bootstrap ARI.

    Design Section 3 vi: GMM on delta-z vectors (with magnitude, not unit
    normalised), k chosen by BIC, ARI bootstrapped across seeds. Pass requires
    >= 3 clusters with bootstrap ARI >= 0.50 and each archetype >= 10 subjects.

    Returns
    -------
    ArchetypeResult.
    """
    rng = np.random.default_rng(seed)
    n = delta_z.shape[0]

    best_k, best_bic, best_labels = k_grid[0], np.inf, np.zeros(n, dtype=int)
    for k in k_grid:
        if k >= n:
            continue
        gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="diag")
        labels = gmm.fit_predict(delta_z)
        bic = gmm.bic(delta_z)
        if bic < best_bic:
            best_k, best_bic, best_labels = k, bic, labels

    # Bootstrap ARI: refit on resampled subjects, score against the reference.
    aris: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        gmm = GaussianMixture(
            n_components=best_k, random_state=int(rng.integers(1_000_000)), covariance_type="diag"
        )
        boot_labels = gmm.fit_predict(delta_z[idx])
        aris.append(adjusted_rand_score(best_labels[idx], boot_labels))

    sizes = {int(c): int(np.sum(best_labels == c)) for c in np.unique(best_labels)}
    ari_mean = float(np.mean(aris))
    passes = (
        best_k >= 3
        and ari_mean >= ARCHETYPE_ARI_PASS
        and all(s >= ARCHETYPE_MIN_SUBJECTS for s in sizes.values())
    )
    return ArchetypeResult(
        best_k=best_k,
        labels=best_labels,
        bootstrap_ari_mean=ari_mean,
        bootstrap_ari_sd=float(np.std(aris)),
        cluster_sizes=sizes,
        passes=passes,
    )
