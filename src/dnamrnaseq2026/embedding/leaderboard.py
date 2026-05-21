"""Six-metric embedding-architecture leaderboard (design Section 3).

Scores each of the three arms on six metrics and aggregates them into the
6-row x 3-column table that Lee uses to pick the winning embedding ``E*``
(design Section 3.7). No composite score: the leaderboard reports per-cell
``value`` + ``PASS/FAIL``, decision is team consensus.

The six metrics:

(i)   Trajectory consistency — :mod:`dnamrnaseq2026.trajectory.geometry`.
(ii)  Trait-state disentanglement — ICC-band counts via the LMM-LRT machinery
      (:mod:`dnamrnaseq2026.embedding.arm_b_mofa`) PLUS CCA cross-subspace
      independence ``rho_max_CCA < 0.30`` (implemented here).
(iii) Leave-one-subject-out reconstruction surrogate — Delta-PCL linear-probe
      MAE (implemented here).
(iv)  Downstream conformal coverage — :mod:`dnamrnaseq2026.conformal.directional`.
(v)   Biological coherence of latent loadings — PLUGGABLE; reports
      ``pending Phase 1 re-run`` when the Phase 1 enrichment artefacts are
      stubs, and does not block metrics i-iv, vi.
(vi)  Trajectory archetype clusterability — :mod:`...trajectory.geometry`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

from dnamrnaseq2026.embedding.arm_b_mofa import MOFAFactors, classify_factors
from dnamrnaseq2026.embedding.feature_selection import PHASE1_ARTEFACT_DIR
from dnamrnaseq2026.trajectory.geometry import (
    across_seed_consistency,
    cluster_archetypes,
    trajectory_consistency,
)

logger = logging.getLogger(__name__)

CCA_PASS_THRESHOLD = 0.30  # design Section 3 ii Part b
BOOTSTRAP_N = 2000
PENDING_PHASE1 = "pending Phase 1 re-run"


# ---------------------------------------------------------------------------
# Metric (ii): trait-state disentanglement (ICC bands + CCA independence)
# ---------------------------------------------------------------------------


def cca_cross_subspace(
    trait_scores: np.ndarray,
    state_deltas: np.ndarray,
) -> float:
    """Maximum canonical correlation between trait scores and state deltas.

    Design Section 3 ii Part b: a clean trait-state decomposition has
    rho_max_CCA near zero. Inputs are subject-aligned:
    ``trait_scores`` is (n_subjects, k_trait), ``state_deltas`` is
    (n_subjects, k_state).
    """
    n = trait_scores.shape[0]
    k = min(trait_scores.shape[1], state_deltas.shape[1])
    if k < 1 or n < 3:
        return 0.0
    cca = CCA(n_components=k)
    u, v = cca.fit_transform(trait_scores, state_deltas)
    corrs = [
        float(np.corrcoef(u[:, j], v[:, j])[0, 1])
        for j in range(k)
        if np.std(u[:, j]) > 0 and np.std(v[:, j]) > 0
    ]
    return float(np.max(np.abs(corrs))) if corrs else 0.0


def trait_state_disentanglement(
    factors: MOFAFactors,
    *,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, object]:
    """Metric (ii): ICC-band counts + CCA cross-subspace independence.

    Works for any arm: ``factors`` carries either the MOFA+ factors (Arm B) or
    the d_latent dimensions of a neural embedding (Arms A / C) treated as
    "factors" for post-hoc classification.

    Returns a dict with ``n_trait``, ``n_state``, ``n_mixed``, ``rho_max_cca``,
    a cluster-bootstrap CI on rho, and PASS/FAIL on both parts.
    """
    classified = classify_factors(factors, n_bootstrap=min(n_bootstrap, 200), seed=seed)
    n_trait = int((classified["classification"] == "trait").sum())
    n_state = int((classified["classification"] == "state").sum())
    n_mixed = int((classified["classification"] == "mixed").sum())

    trait_idx = classified.index[classified["classification"] == "trait"].tolist()
    state_idx = classified.index[classified["classification"] == "state"].tolist()

    rho, rho_ci = _cca_with_bootstrap(factors, trait_idx, state_idx, n_bootstrap, seed)

    part_a_pass = n_trait >= 3 and n_state >= 3
    part_b_pass = rho < CCA_PASS_THRESHOLD and rho_ci[1] < 0.50
    return {
        "n_trait": n_trait,
        "n_state": n_state,
        "n_mixed": n_mixed,
        "rho_max_cca": rho,
        "rho_ci_low": rho_ci[0],
        "rho_ci_high": rho_ci[1],
        "part_a_pass": part_a_pass,
        "part_b_pass": part_b_pass,
        "pass": part_a_pass and part_b_pass,
        "factor_table": classified,
    }


def _subject_factor_arrays(
    factors: MOFAFactors,
    factor_idx: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (subject-mean scores, subject delta scores) for the given factors."""
    subjects = np.unique(factors.subject_ids)
    means = np.zeros((len(subjects), len(factor_idx)))
    deltas = np.zeros((len(subjects), len(factor_idx)))
    for si, subj in enumerate(subjects):
        rows = factors.subject_ids == subj
        pre = factors.scores[rows & (factors.visit == 0)]
        post = factors.scores[rows & (factors.visit == 1)]
        all_rows = factors.scores[rows]
        for fi, k in enumerate(factor_idx):
            means[si, fi] = all_rows[:, k].mean()
            if pre.size and post.size:
                deltas[si, fi] = post[:, k].mean() - pre[:, k].mean()
    return means, deltas


def _cca_with_bootstrap(
    factors: MOFAFactors,
    trait_idx: list[int],
    state_idx: list[int],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, tuple[float, float]]:
    """rho_max_CCA point estimate + subject-clustered bootstrap 95% CI."""
    if not trait_idx or not state_idx:
        return 0.0, (0.0, 0.0)
    trait_scores, _ = _subject_factor_arrays(factors, trait_idx)
    _, state_deltas = _subject_factor_arrays(factors, state_idx)
    rho = cca_cross_subspace(trait_scores, state_deltas)

    rng = np.random.default_rng(seed)
    n = trait_scores.shape[0]
    boot: list[float] = []
    for _ in range(min(n_bootstrap, 500)):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(cca_cross_subspace(trait_scores[idx], state_deltas[idx]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return rho, ci


# ---------------------------------------------------------------------------
# Metric (iii): leave-one-subject-out reconstruction surrogate
# ---------------------------------------------------------------------------


def loso_reconstruction_surrogate(
    delta_z: np.ndarray,
    delta_pcl: np.ndarray,
) -> dict[str, float]:
    """Metric (iii): LOSO Delta-PCL linear-probe mean absolute error.

    Design Section 3 iii: not input-space MSE (intractable for Arms A/C without
    a decoder); instead a predictive surrogate. For each held-out subject, fit
    ``Delta_PCL ~ linear(Delta_z)`` on the in-fold subjects and measure absolute
    error on the held-out subject.

    ``delta_pcl`` may contain NaN; those subjects are dropped.
    """
    valid = ~np.isnan(delta_pcl)
    dz, dp = delta_z[valid], delta_pcl[valid]
    if dz.shape[0] < 3:
        return {"loso_mae": float("nan"), "n_subjects": int(dz.shape[0])}

    loo = LeaveOneOut()
    errors: list[float] = []
    for train_idx, test_idx in loo.split(dz):
        model = LinearRegression().fit(dz[train_idx], dp[train_idx])
        pred = model.predict(dz[test_idx])
        errors.append(float(abs(dp[test_idx][0] - pred[0])))
    return {"loso_mae": float(np.mean(errors)), "n_subjects": int(dz.shape[0])}


# ---------------------------------------------------------------------------
# Metric (v): biological coherence (pluggable; gated on Phase 1)
# ---------------------------------------------------------------------------


def biological_coherence(
    latent_loadings: np.ndarray | None = None,
    *,
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
) -> dict[str, object]:
    """Metric (v): per-axis enrichment in Phase 1 cell-type / pathway / TF / TFBS.

    Pluggable step (design Section 3 v): if the Phase 1 enrichment artefacts are
    stubs (current state), returns ``status = 'pending Phase 1 re-run'`` and does
    NOT block the other five metrics. When Phase 1 lands real artefacts, the
    enrichment-counting logic activates.

    ``latent_loadings`` is (n_features, d_latent); only used once real artefacts
    exist.
    """
    enrichment = artefact_dir / "lola_tfbs_enrichment.csv"
    celldmc = artefact_dir / "celldmc_interaction_results.parquet"
    artefacts_ready = (
        enrichment.exists()
        and enrichment.stat().st_size > 0
        and celldmc.exists()
        and celldmc.stat().st_size > 0
    )
    if not artefacts_ready or latent_loadings is None:
        return {"status": PENDING_PHASE1, "n_coherent_axes": None, "pass": None}

    # Real-data path: count axes with >= 3 annotation channels at FDR < 0.10.
    enrich_df = pd.read_csv(enrichment)
    n_sig = int((enrich_df["fdr"] < 0.10).sum())
    return {
        "status": "scored",
        "n_coherent_axes": n_sig,
        "pass": n_sig >= 1,
    }


# ---------------------------------------------------------------------------
# Aggregation: 6-row x 3-column leaderboard
# ---------------------------------------------------------------------------


@dataclass
class ArmScore:
    """Per-arm metric bundle for the leaderboard."""

    arm: str
    metrics: dict[str, dict[str, object]] = field(default_factory=dict)


def score_arm(
    arm_name: str,
    *,
    delta_z: np.ndarray,
    responder_mask: np.ndarray,
    delta_z_by_seed: list[np.ndarray],
    factors: MOFAFactors,
    delta_pcl: np.ndarray,
    conformal_result: dict[str, object] | None = None,
    latent_loadings: np.ndarray | None = None,
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = 42,
) -> ArmScore:
    """Score one arm on all six metrics.

    ``conformal_result`` is the dict form of a
    :class:`~dnamrnaseq2026.conformal.directional.ConformalResult`; pass None to
    record metric (iv) as not-yet-run. ``n_bootstrap`` controls the
    cluster-bootstrap resample count for metric (ii); lower it for fast
    synthetic test runs, keep the default for the real leaderboard.
    """
    metrics: dict[str, dict[str, object]] = {}

    # (i) trajectory consistency
    within = trajectory_consistency(delta_z, responder_mask)
    across = across_seed_consistency(delta_z_by_seed)
    metrics["i_trajectory_consistency"] = {
        "responder_mean_cos": within.responder_mean_cos,
        "responder_vs_nonresponder_diff": within.responder_vs_nonresponder_diff,
        "across_seed_median": across["median"],
        "across_seed_p05": across["p05"],
        "pass": bool(across["pass"]),
    }

    # (ii) trait-state disentanglement
    metrics["ii_trait_state_disentanglement"] = trait_state_disentanglement(
        factors, n_bootstrap=n_bootstrap, seed=seed
    )

    # (iii) LOSO reconstruction surrogate
    loso: dict[str, object] = dict(loso_reconstruction_surrogate(delta_z, delta_pcl))
    metrics["iii_loso_reconstruction"] = loso

    # (iv) conformal coverage
    if conformal_result is not None:
        metrics["iv_conformal_coverage"] = dict(conformal_result)
    else:
        metrics["iv_conformal_coverage"] = {"status": "not run"}

    # (v) biological coherence (pluggable, gated on Phase 1)
    metrics["v_biological_coherence"] = biological_coherence(
        latent_loadings, artefact_dir=artefact_dir
    )

    # (vi) archetype clusterability
    arch = cluster_archetypes(delta_z, seed=seed)
    metrics["vi_archetype_clusterability"] = {
        "best_k": arch.best_k,
        "bootstrap_ari_mean": arch.bootstrap_ari_mean,
        "cluster_sizes": arch.cluster_sizes,
        "pass": arch.passes,
    }

    return ArmScore(arm=arm_name, metrics=metrics)


def build_leaderboard(arm_scores: list[ArmScore]) -> pd.DataFrame:
    """Aggregate per-arm scores into the 6-row x N-arm leaderboard (Section 3.7).

    Each cell is a short ``value | PASS/FAIL`` string. Metric (v) renders as
    ``pending Phase 1 re-run`` until the Phase 1 enrichment artefacts land.
    """
    metric_order = [
        "i_trajectory_consistency",
        "ii_trait_state_disentanglement",
        "iii_loso_reconstruction",
        "iv_conformal_coverage",
        "v_biological_coherence",
        "vi_archetype_clusterability",
    ]
    rows: list[dict[str, str]] = []
    for metric in metric_order:
        row: dict[str, str] = {"metric": metric}
        for score in arm_scores:
            row[score.arm] = _format_cell(metric, score.metrics.get(metric, {}))
        rows.append(row)
    return pd.DataFrame(rows).set_index("metric")


def _format_cell(metric: str, m: dict[str, object]) -> str:
    """Render one leaderboard cell as a compact string."""
    if not m:
        return "n/a"
    if metric == "i_trajectory_consistency":
        flag = "PASS" if m.get("pass") else "FAIL"
        return f"across-seed med {m['across_seed_median']:.3f} | {flag}"
    if metric == "ii_trait_state_disentanglement":
        flag = "PASS" if m.get("pass") else "FAIL"
        return f"trait={m['n_trait']} state={m['n_state']} rho={m['rho_max_cca']:.3f} | {flag}"
    if metric == "iii_loso_reconstruction":
        mae = m.get("loso_mae")
        return f"Delta-PCL MAE {mae:.3f}" if isinstance(mae, float) else "n/a"
    if metric == "iv_conformal_coverage":
        if m.get("status") in {"not run", None}:
            return "not run"
        cov = m.get("marginal_coverage")
        return f"coverage {cov:.3f}" if isinstance(cov, float) else "n/a"
    if metric == "v_biological_coherence":
        if m.get("status") == PENDING_PHASE1:
            return PENDING_PHASE1
        flag = "PASS" if m.get("pass") else "FAIL"
        return f"coherent axes {m.get('n_coherent_axes')} | {flag}"
    if metric == "vi_archetype_clusterability":
        flag = "PASS" if m.get("pass") else "FAIL"
        return f"k={m['best_k']} ARI={m['bootstrap_ari_mean']:.3f} | {flag}"
    return "n/a"
