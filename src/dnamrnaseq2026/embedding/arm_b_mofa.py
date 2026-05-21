"""Arm B: MOFA+ with trait-state factor decomposition (design Section 2.2).

Owner: Dr. Helen Zhao (statistics, factor identifiability, trait-state core).
This module drafts the implementation; Helen reviews the LMM-LRT machinery.

Two layers:

1. **MOFA+ factorisation.** Two-view probabilistic matrix factorisation
   (DNAm delta-M view + RNA-seq view). CPU-only via ``mofapy2``. The MOFA+ fit
   is wrapped so the scaffold can run a fast synthetic-PCA surrogate when
   ``mofapy2`` training would be too slow for a unit test.

2. **Trait-state classification (Helen override, replaces variance-ratio).**
   Per fitted factor ``k``, a random-intercept LMM is fit on factor scores
   across subjects and visits:

       z_k_{i,t} = beta_0 + b_i + e_{i,t},  b_i ~ N(0, s_between^2)

   Three quantities follow:
   - ICC_k = s_between^2 / (s_between^2 + s_within^2), with cluster-bootstrap CI.
   - State-eligibility LRT of H0: s_within^2 = 0 vs H1: s_within^2 > 0. The null
     statistic follows a 50:50 mixture of chi2_0 and chi2_1 because the variance
     component is on the boundary of the parameter space (Self & Liang JASA 1987;
     Stram & Lee Biometrics 1994). BH-FDR across the K factors.
   - Classification: Trait (ICC > 0.80 AND LRT fails to reject at q >= 0.10),
     State (ICC < 0.50 AND LRT rejects at q < 0.10), Mixed otherwise.

The variance-ratio threshold is deprecated to a footnote synonym in the design
doc and is NOT implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

ICC_TRAIT_THRESHOLD = 0.80
ICC_STATE_THRESHOLD = 0.50
LRT_FDR_THRESHOLD = 0.10
BOOTSTRAP_N = 2000


@dataclass
class MOFAFactors:
    """Output of the MOFA+ factorisation layer.

    Attributes
    ----------
    scores:
        (n_subj_visits, K) factor-score matrix.
    subject_ids:
        Per-row subject id (length n_subj_visits).
    visit:
        Per-row visit code, 0 = PRE, 1 = POST.
    loadings:
        Optional per-view {view_name: (n_features, K)} loading matrices.
    """

    scores: np.ndarray
    subject_ids: np.ndarray
    visit: np.ndarray
    loadings: dict[str, np.ndarray]

    @property
    def n_factors(self) -> int:
        """Number of MOFA+ factors K."""
        return int(self.scores.shape[1])


def fit_mofa(
    views: dict[str, np.ndarray],
    subject_ids: np.ndarray,
    visit: np.ndarray,
    *,
    n_factors: int = 20,
    seed: int = 42,
    use_surrogate: bool = False,
) -> MOFAFactors:
    """Fit a two-view MOFA+ model, or a fast PCA surrogate for scaffolding.

    Parameters
    ----------
    views:
        {view_name: (n_subj_visits, n_features)} matrices, observation-aligned.
    subject_ids, visit:
        Per-observation subject id and visit code (0 = PRE, 1 = POST).
    n_factors:
        MOFA+ factor count K (design Section 2.2: K = 20-30).
    seed:
        Random seed.
    use_surrogate:
        If True, use a concatenated-view PCA surrogate instead of the full
        ``mofapy2`` ELBO fit. The surrogate is for synthetic-fixture unit tests
        and fast smoke runs ONLY; the real leaderboard uses the MOFA+ fit.

    Returns
    -------
    MOFAFactors with the (n_subj_visits, K) score matrix.
    """
    view_names = sorted(views)
    stacked = np.hstack([views[v] for v in view_names])
    k = min(n_factors, stacked.shape[0] - 1, stacked.shape[1])

    if use_surrogate:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=k, random_state=seed)
        scores = pca.fit_transform(stacked)
        loadings: dict[str, np.ndarray] = {}
        col = 0
        for v in view_names:
            width = views[v].shape[1]
            loadings[v] = pca.components_[:, col : col + width].T
            col += width
        logger.info("MOFA+ surrogate (PCA): %d factors on %s", k, stacked.shape)
        return MOFAFactors(scores, subject_ids, visit, loadings)

    scores, loadings = _fit_mofapy2(views, view_names, n_factors=k, seed=seed)
    logger.info("MOFA+ fit: %d factors on %d views", k, len(view_names))
    return MOFAFactors(scores, subject_ids, visit, loadings)


def _fit_mofapy2(
    views: dict[str, np.ndarray],
    view_names: list[str],
    *,
    n_factors: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the real mofapy2 ELBO fit. Separated so the surrogate path is import-light."""
    from mofapy2.run.entry_point import entry_point

    n_obs = views[view_names[0]].shape[0]
    sample_names = [f"obs_{i}" for i in range(n_obs)]
    data_matrix = [[views[v]] for v in view_names]  # [view][group] -> array

    ent = entry_point()
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(
        data_matrix,
        views_names=view_names,
        groups_names=["single_group"],
        samples_names=[sample_names],
        features_names=[[f"{v}_f{j}" for j in range(views[v].shape[1])] for v in view_names],
    )
    ent.set_model_options(factors=n_factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(iter=100, convergence_mode="fast", seed=seed, verbose=False)
    ent.build()
    ent.run()

    expectations = ent.model.getExpectations()
    scores = np.asarray(expectations["Z"]["single_group"]["E"])
    loadings = {v: np.asarray(expectations["W"][i]["E"]) for i, v in enumerate(view_names)}
    return scores, loadings


def _fit_random_intercept_lmm(
    y: np.ndarray,
    subject: np.ndarray,
) -> tuple[float, float, float]:
    """Fit z = beta0 + b_i + e via REML; return (s_between^2, s_within^2, loglik_h1).

    Uses ``statsmodels.MixedLM`` with a random intercept per subject.
    """
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"y": y, "subject": subject})
    model = smf.mixedlm("y ~ 1", df, groups=df["subject"])
    result = model.fit(reml=False, method="lbfgs")
    s_between = float(result.cov_re.iloc[0, 0])
    s_within = float(result.scale)
    return s_between, s_within, float(result.llf)


def _null_loglik(y: np.ndarray) -> float:
    """Log-likelihood of the pooled OLS null (s_within^2 = 0 boundary model)."""
    n = len(y)
    resid = y - y.mean()
    sigma2 = float(np.mean(resid**2))
    if sigma2 <= 0:
        sigma2 = 1e-12
    return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))


def compute_icc(s_between: float, s_within: float) -> float:
    """ICC = s_between^2 / (s_between^2 + s_within^2), clamped to [0, 1]."""
    denom = s_between + s_within
    if denom <= 0:
        return 0.0
    return float(np.clip(s_between / denom, 0.0, 1.0))


def state_eligibility_lrt(loglik_h1: float, loglik_h0: float) -> float:
    """Mixture-chi-square p-value for H0: s_within^2 = 0 (boundary parameter).

    Under H0 the LRT statistic follows a 50:50 mixture of chi2_0 (point mass at
    0) and chi2_1 (Self & Liang JASA 1987; Stram & Lee Biometrics 1994).
    p = 0.5 * P(chi2_1 > stat) for stat > 0; p = 1 for stat <= 0.
    """
    stat = 2.0 * (loglik_h1 - loglik_h0)
    if stat <= 0:
        return 1.0
    return float(0.5 * stats.chi2.sf(stat, df=1))


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted q-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def classify_factors(
    factors: MOFAFactors,
    *,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = 42,
) -> pd.DataFrame:
    """Classify each factor as trait / state / mixed via the LMM-LRT machinery.

    For Arm B this runs on the K MOFA+ factors; for Arms A and C the same
    routine runs post-hoc on the d_latent = 32 latent dimensions (the
    classification is purely descriptive there, no training-time enforcement).

    Returns
    -------
    DataFrame, one row per factor, columns:
    ``factor, icc, icc_ci_low, icc_ci_high, lrt_stat, lrt_pval, lrt_qval,
    classification``.
    """
    rng = np.random.default_rng(seed)
    subject = factors.subject_ids
    unique_subj = np.unique(subject)

    rows: list[dict[str, object]] = []
    pvals: list[float] = []
    for k in range(factors.n_factors):
        y = factors.scores[:, k].astype(np.float64)
        s_between, s_within, llf_h1 = _fit_random_intercept_lmm(y, subject)
        llf_h0 = _null_loglik(y)
        icc = compute_icc(s_between, s_within)
        pval = state_eligibility_lrt(llf_h1, llf_h0)
        pvals.append(pval)

        # Subject-clustered bootstrap CI on ICC (B per Lee's repeated-measures rule).
        boot_icc: list[float] = []
        for _ in range(n_bootstrap):
            picks = rng.choice(unique_subj, size=len(unique_subj), replace=True)
            mask = np.concatenate([np.where(subject == s)[0] for s in picks])
            yb = y[mask]
            sb_b, sw_b, _ = _fit_random_intercept_lmm(yb, subject[mask])
            boot_icc.append(compute_icc(sb_b, sw_b))
        ci_low, ci_high = np.percentile(boot_icc, [2.5, 97.5])

        rows.append(
            {
                "factor": k,
                "icc": icc,
                "icc_ci_low": float(ci_low),
                "icc_ci_high": float(ci_high),
                "lrt_stat": 2.0 * (llf_h1 - llf_h0),
                "lrt_pval": pval,
            }
        )

    qvals = _benjamini_hochberg(np.asarray(pvals))
    for row, q in zip(rows, qvals, strict=True):
        row["lrt_qval"] = float(q)
        icc = float(row["icc"])  # type: ignore[arg-type]
        rejects = q < LRT_FDR_THRESHOLD
        if icc > ICC_TRAIT_THRESHOLD and not rejects:
            row["classification"] = "trait"
        elif icc < ICC_STATE_THRESHOLD and rejects:
            row["classification"] = "state"
        else:
            row["classification"] = "mixed"

    return pd.DataFrame(rows)
