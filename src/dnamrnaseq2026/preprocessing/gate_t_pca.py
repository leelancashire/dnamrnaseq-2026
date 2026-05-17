"""Gate 0-T: PCA of paired delta-vectors with statistical tests.

Runs PCA on the joint (CpG + gene) delta-feature matrix, tests for
responder vs non-responder separation in PC space via PERMANOVA and
Hotelling's T^2 test.

Acceptance criteria (ANALYSIS_PLAN.md Step 0-T):
  - PASS: PERMANOVA p < 0.05 AND centroid Cohen's d >= 0.3 on >=1 of first 5 PCs.
  - MARGINAL: PERMANOVA p in [0.05, 0.15].
  - FAIL: PERMANOVA p >= 0.15.

Analysis plan reference: ANALYSIS_PLAN.md Step 0-T (Method steps 5-8).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Acceptance thresholds (ANALYSIS_PLAN.md Step 0-T)
PERMANOVA_PASS_THRESHOLD = 0.05
PERMANOVA_MARGINAL_THRESHOLD = 0.15
COHENS_D_THRESHOLD = 0.3
N_PCS = 5
BOOTSTRAP_N = 2000


def run_pca(
    delta_matrix: pd.DataFrame,
    n_components: int = N_PCS,
) -> tuple[pd.DataFrame, PCA]:
    """Run PCA on the scaled delta-matrix.

    Parameters
    ----------
    delta_matrix:
        Scaled joint delta-feature matrix (n_subjects x n_features).
        Should already be zero-mean, unit-variance scaled.
    n_components:
        Number of principal components to retain.

    Returns
    -------
    tuple[pd.DataFrame, PCA]
        PC scores DataFrame (n_subjects x n_components) and fitted PCA object.
    """
    pca = PCA(n_components=min(n_components, delta_matrix.shape[0] - 1, delta_matrix.shape[1]))
    scores = pca.fit_transform(delta_matrix.values)
    pc_cols = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores_df = pd.DataFrame(scores, index=delta_matrix.index, columns=pc_cols)
    logger.info(
        "PCA: %d PCs, explained variance: %s",
        pca.n_components_,
        [f"{v:.3f}" for v in pca.explained_variance_ratio_],
    )
    return scores_df, pca


def compute_cohens_d_per_pc(
    pc_scores: pd.DataFrame,
    response: pd.Series,
    r_label: str = "R",
    nr_label: str = "NR",
) -> dict[str, float]:
    """Compute Cohen's d (centroid separation) for each PC.

    Parameters
    ----------
    pc_scores:
        PCA scores DataFrame (n_subjects x n_pcs).
    response:
        Response labels ('R'/'NR') indexed by subject ID.
    r_label:
        Label for responders.
    nr_label:
        Label for non-responders.

    Returns
    -------
    dict
        PC name -> Cohen's d (absolute value).
    """
    aligned = response.reindex(pc_scores.index)
    r_mask = aligned == r_label
    nr_mask = aligned == nr_label

    cohens_d: dict[str, float] = {}
    for col in pc_scores.columns:
        r_vals = pc_scores.loc[r_mask, col].dropna().values
        nr_vals = pc_scores.loc[nr_mask, col].dropna().values
        if len(r_vals) < 2 or len(nr_vals) < 2:
            cohens_d[col] = 0.0
            continue
        pooled_sd = float(
            np.sqrt(
                ((len(r_vals) - 1) * r_vals.std() ** 2 + (len(nr_vals) - 1) * nr_vals.std() ** 2)
                / (len(r_vals) + len(nr_vals) - 2)
            )
        )
        if pooled_sd == 0:
            cohens_d[col] = 0.0
        else:
            cohens_d[col] = float(abs(r_vals.mean() - nr_vals.mean()) / pooled_sd)

    logger.info("Cohen's d per PC: %s", {k: f"{v:.3f}" for k, v in cohens_d.items()})
    return cohens_d


def run_permanova(
    pc_scores: pd.DataFrame,
    response: pd.Series,
    n_permutations: int = BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, Any]:
    """Run PERMANOVA on PC scores, grouped by Response.

    Uses Euclidean distance in PC space. Permutes Response labels.
    Note: full PERMANOVA via skbio requires distance matrices; here we implement
    a pseudo-F permutation test directly on PC space for speed and to avoid
    adding a heavy dependency.

    Parameters
    ----------
    pc_scores:
        PCA scores (n_subjects x n_pcs).
    response:
        Response labels indexed by subject ID.
    n_permutations:
        Number of permutations.
    seed:
        Random seed.

    Returns
    -------
    dict
        Keys: 'f_statistic', 'p_value', 'n_permutations'.
    """
    aligned = response.reindex(pc_scores.index).dropna()
    valid_subjects = aligned.index
    scores = pc_scores.loc[valid_subjects].values
    labels = aligned.values

    def pseudo_f(
        sc: np.ndarray[Any, np.dtype[np.float64]],
        lb: np.ndarray[Any, np.dtype[Any]],
    ) -> float:
        """Compute pseudo-F: ratio of between-group to within-group SS."""
        unique_labels = np.unique(lb)
        grand_centroid = sc.mean(axis=0)
        ss_total = float(np.sum((sc - grand_centroid) ** 2))

        ss_within = 0.0
        for lab in unique_labels:
            group = sc[lb == lab]
            group_centroid = group.mean(axis=0)
            ss_within += float(np.sum((group - group_centroid) ** 2))

        ss_between = ss_total - ss_within
        n_groups = len(unique_labels)
        n_obs = len(lb)
        df_between = float(n_groups - 1)
        df_within = float(n_obs - n_groups)
        if df_within == 0 or ss_within == 0:
            return 0.0
        return float((ss_between / df_between) / (ss_within / df_within))

    obs_f = pseudo_f(scores, labels)

    rng = np.random.default_rng(seed)
    perm_fs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        perm_fs.append(pseudo_f(scores, perm_labels))

    perm_fs_arr = np.array(perm_fs)
    p_value = float(np.mean(perm_fs_arr >= obs_f))

    logger.info("PERMANOVA: F=%.4f, p=%.4f (n_perm=%d)", obs_f, p_value, n_permutations)
    return {
        "f_statistic": obs_f,
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def run_hotelling_t2(
    pc_scores: pd.DataFrame,
    response: pd.Series,
    r_label: str = "R",
    nr_label: str = "NR",
) -> dict[str, Any]:
    """Run Hotelling's T^2 test on multivariate PC centroid difference.

    Per-PC t-tests and combined T^2 via pingouin.multivariate_ttest
    (falls back to per-PC if pingouin unavailable).

    Parameters
    ----------
    pc_scores:
        PCA scores (n_subjects x n_pcs).
    response:
        Response labels indexed by subject ID.
    r_label:
        Responder label.
    nr_label:
        Non-responder label.

    Returns
    -------
    dict
        Keys: 'per_pc_t_p', 'hotelling_p' (or None if pingouin unavailable).
    """
    aligned = response.reindex(pc_scores.index)
    r_mask = aligned == r_label
    nr_mask = aligned == nr_label

    per_pc: dict[str, float] = {}
    for col in pc_scores.columns:
        r_vals = pc_scores.loc[r_mask, col].dropna()
        nr_vals = pc_scores.loc[nr_mask, col].dropna()
        if len(r_vals) < 2 or len(nr_vals) < 2:
            per_pc[col] = 1.0
            continue
        _, p = stats.ttest_ind(r_vals, nr_vals)
        per_pc[col] = float(p)

    # Try Hotelling's T^2 via pingouin
    hotelling_p = None
    try:
        import pingouin

        r_mat = pc_scores.loc[r_mask].dropna().values
        nr_mat = pc_scores.loc[nr_mask].dropna().values
        result = pingouin.multivariate_ttest(r_mat, nr_mat)
        hotelling_p = float(result["pval"].iloc[0])
        logger.info("Hotelling T^2: p=%.4f", hotelling_p)
    except (ImportError, Exception) as e:
        logger.warning("pingouin Hotelling T^2 unavailable (%s); using per-PC t-tests only.", e)

    logger.info("Per-PC t-test p-values: %s", {k: f"{v:.4f}" for k, v in per_pc.items()})
    return {
        "per_pc_t_p": per_pc,
        "hotelling_p": hotelling_p,
    }


def determine_gate_0t_verdict(
    permanova_results: dict[str, Any],
    cohens_d: dict[str, float],
) -> str:
    """Return PASS, MARGINAL, or FAIL verdict for Gate 0-T.

    Parameters
    ----------
    permanova_results:
        Output of run_permanova.
    cohens_d:
        Per-PC Cohen's d dict from compute_cohens_d_per_pc.

    Returns
    -------
    str
        'PASS', 'MARGINAL', or 'FAIL'.
    """
    p = permanova_results["p_value"]
    max_d = max(cohens_d.values()) if cohens_d else 0.0

    if p < PERMANOVA_PASS_THRESHOLD and max_d >= COHENS_D_THRESHOLD:
        return "PASS"
    if PERMANOVA_PASS_THRESHOLD <= p < PERMANOVA_MARGINAL_THRESHOLD:
        return "MARGINAL"
    if p < PERMANOVA_PASS_THRESHOLD:
        # p passes but d doesn't meet threshold
        return "MARGINAL"
    return "FAIL"
