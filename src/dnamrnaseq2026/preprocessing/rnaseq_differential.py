"""Cell-type-corrected RNA-seq differential expression for Phase 1 Step 1.3.

Three contrasts (ANALYSIS_PLAN.md Step 1.3):
  (a) PRE-IOP only:   R vs NR at baseline.
  (b) POST-IOP only:  R vs NR post-treatment.
  (c) Paired delta:   per-subject log-CPM difference, R vs NR.

Strategy:
  - limma-voom is the preferred method (via rpy2) for cross-sectional
    contrasts (a) and (b) to match CellDMC conventions.
  - Python-native fallback (pydeseq2 or statsmodels OLS per gene) when rpy2
    is unavailable.
  - The delta contrast (c) is always implemented in Python: statsmodels OLS
    over genes, parallelised via joblib.
  - Cell-type summary covariate: PC1 of the 6-cell-type proportion matrix
    (avoids rank deficiency from crossing all 7 cell types in the design).
  - FDR: Benjamini-Hochberg per coefficient across genes.

Analysis plan reference: ANALYSIS_PLAN.md Step 1.3.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Covariate columns preferred for the design matrix
COVARIATE_COLS = [
    "Age",
    "sex",
    "smokingScore",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
]

# Cell-type columns used for the summary PC covariate
CELL_TYPE_COLS = ["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]


def _encode_covariates(df: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Encode a covariate DataFrame to a numeric float matrix.

    Categorical/object columns (e.g. sex: M/F) are label-encoded.
    NaN values are filled with column means.
    """
    encoded = df.copy()
    for col in encoded.columns:
        # pandas 2.x may use StringDtype for string columns instead of object;
        # check via api.types helpers so the guard covers object, category, and
        # the newer StringDtype without hardcoding dtype names.
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            valid = encoded[col].notna()
            # Convert to object dtype first: pandas 3.x StringDtype rejects
            # integer assignment via .loc, so we must widen to object before
            # writing the label-encoded integers back.
            encoded[col] = encoded[col].astype(object)
            encoded.loc[valid, col] = le.fit_transform(encoded.loc[valid, col].astype(str))
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    # numeric_only=True: skip any residual non-numeric columns during NaN imputation.
    col_means = encoded.mean(numeric_only=True)
    col_arr: np.ndarray[Any, Any] = encoded.fillna(col_means).fillna(0).values.astype(float)
    return col_arr


# ---------------------------------------------------------------------------
# Cell-type summary covariate
# ---------------------------------------------------------------------------


def make_cell_frac_pc1(cell_fracs: pd.DataFrame, sample_ids: list[str]) -> np.ndarray[Any, Any]:
    """Return PC1 of the cell-fraction matrix as a 1-D covariate.

    Using PC1 rather than all 6 cell types as separate predictors avoids
    rank-deficiency in the design matrix when n is small.

    Parameters
    ----------
    cell_fracs:
        DataFrame of cell fractions, rows = samples.
    sample_ids:
        Sample IDs to extract (must be a subset of cell_fracs.index).

    Returns
    -------
    np.ndarray[Any, Any]
        Shape (n_samples,), PC1 scores.
    """
    cf = cell_fracs.loc[sample_ids]
    available_cols = [c for c in CELL_TYPE_COLS if c in cf.columns]
    if len(available_cols) < 2:
        logger.warning("Fewer than 2 cell-type columns for PC1; returning zeros.")
        return np.zeros(len(sample_ids))
    cf_vals = cf[available_cols].fillna(0.0).values
    # Guard: if all-zero or constant (e.g. rpy2 fallback), return zeros
    if cf_vals.shape[0] == 0 or np.allclose(cf_vals, 0.0) or np.all(cf_vals == cf_vals[0]):
        logger.warning("Cell fraction matrix is constant or empty; returning zeros for PC1.")
        return np.zeros(len(sample_ids))
    scaler = StandardScaler()
    cf_scaled = scaler.fit_transform(cf_vals)
    pca = PCA(n_components=1)
    pc1: np.ndarray[Any, Any] = pca.fit_transform(cf_scaled)[:, 0]
    return pc1


# ---------------------------------------------------------------------------
# DE via Python OLS (per gene)
# ---------------------------------------------------------------------------


def _de_ols_single_gene(
    y: np.ndarray[Any, Any],
    response: np.ndarray[Any, Any],
    covariates: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Fit OLS for a single gene.

    Returns dict with keys: beta_response, p_response.
    Returns NaN if fitting fails.
    """
    from statsmodels.regression.linear_model import OLS

    n = len(y)
    intercept = np.ones((n, 1))
    design = np.hstack([intercept, response.reshape(-1, 1), covariates])
    valid = ~(np.isnan(y) | np.isnan(design).any(axis=1))
    if valid.sum() < design.shape[1] + 2:
        return {"beta_response": np.nan, "p_response": np.nan}
    try:
        fit = OLS(y[valid], design[valid]).fit()
        return {
            "beta_response": float(fit.params[1]),
            "p_response": float(fit.pvalues[1]),
        }
    except Exception:
        return {"beta_response": np.nan, "p_response": np.nan}


def _de_interaction_single_gene(
    y: np.ndarray[Any, Any],
    response: np.ndarray[Any, Any],
    cell_frac_pc1: np.ndarray[Any, Any],
    covariates: np.ndarray[Any, Any],
) -> dict[str, Any]:
    """Fit OLS with Response * cell_frac_PC1 interaction for one gene.

    Returns dict with keys: beta_response, beta_interaction, p_response, p_interaction.
    """
    from statsmodels.regression.linear_model import OLS

    n = len(y)
    intercept = np.ones((n, 1))
    interaction = (response * cell_frac_pc1).reshape(-1, 1)
    design = np.hstack(
        [
            intercept,
            response.reshape(-1, 1),
            cell_frac_pc1.reshape(-1, 1),
            interaction,
            covariates,
        ]
    )
    valid = ~(np.isnan(y) | np.isnan(design).any(axis=1))
    if valid.sum() < design.shape[1] + 2:
        return {
            "beta_response": np.nan,
            "beta_interaction": np.nan,
            "p_response": np.nan,
            "p_interaction": np.nan,
        }
    try:
        fit = OLS(y[valid], design[valid]).fit()
        return {
            "beta_response": float(fit.params[1]),
            "beta_interaction": float(fit.params[3]),
            "p_response": float(fit.pvalues[1]),
            "p_interaction": float(fit.pvalues[3]),
        }
    except Exception:
        return {
            "beta_response": np.nan,
            "beta_interaction": np.nan,
            "p_response": np.nan,
            "p_interaction": np.nan,
        }


def run_de_ols(
    log_cpm: np.ndarray[Any, Any],
    gene_ids: list[str],
    pdata: pd.DataFrame,
    cell_fracs: pd.DataFrame,
    response_col: str = "Response",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run per-gene OLS differential expression (cross-sectional).

    Fits: log_CPM ~ Response * cell_frac_PC1 + Age + sex + covariates.

    Parameters
    ----------
    log_cpm:
        2-D array (n_genes, n_samples), log-CPM values.
    gene_ids:
        Gene identifiers.
    pdata:
        pData2 with response column and covariates.
    cell_fracs:
        Cell-fraction DataFrame.
    response_col:
        Column name for R/NR label.
    n_jobs:
        joblib workers.

    Returns
    -------
    pd.DataFrame
        Columns: gene, beta_response, beta_interaction, p_response, p_interaction,
        q_response, q_interaction, n.
    """
    from joblib import Parallel, delayed

    resp_raw = pdata[response_col].astype(str).str.strip().str.upper()
    valid_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    pdata_sub = pdata[valid_mask]
    resp_enc: np.ndarray[Any, Any] = np.asarray(
        resp_raw[valid_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .to_numpy(),
        dtype=float,
    )

    sample_ids = list(pdata_sub.index)
    col_pos = [list(pdata.index).index(s) for s in sample_ids]
    lc_sub = log_cpm[:, col_pos]

    cf_pc1 = make_cell_frac_pc1(cell_fracs, sample_ids)

    cov_cols = [c for c in COVARIATE_COLS if c in pdata_sub.columns]
    cov_matrix: np.ndarray[Any, Any] = (
        _encode_covariates(pdata_sub[cov_cols]) if cov_cols else np.empty((len(sample_ids), 0))
    )

    def process_gene(gi: int) -> dict[str, Any]:
        res = _de_interaction_single_gene(lc_sub[gi], resp_enc, cf_pc1, cov_matrix)
        res["gene"] = gene_ids[gi]
        res["n"] = int(valid_mask.sum())
        return res

    rows = Parallel(n_jobs=n_jobs)(delayed(process_gene)(gi) for gi in range(len(gene_ids)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # FDR
    df["q_response"] = multipletests(df["p_response"].fillna(1.0).values, method="fdr_bh")[1]
    df["q_interaction"] = multipletests(df["p_interaction"].fillna(1.0).values, method="fdr_bh")[1]
    return df[
        [
            "gene",
            "beta_response",
            "beta_interaction",
            "p_response",
            "p_interaction",
            "q_response",
            "q_interaction",
            "n",
        ]
    ]


def run_de_delta(
    delta_log_cpm: np.ndarray[Any, Any],
    gene_ids: list[str],
    pdata_paired: pd.DataFrame,
    delta_cell_fracs: pd.DataFrame,
    response_col: str = "Response",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run per-gene OLS delta-contrast differential expression.

    Fits: delta_logCPM ~ Response * delta_cell_frac_PC1 + covariates (per gene).

    Parameters
    ----------
    delta_log_cpm:
        2-D array (n_genes, n_paired_subjects), log-CPM deltas.
    gene_ids:
        Gene identifiers.
    pdata_paired:
        pData2 for paired subjects only, one row per subject.
    delta_cell_fracs:
        Delta cell fractions per paired subject.
    response_col:
        Response column in pdata_paired.
    n_jobs:
        joblib workers.

    Returns
    -------
    pd.DataFrame
        Columns: gene, beta_response, beta_interaction, p_response, p_interaction,
        q_response, q_interaction, n.
    """
    from joblib import Parallel, delayed

    resp_raw = pdata_paired[response_col].astype(str).str.strip().str.upper()
    valid_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    pdata_sub = pdata_paired[valid_mask]
    resp_enc: np.ndarray[Any, Any] = np.asarray(
        resp_raw[valid_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .to_numpy(),
        dtype=float,
    )

    sample_ids = list(pdata_sub.index)
    col_pos_all = list(pdata_paired.index)
    col_pos = [col_pos_all.index(s) for s in sample_ids]
    dlc_sub = delta_log_cpm[:, col_pos]

    cf_pc1 = make_cell_frac_pc1(delta_cell_fracs, sample_ids)

    cov_cols = [c for c in COVARIATE_COLS if c in pdata_sub.columns]
    cov_matrix: np.ndarray[Any, Any] = (
        _encode_covariates(pdata_sub[cov_cols]) if cov_cols else np.empty((len(sample_ids), 0))
    )

    def process_gene(gi: int) -> dict[str, Any]:
        res = _de_interaction_single_gene(dlc_sub[gi], resp_enc, cf_pc1, cov_matrix)
        res["gene"] = gene_ids[gi]
        res["n"] = int(valid_mask.sum())
        return res

    rows = Parallel(n_jobs=n_jobs)(delayed(process_gene)(gi) for gi in range(len(gene_ids)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_response"] = multipletests(df["p_response"].fillna(1.0).values, method="fdr_bh")[1]
    df["q_interaction"] = multipletests(df["p_interaction"].fillna(1.0).values, method="fdr_bh")[1]
    return df[
        [
            "gene",
            "beta_response",
            "beta_interaction",
            "p_response",
            "p_interaction",
            "q_response",
            "q_interaction",
            "n",
        ]
    ]


# ---------------------------------------------------------------------------
# Rescue check 1.3.5: 0-X centroid permutation on corrected matrices
# ---------------------------------------------------------------------------


def rescue_check_1_3_5(
    corrected_rnaseq_emory: np.ndarray[Any, Any],
    gene_ids_emory: list[str],
    pdata_emory: pd.DataFrame,
    gse_matrix: np.ndarray[Any, Any],
    gene_ids_gse: np.ndarray[Any, Any],
    gse_labels: np.ndarray[Any, Any],
    response_col: str = "Response",
    seed: int = 42,
    n_permutations: int = 2000,
    top_genes: int = 2000,
) -> dict[str, Any]:
    """Rescue check 1.3.5: 0-X centroid permutation on corrected RNA-seq.

    After cell-type correction, re-run the cross-disorder centroid test
    (Gate 0-X) on corrected matrices.

    Parameters
    ----------
    corrected_rnaseq_emory:
        2-D array (n_genes_emory, n_samples), cell-type-corrected log-CPM.
    gene_ids_emory:
        Gene IDs for rows of corrected_rnaseq_emory.
    pdata_emory:
        pData2 for Emory; must contain response_col and Visit columns.
    gse_matrix:
        2-D array (n_genes_gse, n_gse_samples), harmonised GSE98793 expression.
    gene_ids_gse:
        Gene IDs for rows of gse_matrix.
    gse_labels:
        1-D array: 1 = TRD-inflammatory, 0 = control, per column of gse_matrix.
    response_col:
        Response column in pdata_emory.
    seed, n_permutations, top_genes:
        PCA and permutation parameters.

    Returns
    -------
    dict with keys: d_NR_TRD, d_R_TRD, permanova_p, verdict, rescue_passed.
    """
    # Find PRE-IOP samples in Emory
    visit_col = next(
        (
            c
            for c in ["Visit", "visit", "VISIT", "TimePoint", "time_point"]
            if c in pdata_emory.columns
        ),
        None,
    )
    pre_mask_arr: np.ndarray[Any, Any]
    if visit_col is None:
        pre_mask_arr = np.ones(len(pdata_emory), dtype=bool)
    else:
        pre_mask_arr = np.asarray(
            pdata_emory[visit_col]
            .astype(str)
            .str.upper()
            .isin(["PRE", "PRE-IOP", "BL", "BASELINE", "T0", "0"])
            .to_numpy(),
            dtype=bool,
        )

    resp_raw = pdata_emory[response_col].astype(str).str.strip().str.upper()
    r_mask_arr: np.ndarray[Any, Any] = pre_mask_arr & np.asarray(
        resp_raw.isin(["R", "RESPONDER", "1"]).to_numpy(), dtype=bool
    )
    nr_mask_arr: np.ndarray[Any, Any] = pre_mask_arr & np.asarray(
        resp_raw.isin(["NR", "NON-RESPONDER", "0"]).to_numpy(), dtype=bool
    )

    emory_col_idx = np.arange(len(pdata_emory))
    r_cols = emory_col_idx[r_mask_arr]
    nr_cols = emory_col_idx[nr_mask_arr]

    # Gene intersection
    common_genes = list(set(gene_ids_emory) & set(gene_ids_gse.tolist()))
    if len(common_genes) < 100:
        return {
            "d_NR_TRD": np.nan,
            "d_R_TRD": np.nan,
            "permanova_p": np.nan,
            "verdict": "INSUFFICIENT_OVERLAP",
            "rescue_passed": False,
        }

    emory_gene_pos = {g: i for i, g in enumerate(gene_ids_emory)}
    gse_gene_pos = {g: i for i, g in enumerate(gene_ids_gse.tolist())}
    e_idx = np.array([emory_gene_pos[g] for g in common_genes])
    g_idx = np.array([gse_gene_pos[g] for g in common_genes])

    emory_sub = corrected_rnaseq_emory[e_idx, :]
    gse_sub = gse_matrix[g_idx, :]

    # Variance filter
    combined = np.hstack([emory_sub, gse_sub])
    var = np.nanvar(combined, axis=1)
    top_idx = np.argsort(var)[-top_genes:]
    emory_top = emory_sub[top_idx, :].T
    gse_top = gse_sub[top_idx, :].T

    # Centroids
    centroid_r = emory_top[r_cols].mean(axis=0)
    centroid_nr = emory_top[nr_cols].mean(axis=0)
    trd_mask = gse_labels == 1
    centroid_trd = gse_top[trd_mask].mean(axis=0)

    d_nr_trd = float(np.linalg.norm(centroid_nr - centroid_trd))
    d_r_trd = float(np.linalg.norm(centroid_r - centroid_trd))

    # Permutation test: is d_nr_trd < d_r_trd by chance?
    rng = np.random.default_rng(seed)
    all_emory = emory_top  # n_samples x n_genes
    all_labels = np.array(["R"] * len(r_cols) + ["NR"] * len(nr_cols))
    pre_samples = np.concatenate([r_cols, nr_cols])
    all_emory_pre = all_emory[pre_samples]

    obs_diff = d_nr_trd - d_r_trd  # negative = NR closer to TRD (expected direction)
    null_diffs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(all_labels)
        c_r = all_emory_pre[perm_labels == "R"].mean(axis=0)
        c_nr = all_emory_pre[perm_labels == "NR"].mean(axis=0)
        null_diffs.append(
            float(np.linalg.norm(c_nr - centroid_trd) - np.linalg.norm(c_r - centroid_trd))
        )

    perm_p = float((np.array(null_diffs) <= obs_diff).mean())  # one-tailed: NR closer to TRD

    rescue_passed = perm_p < 0.05 and obs_diff < 0
    if rescue_passed:
        verdict = "RESCUE_PASS"
    elif perm_p < 0.15 and obs_diff < 0:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        "d_NR_TRD": d_nr_trd,
        "d_R_TRD": d_r_trd,
        "obs_diff": obs_diff,
        "perm_p": perm_p,
        "verdict": verdict,
        "rescue_passed": rescue_passed,
        "n_genes": len(common_genes),
    }
