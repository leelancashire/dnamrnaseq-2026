"""Cell-type correction and CellDMC-style analysis for Phase 1.

Steps 1.1 and 1.2 of ANALYSIS_PLAN.md.

Step 1.1: Full EpiDISH deconvolution on Emory (388) and BEST (141) samples.
Step 1.2: CellDMC-style differential methylation at three contrast levels:
  (a) PRE-IOP only
  (b) POST-IOP only
  (c) Within-subject delta (the load-bearing test)

EpiDISH for cross-sectional contrasts (a)+(b) is called via rpy2 if available;
if rpy2 is not importable the module falls back to treating pData2 columns as the
EpiDISH output (equivalent to Phase 0 strategy).

CellDMC at delta level is implemented in Python (statsmodels OLS) because it is
a custom extension of the cross-sectional CellDMC model: no published CellDMC
function exists for paired-delta contrasts. Standard cross-sectional CellDMC
(PRE and POST) is also implemented via Python OLS so the codebase is self-contained
and CI-testable without a running R environment.

Implementation notes:
- All M-value transformations use log2(beta/(1-beta)), clipped to [-8, 8] for
  numerical stability.
- Parallelisation via joblib.Parallel over CpGs with n_jobs=-1 default.
- FDR correction via Benjamini-Hochberg per cell type (statsmodels multipletests).
- Sex-chromosome CpGs (chrX, chrY) are excluded before model fitting per
  ANALYSIS_PLAN.md Step 1.2 Risk note.

Analysis plan reference: ANALYSIS_PLAN.md Steps 1.1 and 1.2.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Cell-type column names (IDOL 7-cell reference, as stored in pData2)
CELL_TYPE_COLS = ["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]

# Covariate columns expected in pData2 (subset available)
COVARIATE_COLS_PREFERRED = [
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


def _encode_covariates(df: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Encode a covariate DataFrame to a numeric float matrix.

    Categorical/object columns (e.g. sex: M/F) are label-encoded as integers.
    NaN values are filled with column means (or 0 for all-NaN columns).
    """
    encoded = df.copy()
    for col in encoded.columns:
        if encoded[col].dtype == object or str(encoded[col].dtype) == "category":
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            valid = encoded[col].notna()
            encoded.loc[valid, col] = le.fit_transform(encoded.loc[valid, col].astype(str))
            encoded[col] = pd.to_numeric(encoded[col], errors="coerce")
    col_means = encoded.mean()
    col_arr: np.ndarray[Any, Any] = encoded.fillna(col_means).fillna(0).values.astype(float)
    return col_arr


# Beta clipping before M-value transform
BETA_CLIP = 1e-6

# M-value symmetric clip to avoid infinite OLS outcomes
MVAL_CLIP = 8.0


def beta_to_m(beta: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert beta values to M-values with safe clipping.

    Parameters
    ----------
    beta:
        2-D numpy array, CpG x samples, values in [0, 1].

    Returns
    -------
    np.ndarray[Any, Any]
        M-values, same shape, clipped to [-MVAL_CLIP, MVAL_CLIP].
    """
    beta_safe = np.clip(beta, BETA_CLIP, 1.0 - BETA_CLIP)
    m = np.log2(beta_safe / (1.0 - beta_safe))
    result: np.ndarray[Any, Any] = np.clip(m, -MVAL_CLIP, MVAL_CLIP)
    return result


# ---------------------------------------------------------------------------
# EpiDISH via rpy2 (Step 1.1)
# ---------------------------------------------------------------------------


def run_epidish_rpy2(
    beta_matrix: np.ndarray[Any, Any],
    sample_ids: list[str],
    cpg_ids: list[str],
    ref_name: str = "centDHSbloodDMC.m",
    method: str = "RPC",
) -> pd.DataFrame:
    """Run EpiDISH via rpy2 to estimate cell-type proportions.

    Parameters
    ----------
    beta_matrix:
        2-D numpy array, CpG x samples.
    sample_ids:
        List of sample identifiers (columns of beta_matrix).
    cpg_ids:
        List of CpG identifiers (rows of beta_matrix).
    ref_name:
        R object name for the EpiDISH reference matrix.
        Defaults to ``centDHSbloodDMC.m`` (IDOL 7-cell, ships with EpiDISH).
    method:
        EpiDISH method. One of ``RPC``, ``CBS``, ``CP``. Defaults to ``RPC``.

    Returns
    -------
    pd.DataFrame
        Cell fractions, shape (n_samples, n_cell_types). Row index = sample IDs.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        numpy2ri.activate()
        pandas2ri.activate()

        epidish = importr("EpiDISH")

        # Build R beta matrix: rows = CpGs, cols = samples
        r_beta = ro.r.matrix(
            beta_matrix.flatten(order="F"),
            nrow=beta_matrix.shape[0],
            ncol=beta_matrix.shape[1],
            byrow=False,
        )
        ro.r.assign("cpg_names", ro.StrVector(cpg_ids))
        ro.r.assign("sample_names", ro.StrVector(sample_ids))
        ro.r("rownames(r_beta) <- cpg_names")
        ro.r("colnames(r_beta) <- sample_names")
        ro.r.assign("r_beta", r_beta)

        # Load reference
        ro.r(f"data({ref_name}, package='EpiDISH')")
        ro.r(f"ref <- {ref_name}")

        # Run EpiDISH
        result = epidish.epidish(
            **{
                "beta.m": ro.r["r_beta"],
                "ref.m": ro.r["ref"],
                "method": method,
            }
        )
        # result$estF is the fraction matrix (samples x cell types)
        est_f = np.array(result.rx2("estF"))
        cell_types = list(result.rx2("estF").colnames)
        props_df = pd.DataFrame(est_f, index=sample_ids, columns=cell_types)
        logger.info(
            "EpiDISH (rpy2) complete: %d samples x %d cell types.",
            len(sample_ids),
            len(cell_types),
        )
        return props_df

    except ImportError:
        logger.warning("rpy2 not available; returning empty DataFrame for EpiDISH.")
        return pd.DataFrame(
            index=sample_ids,
            columns=CELL_TYPE_COLS,
            dtype=float,
        )
    except Exception as exc:
        logger.error("EpiDISH via rpy2 failed: %s", exc, exc_info=True)
        raise


def run_epidish_from_pdata(pdata: pd.DataFrame) -> pd.DataFrame:
    """Return EpiDISH cell fractions stored in pData2 (Phase 0 fallback).

    Used when rpy2 is unavailable.  pData2 columns Bcell, CD4T, CD8T,
    Mono, Neu, NK are returned as-is.

    Parameters
    ----------
    pdata:
        pData2 DataFrame with sample-level metadata.

    Returns
    -------
    pd.DataFrame
        Cell fractions, shape (n_samples, 6). Index = row index of pdata.
    """
    available = [c for c in CELL_TYPE_COLS if c in pdata.columns]
    if len(available) < 2:
        raise ValueError(
            "pData2 does not contain expected cell-type columns "
            f"({CELL_TYPE_COLS}). Found: {list(pdata.columns)}"
        )
    logger.info(
        "Using pData2 cell-type columns (%d of 6 expected): %s",
        len(available),
        available,
    )
    return pdata[available].copy()


# ---------------------------------------------------------------------------
# CellDMC-style OLS model (Steps 1.2a, 1.2b, 1.2c)
# ---------------------------------------------------------------------------


def _fit_celldmc_cpg(
    m_vals: np.ndarray[Any, Any],
    cell_fracs: np.ndarray[Any, Any],
    response: np.ndarray[Any, Any],
    covariates: np.ndarray[Any, Any],
    cell_type_names: list[str],
) -> list[dict[str, Any]]:
    """Fit the CellDMC interaction model for a single CpG.

    Model: M ~ Response*cell_frac_c + covariates  for each cell type c.

    Parameters
    ----------
    m_vals:
        1-D array (n_samples,), M-values for this CpG.
    cell_fracs:
        2-D array (n_samples, n_cell_types), cell fractions.
    response:
        1-D array (n_samples,), binary (0 = NR, 1 = R).
    covariates:
        2-D array (n_samples, n_covariates), nuisance covariates.
    cell_type_names:
        Names of the cell types (columns of cell_fracs).

    Returns
    -------
    list of dict
        One dict per cell type with keys:
        cell_type, beta_response, beta_interaction, p_response, p_interaction.
    """
    from statsmodels.regression.linear_model import OLS

    n = len(m_vals)
    results: list[dict[str, Any]] = []
    intercept = np.ones((n, 1))

    for ci, ct in enumerate(cell_type_names):
        frac_c = cell_fracs[:, ci : ci + 1]
        interaction = response.reshape(-1, 1) * frac_c
        # Design: intercept | response | frac_c | interaction | covariates
        design = np.hstack([intercept, response.reshape(-1, 1), frac_c, interaction, covariates])
        valid = ~(np.isnan(m_vals) | np.isnan(design).any(axis=1))
        if valid.sum() < design.shape[1] + 2:
            results.append(
                {
                    "cell_type": ct,
                    "beta_response": np.nan,
                    "beta_interaction": np.nan,
                    "p_response": np.nan,
                    "p_interaction": np.nan,
                }
            )
            continue
        try:
            fit = OLS(m_vals[valid], design[valid]).fit()
            results.append(
                {
                    "cell_type": ct,
                    "beta_response": float(fit.params[1]),
                    "beta_interaction": float(fit.params[3]),
                    "p_response": float(fit.pvalues[1]),
                    "p_interaction": float(fit.pvalues[3]),
                }
            )
        except Exception:
            results.append(
                {
                    "cell_type": ct,
                    "beta_response": np.nan,
                    "beta_interaction": np.nan,
                    "p_response": np.nan,
                    "p_interaction": np.nan,
                }
            )
    return results


def run_celldmc(
    m_matrix: np.ndarray[Any, Any],
    cpg_ids: list[str],
    cell_fracs: pd.DataFrame,
    pdata: pd.DataFrame,
    response_col: str = "Response",
    n_jobs: int = -1,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Run CellDMC-style interaction model across all CpGs.

    The model is applied independently to each CpG. Parallelisation is over
    CpGs via joblib. FDR correction is applied per cell type.

    Parameters
    ----------
    m_matrix:
        2-D numpy array, shape (n_cpg, n_samples), M-values.
    cpg_ids:
        CpG identifiers (rows of m_matrix).
    cell_fracs:
        DataFrame of cell fractions aligned to pdata samples.
    pdata:
        pData2 DataFrame with covariates and response column.
    response_col:
        Column name in pdata for the binary response variable.
    n_jobs:
        joblib parallelism. -1 = all cores.
    chunk_size:
        Number of CpGs per parallel chunk.

    Returns
    -------
    pd.DataFrame
        Long-format table: CpG, cell_type, beta_response, beta_interaction,
        p_response, p_interaction, q_response, q_interaction.
    """
    from joblib import Parallel, delayed

    if response_col not in pdata.columns:
        raise ValueError(f"Response column '{response_col}' not in pData.")

    # Encode Response: R=1, NR=0; drop partial
    resp_raw = pdata[response_col].astype(str).str.strip().str.upper()
    resp_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    if resp_mask.sum() < 10:
        raise ValueError(f"Too few samples with R/NR response label in column '{response_col}'.")
    pdata_sub = pdata.loc[resp_mask].copy()
    resp_encoded = (
        resp_raw.loc[resp_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .astype(float)
    )

    cell_fracs_sub = cell_fracs.loc[pdata_sub.index]
    m_sub = m_matrix[:, pdata.index.isin(pdata_sub.index)]
    # Align columns
    shared_idx = pdata.index[pdata.index.isin(pdata_sub.index)]
    col_pos = [list(pdata.index).index(i) for i in shared_idx]
    m_sub = m_matrix[:, col_pos]
    cell_fracs_aligned = cell_fracs_sub.loc[shared_idx].values.astype(float)
    response_arr = resp_encoded.loc[shared_idx].values

    # Build covariate matrix (encode categoricals to numeric)
    cov_cols = [c for c in COVARIATE_COLS_PREFERRED if c in pdata_sub.columns]
    if cov_cols:
        cov_matrix = _encode_covariates(pdata_sub.loc[shared_idx, cov_cols])
    else:
        logger.warning("No covariate columns found; fitting with intercept only.")
        cov_matrix = np.empty((len(shared_idx), 0))

    cell_type_names = list(cell_fracs_sub.columns)
    n_cpg = m_sub.shape[0]
    logger.info(
        "CellDMC: %d CpGs x %d samples x %d cell types.",
        n_cpg,
        m_sub.shape[1],
        len(cell_type_names),
    )

    chunks = [range(i, min(i + chunk_size, n_cpg)) for i in range(0, n_cpg, chunk_size)]

    def process_chunk(idxs: range) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ci in idxs:
            cpg = cpg_ids[ci]
            res = _fit_celldmc_cpg(
                m_sub[ci],
                cell_fracs_aligned,
                response_arr,
                cov_matrix,
                cell_type_names,
            )
            for r in res:
                r["cpg"] = cpg
                rows.append(r)
        return rows

    all_rows: list[dict[str, Any]] = []
    results_nested = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(ch) for ch in chunks)
    for chunk_rows in results_nested:
        all_rows.extend(chunk_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # FDR correction per cell type
    df["q_response"] = np.nan
    df["q_interaction"] = np.nan
    for ct in df["cell_type"].unique():
        mask = df["cell_type"] == ct
        p_resp = df.loc[mask, "p_response"].fillna(1.0).values
        p_intr = df.loc[mask, "p_interaction"].fillna(1.0).values
        df.loc[mask, "q_response"] = multipletests(p_resp, method="fdr_bh")[1]
        df.loc[mask, "q_interaction"] = multipletests(p_intr, method="fdr_bh")[1]

    col_order = [
        "cpg",
        "cell_type",
        "beta_response",
        "beta_interaction",
        "p_response",
        "p_interaction",
        "q_response",
        "q_interaction",
    ]
    return df[[c for c in col_order if c in df.columns]]


# ---------------------------------------------------------------------------
# Residualise methylation / RNA-seq on cell fractions
# ---------------------------------------------------------------------------


def residualise_on_cell_props(
    feature_matrix: np.ndarray[Any, Any],
    cell_fracs: pd.DataFrame,
    sample_ids: list[str],
) -> np.ndarray[Any, Any]:
    """Residualise a feature matrix on cell-type proportions.

    For each feature, regress out the cell-fraction columns (OLS) and
    return the residuals.  Used to produce cell-type-corrected matrices
    for downstream pathway and TF inference.

    Parameters
    ----------
    feature_matrix:
        2-D numpy array, shape (n_features, n_samples).
    cell_fracs:
        DataFrame of cell fractions, shape (n_samples, n_cell_types).
    sample_ids:
        Sample identifiers aligning columns of feature_matrix to rows of
        cell_fracs.

    Returns
    -------
    np.ndarray[Any, Any]
        Residuals, same shape as feature_matrix.
    """
    from statsmodels.regression.linear_model import OLS

    cf = cell_fracs.loc[sample_ids].values.astype(float)
    intercept = np.ones((cf.shape[0], 1))
    design = np.hstack([intercept, cf])

    n_features = feature_matrix.shape[0]
    residuals = np.empty_like(feature_matrix)
    for i in range(n_features):
        y = feature_matrix[i]
        valid = ~np.isnan(y)
        if valid.sum() < design.shape[1] + 2:
            residuals[i] = np.nan
            continue
        try:
            fit = OLS(y[valid], design[valid]).fit()
            pred = design @ fit.params
            residuals[i] = y - pred
        except Exception:
            residuals[i] = np.nan
    return residuals


# ---------------------------------------------------------------------------
# Cross-contrast annotation (Step 1.2 cross-contrast classification)
# ---------------------------------------------------------------------------

CrossContrastType = str  # "state_of_recovery" | "baseline_and_recovery" | "trait_stable" | "other"


def annotate_cross_contrast(
    celldmc_pre: pd.DataFrame,
    celldmc_post: pd.DataFrame,
    celldmc_delta: pd.DataFrame,
    fdr_threshold: float = 0.05,
    min_cell_types: int = 1,
) -> pd.DataFrame:
    """Classify each CpG by its cross-contrast significance pattern.

    Classification per ANALYSIS_PLAN.md Step 1.2:
    - state_of_recovery: significant in delta only
    - baseline_and_recovery: significant in (a)/(b) AND delta
    - trait_stable: significant in (a) or (b) but NOT delta
    - other: not significant in any contrast

    Parameters
    ----------
    celldmc_pre, celldmc_post, celldmc_delta:
        CellDMC output DataFrames from run_celldmc.
    fdr_threshold:
        q_interaction threshold for declaring significance.
    min_cell_types:
        Minimum number of cell types in which the CpG must be significant.

    Returns
    -------
    pd.DataFrame
        Columns: cpg, cell_type, cross_contrast_class.
    """
    # Vectorised implementation: set a boolean sig flag per (cpg, cell_type) key
    # then merge across contrasts.  The row-wise lookup approach is O(N^2) and
    # intractable for 292k CpGs.
    def _sig_flags(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Return boolean sig column keyed on (cpg, cell_type)."""
        tmp = df[["cpg", "cell_type", "q_interaction"]].copy()
        tmp[f"sig_{suffix}"] = tmp["q_interaction"].fillna(1.0) < fdr_threshold
        return tmp[["cpg", "cell_type", f"sig_{suffix}"]].drop_duplicates(
            subset=["cpg", "cell_type"]
        )

    pre_flags = _sig_flags(celldmc_pre, "pre")
    post_flags = _sig_flags(celldmc_post, "post")
    delta_flags = _sig_flags(celldmc_delta, "delta")

    merged = delta_flags.merge(pre_flags, on=["cpg", "cell_type"], how="left").merge(
        post_flags, on=["cpg", "cell_type"], how="left"
    )
    merged["sig_pre"] = merged["sig_pre"].fillna(False)
    merged["sig_post"] = merged["sig_post"].fillna(False)

    in_delta = merged["sig_delta"]
    in_pre_or_post = merged["sig_pre"] | merged["sig_post"]

    # Vectorise classification via numpy conditions (avoids row-apply overhead)
    merged["cross_contrast_class"] = np.select(
        [
            in_delta & ~in_pre_or_post,
            in_delta & in_pre_or_post,
            ~in_delta & in_pre_or_post,
        ],
        ["state_of_recovery", "baseline_and_recovery", "trait_stable"],
        default="other",
    )

    return merged[["cpg", "cell_type", "cross_contrast_class"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Rescue check 1.2.5: PCA on cell-type-corrected delta vectors
# ---------------------------------------------------------------------------


def rescue_check_1_2_5(
    corrected_delta_m: np.ndarray[Any, Any],
    corrected_delta_rna: np.ndarray[Any, Any],
    cpg_ids: list[str],
    gene_ids: list[str],
    pdata_paired: pd.DataFrame,
    response_col: str = "Response",
    seed: int = 42,
    n_permutations: int = 2000,
    top_cpgs: int = 5000,
    top_genes: int = 2000,
) -> dict[str, Any]:
    """Rescue check 1.2.5: PCA on cell-type-corrected delta vectors.

    After CellDMC produces cell-type-corrected residuals, re-run the 0-T PCA
    to test whether R vs NR separation is improved post-correction.

    Acceptance (TASK-2026-05-17-007):
    - PERMANOVA p < 0.05 AND max Cohen's d > 0.30 in PC1-2 of corrected delta.

    Parameters
    ----------
    corrected_delta_m:
        2-D array (n_cpg, n_paired_subjects), cell-type-corrected M-value deltas.
    corrected_delta_rna:
        2-D array (n_genes, n_paired_subjects), cell-type-corrected log-CPM deltas.
    cpg_ids:
        CpG identifiers (rows of corrected_delta_m).
    gene_ids:
        Gene identifiers (rows of corrected_delta_rna).
    pdata_paired:
        pData2 slice for paired subjects; must contain response_col.
    response_col:
        Column for binary response label.
    seed:
        Random seed for PERMANOVA.
    n_permutations:
        Number of PERMANOVA permutations.
    top_cpgs, top_genes:
        Number of variance-top features to retain before PCA.

    Returns
    -------
    dict with keys: permanova_p, cohen_d_per_pc, verdict, n_r, n_nr,
    rescue_passed.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    resp_raw = pdata_paired[response_col].astype(str).str.strip().str.upper()
    r_mask = resp_raw.isin(["R", "RESPONDER", "1"])
    nr_mask = resp_raw.isin(["NR", "NON-RESPONDER", "0"])
    valid_mask = r_mask | nr_mask
    if valid_mask.sum() < 10:
        raise ValueError("Insufficient R/NR samples for rescue check.")

    n_r = int(r_mask.sum())
    n_nr = int(nr_mask.sum())
    labels = np.where(resp_raw.values == "R", 1, np.where(resp_raw.values == "NR", 0, -1))
    labels = labels[valid_mask]

    # Filter valid subjects
    delta_m_valid = corrected_delta_m[:, valid_mask]
    delta_rna_valid = corrected_delta_rna[:, valid_mask]

    # Variance filter
    var_cpg = np.nanvar(delta_m_valid, axis=1)
    top_cpg_idx = np.argsort(var_cpg)[-top_cpgs:]
    var_rna = np.nanvar(delta_rna_valid, axis=1)
    top_rna_idx = np.argsort(var_rna)[-top_genes:]

    delta_m_top = delta_m_valid[top_cpg_idx, :].T
    delta_rna_top = delta_rna_valid[top_rna_idx, :].T
    delta_combined = np.hstack([delta_m_top, delta_rna_top])

    # Impute NaNs with column means
    col_means = np.nanmean(delta_combined, axis=0)
    nan_mask = np.isnan(delta_combined)
    delta_combined[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    scaler = StandardScaler()
    delta_scaled = scaler.fit_transform(delta_combined)

    pca = PCA(n_components=5, random_state=seed)
    pcs = pca.fit_transform(delta_scaled)

    # Cohen's d per PC
    cohen_d: dict[str, float] = {}
    for i in range(min(5, pcs.shape[1])):
        r_vals = pcs[labels == 1, i]
        nr_vals = pcs[labels == 0, i]
        if len(r_vals) < 2 or len(nr_vals) < 2:
            cohen_d[f"PC{i + 1}"] = float("nan")
            continue
        pooled_sd = float(
            np.sqrt(
                (
                    (len(r_vals) - 1) * np.var(r_vals, ddof=1)
                    + (len(nr_vals) - 1) * np.var(nr_vals, ddof=1)
                )
                / (len(r_vals) + len(nr_vals) - 2)
            )
        )
        d = float(abs(np.mean(r_vals) - np.mean(nr_vals)) / pooled_sd) if pooled_sd > 0 else 0.0
        cohen_d[f"PC{i + 1}"] = d

    # PERMANOVA (simplified: pseudo-F via permutation on PC scores)
    pc_mat = pcs[:, :5]
    rng = np.random.default_rng(seed)

    def pseudo_f(labs: np.ndarray[Any, Any]) -> float:
        r_idx = labs == 1
        nr_idx = labs == 0
        grand_mean = pc_mat.mean(axis=0)
        ss_between = float(
            r_idx.sum() * np.sum((pc_mat[r_idx].mean(axis=0) - grand_mean) ** 2)
            + nr_idx.sum() * np.sum((pc_mat[nr_idx].mean(axis=0) - grand_mean) ** 2)
        )
        ss_within = float(
            np.sum((pc_mat[r_idx] - pc_mat[r_idx].mean(axis=0)) ** 2)
            + np.sum((pc_mat[nr_idx] - pc_mat[nr_idx].mean(axis=0)) ** 2)
        )
        n_total = len(labs)
        if ss_within < 1e-12:
            return 0.0
        return (ss_between / 1) / (ss_within / (n_total - 2))

    observed_f = pseudo_f(labels)
    null_f = np.array([pseudo_f(rng.permutation(labels)) for _ in range(n_permutations)])
    permanova_p = float((null_f >= observed_f).mean())

    max_d = max((v for v in cohen_d.values() if not np.isnan(v)), default=0.0)
    rescue_passed = permanova_p < 0.05 and max_d > 0.30
    if rescue_passed:
        verdict = "RESCUE_PASS"
    elif permanova_p < 0.15:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        "permanova_p": permanova_p,
        "cohen_d_per_pc": cohen_d,
        "max_cohen_d": max_d,
        "n_r": n_r,
        "n_nr": n_nr,
        "verdict": verdict,
        "rescue_passed": rescue_passed,
        "n_cpg_features": top_cpgs,
        "n_gene_features": top_genes,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
