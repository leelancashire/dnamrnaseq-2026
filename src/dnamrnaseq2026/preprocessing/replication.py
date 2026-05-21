"""Cross-cohort BEST replication for Phase 1 Step 1.7.

Applies Emory CellDMC(delta) significant CpG list to BEST paired subjects.
Stratified by Therapy_type (CPT, PE, None).

Replication criteria (ANALYSIS_PLAN.md Step 1.7):
  - Overall: >= 40% of Emory-significant CpGs show same direction in BEST at
    uncorrected p < 0.10.
  - At least one of CPT or PE strata replicates the overall pattern.

BEST pData gotcha (from ANALYSIS_PLAN.md Step 1.7 Risk notes + Phase 0 note):
  - BEST sample-ID format uses {Subcode}-{BL|12W}; verify after rdata load.
  - pData factor columns may not survive rdata 1.0.0 conversion; check for
    Therapy_type and Response columns and handle missing values.

Analysis plan reference: ANALYSIS_PLAN.md Step 1.7.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BEST sample helpers
# ---------------------------------------------------------------------------


def build_best_paired_ids(
    pdata_best: pd.DataFrame,
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    pre_label: str = "BL",
    post_label: str = "12W",
) -> tuple[list[str], list[str], list[str]]:
    """Identify paired PRE/POST sample IDs in BEST cohort.

    BEST uses {Subcode}-{BL|12W} sample naming per ANALYSIS_PLAN.md Risk note.
    Tries that pattern first; if not present, falls back to matching on
    Subcode + Visit columns.

    Parameters
    ----------
    pdata_best:
        BEST pData2 DataFrame, index = SampleName.
    subcode_col:
        Column with per-subject ID.
    visit_col:
        Column with visit label.
    pre_label, post_label:
        Visit label strings for PRE and POST.

    Returns
    -------
    Tuple (paired_subjects, pre_ids, post_ids):
        paired_subjects: shared subject identifiers.
        pre_ids: PRE sample IDs aligned to paired_subjects.
        post_ids: POST sample IDs aligned to paired_subjects.
    """
    if subcode_col not in pdata_best.columns or visit_col not in pdata_best.columns:
        # Try to infer from SampleName index: {Subcode}-{BL|12W}
        pre_samples = {
            s.replace(f"-{pre_label}", ""): s
            for s in pdata_best.index
            if str(s).endswith(f"-{pre_label}")
        }
        post_samples = {
            s.replace(f"-{post_label}", ""): s
            for s in pdata_best.index
            if str(s).endswith(f"-{post_label}")
        }
        subjects = sorted(set(pre_samples) & set(post_samples))
        logger.info("BEST pairing via SampleName pattern: %d paired subjects.", len(subjects))
        return (
            subjects,
            [pre_samples[s] for s in subjects],
            [post_samples[s] for s in subjects],
        )

    pre_mask = (
        pdata_best[visit_col]
        .astype(str)
        .str.upper()
        .isin([pre_label.upper(), "PRE", "PRE-IOP", "BL", "BASELINE", "T0", "0"])
    )
    post_mask = (
        pdata_best[visit_col]
        .astype(str)
        .str.upper()
        .isin([post_label.upper(), "POST", "POST-IOP", "12W", "12WEEKS", "T1", "1"])
    )
    pre_map = dict(
        zip(pdata_best.loc[pre_mask, subcode_col].values, pdata_best.index[pre_mask], strict=False)
    )
    post_map = dict(
        zip(
            pdata_best.loc[post_mask, subcode_col].values, pdata_best.index[post_mask], strict=False
        )
    )

    subjects = sorted(set(pre_map) & set(post_map))
    logger.info("BEST pairing via Subcode+Visit: %d paired subjects.", len(subjects))
    return (
        subjects,
        [pre_map[s] for s in subjects],
        [post_map[s] for s in subjects],
    )


# ---------------------------------------------------------------------------
# Per-CpG delta-M and OLS replication
# ---------------------------------------------------------------------------


def compute_best_delta_m(
    bvals_best: np.ndarray[Any, Any],
    cpg_ids_best: list[str],
    pre_ids: list[str],
    post_ids: list[str],
    pdata_best: pd.DataFrame,
    sig_cpg_ids: list[str],
    beta_clip: float = 1e-6,
    mval_clip: float = 8.0,
) -> tuple[np.ndarray[Any, Any], list[str]]:
    """Compute paired delta-M for BEST on the Emory-significant CpG list.

    Parameters
    ----------
    bvals_best:
        2-D array (n_cpg, n_samples), beta values for BEST.
    cpg_ids_best:
        CpG identifiers for rows of bvals_best.
    pre_ids, post_ids:
        PRE and POST BEST sample IDs aligned to paired subjects.
    pdata_best:
        BEST pData2, index = SampleName.
    sig_cpg_ids:
        Emory-significant CpG IDs to extract.
    beta_clip, mval_clip:
        Numerical safety parameters for M-value computation.

    Returns
    -------
    Tuple (delta_m, kept_cpg_ids):
        delta_m: 2-D array (n_kept_cpgs, n_paired_subjects).
        kept_cpg_ids: CpGs that were found in the BEST array.
    """
    cpg_index = {c: i for i, c in enumerate(cpg_ids_best)}
    sample_index = {s: i for i, s in enumerate(pdata_best.index)}

    kept: list[str] = [c for c in sig_cpg_ids if c in cpg_index]
    if not kept:
        return np.empty((0, len(pre_ids))), []

    cpg_rows = np.array([cpg_index[c] for c in kept])
    pre_cols = np.array([sample_index[s] for s in pre_ids if s in sample_index])
    post_cols = np.array([sample_index[s] for s in post_ids if s in sample_index])

    n_pairs = min(len(pre_cols), len(post_cols))
    pre_cols = pre_cols[:n_pairs]
    post_cols = post_cols[:n_pairs]

    bvals_pre = bvals_best[np.ix_(cpg_rows, pre_cols)]
    bvals_post = bvals_best[np.ix_(cpg_rows, post_cols)]

    def to_m(b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        b_safe = np.clip(b, beta_clip, 1.0 - beta_clip)
        result: np.ndarray[Any, Any] = np.clip(
            np.log2(b_safe / (1.0 - b_safe)), -mval_clip, mval_clip
        )
        return result

    delta_m = to_m(bvals_post) - to_m(bvals_pre)
    return delta_m, kept


def _ols_replication_cpg(
    delta_m: np.ndarray[Any, Any],
    response: np.ndarray[Any, Any],
    covariates: np.ndarray[Any, Any],
) -> dict[str, float]:
    """OLS for a single CpG in BEST replication."""
    from statsmodels.regression.linear_model import OLS

    n = len(delta_m)
    intercept = np.ones((n, 1))
    design = np.hstack([intercept, response.reshape(-1, 1), covariates])
    valid = ~(np.isnan(delta_m) | np.isnan(design).any(axis=1))
    if valid.sum() < design.shape[1] + 2:
        return {"beta_best": np.nan, "p_best": np.nan}
    try:
        fit = OLS(delta_m[valid], design[valid]).fit()
        return {"beta_best": float(fit.params[1]), "p_best": float(fit.pvalues[1])}
    except Exception:
        return {"beta_best": np.nan, "p_best": np.nan}


# ---------------------------------------------------------------------------
# Overall and stratum-level replication
# ---------------------------------------------------------------------------


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


def run_replication(
    delta_m_best: np.ndarray[Any, Any],
    kept_cpg_ids: list[str],
    emory_betas: pd.DataFrame,
    pdata_best_paired: pd.DataFrame,
    response_col: str = "Response",
    therapy_col: str = "Therapy_type",
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run overall and within-modality replication for BEST.

    Parameters
    ----------
    delta_m_best:
        2-D array (n_cpg, n_paired_subjects), BEST delta-M values.
    kept_cpg_ids:
        CpG IDs (rows of delta_m_best).
    emory_betas:
        Emory CellDMC(delta) output DataFrame with columns:
        cpg, cell_type, beta_interaction (Emory beta reference).
    pdata_best_paired:
        pData2 for BEST paired subjects (one row per subject).
    response_col:
        Column for R/NR label.
    therapy_col:
        Column for therapy type (CPT, PE, None).
    n_jobs:
        joblib workers.

    Returns
    -------
    Tuple of three DataFrames:
        overall: per-CpG overall replication.
        within_modality: per-CpG per-modality replication.
        modality_interaction: per-CpG therapy × response interaction.
    """
    from joblib import Parallel, delayed

    resp_raw = pdata_best_paired[response_col].astype(str).str.strip().str.upper()
    valid_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    pdata_sub = pdata_best_paired[valid_mask]
    resp_enc = (
        resp_raw[valid_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .values.astype(float)
    )

    col_pos_all = list(pdata_best_paired.index)
    col_pos = [col_pos_all.index(s) for s in pdata_sub.index]
    dm_sub = delta_m_best[:, col_pos]

    cov_cols = [c for c in COVARIATE_COLS if c in pdata_sub.columns]
    cov_matrix = (
        pdata_sub[cov_cols].fillna(0).values.astype(float)
        if cov_cols
        else np.empty((len(pdata_sub), 0))
    )

    # Build per-CpG Emory beta reference (use first cell type or mean)
    emory_beta_map: dict[str, float] = {}
    if not emory_betas.empty and "cpg" in emory_betas.columns:
        grp = emory_betas.groupby("cpg")["beta_interaction"].mean()
        emory_beta_map = grp.to_dict()

    def process_cpg(ci: int) -> dict[str, Any]:
        cpg = kept_cpg_ids[ci]
        res = _ols_replication_cpg(dm_sub[ci], resp_enc, cov_matrix)
        emory_b = emory_beta_map.get(cpg, np.nan)
        same_direction = (
            bool(np.sign(res["beta_best"]) == np.sign(emory_b))
            if not (np.isnan(res["beta_best"]) or np.isnan(emory_b))
            else False
        )
        return {
            "cpg": cpg,
            "emory_beta": emory_b,
            **res,
            "same_direction": same_direction,
        }

    rows = Parallel(n_jobs=n_jobs)(delayed(process_cpg)(ci) for ci in range(len(kept_cpg_ids)))
    overall = pd.DataFrame(rows)
    if not overall.empty:
        overall["q_best"] = multipletests(overall["p_best"].fillna(1.0).values, method="fdr_bh")[1]

    # Within-modality replication
    modality_rows: list[dict[str, Any]] = []
    if therapy_col in pdata_best_paired.columns:
        modalities = pdata_best_paired[therapy_col].dropna().unique()
        for modality in modalities:
            mod_mask = (
                pdata_best_paired[therapy_col].astype(str).str.strip().str.upper()
                == str(modality).strip().upper()
            )
            mod_resp_mask = valid_mask & mod_mask
            if mod_resp_mask.sum() < 5:
                continue
            pdata_mod = pdata_best_paired[mod_resp_mask]
            resp_mod = (
                resp_raw[mod_resp_mask]
                .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
                .values.astype(float)
            )
            col_pos_mod = [col_pos_all.index(s) for s in pdata_mod.index]
            dm_mod = delta_m_best[:, col_pos_mod]
            cov_mod = (
                pdata_mod[cov_cols].fillna(0).values.astype(float)
                if cov_cols
                else np.empty((len(pdata_mod), 0))
            )

            for ci, cpg in enumerate(kept_cpg_ids):
                res = _ols_replication_cpg(dm_mod[ci], resp_mod, cov_mod)
                emory_b = emory_beta_map.get(cpg, np.nan)
                modality_rows.append(
                    {
                        "cpg": cpg,
                        "modality": modality,
                        "emory_beta": emory_b,
                        **res,
                        "same_direction": (
                            bool(np.sign(res["beta_best"]) == np.sign(emory_b))
                            if not (np.isnan(res["beta_best"]) or np.isnan(emory_b))
                            else False
                        ),
                    }
                )

    within_modality = pd.DataFrame(modality_rows)

    # Modality interaction (Therapy_type x Response)
    interaction_rows: list[dict[str, Any]] = []
    if therapy_col in pdata_best_paired.columns and not overall.empty:
        from statsmodels.regression.linear_model import OLS

        therapy_vals = pdata_best_paired.loc[pdata_sub.index, therapy_col].astype(str).str.strip()
        # Encode therapy: CPT=1, others=0 for binary interaction
        therapy_enc = (therapy_vals.str.upper() == "CPT").astype(float).values

        for ci, cpg in enumerate(kept_cpg_ids):
            y = dm_sub[ci]
            interaction = resp_enc * therapy_enc
            n = len(y)
            intercept = np.ones((n, 1))
            design_int = np.hstack(
                [
                    intercept,
                    resp_enc.reshape(-1, 1),
                    therapy_enc.reshape(-1, 1),
                    interaction.reshape(-1, 1),
                    cov_matrix,
                ]
            )
            valid = ~(np.isnan(y) | np.isnan(design_int).any(axis=1))
            if valid.sum() < design_int.shape[1] + 2:
                interaction_rows.append(
                    {"cpg": cpg, "beta_interaction": np.nan, "p_interaction": np.nan}
                )
                continue
            try:
                fit = OLS(y[valid], design_int[valid]).fit()
                interaction_rows.append(
                    {
                        "cpg": cpg,
                        "beta_interaction": float(fit.params[3]),
                        "p_interaction": float(fit.pvalues[3]),
                    }
                )
            except Exception:
                interaction_rows.append(
                    {"cpg": cpg, "beta_interaction": np.nan, "p_interaction": np.nan}
                )

    modality_interaction = pd.DataFrame(interaction_rows)
    if not modality_interaction.empty:
        modality_interaction["q_interaction"] = multipletests(
            modality_interaction["p_interaction"].fillna(1.0).values, method="fdr_bh"
        )[1]

    return overall, within_modality, modality_interaction


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarise_replication(
    overall: pd.DataFrame,
    uncorrected_p_threshold: float = 0.10,
) -> dict[str, Any]:
    """Compute replication summary statistics.

    Replication criterion: >= 40% of Emory-significant CpGs same direction
    in BEST at uncorrected p < 0.10.

    Parameters
    ----------
    overall:
        Per-CpG overall replication DataFrame.
    uncorrected_p_threshold:
        p-value threshold for counting same-direction replication.

    Returns
    -------
    dict with keys: n_tested, n_replicated, pct_replicated, verdict.
    """
    if overall.empty:
        return {"n_tested": 0, "n_replicated": 0, "pct_replicated": 0.0, "verdict": "NO_DATA"}

    n_tested = int((~overall["p_best"].isna()).sum())
    n_replicated = int(
        (overall["same_direction"] & (overall["p_best"] < uncorrected_p_threshold)).sum()
    )
    pct = (n_replicated / n_tested * 100) if n_tested > 0 else 0.0
    verdict = "PASS" if pct >= 40.0 else "FAIL"
    # Jointly drop rows where either beta is NaN to keep arrays aligned.
    _both = overall[["emory_beta", "beta_best"]].dropna()
    spearman_rho = (
        float(stats.spearmanr(_both["emory_beta"], _both["beta_best"])[0])
        if len(_both) >= 5
        else np.nan
    )

    return {
        "n_tested": n_tested,
        "n_replicated": n_replicated,
        "pct_replicated": round(pct, 2),
        "verdict": verdict,
        "spearman_rho_emory_best": spearman_rho,
    }
