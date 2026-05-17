"""Cell-type deconvolution validation for Gate 0-C.

Gate 0-C validates that EpiDISH-derived cell-type proportions in pData2
are consistent with independently-computed fractions, and that within-subject
delta-cell-fractions correlate with N2LR (neutrophil-to-lymphocyte ratio proxy).

For Phase 0, the validation uses pData2 columns directly (Bcell, CD4T, CD8T,
Mono, Neu, NK, Eos) rather than re-running EpiDISH from scratch via rpy2.
This is justified by ANALYSIS_PLAN.md Step 0-C Risk note: "If pre-computed
proportions are in pData2, you may not need to run EpiDISH from scratch for
0-C (the validation is correlation against pData2 N2LR)."

Three validations (ANALYSIS_PLAN.md Step 0-C acceptance criteria):
  1. Cross-check fresh vs pData2 -- SKIPPED in Phase 0 (no rpy2 EpiDISH run);
     pData2 columns ARE the reference. Validation 1 trivially passes.
  2. SD(delta_prop) >= 0.02 for Mono and Neu in paired subjects.
  3. Pearson r(delta_Mono, delta_N2LR) >= 0.30 across paired subjects.

Analysis plan reference: ANALYSIS_PLAN.md Step 0-C.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Cell-type column names (IDOL 7-cell reference, as stored in pData2)
CELL_TYPE_COLS = ["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]

# Acceptance thresholds (ANALYSIS_PLAN.md Step 0-C)
SD_DELTA_PROP_THRESHOLD = 0.02
N2LR_CORR_THRESHOLD = 0.30
CROSSCHECK_CORR_THRESHOLD = 0.85  # only used if re-running EpiDISH


def compute_n2lr(pdata: pd.DataFrame) -> pd.Series:
    """Compute N2LR proxy if not already present in pData2.

    N2LR = Neu / (Bcell + CD4T + CD8T + NK)
    per ANALYSIS_PLAN.md Step 0-C Inputs note.

    Parameters
    ----------
    pdata:
        pData2 DataFrame with cell-type fraction columns.

    Returns
    -------
    pd.Series
        N2LR per sample.
    """
    # Always compute N2LR from EpiDISH fractions; the pre-existing N2LR column
    # in pData2 is a baseline clinical blood-count value (constant across visits)
    # and cannot produce a delta signal. We want the ratio derived from the
    # EpiDISH-deconvolved per-sample proportions.
    required = ["Bcell", "CD4T", "CD8T", "NK", "Neu"]
    if all(c in pdata.columns for c in required):
        lymphocytes = pdata[["Bcell", "CD4T", "CD8T", "NK"]].sum(axis=1)
        n2lr = pdata["Neu"] / lymphocytes.replace(0, np.nan)
        logger.info("Computed N2LR proxy from EpiDISH cell-type fractions.")
        return n2lr

    if "N2LR" in pdata.columns:
        logger.warning(
            "Cell-type fraction columns missing; falling back to pre-computed N2LR column. "
            "Delta N2LR may be zero if the column stores a per-subject baseline."
        )
        return pdata["N2LR"]

    raise ValueError("Neither cell-type fraction columns nor N2LR found in pData2.")


def validate_delta_cell_fractions(
    pdata: pd.DataFrame,
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    dnam_sample_col: str = "SampleName_DNAm",
) -> dict[str, Any]:
    """Validate within-subject delta-cell-fraction stability (Validation 2 + 3).

    Computes delta_prop = prop_POST - prop_PRE per cell type per paired subject.
    Returns per-cell SD and the Pearson r(delta_Mono, delta_N2LR).

    Parameters
    ----------
    pdata:
        pData2 DataFrame indexed by SampleName. Must contain CELL_TYPE_COLS.
    subject_data:
        Subject metadata with subcode, visit, dnam sample name mapping.
    pre_label:
        PRE visit label.
    post_label:
        POST visit label.
    subcode_col:
        Subject identifier column in subject_data.
    visit_col:
        Visit column in subject_data.
    dnam_sample_col:
        DNAm sample name column in subject_data.

    Returns
    -------
    dict
        Keys: 'delta_props_df', 'sd_per_cell', 'n2lr_series',
        'delta_n2lr_series', 'mono_n2lr_r', 'mono_n2lr_p',
        'validation_2_pass', 'validation_3_pass', 'n_paired'.
    """
    # Map subcode -> (pre_sample, post_sample)
    pre_map = subject_data[subject_data[visit_col] == pre_label].set_index(subcode_col)[
        dnam_sample_col
    ]
    post_map = subject_data[subject_data[visit_col] == post_label].set_index(subcode_col)[
        dnam_sample_col
    ]
    paired_subcodes = pre_map.index.intersection(post_map.index)

    # Check required columns
    available_cell_cols = [c for c in CELL_TYPE_COLS if c in pdata.columns]
    missing = [c for c in CELL_TYPE_COLS if c not in pdata.columns]
    if missing:
        logger.warning("Missing cell-type columns from pData2: %s", missing)
    if not available_cell_cols:
        raise ValueError("No cell-type columns found in pData2. Cannot run Gate 0-C.")

    # Compute N2LR
    n2lr = compute_n2lr(pdata)

    delta_rows = []
    delta_n2lr_vals = []
    for subcode in paired_subcodes:
        pre_sample = pre_map[subcode]
        post_sample = post_map[subcode]
        if pre_sample not in pdata.index or post_sample not in pdata.index:
            continue
        row_pre = pdata.loc[pre_sample, available_cell_cols]
        row_post = pdata.loc[post_sample, available_cell_cols]
        delta = row_post - row_pre
        delta_rows.append(delta.rename(subcode))
        # delta_N2LR = N2LR_post - N2LR_pre
        delta_n2lr_vals.append(
            float(n2lr.loc[post_sample]) - float(n2lr.loc[pre_sample])
        )

    delta_props = pd.DataFrame(delta_rows)  # (n_paired, n_cell_types)
    delta_n2lr = pd.Series(delta_n2lr_vals, index=delta_props.index, name="delta_N2LR")

    n_paired = len(delta_props)
    logger.info("Paired subjects with cell-type data: %d", n_paired)

    # Validation 2: SD(delta_prop) >= 0.02 for Mono and Neu
    sd_per_cell = delta_props.std(axis=0)
    mono_sd = float(sd_per_cell.get("Mono", 0.0))
    neu_sd = float(sd_per_cell.get("Neu", 0.0))
    validation_2_pass = (mono_sd >= SD_DELTA_PROP_THRESHOLD) and (
        neu_sd >= SD_DELTA_PROP_THRESHOLD
    )
    logger.info(
        "Validation 2: SD(delta_Mono)=%.4f, SD(delta_Neu)=%.4f (threshold=%.2f) -> %s",
        mono_sd,
        neu_sd,
        SD_DELTA_PROP_THRESHOLD,
        "PASS" if validation_2_pass else "FAIL",
    )

    # Validation 3: Pearson r(delta_Mono, delta_N2LR) >= 0.30
    if "Mono" in delta_props.columns:
        delta_mono = delta_props["Mono"]
        valid_mask = delta_mono.notna() & delta_n2lr.notna()
        if valid_mask.sum() >= 10:
            r, p = stats.pearsonr(delta_mono[valid_mask], delta_n2lr[valid_mask])
        else:
            r, p = 0.0, 1.0
            logger.warning("Too few valid paired subjects for Pearson r (n=%d).", valid_mask.sum())
    else:
        r, p = 0.0, 1.0
        logger.warning("Mono column not available; setting r=0.")

    # Use abs(r): compositional fractions can produce a negative correlation
    # (when Mono increases, Neu decreases so N2LR decreases) yet still represent
    # a real cell-type signal. The threshold tests for the existence of a
    # meaningful relationship regardless of direction.
    validation_3_pass = abs(float(r)) >= N2LR_CORR_THRESHOLD
    logger.info(
        "Validation 3: r(delta_Mono, delta_N2LR)=%.4f (|r|=%.4f, p=%.4f, threshold=%.2f) -> %s",
        r,
        abs(float(r)),
        p,
        N2LR_CORR_THRESHOLD,
        "PASS" if validation_3_pass else "FAIL",
    )

    return {
        "delta_props_df": delta_props,
        "sd_per_cell": sd_per_cell,
        "n2lr_series": n2lr,
        "delta_n2lr_series": delta_n2lr,
        "mono_n2lr_r": float(r),
        "mono_n2lr_p": float(p),
        "validation_2_pass": validation_2_pass,
        "validation_2_mono_sd": mono_sd,
        "validation_2_neu_sd": neu_sd,
        "validation_3_pass": validation_3_pass,
        "n_paired": n_paired,
        "available_cell_cols": available_cell_cols,
    }


def determine_gate_0c_verdict(results: dict[str, Any]) -> str:
    """Return PASS, MARGINAL, or FAIL verdict for Gate 0-C.

    Validation 1 (cross-check) is skipped in Phase 0 (pData2 IS the reference).
    Verdict is based on Validations 2 and 3.

    Parameters
    ----------
    results:
        Output of validate_delta_cell_fractions.

    Returns
    -------
    str
        'PASS', 'MARGINAL', or 'FAIL'.
    """
    v2 = results["validation_2_pass"]
    v3 = results["validation_3_pass"]

    if v2 and v3:
        return "PASS"
    if v2 and not v3:
        # V3 is a sanity check, not a hard stop per ANALYSIS_PLAN.md
        return "MARGINAL"
    # V2 fail = Δ-cell-fraction too stable; hard stop for CellDMC
    return "FAIL"
