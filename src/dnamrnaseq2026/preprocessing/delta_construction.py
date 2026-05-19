"""Delta-vector construction for paired PRE/POST subjects.

Constructs within-subject difference vectors (POST - PRE) for both DNAm
(M-values converted from beta) and RNA-seq (log-CPM). Used by Gate 0-T
(PCA of delta) and Gate 0-S (source-domain shift in delta-space).

All functions return DataFrames indexed by subject ID (Subcode / paired ID),
with feature columns. Paired subjects are those with both PRE and POST samples
in the data.

Analysis plan reference: ANALYSIS_PLAN.md Step 0-T (Method, steps 1-4).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Acceptable M-value clamp range (ANALYSIS_PLAN.md Step 0-T, Risks)
M_VALUE_CLIP_MIN = -3.0
M_VALUE_CLIP_MAX = 3.0

# Default variance-filter sizes (ANALYSIS_PLAN.md Step 0-T, Method step 3)
DEFAULT_TOP_CPGS = 5000
DEFAULT_TOP_GENES = 2000


def beta_to_mvalue(
    beta: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Convert beta values [0,1] to M-values via logit transformation.

    M = log2(beta / (1 - beta))

    Clips betas to [0.001, 0.999] to avoid log(0) / log(inf).
    Returns M-values clipped to [M_VALUE_CLIP_MIN, M_VALUE_CLIP_MAX]
    per ANALYSIS_PLAN.md Step 0-T (Risks: outlier CpGs).

    Parameters
    ----------
    beta:
        Array of beta values in [0, 1].

    Returns
    -------
    np.ndarray
        M-values, clipped to [-3, 3].
    """
    beta_clipped = np.clip(beta, 0.001, 0.999)
    m = np.log2(beta_clipped / (1.0 - beta_clipped))
    clipped: np.ndarray[Any, np.dtype[np.float64]] = np.clip(m, M_VALUE_CLIP_MIN, M_VALUE_CLIP_MAX)
    return clipped


def identify_paired_subjects(
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    response_col: str = "Response",
) -> pd.DataFrame:
    """Return a DataFrame of paired subjects with their Response labels.

    A paired subject has exactly one PRE and one POST sample. The function
    returns one row per subject with columns: Subcode, Response.

    Parameters
    ----------
    subject_data:
        Subject metadata DataFrame. Must contain subcode_col, visit_col,
        and response_col columns.
    pre_label:
        Label for PRE visit in the Visit column.
    post_label:
        Label for POST visit in the Visit column.
    subcode_col:
        Column name for subject identifier.
    visit_col:
        Column name for visit.
    response_col:
        Column name for response label.

    Returns
    -------
    pd.DataFrame
        Columns: [subcode_col, response_col]. One row per paired subject.
    """
    visits_per_subject = subject_data.groupby(subcode_col)[visit_col].apply(set)
    paired_mask = visits_per_subject.apply(lambda v: pre_label in v and post_label in v)
    paired_subcodes = visits_per_subject[paired_mask].index.tolist()

    # Get Response for each paired subject (take the PRE row; Response is constant per subject)
    paired_info = (
        subject_data[
            (subject_data[subcode_col].isin(paired_subcodes))
            & (subject_data[visit_col] == pre_label)
        ][[subcode_col, response_col]]
        .drop_duplicates(subcode_col)
        .reset_index(drop=True)
    )

    logger.info(
        "Paired subjects: %d total (%s)",
        len(paired_info),
        paired_info[response_col].value_counts().to_dict(),
    )
    return paired_info


def build_dnam_delta_matrix(
    bvals: pd.DataFrame,
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    dnam_sample_col: str = "SampleName_DNAm",
    top_n_cpgs: int = DEFAULT_TOP_CPGS,
) -> pd.DataFrame:
    """Build a paired delta-M matrix: POST_M - PRE_M per subject.

    CpGs are variance-filtered to top_n_cpgs by within-paired-subject
    variance of the delta values. M-values are computed from betas and
    clipped to [-3, 3].

    Parameters
    ----------
    bvals:
        Beta value DataFrame, shape (n_cpgs, n_samples). Columns are
        SampleName_DNAm identifiers.
    subject_data:
        Subject metadata with subcode_col, visit_col, dnam_sample_col.
    pre_label:
        Visit label for PRE samples.
    post_label:
        Visit label for POST samples.
    subcode_col:
        Subject identifier column in subject_data.
    visit_col:
        Visit column in subject_data.
    dnam_sample_col:
        Column in subject_data giving the DNAm SampleName identifier.
    top_n_cpgs:
        Number of top-variance CpGs to retain.

    Returns
    -------
    pd.DataFrame
        Shape (n_paired_subjects, top_n_cpgs). Index: subject Subcode.
        Columns: CpG site IDs.
    """
    logger.info("Building DNAm delta-M matrix (top %d CpGs)...", top_n_cpgs)

    # Build mapping: subcode -> (pre_sample, post_sample)
    pre_map = subject_data[subject_data[visit_col] == pre_label].set_index(subcode_col)[
        dnam_sample_col
    ]
    post_map = subject_data[subject_data[visit_col] == post_label].set_index(subcode_col)[
        dnam_sample_col
    ]
    paired_subcodes = pre_map.index.intersection(post_map.index)

    delta_rows: list[pd.Series] = []
    for subcode in paired_subcodes:
        pre_sample = pre_map[subcode]
        post_sample = post_map[subcode]
        if pre_sample not in bvals.columns or post_sample not in bvals.columns:
            logger.warning(
                "Subject %s: sample(s) missing from bVals (%s, %s). Skipping.",
                subcode,
                pre_sample,
                post_sample,
            )
            continue
        pre_m = beta_to_mvalue(bvals[pre_sample].values)
        post_m = beta_to_mvalue(bvals[post_sample].values)
        delta = post_m - pre_m
        delta_rows.append(pd.Series(delta, index=bvals.index, name=subcode))

    delta_df = pd.DataFrame(delta_rows)  # (n_subjects, n_cpgs)
    logger.info("  Raw delta-M matrix: %s", delta_df.shape)

    # Variance filter: top_n_cpgs by variance across subjects
    cpg_var = delta_df.var(axis=0)
    top_cpgs = cpg_var.nlargest(min(top_n_cpgs, len(cpg_var))).index
    delta_df = delta_df[top_cpgs]
    logger.info("  After CpG variance filter: %s", delta_df.shape)
    return delta_df


def build_rnaseq_delta_matrix(
    rnaseq: pd.DataFrame,
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    rnaseq_sample_col: str = "SampleName_RNASeq",
    top_n_genes: int = DEFAULT_TOP_GENES,
) -> pd.DataFrame:
    """Build a paired delta-logCPM matrix: POST - PRE per subject.

    RNA-seq columns in the CSV are formatted as '{SubjectID}-{Visit}'.
    This function parses that format or uses subject_data mapping.

    Parameters
    ----------
    rnaseq:
        Log-CPM DataFrame, shape (n_genes, n_samples). Columns are
        '{SubjectID}-{Visit}' or identifiers from rnaseq_sample_col.
    subject_data:
        Subject metadata with subcode_col, visit_col, rnaseq_sample_col.
    pre_label:
        Visit label for PRE samples (as stored in Visit column).
    post_label:
        Visit label for POST samples.
    subcode_col:
        Subject identifier column.
    visit_col:
        Visit column.
    rnaseq_sample_col:
        Column in subject_data giving the RNA-seq sample identifier.
    top_n_genes:
        Number of top-variance genes to retain.

    Returns
    -------
    pd.DataFrame
        Shape (n_paired_subjects, top_n_genes). Index: subject Subcode.
        Columns: gene IDs.
    """
    logger.info("Building RNA-seq delta-logCPM matrix (top %d genes)...", top_n_genes)

    pre_map = subject_data[subject_data[visit_col] == pre_label].set_index(subcode_col)[
        rnaseq_sample_col
    ]
    post_map = subject_data[subject_data[visit_col] == post_label].set_index(subcode_col)[
        rnaseq_sample_col
    ]
    paired_subcodes = pre_map.index.intersection(post_map.index)

    # Detect whether the rnaseq_sample_col values match RNA-seq columns.
    # If not, fall back to constructing '{Subcode}-{Visit}' which is the
    # format used when the mmVAE CSV is indexed as {subcode}-{visit}.
    rnaseq_col_set = set(rnaseq.columns)
    sample_col_values = set(pre_map.values) | set(post_map.values)
    use_sample_col = bool(sample_col_values & rnaseq_col_set)
    if not use_sample_col:
        logger.info(
            "SampleName_RNASeq values (%s ...) not found in RNA-seq columns (%s ...). "
            "Falling back to '{Subcode}-{Visit}' column format.",
            list(sample_col_values)[:2],
            list(rnaseq.columns[:2]),
        )

    delta_rows: list[pd.Series] = []
    for subcode in paired_subcodes:
        if use_sample_col:
            pre_sample = str(pre_map[subcode])
            post_sample = str(post_map[subcode])
        else:
            pre_sample = f"{subcode}-{pre_label}"
            post_sample = f"{subcode}-{post_label}"
        if pre_sample not in rnaseq.columns or post_sample not in rnaseq.columns:
            logger.warning(
                "Subject %s: RNA-seq sample(s) missing (%s, %s). Skipping.",
                subcode,
                pre_sample,
                post_sample,
            )
            continue
        delta = rnaseq[post_sample].values - rnaseq[pre_sample].values
        delta_rows.append(pd.Series(delta, index=rnaseq.index, name=subcode))

    delta_df = pd.DataFrame(delta_rows)  # (n_subjects, n_genes)
    logger.info("  Raw delta-logCPM matrix: %s", delta_df.shape)

    gene_var = delta_df.var(axis=0)
    top_genes = gene_var.nlargest(min(top_n_genes, len(gene_var))).index
    delta_df = delta_df[top_genes]
    logger.info("  After gene variance filter: %s", delta_df.shape)
    return delta_df


def build_joint_delta_matrix(
    dnam_delta: pd.DataFrame,
    rnaseq_delta: pd.DataFrame,
    scale: bool = True,
) -> pd.DataFrame:
    """Concatenate and optionally scale CpG + gene delta matrices.

    Aligns on subject index (intersection). Applies StandardScaler
    per-feature if scale=True (ANALYSIS_PLAN.md Step 0-T, Method step 4).

    Parameters
    ----------
    dnam_delta:
        Delta-M matrix, shape (n_subjects, n_cpgs).
    rnaseq_delta:
        Delta-logCPM matrix, shape (n_subjects, n_genes).
    scale:
        If True, centre and scale each feature to zero mean, unit variance.

    Returns
    -------
    pd.DataFrame
        Shape (n_subjects, n_cpgs + n_genes). Index: subject IDs.
    """
    shared_subjects = dnam_delta.index.intersection(rnaseq_delta.index)
    logger.info("Shared paired subjects (DNAm and RNA-seq): %d", len(shared_subjects))
    joint = pd.concat(
        [dnam_delta.loc[shared_subjects], rnaseq_delta.loc[shared_subjects]],
        axis=1,
    )
    logger.info("  Joint delta matrix: %s", joint.shape)

    if scale:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(joint.values)
        joint = pd.DataFrame(scaled, index=joint.index, columns=joint.columns)
        logger.info("  Scaled (zero mean, unit variance) per feature.")

    return joint


# ---------------------------------------------------------------------------
# Phase 1 compatibility shims
# ---------------------------------------------------------------------------


def filter_paired_ids(
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    dnam_sample_col: str = "SampleName_DNAm",
    rna_sample_col: str = "SampleName_RNASeq",
) -> tuple[list[str], list[str], list[str]]:
    """Return (paired_subjects, pre_sample_ids, post_sample_ids).

    Compatibility shim for Phase 1 scripts. Wraps ``identify_paired_subjects``
    and returns parallel lists of subject codes and their PRE/POST sample IDs
    for use in matrix subsetting.

    Parameters
    ----------
    subject_data:
        Subject metadata DataFrame. Must contain subcode_col, visit_col,
        and dnam_sample_col (or rna_sample_col) columns.
    pre_label, post_label:
        Visit labels for PRE and POST respectively.
    subcode_col, visit_col, dnam_sample_col, rna_sample_col:
        Column names in subject_data.

    Returns
    -------
    tuple of (paired_subjects, pre_sample_ids, post_sample_ids)
        - paired_subjects: list of subject Subcode strings
        - pre_sample_ids: list of PRE sample IDs (dnam_sample_col or index)
        - post_sample_ids: list of POST sample IDs (matching order)
    """
    # Resolve sample ID column: prefer dnam_sample_col if present
    sample_col = dnam_sample_col if dnam_sample_col in subject_data.columns else None
    if sample_col is None and visit_col in subject_data.columns:
        # Fall back to index-based IDs
        sample_col = None

    if visit_col in subject_data.columns:
        pre_mask = subject_data[visit_col].astype(str) == pre_label
        post_mask = subject_data[visit_col].astype(str) == post_label
    else:
        pre_mask = pd.Series(False, index=subject_data.index)
        post_mask = pd.Series(False, index=subject_data.index)

    pre_df = subject_data[pre_mask]
    post_df = subject_data[post_mask]

    if subcode_col in subject_data.columns:
        pre_subcodes = set(pre_df[subcode_col].astype(str))
        post_subcodes = set(post_df[subcode_col].astype(str))
        paired_subcodes = sorted(pre_subcodes & post_subcodes)
    else:
        paired_subcodes = []

    pre_rows = (
        pre_df[pre_df[subcode_col].astype(str).isin(paired_subcodes)].set_index(subcode_col)
    )
    post_rows = (
        post_df[post_df[subcode_col].astype(str).isin(paired_subcodes)].set_index(subcode_col)
    )

    pre_ids: list[str] = []
    post_ids: list[str] = []
    paired_subjects: list[str] = []

    for sc in paired_subcodes:
        if sc not in pre_rows.index or sc not in post_rows.index:
            continue
        if sample_col and sample_col in pre_rows.columns:
            pre_id = str(pre_rows.loc[sc, sample_col])
            post_id = str(post_rows.loc[sc, sample_col])
        else:
            pre_id = str(pre_rows.index[pre_rows.index.get_loc(sc)])
            post_id = str(post_rows.index[post_rows.index.get_loc(sc)])
        paired_subjects.append(sc)
        pre_ids.append(pre_id)
        post_ids.append(post_id)

    logger.info(
        "filter_paired_ids: %d paired subjects (%s vs %s).",
        len(paired_subjects), pre_label, post_label,
    )
    return paired_subjects, pre_ids, post_ids


def filter_paired_ids_rna(
    subject_data: pd.DataFrame,
    pre_label: str = "PRE-IOP",
    post_label: str = "POST-IOP",
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
) -> tuple[list[str], list[str], list[str]]:
    """Return (paired_subjects, pre_sample_ids, post_sample_ids) using index.

    Like filter_paired_ids but uses the DataFrame index as sample ID rather
    than a dedicated SampleName column. Correct for RNA-seq alignment where
    the index is already the {Subcode}-{Visit} sample identifier.

    Returns
    -------
    tuple of (paired_subjects, pre_index_ids, post_index_ids)
    """
    if visit_col not in subject_data.columns or subcode_col not in subject_data.columns:
        logger.warning(
            "filter_paired_ids_rna: missing %s or %s columns; returning empty.",
            visit_col,
            subcode_col,
        )
        return [], [], []

    pre_mask = subject_data[visit_col].astype(str) == pre_label
    post_mask = subject_data[visit_col].astype(str) == post_label
    pre_df = subject_data[pre_mask]
    post_df = subject_data[post_mask]

    pre_subcodes = set(pre_df[subcode_col].astype(str))
    post_subcodes = set(post_df[subcode_col].astype(str))
    common = sorted(pre_subcodes & post_subcodes)

    pre_by_subcode: dict[str, str] = {
        str(row[subcode_col]): idx for idx, row in pre_df.iterrows()
    }
    post_by_subcode: dict[str, str] = {
        str(row[subcode_col]): idx for idx, row in post_df.iterrows()
    }

    paired_subjects: list[str] = []
    pre_ids: list[str] = []
    post_ids: list[str] = []
    for sc in common:
        if sc in pre_by_subcode and sc in post_by_subcode:
            paired_subjects.append(sc)
            pre_ids.append(pre_by_subcode[sc])
            post_ids.append(post_by_subcode[sc])

    logger.info(
        "filter_paired_ids_rna: %d paired subjects (%s vs %s).",
        len(paired_subjects),
        pre_label,
        post_label,
    )
    return paired_subjects, pre_ids, post_ids


def build_paired_delta(
    feature_matrix: np.ndarray[Any, np.dtype[np.float64]],
    feature_ids: list[str],
    pre_ids: list[str],
    post_ids: list[str],
    sample_ids: list[str],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], list[str]]:
    """Build a paired POST-PRE delta matrix from a feature matrix.

    Parameters
    ----------
    feature_matrix:
        Shape (n_features, n_samples). Columns match sample_ids ordering.
    feature_ids:
        Feature identifier list (length n_features).
    pre_ids, post_ids:
        Parallel lists of PRE and POST sample IDs (from filter_paired_ids).
    sample_ids:
        Ordered list of all sample IDs matching feature_matrix columns.

    Returns
    -------
    tuple of (delta_matrix, kept_feature_ids)
        delta_matrix: shape (n_features, n_pairs)
        kept_feature_ids: same as feature_ids (all retained)
    """
    sample_idx = {s: i for i, s in enumerate(sample_ids)}
    valid_pairs = [
        (p, q)
        for p, q in zip(pre_ids, post_ids, strict=False)
        if p in sample_idx and q in sample_idx
    ]
    if not valid_pairs:
        empty: np.ndarray[Any, np.dtype[np.float64]] = np.empty(
            (len(feature_ids), 0), dtype=np.float64
        )
        return empty, feature_ids
    pre_pos = [sample_idx[p] for p, _ in valid_pairs]
    post_pos = [sample_idx[q] for _, q in valid_pairs]
    delta: np.ndarray[Any, np.dtype[np.float64]] = (
        feature_matrix[:, post_pos] - feature_matrix[:, pre_pos]
    )
    logger.info("build_paired_delta: %d features x %d pairs.", delta.shape[0], delta.shape[1])
    return delta, feature_ids
