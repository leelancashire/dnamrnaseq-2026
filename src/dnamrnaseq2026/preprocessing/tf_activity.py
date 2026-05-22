"""Transcription-factor activity inference for Phase 1 Step 1.5.

Uses CollecTRI regulons via decoupler-py ULM method to infer per-sample TF
activity from cell-type-corrected RNA-seq matrices.

Priority TF families (from mmVAE supplementary priors):
  - NFAT family: NFATC1, NFATC2, NFATC3, NFATC4, NFAT5
  - WNT pathway TFs: TCF7L2, LEF1, TCF7, TCF7L1

Acceptance criteria (ANALYSIS_PLAN.md Step 1.5):
  - At least 1 NFAT or WNT TF with FDR < 0.10 Response association at delta.
  - TF activity matrices have non-null values for >= 1,000 TFs per sample.

Analysis plan reference: ANALYSIS_PLAN.md Step 1.5.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# Priority TF families per mmVAE supplementary (ANALYSIS_PLAN.md Step 1.5)
NFAT_FAMILY = ["NFATC1", "NFATC2", "NFATC3", "NFATC4", "NFAT5"]
WNT_TFS = ["TCF7L2", "LEF1", "TCF7", "TCF7L1"]
PRIORITY_TFS = set(NFAT_FAMILY + WNT_TFS)


# ---------------------------------------------------------------------------
# CollecTRI network loading
# ---------------------------------------------------------------------------


def get_collectri_net(
    organism: str = "human",
    split_complexes: bool = False,
    min_targets: int = 5,
) -> pd.DataFrame:
    """Load CollecTRI regulon network via decoupler-py.

    Compatible with both decoupler 1.x (``decoupler.get_collectri``) and
    decoupler 2.x (``decoupler.op.collectri``).

    Parameters
    ----------
    organism:
        ``'human'`` or ``'mouse'``.
    split_complexes:
        Whether to split TF complexes into individual subunits.  In
        decoupler 2.x this parameter is named ``remove_complexes`` (inverted
        semantics); both are handled transparently.
    min_targets:
        Drop TFs with fewer than this many unique targets.

    Returns
    -------
    pd.DataFrame
        Columns: source (TF), target (gene), weight (positive or negative).
    """
    try:
        import decoupler

        # decoupler 2.x: decoupler.op.collectri
        if hasattr(decoupler, "op") and hasattr(decoupler.op, "collectri"):
            # 2.x uses remove_complexes (inverted logic relative to split_complexes)
            net: pd.DataFrame = decoupler.op.collectri(
                organism=organism, remove_complexes=not split_complexes
            )
        elif hasattr(decoupler, "get_collectri"):
            # decoupler 1.x legacy
            net = decoupler.get_collectri(organism=organism, split_complexes=split_complexes)
        else:
            raise AttributeError("Could not find CollecTRI loader in decoupler API.")

        # Filter to TFs with at least min_targets
        counts = net.groupby("source")["target"].nunique()
        keep_tfs = counts[counts >= min_targets].index
        net = net[net["source"].isin(keep_tfs)]
        logger.info(
            "CollecTRI loaded: %d TFs, %d TF-gene pairs (min_targets=%d).",
            net["source"].nunique(),
            len(net),
            min_targets,
        )
        return net
    except ImportError:
        logger.warning("decoupler not installed; returning synthetic CollecTRI stub.")
        return _synthetic_collectri_net()


def _synthetic_collectri_net(n_tfs: int = 5, n_targets: int = 10) -> pd.DataFrame:
    """Minimal synthetic CollecTRI network for CI smoke tests."""
    rows = []
    tfs = [f"TF_{i}" for i in range(n_tfs)]
    for tf in tfs:
        for j in range(n_targets):
            rows.append(
                {
                    "source": tf,
                    "target": f"GENE_{tf}_{j}",
                    "weight": 1.0 if j % 2 == 0 else -1.0,
                }
            )
    # Embed priority TFs so flag tests work
    for tf in ["NFATC1", "TCF7L2"]:
        for j in range(n_targets):
            rows.append({"source": tf, "target": f"GENE_{tf}_{j}", "weight": 1.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TF activity inference
# ---------------------------------------------------------------------------


def _strip_ensembl_prefix(gene_ids: list[str]) -> list[str]:
    """Convert Ensembl+symbol IDs to gene symbols.

    Handles the 'ENSG00000..._SYMBOL' format from the mmVAE RNA-seq pipeline.
    Duplicates after stripping are made unique with '_1', '_2' suffixes.
    """
    symbols = []
    for g in gene_ids:
        parts = g.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("ENSG"):
            symbols.append(parts[1])
        else:
            symbols.append(g)
    seen: dict[str, int] = {}
    result: list[str] = []
    for sym in symbols:
        if sym in seen:
            seen[sym] += 1
            result.append(f"{sym}_{seen[sym]}")
        else:
            seen[sym] = 0
            result.append(sym)
    return result


def run_tf_ulm(
    log_cpm: np.ndarray[Any, Any],
    gene_ids: list[str],
    sample_ids: list[str],
    net: pd.DataFrame,
) -> pd.DataFrame:
    """Run TF activity inference via decoupler ULM on CollecTRI regulons.

    Gene IDs are converted to gene symbols before matching against the
    CollecTRI network (handles Ensembl+symbol format from the mmVAE pipeline).

    Compatible with both decoupler 1.x and 2.x APIs.

    Parameters
    ----------
    log_cpm:
        2-D array (n_genes, n_samples), log-CPM expression.
    gene_ids:
        Gene identifiers (rows).  May be in 'ENSG..._SYMBOL' format.
    sample_ids:
        Sample identifiers (columns).
    net:
        CollecTRI network DataFrame (source, target, weight).

    Returns
    -------
    pd.DataFrame
        TF activity matrix, shape (n_samples, n_tfs). Index = sample_ids.
    """
    try:
        import decoupler

        symbols = _strip_ensembl_prefix(gene_ids)
        mat = pd.DataFrame(log_cpm.T, index=sample_ids, columns=symbols)

        # decoupler 2.x: decoupler.mt.ulm(data, net) returns (estimates, pvals)
        if hasattr(decoupler, "mt") and hasattr(decoupler.mt, "ulm"):
            out = decoupler.mt.ulm(data=mat, net=net)
            estimates = out[0] if isinstance(out, tuple) else out
            if isinstance(estimates, pd.DataFrame):
                result_df = estimates
            else:
                result_df = pd.DataFrame(estimates)
        elif hasattr(decoupler, "run_ulm"):
            # decoupler 1.x legacy
            estimates, _ = decoupler.run_ulm(mat=mat, net=net, verbose=False)
            result_df = pd.DataFrame(estimates)
        else:
            raise AttributeError("Could not find ULM runner in decoupler API.")

        logger.info(
            "CollecTRI ULM complete: %d samples x %d TFs.",
            result_df.shape[0],
            result_df.shape[1],
        )
        return result_df
    except ImportError:
        logger.warning("decoupler not available; returning NaN TF activity matrix.")
        tfs = net["source"].unique().tolist() if not net.empty else ["TF_0"]
        return pd.DataFrame(np.nan, index=sample_ids, columns=tfs)


# ---------------------------------------------------------------------------
# Delta TF activity
# ---------------------------------------------------------------------------


def compute_delta_tf_activity(
    tf_activity: pd.DataFrame,
    pre_ids: list[str],
    post_ids: list[str],
    paired_subjects: list[str],
) -> pd.DataFrame:
    """Compute delta TF activity (POST minus PRE) per paired subject.

    Parameters
    ----------
    tf_activity:
        Activity DataFrame, index = sample IDs, columns = TF names.
    pre_ids, post_ids:
        PRE and POST sample IDs aligned to paired_subjects order.
    paired_subjects:
        Paired subject identifiers.

    Returns
    -------
    pd.DataFrame
        Delta TF activity, shape (n_pairs, n_tfs). Index = paired_subjects.
    """
    pre = tf_activity.loc[pre_ids].values
    post = tf_activity.loc[post_ids].values
    return pd.DataFrame(post - pre, index=paired_subjects, columns=tf_activity.columns)


# ---------------------------------------------------------------------------
# Per-TF Response association
# ---------------------------------------------------------------------------


def test_tf_response_association(
    delta_activity: pd.DataFrame,
    pdata_paired: pd.DataFrame,
    response_col: str = "Response",
    extra_covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Per-TF OLS association with Response in delta space.

    Model: delta_TF_activity_k ~ Response + [covariates] for each TF k.
    FDR: BH per TF set.

    Parameters
    ----------
    delta_activity:
        Delta TF activity, index = paired subject IDs.
    pdata_paired:
        pData2 for paired subjects; must contain response_col.
    response_col:
        Column for binary R/NR label.
    extra_covariates:
        Additional covariate columns from pdata_paired.

    Returns
    -------
    pd.DataFrame
        Columns: tf, beta_response, p_response, q_response, priority_family.
    """
    from statsmodels.regression.linear_model import OLS

    resp_raw = pdata_paired[response_col].astype(str).str.strip().str.upper()
    valid_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    pdata_sub = pdata_paired[valid_mask]
    resp_enc = (
        resp_raw[valid_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .values.astype(float)
    )

    shared_idx = delta_activity.index.intersection(pdata_sub.index)
    da_sub = delta_activity.loc[shared_idx]
    resp_aligned: np.ndarray[Any, Any] = np.asarray(
        pd.Series(resp_enc, index=pdata_sub.index).loc[shared_idx].values,
        dtype=float,
    )

    cov_cols = [c for c in (extra_covariates or []) if c in pdata_sub.columns]
    cov_matrix = (
        pdata_sub.loc[shared_idx, cov_cols].fillna(0).to_numpy().astype(float)
        if cov_cols
        else np.empty((len(shared_idx), 0))
    )
    n = len(shared_idx)
    intercept = np.ones((n, 1))
    design = np.hstack([intercept, resp_aligned.reshape(-1, 1), cov_matrix])

    rows = []
    for tf_name in da_sub.columns:
        y = da_sub[tf_name].values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < design.shape[1] + 2:
            rows.append({"tf": tf_name, "beta_response": np.nan, "p_response": np.nan})
            continue
        try:
            fit = OLS(y[valid], design[valid]).fit()
            rows.append(
                {
                    "tf": tf_name,
                    "beta_response": float(fit.params[1]),
                    "p_response": float(fit.pvalues[1]),
                }
            )
        except Exception:
            rows.append({"tf": tf_name, "beta_response": np.nan, "p_response": np.nan})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["q_response"] = multipletests(df["p_response"].fillna(1.0).values, method="fdr_bh")[1]

    # Priority family flag
    df["priority_family"] = df["tf"].apply(_priority_family_label)

    return df[["tf", "beta_response", "p_response", "q_response", "priority_family"]]


def _priority_family_label(tf: str) -> str:
    """Return priority family label for known TFs, else empty string."""
    if tf in NFAT_FAMILY:
        return "NFAT"
    if tf in WNT_TFS:
        return "WNT"
    return ""


# ---------------------------------------------------------------------------
# Priority TF flag table
# ---------------------------------------------------------------------------


def build_priority_tf_table(
    tf_response_test: pd.DataFrame,
    fdr_threshold: float = 0.10,
) -> pd.DataFrame:
    """Extract priority TF rows and flag significance.

    Parameters
    ----------
    tf_response_test:
        Output of test_tf_response_association.
    fdr_threshold:
        Significance threshold for the flag.

    Returns
    -------
    pd.DataFrame
        Rows for NFAT and WNT TFs with a ``significant`` column.
    """
    priority = tf_response_test[tf_response_test["priority_family"].isin(["NFAT", "WNT"])].copy()
    if priority.empty:
        # Return empty with the expected schema
        return pd.DataFrame(
            columns=[
                "tf",
                "beta_response",
                "p_response",
                "q_response",
                "priority_family",
                "significant",
            ]
        )
    priority["significant"] = priority["q_response"] < fdr_threshold
    return priority.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def check_acceptance_criteria(
    tf_activity: pd.DataFrame,
    tf_response_test: pd.DataFrame,
    fdr_threshold: float = 0.10,
    min_tfs_per_sample: int = 1000,
) -> dict[str, Any]:
    """Check Step 1.5 acceptance criteria.

    Returns
    -------
    dict with keys: criteria_1_pass (non-null TFs per sample), criteria_2_pass
    (NFAT or WNT significant), overall_pass.
    """
    # Criterion 1: >= 1000 non-null TF values per sample
    non_null_per_sample = tf_activity.notna().sum(axis=1)
    min_non_null = int(non_null_per_sample.min()) if not tf_activity.empty else 0
    crit1 = min_non_null >= min_tfs_per_sample

    # Criterion 2: at least 1 priority TF with FDR < threshold
    priority_hits = tf_response_test[
        (tf_response_test["priority_family"].isin(["NFAT", "WNT"]))
        & (tf_response_test["q_response"] < fdr_threshold)
    ]
    crit2 = len(priority_hits) > 0

    return {
        "criteria_1_pass": crit1,
        "min_non_null_tfs_per_sample": min_non_null,
        "criteria_2_pass": crit2,
        "n_priority_tfs_significant": len(priority_hits),
        "overall_pass": crit1 and crit2,
    }
