"""Pathway activity inference for Phase 1 Step 1.4.

Infers per-sample pathway activity using:
  1. PROGENy 14 cancer/inflammation pathways via decoupler-py (ULM method).
  2. GSVA on a MetaBase / Reactome / KEGG gene-set collection via decoupler-py
     GSVA wrapper (or gseapy fallback).

Both Emory and BEST cohorts are processed. Delta-activity per paired subject
is computed. Response association test (OLS) is run at PRE, POST, and delta
contrasts.

Analysis plan reference: ANALYSIS_PLAN.md Step 1.4.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PROGENy via decoupler-py
# ---------------------------------------------------------------------------


def get_progeny_net(organism: str = "human", top: int = 100) -> pd.DataFrame:
    """Load PROGENy regulon network via decoupler-py.

    Falls back to a minimal synthetic network if decoupler is not installed
    (used in CI without the full scientific stack).

    Parameters
    ----------
    organism:
        Organism string passed to ``decoupler.get_progeny``.
    top:
        Number of top targets per pathway.

    Returns
    -------
    pd.DataFrame
        Columns: source, target, weight.
    """
    try:
        import decoupler

        net: pd.DataFrame = decoupler.get_progeny(organism=organism, top=top)
        logger.info("Loaded PROGENy network: %d gene-pathway pairs.", len(net))
        return net
    except ImportError:
        logger.warning("decoupler not installed; returning synthetic 2-pathway PROGENy stub.")
        return _synthetic_progeny_net()


def _synthetic_progeny_net() -> pd.DataFrame:
    """Minimal synthetic PROGENy network for CI smoke tests."""
    rows = []
    for pathway in ["TNFa", "MAPK"]:
        for i, gene in enumerate([f"GENE_{pathway}_{j}" for j in range(5)]):
            rows.append({"source": pathway, "target": gene, "weight": float(i + 1)})
    return pd.DataFrame(rows)


def run_progeny_ulm(
    log_cpm: np.ndarray[Any, Any],
    gene_ids: list[str],
    sample_ids: list[str],
    net: pd.DataFrame,
) -> pd.DataFrame:
    """Run PROGENy via decoupler ULM method.

    Parameters
    ----------
    log_cpm:
        2-D array (n_genes, n_samples), log-CPM expression matrix.
    gene_ids:
        Gene identifiers (rows).
    sample_ids:
        Sample identifiers (columns).
    net:
        PROGENy network DataFrame (source, target, weight).

    Returns
    -------
    pd.DataFrame
        Activity matrix, shape (n_samples, n_pathways). Index = sample_ids.
    """
    try:
        import decoupler

        mat = pd.DataFrame(log_cpm.T, index=sample_ids, columns=gene_ids)
        estimates, pvals = decoupler.run_ulm(mat=mat, net=net, verbose=False)
        logger.info(
            "PROGENy ULM complete: %d samples x %d pathways.",
            estimates.shape[0],
            estimates.shape[1],
        )
        return estimates
    except ImportError:
        logger.warning("decoupler not available; returning NaN activity matrix.")
        pathways = net["source"].unique().tolist() if not net.empty else ["TNFa"]
        return pd.DataFrame(
            np.nan,
            index=sample_ids,
            columns=pathways,
        )


# ---------------------------------------------------------------------------
# GSVA via decoupler-py or gseapy
# ---------------------------------------------------------------------------


def run_gsva(
    log_cpm: np.ndarray[Any, Any],
    gene_ids: list[str],
    sample_ids: list[str],
    gene_sets: dict[str, list[str]],
    method: str = "gsva",
    min_targets: int = 5,
) -> pd.DataFrame:
    """Run GSVA on a gene-set dictionary.

    Tries decoupler.run_gsva first; falls back to gseapy.gsva.
    If both are unavailable, returns an empty NaN DataFrame.

    Parameters
    ----------
    log_cpm:
        2-D array (n_genes, n_samples).
    gene_ids:
        Gene identifiers.
    sample_ids:
        Sample identifiers.
    gene_sets:
        Dict mapping gene-set name to list of gene IDs.
    method:
        GSVA method string passed to decoupler (``gsva``, ``ssgsea``).
    min_targets:
        Minimum genes per gene set to keep.

    Returns
    -------
    pd.DataFrame
        GSVA scores, shape (n_samples, n_gene_sets). Index = sample_ids.
    """
    # Filter gene sets to those with at least min_targets in the gene panel
    gene_panel_set = set(gene_ids)
    filtered_sets = {
        k: [g for g in v if g in gene_panel_set]
        for k, v in gene_sets.items()
        if len([g for g in v if g in gene_panel_set]) >= min_targets
    }
    if not filtered_sets:
        logger.warning("No gene sets with >= %d genes in panel; returning empty.", min_targets)
        return pd.DataFrame(index=sample_ids)

    try:
        import decoupler

        # Convert to decoupler network format
        rows = []
        for gs_name, genes in filtered_sets.items():
            for g in genes:
                rows.append({"source": gs_name, "target": g, "weight": 1.0})
        net = pd.DataFrame(rows)

        mat = pd.DataFrame(log_cpm.T, index=sample_ids, columns=gene_ids)
        estimates, _ = decoupler.run_gsva(mat=mat, net=net, verbose=False)
        logger.info("GSVA (decoupler) complete: %d samples x %d sets.", *estimates.shape)
        return estimates

    except (ImportError, AttributeError):
        pass

    try:
        import gseapy

        mat_df = pd.DataFrame(log_cpm, index=gene_ids, columns=sample_ids)
        result = gseapy.ssgsea(data=mat_df, gene_sets=filtered_sets, outdir=None, no_plot=True)
        scores = result.res2d.pivot(index="Name", columns="Term", values="ES")
        logger.info("GSVA (gseapy ssGSEA) complete.")
        return scores.T

    except ImportError:
        logger.warning("Neither decoupler nor gseapy available; returning NaN GSVA matrix.")
        return pd.DataFrame(np.nan, index=sample_ids, columns=list(filtered_sets.keys()))


# ---------------------------------------------------------------------------
# Delta-pathway activity
# ---------------------------------------------------------------------------


def compute_delta_activity(
    activity: pd.DataFrame,
    pre_ids: list[str],
    post_ids: list[str],
    paired_subjects: list[str],
) -> pd.DataFrame:
    """Compute paired-delta activity scores (POST minus PRE).

    Parameters
    ----------
    activity:
        Activity DataFrame, index = sample IDs, columns = pathways/TFs.
    pre_ids:
        PRE sample IDs aligned to paired_subjects order.
    post_ids:
        POST sample IDs aligned to paired_subjects order.
    paired_subjects:
        Paired subject IDs (for output index).

    Returns
    -------
    pd.DataFrame
        Delta activity, shape (n_pairs, n_pathways). Index = paired_subjects.
    """
    pre = activity.loc[pre_ids].values
    post = activity.loc[post_ids].values
    delta = post - pre
    return pd.DataFrame(delta, index=paired_subjects, columns=activity.columns)


# ---------------------------------------------------------------------------
# Response association test (OLS, per pathway)
# ---------------------------------------------------------------------------


def test_response_association(
    activity: pd.DataFrame,
    pdata: pd.DataFrame,
    response_col: str = "Response",
    extra_covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Test Response association with each pathway/TF activity column.

    Fits: activity_k ~ Response + [extra_covariates] per pathway k.

    Parameters
    ----------
    activity:
        Activity DataFrame, index = sample IDs.
    pdata:
        pData2 DataFrame with response_col and covariates.
    response_col:
        Column name for binary R/NR label.
    extra_covariates:
        Additional covariate columns from pdata to include.

    Returns
    -------
    pd.DataFrame
        Columns: pathway, beta_response, p_response, q_response.
    """
    from statsmodels.regression.linear_model import OLS

    resp_raw = pdata[response_col].astype(str).str.strip().str.upper()
    valid_mask = resp_raw.isin(["R", "NR", "RESPONDER", "NON-RESPONDER", "0", "1"])
    pdata_sub = pdata[valid_mask]
    resp_enc = (
        resp_raw[valid_mask]
        .map({"R": 1, "NR": 0, "RESPONDER": 1, "NON-RESPONDER": 0, "1": 1, "0": 0})
        .values.astype(float)
    )

    shared_idx = activity.index.intersection(pdata_sub.index)
    act_sub = activity.loc[shared_idx]
    resp_aligned = pd.Series(resp_enc, index=pdata_sub.index).loc[shared_idx].values

    cov_cols = extra_covariates or []
    cov_cols = [c for c in cov_cols if c in pdata_sub.columns]
    cov_matrix = (
        pdata_sub.loc[shared_idx, cov_cols].fillna(0).values.astype(float)
        if cov_cols
        else np.empty((len(shared_idx), 0))
    )

    n = len(shared_idx)
    intercept = np.ones((n, 1))
    design = np.hstack([intercept, resp_aligned.reshape(-1, 1), cov_matrix])

    rows = []
    for col in act_sub.columns:
        y = act_sub[col].values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() < design.shape[1] + 2:
            rows.append({"pathway": col, "beta_response": np.nan, "p_response": np.nan})
            continue
        try:
            fit = OLS(y[valid], design[valid]).fit()
            rows.append(
                {
                    "pathway": col,
                    "beta_response": float(fit.params[1]),
                    "p_response": float(fit.pvalues[1]),
                }
            )
        except Exception:
            rows.append({"pathway": col, "beta_response": np.nan, "p_response": np.nan})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_response"] = multipletests(df["p_response"].fillna(1.0).values, method="fdr_bh")[1]
    return df[["pathway", "beta_response", "p_response", "q_response"]]


# ---------------------------------------------------------------------------
# Convenience: build empty gene-sets stub for CI
# ---------------------------------------------------------------------------


def stub_gene_sets(n_sets: int = 5, genes_per_set: int = 10) -> dict[str, list[str]]:
    """Return a synthetic gene-set dict for CI smoke tests."""
    return {f"SET_{i}": [f"GENE_{i}_{j}" for j in range(genes_per_set)] for i in range(n_sets)}
