"""ENCODE TFBS / EpiMap regulatory enrichment for Phase 1 Step 1.6.

Tests whether top CellDMC(delta) CpGs are enriched in ENCODE / EpiMap
regulatory annotations (TFBS, enhancer marks, open chromatin).

Method:
  - Convert significant CellDMC(delta) CpG positions to BED intervals.
  - For each (cell-type, regulatory-feature) pair:
      observed = count of CpGs overlapping the feature.
      expected = (size_feature / background_size) * n_sig_cpgs.
      enrichment = observed / expected.
      p-value = hypergeometric test.
  - FDR across all (cell-type, regulatory-feature) pairs via BH.

If pybedtools is not installed, falls back to numpy-based interval overlap
counting (suitable for sorted BED files or small feature sets).

Coordinate system: hg38. EPIC array CpG positions are provided as input;
caller is responsible for lifting over from hg19 if needed (use pyliftover).

Background: EPIC v1 850k array CpG background (user-supplied; a file of all
array CpG positions in BED format).

Analysis plan reference: ANALYSIS_PLAN.md Step 1.6.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BED interval overlap (numpy fallback, no pybedtools required)
# ---------------------------------------------------------------------------


def _count_overlaps_numpy(
    query_chrom: np.ndarray[Any, Any],
    query_start: np.ndarray[Any, Any],
    query_end: np.ndarray[Any, Any],
    target_chrom: np.ndarray[Any, Any],
    target_start: np.ndarray[Any, Any],
    target_end: np.ndarray[Any, Any],
) -> int:
    """Count CpGs overlapping any target interval.

    Both query and target are assumed to be sorted by chromosome then start.
    For 1-bp CpG intervals this reduces to: for each CpG, test whether any
    target interval contains it.

    Time complexity: O(n_query * n_target) in worst case; suitable for
    feature sets < 1M entries. For large feature sets use pybedtools.
    """
    count = 0
    for i in range(len(query_chrom)):
        mask = (
            (target_chrom == query_chrom[i])
            & (target_start <= query_end[i])
            & (target_end >= query_start[i])
        )
        if mask.any():
            count += 1
    return count


def count_cpg_overlaps(
    sig_cpg_bed: pd.DataFrame,
    feature_bed: pd.DataFrame,
    use_pybedtools: bool = True,
) -> int:
    """Count significant CpGs overlapping a regulatory feature BED set.

    Parameters
    ----------
    sig_cpg_bed:
        DataFrame with columns: chrom, start, end (1-based or 0-based; must
        match feature_bed convention).
    feature_bed:
        DataFrame with columns: chrom, start, end.
    use_pybedtools:
        Whether to try pybedtools first (faster for large feature sets).

    Returns
    -------
    int
        Number of sig_cpg entries overlapping any feature_bed entry.
    """
    if sig_cpg_bed.empty or feature_bed.empty:
        return 0

    if use_pybedtools:
        try:
            import pybedtools

            query_bt = pybedtools.BedTool.from_dataframe(sig_cpg_bed[["chrom", "start", "end"]])
            target_bt = pybedtools.BedTool.from_dataframe(feature_bed[["chrom", "start", "end"]])
            intersection = query_bt.intersect(target_bt, u=True)
            return len(intersection)
        except ImportError:
            pass  # fall through to numpy

    return _count_overlaps_numpy(
        sig_cpg_bed["chrom"].values,
        sig_cpg_bed["start"].values,
        sig_cpg_bed["end"].values,
        feature_bed["chrom"].values,
        feature_bed["start"].values,
        feature_bed["end"].values,
    )


# ---------------------------------------------------------------------------
# Hypergeometric enrichment test
# ---------------------------------------------------------------------------


def hypergeometric_enrichment(
    n_sig: int,
    n_background: int,
    feature_size: int,
    n_overlap: int,
) -> dict[str, float]:
    """Compute enrichment statistics via the hypergeometric test.

    Parameters
    ----------
    n_sig:
        Number of significant CpGs (our test set size).
    n_background:
        Total number of CpGs on the array (background population).
    feature_size:
        Size of the regulatory feature set (number of CpGs in background
        that overlap the feature).
    n_overlap:
        Observed overlap between sig_cpgs and feature.

    Returns
    -------
    dict
        Keys: observed, expected, enrichment, p_hypergeom.
    """
    if n_background == 0 or feature_size == 0:
        return {"observed": n_overlap, "expected": 0.0, "enrichment": np.nan, "p_hypergeom": np.nan}

    expected = n_sig * (feature_size / n_background)
    enrichment = (n_overlap / expected) if expected > 0 else np.nan

    # Hypergeometric: PMF of X >= n_overlap given (n_background, feature_size, n_sig)
    # scipy.hypergeom: hypergeom.sf(k-1, M, n, N) = P(X >= k)
    p_val = float(hypergeom.sf(n_overlap - 1, n_background, feature_size, n_sig))

    return {
        "observed": float(n_overlap),
        "expected": float(expected),
        "enrichment": float(enrichment),
        "p_hypergeom": p_val,
    }


# ---------------------------------------------------------------------------
# Full enrichment analysis
# ---------------------------------------------------------------------------


def run_regulatory_enrichment(
    celldmc_delta: pd.DataFrame,
    cpg_positions: pd.DataFrame,
    background_cpg_positions: pd.DataFrame,
    encode_features: dict[str, pd.DataFrame],
    cell_type_col: str = "cell_type",
    fdr_col: str = "q_interaction",
    fdr_threshold: float = 0.05,
    cpg_id_col: str | None = None,
) -> pd.DataFrame:
    """Run ENCODE / EpiMap enrichment for all (cell_type, feature) pairs.

    Parameters
    ----------
    celldmc_delta:
        CellDMC delta output from run_celldmc. Must have columns:
        cpg, cell_type, q_interaction.
    cpg_positions:
        BED-format CpG positions for all CpGs in the analysis (chrom, start,
        end columns; cpg name in index).
    background_cpg_positions:
        BED-format CpG positions for all 850k EPIC array CpGs (background).
    encode_features:
        Dict mapping feature_name -> BED DataFrame (chrom, start, end).
    cell_type_col:
        Column name for cell type in celldmc_delta.
    fdr_col:
        Column name for FDR values in celldmc_delta.
    fdr_threshold:
        Significance threshold for selecting CpGs.

    Returns
    -------
    pd.DataFrame
        Columns: cell_type, feature, observed, expected, enrichment,
        p_hypergeom, q_hypergeom.
    """
    if celldmc_delta.empty or not encode_features:
        logger.warning(
            "Empty CellDMC output or no ENCODE features; returning empty enrichment table."
        )
        return pd.DataFrame(
            columns=[
                "cell_type",
                "feature",
                "observed",
                "expected",
                "enrichment",
                "p_hypergeom",
                "q_hypergeom",
            ]
        )

    # Resolve CpG ID column: prefer explicit param, then try common names
    if cpg_id_col is None:
        for candidate in ("cpg", "cpg_id"):
            if candidate in celldmc_delta.columns:
                cpg_id_col = candidate
                break
        else:
            cpg_id_col = celldmc_delta.columns[0]

    n_background = len(background_cpg_positions)
    rows: list[dict[str, Any]] = []

    cell_types = celldmc_delta[cell_type_col].unique()
    for ct in cell_types:
        ct_mask = (celldmc_delta[cell_type_col] == ct) & (
            celldmc_delta[fdr_col].fillna(1.0) < fdr_threshold
        )
        sig_cpgs = celldmc_delta.loc[ct_mask, cpg_id_col].values
        if len(sig_cpgs) == 0:
            logger.debug("No significant CpGs for cell type %s; skipping.", ct)
            continue

        # Get BED rows for significant CpGs
        avail = [c for c in sig_cpgs if c in cpg_positions.index]
        if not avail:
            continue
        sig_bed = cpg_positions.loc[avail, ["chrom", "start", "end"]].reset_index(drop=True)

        for feat_name, feat_bed in encode_features.items():
            # Count background CpGs overlapping the feature
            n_feature_bg = count_cpg_overlaps(
                background_cpg_positions.reset_index(drop=True),
                feat_bed,
                use_pybedtools=True,
            )
            n_overlap = count_cpg_overlaps(sig_bed, feat_bed, use_pybedtools=True)
            stats = hypergeometric_enrichment(
                n_sig=len(avail),
                n_background=n_background,
                feature_size=n_feature_bg,
                n_overlap=n_overlap,
            )
            rows.append(
                {
                    "cell_type": ct,
                    "feature": feat_name,
                    "n_sig_cpgs": len(avail),
                    **stats,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "cell_type",
                "feature",
                "observed",
                "expected",
                "enrichment",
                "p_hypergeom",
                "q_hypergeom",
            ]
        )

    df = pd.DataFrame(rows)
    df["q_hypergeom"] = multipletests(df["p_hypergeom"].fillna(1.0).values, method="fdr_bh")[1]
    return df.sort_values("p_hypergeom")


# ---------------------------------------------------------------------------
# CpG position utilities
# ---------------------------------------------------------------------------


def cpg_ids_to_bed(
    cpg_ids: list[str],
    cpg_manifest: pd.DataFrame,
    chrom_col: str = "CHR",
    pos_col: str = "MAPINFO",
) -> pd.DataFrame:
    """Convert CpG IDs to 1-bp BED intervals using an EPIC manifest.

    Parameters
    ----------
    cpg_ids:
        List of CpG identifiers (e.g. ``cg00000029``).
    cpg_manifest:
        EPIC manifest DataFrame; must have chrom_col and pos_col columns.
    chrom_col:
        Chromosome column in manifest. Chromosome values may or may not
        have ``chr`` prefix; we normalise to ``chr<N>`` format.
    pos_col:
        0-based start position column in manifest.

    Returns
    -------
    pd.DataFrame
        Columns: chrom, start, end. One row per CpG. Index = cpg IDs.
    """
    avail = [c for c in cpg_ids if c in cpg_manifest.index]
    if not avail:
        return pd.DataFrame(columns=["chrom", "start", "end"])

    sub = cpg_manifest.loc[avail, [chrom_col, pos_col]].copy()
    sub = sub.rename(columns={chrom_col: "chrom", pos_col: "start"})
    sub["chrom"] = sub["chrom"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    sub["start"] = sub["start"].astype(int)
    sub["end"] = sub["start"] + 1  # 1-bp interval
    return sub[["chrom", "start", "end"]]


# ---------------------------------------------------------------------------
# Synthetic stubs for CI
# ---------------------------------------------------------------------------


def stub_encode_features(
    n_features: int = 3,
    n_intervals: int = 100,
) -> dict[str, pd.DataFrame]:
    """Return a synthetic ENCODE feature dict for CI smoke tests."""
    rng = np.random.default_rng(42)
    features = {}
    for i in range(n_features):
        starts = rng.integers(1_000_000, 200_000_000, size=n_intervals)
        features[f"ENCODE_feature_{i}"] = pd.DataFrame(
            {
                "chrom": [f"chr{rng.integers(1, 23)}" for _ in range(n_intervals)],
                "start": starts,
                "end": starts + 500,
            }
        )
    return features


def stub_cpg_positions(
    cpg_ids: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Return synthetic CpG position BED rows for CI smoke tests."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(1_000_000, 200_000_000, size=len(cpg_ids))
    return pd.DataFrame(
        {
            "chrom": [f"chr{rng.integers(1, 23)}" for _ in range(len(cpg_ids))],
            "start": starts,
            "end": starts + 1,
        },
        index=cpg_ids,
    )
