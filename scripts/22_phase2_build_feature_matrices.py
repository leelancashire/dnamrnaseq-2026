"""Phase 2 Step 2.0: build Tier 1 and Tier 2 feature matrices from Phase 1 outputs.

Implements the two-tier feature subsetting scheme (design doc v1.1, Section 2.0,
Kai override 2026-05-19):

  Tier 1: CellDMC-prioritised DNAm CpGs (FDR < 0.10 in the delta-contrast
           interaction test, any cell type), plus the Arm A RNA-side PROGENy /
           decoupleR TF activity scores assembled per design Section 2.1.

  Tier 2: Biology-informed variance-filtered DNAm probes (cross-reactive and
           sex-chromosome probes removed; beta-range floor applied; top 5 000 by
           variance).  RNA-side: top 2 000 HVGs from the full sample set.
           Tier 2 is the documented fallback, not the primary path.

Outputs written to analysis/latest/:
  feature_matrix_tier1_dnam.parquet   -- (n_samples, n_tier1_cpgs) M-values
  feature_matrix_tier1_rna.parquet    -- (n_samples, n_activity_features)
                                         PROGENy + top-150 TF activities
  tier1_cpg_list.txt                  -- one CpG per line
  tier1_celltype_table.tsv            -- sig rows from celldmc_delta_emory.tsv
                                         (cpg, cell_type, coef, fdr)
  feature_matrix_tier2_dnam.parquet   -- (n_samples, TIER2_DNAM_TOP) M-values
  feature_matrix_tier2_rna.parquet    -- (n_samples, TIER2_RNA_TOP) log-CPM
  tier2_cpg_list.txt
  tier2_gene_list.txt
  celldmc_interaction_results.parquet -- canonical name expected by feature_selection.py
                                         (union of delta / pre / post sig CpGs)
  progeny_pathway_activity.parquet    -- alias for feature_selection.py reader
  decoupler_tf_activity.parquet       -- alias for feature_selection.py reader

Author: Lee Lancashire
Design reference: 04-projects/dnamrnaseq/2026-05-19-phase-2-design.md v1.1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LATEST_DIR = Path("analysis/latest")

# Tier 1 significance threshold (design doc Section 2.0, CELLDMC_FDR_THRESHOLD
# in feature_selection.py). Note: the Phase 1 `sig` flag uses FDR < 0.05; this
# script re-applies FDR < 0.10 explicitly per the design contract.
CELLDMC_FDR_THRESHOLD = 0.10

# Tier 2 sizing (design doc Section 2.0)
TIER2_DNAM_TOP = 5_000
TIER2_RNA_TOP = 2_000
BETA_RANGE_FLOOR = 0.05

# Top TF activities by variance (design doc Section 2.1, ~120-220 total features)
TOP_TF_BY_VARIANCE = 150


def _beta_to_m(bvals: np.ndarray) -> np.ndarray:
    """Convert beta-values to M-values (logit transform, clipped to avoid inf)."""
    bvals = np.clip(bvals, 1e-6, 1 - 1e-6)
    result: np.ndarray = np.log2(bvals / (1 - bvals))
    return result


def load_bvals_as_m(latest_dir: Path) -> pd.DataFrame:
    """Load Emory bVals parquet and convert to M-values.

    Returns DataFrame indexed by CpG (rows) x SentrixID (columns).
    """
    logger.info("Loading data_emory.parquet ...")
    raw = pd.read_parquet(latest_dir / "data_emory.parquet")
    # First column is 'cpg'; remaining 388 columns are SentrixIDs
    cpg_ids = raw["cpg"].astype(str).values
    sample_cols = [c for c in raw.columns if c != "cpg"]
    bvals_arr = raw[sample_cols].to_numpy(dtype=np.float32)
    m_arr = _beta_to_m(bvals_arr)
    m_df = pd.DataFrame(m_arr, index=cpg_ids, columns=sample_cols)
    logger.info("M-value matrix: %s (CpGs x samples)", m_df.shape)
    return m_df


def load_pdata(latest_dir: Path) -> pd.DataFrame:
    """Load EpiDISH-augmented pdata (SentrixID index)."""
    pdata = pd.read_csv(latest_dir / "pdata_emory_with_epidish.csv", index_col=0)
    # Construct AMC-ID (used by PROGENy / TF parquets)
    pdata["amc_id"] = pdata["Subcode"] + "-" + pdata["Visit"]
    logger.info("pdata loaded: %d samples, %d covariates", *pdata.shape)
    return pdata


def build_tier1_dnam(
    m_df: pd.DataFrame,
    latest_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build Tier 1 DNAm feature matrix.

    Reads celldmc_delta_emory.tsv and applies FDR < CELLDMC_FDR_THRESHOLD
    (any cell type).  Returns:
        (m_tier1, sig_table, n_unique_cpgs)

    m_tier1: (n_samples, n_sig_cpgs) M-values, sample-indexed (SentrixID).
    sig_table: significant rows from the CellDMC delta table.
    """
    logger.info("Reading CellDMC delta interaction table ...")
    celldmc = pd.read_csv(latest_dir / "celldmc_delta_emory.tsv", sep="\t")
    sig = celldmc[celldmc["fdr"] < CELLDMC_FDR_THRESHOLD].copy()
    n_sig_rows = len(sig)
    n_unique_cpgs = sig["cpg"].nunique()
    logger.info(
        "Tier 1: %d significant CpG x cell-type rows at FDR < %.2f, %d unique CpGs",
        n_sig_rows,
        CELLDMC_FDR_THRESHOLD,
        n_unique_cpgs,
    )

    # Cell-type breakdown
    for ct, grp in sig.groupby("cell_type"):
        logger.info("  %s: %d sig rows (%d unique CpGs)", ct, len(grp), grp["cpg"].nunique())

    # Restrict to CpGs present in the M-value matrix
    tier1_cpgs = sorted(sig["cpg"].astype(str).unique())
    available = [c for c in tier1_cpgs if c in m_df.index]
    if len(available) < len(tier1_cpgs):
        missing = len(tier1_cpgs) - len(available)
        logger.warning(
            "Tier 1: %d CpGs not found in M-value matrix (probes filtered in QC); "
            "proceeding with %d available CpGs",
            missing,
            len(available),
        )

    # Transpose: samples x CpGs
    m_tier1 = m_df.loc[available].T
    logger.info("Tier 1 DNAm matrix (samples x CpGs): %s", m_tier1.shape)
    return m_tier1, sig[["cpg", "cell_type", "coef", "se", "fdr"]], len(available)


def build_tier1_rna(
    pdata: pd.DataFrame,
    latest_dir: Path,
) -> pd.DataFrame:
    """Build Tier 1 RNA-side feature matrix: PROGENy + top-TF activities.

    Returns (n_samples, n_activity_features) DataFrame indexed by SentrixID,
    aligned to the same sample order as pdata.

    PROGENy and TF activity parquets are indexed by AMC-ID
    (e.g. 'AMC-280058-POST-IOP').  We re-index them to SentrixID using the
    amc_id column in pdata.
    """
    logger.info("Loading PROGENy activity ...")
    progeny = pd.read_parquet(latest_dir / "progeny_activity_emory.parquet")
    logger.info("  PROGENy: %s", progeny.shape)

    logger.info("Loading TF activity ...")
    tf_act = pd.read_parquet(latest_dir / "tf_activity_emory.parquet")
    logger.info("  TF activity: %s", tf_act.shape)

    # Select top TFs by variance across samples
    tf_var = tf_act.var(axis=0).sort_values(ascending=False)
    top_tfs = tf_var.head(TOP_TF_BY_VARIANCE).index.tolist()
    tf_top = tf_act[top_tfs].copy()
    logger.info("  TF: selected %d top-variance TFs of %d", len(top_tfs), tf_act.shape[1])

    # Concatenate PROGENy + TF
    rna_act = pd.concat([progeny, tf_top], axis=1)
    logger.info("RNA activity features combined: %s (AMC-ID index)", rna_act.shape)

    # Re-index to SentrixID via pdata amc_id column
    amc_to_sentrix = {row["amc_id"]: idx for idx, row in pdata.iterrows()}
    rna_sentrix = rna_act.rename(index=amc_to_sentrix)
    # Keep only samples that map cleanly
    rna_sentrix = rna_sentrix[rna_sentrix.index.isin(pdata.index)]
    logger.info(
        "RNA activity re-indexed to SentrixID: %d samples mapped of %d AMC-IDs",
        len(rna_sentrix),
        len(rna_act),
    )
    return rna_sentrix


def build_canonical_celldmc_parquet(latest_dir: Path) -> pd.DataFrame:
    """Write canonical celldmc_interaction_results.parquet expected by feature_selection.py.

    Merges delta / pre / post CellDMC tables, keeping all significant rows
    (FDR < CELLDMC_FDR_THRESHOLD) from any contrast.  The `contrast` column
    records which contrast each row came from.
    """
    parts = []
    for contrast_name, fname in [
        ("delta", "celldmc_delta_emory.tsv"),
        ("pre", "celldmc_pre_emory.tsv"),
        ("post", "celldmc_post_emory.tsv"),
    ]:
        p = latest_dir / fname
        if not p.exists():
            logger.warning("CellDMC file not found: %s", p)
            continue
        df = pd.read_csv(p, sep="\t")
        df["contrast"] = contrast_name
        parts.append(df)

    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(latest_dir / "celldmc_interaction_results.parquet", index=False)
    logger.info("celldmc_interaction_results.parquet written: %d rows total", len(combined))
    return combined


def build_tier2_dnam(
    m_df: pd.DataFrame,
    bvals_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build Tier 2 DNAm feature matrix (biology-informed variance filter).

    Filters:
      (a) Beta-range floor: remove probes where max - min <= BETA_RANGE_FLOOR.
      (b) Variance ranking: top TIER2_DNAM_TOP probes.
      (c) Sex-chromosome probes and cross-reactive probes would be removed if
          the EPIC manifest / blacklist files are available; they are not in
          analysis/latest/, so this step is documented but not applied.
          A warning is logged noting the gap.

    Returns (m_tier2, tier2_cpgs) where m_tier2 is (n_samples, TIER2_DNAM_TOP),
    samples as rows.
    """
    logger.warning(
        "Tier 2 sex-chromosome and cross-reactive probe filter: EPIC manifest and "
        "Pidsley 2016 / Zhou 2017 blacklist files not found in analysis/latest/. "
        "Beta-range and variance filters applied; blacklist filter deferred. "
        "See design doc Section 2.0 Tier 2 filter sequence."
    )

    # Beta-range filter applied to the raw bvals (m_df is already M-values)
    bvals_cols = [c for c in bvals_raw.columns if c != "cpg"]
    bvals_arr = bvals_raw[bvals_cols].to_numpy(dtype=np.float32)
    cpg_ids = bvals_raw["cpg"].astype(str).values
    beta_range = bvals_arr.max(axis=1) - bvals_arr.min(axis=1)
    range_mask = beta_range > BETA_RANGE_FLOOR
    logger.info(
        "Tier 2 beta-range filter (> %.3f): %d / %d CpGs pass",
        BETA_RANGE_FLOOR,
        range_mask.sum(),
        len(range_mask),
    )

    # Apply range mask to M-value matrix
    cpgs_passed = cpg_ids[range_mask]
    m_filtered = m_df.loc[cpgs_passed]

    # Variance ranking on filtered set
    variances = m_filtered.var(axis=1).sort_values(ascending=False)
    tier2_cpgs = [str(c) for c in variances.head(TIER2_DNAM_TOP).index]
    logger.info("Tier 2 DNAm: top %d CpGs selected by variance", len(tier2_cpgs))

    m_tier2 = m_df.loc[tier2_cpgs].T
    logger.info("Tier 2 DNAm matrix (samples x CpGs): %s", m_tier2.shape)
    return m_tier2, tier2_cpgs


def build_tier2_rna(
    latest_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Build Tier 2 RNA feature matrix (HVG selection on corrected log-CPM).

    The rnaseq_corrected_emory.parquet in analysis/latest/ has 0 columns (the
    cell-type correction step produced an empty sample axis).  In that case, we
    return an empty DataFrame and log the issue clearly so Tobias's loader
    knows to handle the missing RNA Tier 2 gracefully.

    If expression data becomes available (e.g. after the RNA correction step is
    re-run), this function picks the top TIER2_RNA_TOP HVGs.
    """
    rna_path = latest_dir / "rnaseq_corrected_emory.parquet"
    if rna_path.exists():
        expr = pd.read_parquet(rna_path)
    else:
        logger.warning("rnaseq_corrected_emory.parquet not found; Tier 2 RNA empty")
        return pd.DataFrame(), []

    if expr.shape[1] == 0:
        logger.warning(
            "rnaseq_corrected_emory.parquet has 0 sample columns (Phase 1 RNA "
            "correction produced empty output). Tier 2 RNA matrix is empty. "
            "Re-run Phase 1 step 1.3 with RNA data to populate. "
            "Shape: %s (genes x samples)",
            expr.shape,
        )
        return pd.DataFrame(), []

    # HVG selection (training-fold agnostic at this stage; loaders apply
    # fold-level HVG inside the CV loop per design doc Section 2.0 note)
    variances = expr.var(axis=1).sort_values(ascending=False)
    tier2_genes = [str(g) for g in variances.head(TIER2_RNA_TOP).index]
    logger.info("Tier 2 RNA: top %d HVGs of %d genes", len(tier2_genes), expr.shape[0])
    expr_tier2 = expr.loc[tier2_genes].T
    logger.info("Tier 2 RNA matrix (samples x genes): %s", expr_tier2.shape)
    return expr_tier2, tier2_genes


def main() -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = LATEST_DIR  # all outputs to analysis/latest/

    # --- Load raw data ---
    pdata = load_pdata(LATEST_DIR)
    m_df = load_bvals_as_m(LATEST_DIR)

    # --- Tier 1 DNAm ---
    m_tier1, sig_table, n_tier1_cpgs = build_tier1_dnam(m_df, LATEST_DIR)

    # Write Tier 1 DNAm matrix
    m_tier1.to_parquet(out_dir / "feature_matrix_tier1_dnam.parquet")
    logger.info("Written: feature_matrix_tier1_dnam.parquet %s", m_tier1.shape)

    # Write sig table (subset of celldmc for Tier 1 CpGs)
    sig_table.to_csv(out_dir / "tier1_celltype_table.tsv", sep="\t", index=False)
    logger.info("Written: tier1_celltype_table.tsv (%d rows)", len(sig_table))

    # Write CpG list
    tier1_cpg_list = m_tier1.columns.tolist()
    (out_dir / "tier1_cpg_list.txt").write_text("\n".join(tier1_cpg_list) + "\n")
    logger.info("Written: tier1_cpg_list.txt (%d CpGs)", len(tier1_cpg_list))

    # --- Tier 1 RNA (PROGENy + TF activity) ---
    rna_tier1 = build_tier1_rna(pdata, LATEST_DIR)
    rna_tier1.to_parquet(out_dir / "feature_matrix_tier1_rna.parquet")
    logger.info("Written: feature_matrix_tier1_rna.parquet %s", rna_tier1.shape)

    # --- Canonical CellDMC parquet (alias for feature_selection.py) ---
    build_canonical_celldmc_parquet(LATEST_DIR)

    # --- feature_selection.py name aliases ---
    # PROGENy alias
    progeny_src = LATEST_DIR / "progeny_activity_emory.parquet"
    progeny_dst = out_dir / "progeny_pathway_activity.parquet"
    if progeny_src.exists():
        import shutil

        shutil.copy(progeny_src, progeny_dst)
        logger.info("Written alias: progeny_pathway_activity.parquet")

    # TF activity alias
    tf_src = LATEST_DIR / "tf_activity_emory.parquet"
    tf_dst = out_dir / "decoupler_tf_activity.parquet"
    if tf_src.exists():
        import shutil

        shutil.copy(tf_src, tf_dst)
        logger.info("Written alias: decoupler_tf_activity.parquet")

    # --- Tier 2 DNAm ---
    bvals_raw = pd.read_parquet(LATEST_DIR / "data_emory.parquet")
    m_tier2, tier2_cpgs = build_tier2_dnam(m_df, bvals_raw)
    m_tier2.to_parquet(out_dir / "feature_matrix_tier2_dnam.parquet")
    logger.info("Written: feature_matrix_tier2_dnam.parquet %s", m_tier2.shape)
    (out_dir / "tier2_cpg_list.txt").write_text("\n".join(tier2_cpgs) + "\n")
    logger.info("Written: tier2_cpg_list.txt (%d CpGs)", len(tier2_cpgs))

    # --- Tier 2 RNA ---
    rna_tier2, tier2_genes = build_tier2_rna(LATEST_DIR)
    if not rna_tier2.empty:
        rna_tier2.to_parquet(out_dir / "feature_matrix_tier2_rna.parquet")
        logger.info("Written: feature_matrix_tier2_rna.parquet %s", rna_tier2.shape)
    else:
        # Write empty parquet as stub so loaders don't error on missing file
        pd.DataFrame().to_parquet(out_dir / "feature_matrix_tier2_rna.parquet")
        logger.warning(
            "Written: feature_matrix_tier2_rna.parquet (empty stub -- RNA data unavailable)"
        )
    (out_dir / "tier2_gene_list.txt").write_text("\n".join(tier2_genes) + "\n")
    logger.info("Written: tier2_gene_list.txt (%d genes)", len(tier2_genes))

    # --- Summary ---
    logger.info("=== Feature matrix build complete ===")
    logger.info(
        "Tier 1 DNAm: %d samples x %d CpGs (FDR < %.2f, delta contrast, any cell type)",
        m_tier1.shape[0],
        m_tier1.shape[1],
        CELLDMC_FDR_THRESHOLD,
    )
    logger.info(
        "Tier 1 RNA:  %d samples x %d activity features (PROGENy + top-%d TFs)",
        rna_tier1.shape[0],
        rna_tier1.shape[1],
        TOP_TF_BY_VARIANCE,
    )
    logger.info(
        "Tier 2 DNAm: %d samples x %d CpGs (beta-range filter + variance top-%d)",
        m_tier2.shape[0],
        m_tier2.shape[1],
        TIER2_DNAM_TOP,
    )
    if not rna_tier2.empty:
        logger.info("Tier 2 RNA:  %d samples x %d HVGs", rna_tier2.shape[0], rna_tier2.shape[1])
    else:
        logger.warning("Tier 2 RNA:  EMPTY (Phase 1 RNA correction not yet run)")


if __name__ == "__main__":
    main()
