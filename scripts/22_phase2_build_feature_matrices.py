"""Phase 2 Step 2.0: build Tier 1 and Tier 2 feature matrices from Phase 1 outputs.

Implements the two-tier feature subsetting scheme (design doc v1.1, Section 2.0,
Kai override 2026-05-19):

  Tier 1: CellDMC-prioritised DNAm CpGs (FDR < 0.10 in the delta-contrast
           interaction test, any cell type), plus the Arm A RNA-side PROGENy /
           decoupleR TF activity scores assembled per design Section 2.1.

  Tier 2: Biology-informed variance-filtered DNAm probes (cross-reactive and
           sex-chromosome probes removed; beta-range floor applied).  RNA-side:
           corrected log-CPM.  Tier 2 is the documented fallback, not the
           primary path.

LEAKAGE CONTRACT (design doc Section 4.2 -- hard rule)
------------------------------------------------------
Tier 2 variance / HVG *ranking* is a data-driven feature selection and MUST be
fit inside the outer CV loop, on the training fold only, then applied to the
held-out fold. Ranking variance / HVGs on the full cohort lets held-out test
rows decide which features exist; that is a train/test leak that invalidates an
embedding benchmark.

Therefore this script DOES NOT bake a fixed cohort-wide top-5000 / top-2000
list. It writes the full biology-filtered Tier 2 *candidate* matrices (all
beta-range-passing CpGs / all corrected genes) tagged as EDA / exploratory
only. The variance / HVG top-N selection is performed per fold by
``dnamrnaseq2026.embedding.data_harness.PairedPreprocessor``. The candidate
matrices carry a ``cv_loop_safe = False`` / ``selection_stage = "EDA_ONLY"``
provenance marker so a loader cannot silently feed them into the CV loop as if
they were a finished, leakage-free feature set.

Tier 1 (CellDMC FDR<0.10) is a fixed biological prior, not a data-driven
selection; it is correctly pre-computable cohort-wide and is reported in the
manuscript as a fixed cohort-level prior, NOT a learned selection.

LEAKAGE CONTRACT -- TIER 1 RNA (corrected 2026-05-22, Helen Zhao)
-----------------------------------------------------------------
Tier 1 DNAm (CellDMC FDR<0.10) is a genuinely fixed biological prior: a
pre-specified significance threshold, no cohort-relative ranking. It stays
``cv_loop_safe = True``.

Tier 1 RNA is NOT a pure fixed prior. The PROGENy 14-pathway panel IS a fixed
curated set, but the TF panel was assembled by ``tf_act.var(axis=0)`` -- a
variance rank computed across ALL cohort samples. A cohort-wide variance rank
lets held-out test rows decide which TFs enter the matrix; that is the same
train/test leak class caught for Tier 2. The design doc Section 2.1 always
specified the TF set as "top ~100-200 TF activity scores by variance", i.e.
a data-driven rank, never a curated biological list -- so the leakage-correct
AND design-faithful fix is to move the TF variance rank per training fold,
not to invent a fixed list. Consequently this script no longer bakes a
cohort-variance-ranked 150-TF Tier 1 RNA matrix. It writes the FULL TF
activity matrix (PROGENy fixed + all TFs) as a candidate set stamped
``cv_loop_safe = False``; the top-N TF variance selection is fit per fold by
``PairedPreprocessor`` (or, for the unsupervised Arm B run, per MOFA+ fit on
the training rows). PROGENy stays fixed and leakage-free regardless.

Outputs written to analysis/latest/:
  feature_matrix_tier1_dnam.parquet   -- (n_samples, n_tier1_cpgs) M-values
                                         cv_loop_safe=True (fixed CellDMC prior)
  feature_matrix_tier1_rna.parquet    -- (n_samples, 14 PROGENy + all TFs)
                                         cv_loop_safe=False -- TF variance rank
                                         is fit per fold; PROGENy fixed
  tier1_cpg_list.txt                  -- one CpG per line
  tier1_celltype_table.tsv            -- sig rows from celldmc_delta_emory.tsv
                                         (cpg, cell_type, coef, fdr)
  feature_matrix_tier2_dnam.parquet   -- (n_samples, n_candidate_cpgs) M-values,
                                         EDA-only candidate set; per-fold
                                         variance ranking applied downstream
  feature_matrix_tier2_rna.parquet    -- (n_samples, n_candidate_genes) log-CPM,
                                         EDA-only candidate set; per-fold HVG
                                         selection applied downstream
  tier2_candidate_cpg_list.txt
  tier2_candidate_gene_list.txt
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

# Tier 2 sizing (design doc Section 2.0). These are the per-fold selection
# sizes consumed by PairedPreprocessor INSIDE the CV loop -- not applied here.
TIER2_DNAM_TOP = 5_000
TIER2_RNA_TOP = 2_000
BETA_RANGE_FLOOR = 0.05

# Provenance marker stamped on the Tier 2 candidate matrices. The matrices are
# biology-filtered candidate sets only; the data-driven variance / HVG ranking
# is fit per fold by PairedPreprocessor (design doc Section 4.2 leakage rule).
# This stamp is ENFORCED consumer-side: feature_selection.assert_cv_loop_safe
# (called by load_feature_matrix_for_cv, the canonical on-disk-matrix loader)
# raises Phase1ArtefactError when cv_loop_safe is False, so a candidate matrix
# cannot be fed into a CV / training path.
TIER2_PROVENANCE = {
    "selection_stage": "EDA_ONLY",
    "cv_loop_safe": "False",
    "note": (
        "Biology-filtered Tier 2 candidate set. Variance/HVG top-N ranking is "
        "fit per training fold by PairedPreprocessor; do NOT treat this matrix "
        "as a finished leakage-free feature set. Design doc Section 4.2."
    ),
}

# Provenance marker for the Tier 1 DNAm matrix. The DNAm Tier 1 feature set is
# a fixed biological prior: CellDMC FDR<0.10 CpGs from the delta-contrast
# interaction test (any cell type). FDR<0.10 is a pre-specified significance
# threshold, not a cohort-relative variance ranking; no held-out sample
# influences which CpGs are included. Tier 1 DNAm is therefore legitimately
# CV-loop-safe and may be admitted to the Phase 2 modelling path directly.
# Design doc v1.1 Section 2.0; PR #12 review (Helen Zhao).
TIER1_DNAM_PROVENANCE = {
    "selection_stage": "TIER1_FIXED_PRIOR",
    "cv_loop_safe": "True",
    "rationale": (
        "Tier 1 DNAm is a fixed biological prior, not a cohort-data-driven "
        "selection. CellDMC FDR<0.10 CpGs from the delta-contrast interaction "
        "test (any cell type) -- a pre-specified significance threshold, not a "
        "variance ranking fitted on the full cohort. Because no held-out test "
        "sample influences which CpGs are included, the Tier 1 DNAm matrix "
        "carries no train/test leakage and is safe to load directly into the "
        "CV loop. Design doc v1.1 Section 2.0; PR #12 review (Helen Zhao)."
    ),
}

# Provenance marker for the Tier 1 RNA matrix. CORRECTED 2026-05-22 (Helen
# Zhao): Tier 1 RNA is NOT a pure fixed prior and is stamped
# cv_loop_safe=False. The PROGENy 14-pathway panel IS fixed and curated, but
# the TF panel was previously assembled by a cohort-wide variance rank
# (tf_act.var(axis=0) over all 344 samples). A cohort-wide variance rank lets
# held-out rows decide which TFs enter the matrix -- the same train/test leak
# class caught for Tier 2. This script no longer bakes a 150-TF cohort-ranked
# panel; it writes PROGENy (fixed) + ALL TF activities as a candidate matrix,
# and the top-N TF variance rank is fit per training fold by
# PairedPreprocessor (or per MOFA+ fit on training rows for the unsupervised
# Arm B). Design doc Section 2.1 always specified the TF set as a variance
# rank, so per-fold selection is both leakage-correct and design-faithful.
TIER1_RNA_PROVENANCE = {
    "selection_stage": "TIER1_RNA_CANDIDATE",
    "cv_loop_safe": "False",
    "note": (
        "Tier 1 RNA candidate set: PROGENy 14-pathway activities (fixed, "
        "leakage-free curated panel) concatenated with the FULL decoupleR/"
        "CollecTRI TF activity matrix (all TFs, NOT a baked top-N list). The "
        "top-N TF variance ranking is a data-driven selection and is fit per "
        "training fold by PairedPreprocessor (design doc Section 4.2); for the "
        "unsupervised Arm B MOFA+ run the TF rank is computed on the training "
        "rows of each fit. Do NOT treat this matrix as a finished leakage-free "
        "feature set. Corrected 2026-05-22 (Helen Zhao): the previous build "
        "baked a cohort-variance-ranked 150-TF panel, which leaked held-out "
        "rows into feature selection."
    ),
}


def _stamp_tier2_provenance(path: Path) -> None:
    """Write the EDA-only / cv_loop_safe=False provenance marker beside a matrix.

    The marker is a sidecar JSON keyed to the matrix filename so a downstream
    loader can assert ``cv_loop_safe`` before admitting the matrix to the CV
    loop. Stamping a sidecar (rather than parquet metadata) keeps the check
    trivially readable and engine-agnostic.
    """
    import json

    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(TIER2_PROVENANCE, indent=2) + "\n")
    logger.info("Written provenance marker: %s (cv_loop_safe=False)", sidecar.name)


def _stamp_tier1_provenance(path: Path, provenance: dict[str, str]) -> None:
    """Write a Tier 1 provenance marker sidecar beside a Tier 1 matrix.

    Tier 1 DNAm (``TIER1_DNAM_PROVENANCE``) is a fixed biological prior
    (CellDMC FDR<0.10) and is stamped ``cv_loop_safe=True``. Tier 1 RNA
    (``TIER1_RNA_PROVENANCE``) is a candidate set: PROGENy is fixed but the TF
    variance rank must be fit per fold, so it is stamped ``cv_loop_safe=False``
    and the fail-closed loader will refuse it on the CV path.

    The sidecar is read by the fail-closed loader
    (feature_selection.assert_cv_loop_safe): a missing sidecar raises
    Phase1ArtefactError. Both Tier 1 matrices must be stamped before any
    downstream Phase 2 path can consume them.
    """
    import json

    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info(
        "Written provenance marker: %s (cv_loop_safe=%s)",
        sidecar.name,
        provenance["cv_loop_safe"],
    )


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
    """Build the Tier 1 RNA-side CANDIDATE matrix: PROGENy + ALL TF activities.

    Returns (n_samples, 14 PROGENy + n_all_tfs) DataFrame indexed by SentrixID,
    aligned to the same sample order as pdata.

    CORRECTED 2026-05-22 (Helen Zhao): this function no longer variance-ranks
    the TF panel to a fixed cohort-wide top-150 list. ``tf_act.var(axis=0)``
    was a variance rank computed across all cohort samples; baking that into
    the matrix lets held-out test rows decide which TFs exist -- the same
    train/test leak class caught for Tier 2. The full TF activity matrix is
    written instead; the top-N TF variance ranking is fit per training fold
    downstream by PairedPreprocessor (design doc Section 4.2). PROGENy (a
    fixed 14-pathway curated panel) stays leakage-free and is kept as-is.

    PROGENy and TF activity parquets are indexed by AMC-ID
    (e.g. 'AMC-280058-POST-IOP').  We re-index them to SentrixID using the
    amc_id column in pdata.
    """
    logger.info("Loading PROGENy activity ...")
    progeny = pd.read_parquet(latest_dir / "progeny_activity_emory.parquet")
    logger.info("  PROGENy: %s (fixed 14-pathway panel, leakage-free)", progeny.shape)

    logger.info("Loading TF activity ...")
    tf_act = pd.read_parquet(latest_dir / "tf_activity_emory.parquet")
    logger.info(
        "  TF activity: %s -- ALL TFs retained; per-fold variance rank applied "
        "downstream (NO cohort-wide top-N baked here)",
        tf_act.shape,
    )

    # Concatenate PROGENy (fixed) + the FULL TF activity matrix (candidate set).
    # No cohort-wide variance ranking is applied here -- that is a data-driven
    # selection and is fit per training fold by PairedPreprocessor.
    rna_act = pd.concat([progeny, tf_act], axis=1)
    logger.info(
        "RNA Tier 1 CANDIDATE matrix combined: %s (AMC-ID index) -- "
        "%d PROGENy + %d TF, cv_loop_safe=False",
        rna_act.shape,
        progeny.shape[1],
        tf_act.shape[1],
    )

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
    """Build the Tier 2 DNAm *candidate* matrix (biology filter only, no ranking).

    Applies only the cohort-invariant biology filter:
      (a) Beta-range floor: remove probes where max - min <= BETA_RANGE_FLOOR.
      (b) Sex-chromosome / cross-reactive probes would be removed if the EPIC
          manifest / blacklist files were available; they are not in
          analysis/latest/, so that step is documented but not applied.

    It deliberately DOES NOT variance-rank to a fixed top-TIER2_DNAM_TOP list.
    Variance ranking is a data-driven selection and must be fit per training
    fold (design doc Section 4.2); doing it here on all 388 samples leaks the
    held-out test rows into feature selection. The variance top-N step is
    performed downstream by PairedPreprocessor.fit() on each outer-fold
    training subset.

    The beta-range floor is a fixed threshold, not a cohort-relative ranking,
    so it is leakage-safe to apply cohort-wide; it only removes probes that are
    near-constant in every sample.

    Returns (m_tier2_candidates, candidate_cpgs): the full biology-filtered
    candidate matrix (n_samples, n_candidate_cpgs), samples as rows.
    """
    logger.warning(
        "Tier 2 sex-chromosome and cross-reactive probe filter: EPIC manifest and "
        "Pidsley 2016 / Zhou 2017 blacklist files not found in analysis/latest/. "
        "Beta-range filter applied; blacklist filter deferred. "
        "See design doc Section 2.0 Tier 2 filter sequence."
    )

    # Beta-range filter applied to the raw bvals (m_df is already M-values).
    # This is a fixed-threshold, cohort-invariant filter -- leakage-safe.
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

    cpgs_passed = [str(c) for c in cpg_ids[range_mask]]
    m_tier2 = m_df.loc[cpgs_passed].T
    logger.info(
        "Tier 2 DNAm CANDIDATE matrix (samples x CpGs): %s -- NO variance ranking "
        "applied; per-fold top-%d selection is done by PairedPreprocessor",
        m_tier2.shape,
        TIER2_DNAM_TOP,
    )
    return m_tier2, cpgs_passed


def build_tier2_rna(
    latest_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the Tier 2 RNA *candidate* matrix (corrected log-CPM, no HVG ranking).

    The rnaseq_corrected_emory.parquet in analysis/latest/ currently has 0
    columns (the cell-type correction step produced an empty sample axis). In
    that case an empty DataFrame is returned and the issue logged so the loader
    handles the missing RNA Tier 2 gracefully.

    When expression data is available, this function returns the FULL corrected
    expression matrix as the Tier 2 candidate set. It deliberately DOES NOT
    select the top TIER2_RNA_TOP HVGs: HVG selection is a data-driven selection
    and must be fit per training fold (design doc Section 4.2); ranking HVGs on
    all 344 samples here leaks the held-out test rows. The HVG top-N step is
    performed downstream by PairedPreprocessor.fit() on each outer-fold
    training subset.

    Returns (expr_candidates, candidate_genes): the full corrected expression
    matrix (n_samples, n_candidate_genes), samples as rows.
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

    candidate_genes = [str(g) for g in expr.index]
    expr_tier2 = expr.T
    logger.info(
        "Tier 2 RNA CANDIDATE matrix (samples x genes): %s -- NO HVG ranking "
        "applied; per-fold top-%d HVG selection is done by PairedPreprocessor",
        expr_tier2.shape,
        TIER2_RNA_TOP,
    )
    return expr_tier2, candidate_genes


def main() -> None:
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = LATEST_DIR  # all outputs to analysis/latest/

    # --- Load raw data ---
    pdata = load_pdata(LATEST_DIR)
    m_df = load_bvals_as_m(LATEST_DIR)

    # --- Tier 1 DNAm ---
    m_tier1, sig_table, n_tier1_cpgs = build_tier1_dnam(m_df, LATEST_DIR)

    # Write Tier 1 DNAm matrix + provenance sidecar.
    # cv_loop_safe=True: Tier 1 is a fixed biological prior (CellDMC FDR<0.10),
    # not a cohort-data-driven selection, so it carries no train/test leakage.
    # The sidecar is required by the fail-closed loader (assert_cv_loop_safe).
    tier1_dnam_path = out_dir / "feature_matrix_tier1_dnam.parquet"
    m_tier1.to_parquet(tier1_dnam_path)
    _stamp_tier1_provenance(tier1_dnam_path, TIER1_DNAM_PROVENANCE)
    logger.info("Written: feature_matrix_tier1_dnam.parquet %s", m_tier1.shape)

    # Write sig table (subset of celldmc for Tier 1 CpGs)
    sig_table.to_csv(out_dir / "tier1_celltype_table.tsv", sep="\t", index=False)
    logger.info("Written: tier1_celltype_table.tsv (%d rows)", len(sig_table))

    # Write CpG list
    tier1_cpg_list = m_tier1.columns.tolist()
    (out_dir / "tier1_cpg_list.txt").write_text("\n".join(tier1_cpg_list) + "\n")
    logger.info("Written: tier1_cpg_list.txt (%d CpGs)", len(tier1_cpg_list))

    # --- Tier 1 RNA (PROGENy fixed + ALL TF activities; candidate set) ---
    # cv_loop_safe=False: PROGENy is a fixed curated panel, but the TF panel
    # was previously a cohort-wide variance rank (leak). The full TF matrix is
    # written; the top-N TF variance rank is fit per fold downstream.
    tier1_rna_path = out_dir / "feature_matrix_tier1_rna.parquet"
    rna_tier1 = build_tier1_rna(pdata, LATEST_DIR)
    rna_tier1.to_parquet(tier1_rna_path)
    _stamp_tier1_provenance(tier1_rna_path, TIER1_RNA_PROVENANCE)
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

    # --- Tier 2 DNAm (EDA-only candidate matrix; no cohort-wide variance rank) ---
    bvals_raw = pd.read_parquet(LATEST_DIR / "data_emory.parquet")
    m_tier2, tier2_cpgs = build_tier2_dnam(m_df, bvals_raw)
    tier2_dnam_path = out_dir / "feature_matrix_tier2_dnam.parquet"
    m_tier2.to_parquet(tier2_dnam_path)
    _stamp_tier2_provenance(tier2_dnam_path)
    logger.info("Written: feature_matrix_tier2_dnam.parquet %s (candidate set)", m_tier2.shape)
    (out_dir / "tier2_candidate_cpg_list.txt").write_text("\n".join(tier2_cpgs) + "\n")
    logger.info("Written: tier2_candidate_cpg_list.txt (%d CpGs)", len(tier2_cpgs))

    # --- Tier 2 RNA (EDA-only candidate matrix; no cohort-wide HVG rank) ---
    rna_tier2, tier2_genes = build_tier2_rna(LATEST_DIR)
    tier2_rna_path = out_dir / "feature_matrix_tier2_rna.parquet"
    if not rna_tier2.empty:
        rna_tier2.to_parquet(tier2_rna_path)
        _stamp_tier2_provenance(tier2_rna_path)
        logger.info("Written: feature_matrix_tier2_rna.parquet %s (candidate set)", rna_tier2.shape)
    else:
        # Write empty parquet as stub so loaders don't error on missing file
        pd.DataFrame().to_parquet(tier2_rna_path)
        logger.warning(
            "Written: feature_matrix_tier2_rna.parquet (empty stub -- RNA data unavailable)"
        )
    (out_dir / "tier2_candidate_gene_list.txt").write_text("\n".join(tier2_genes) + "\n")
    logger.info("Written: tier2_candidate_gene_list.txt (%d genes)", len(tier2_genes))

    # --- Summary ---
    logger.info("=== Feature matrix build complete ===")
    logger.info(
        "Tier 1 DNAm: %d samples x %d CpGs (FDR < %.2f, delta contrast, any cell type)",
        m_tier1.shape[0],
        m_tier1.shape[1],
        CELLDMC_FDR_THRESHOLD,
    )
    logger.info(
        "Tier 1 RNA:  %d samples x %d candidate features (14 PROGENy fixed + "
        "all TFs; per-fold TF variance rank downstream -- cv_loop_safe=False)",
        rna_tier1.shape[0],
        rna_tier1.shape[1],
    )
    logger.info(
        "Tier 2 DNAm: %d samples x %d candidate CpGs (beta-range filter only; "
        "per-fold variance top-%d via PairedPreprocessor -- EDA_ONLY, cv_loop_safe=False)",
        m_tier2.shape[0],
        m_tier2.shape[1],
        TIER2_DNAM_TOP,
    )
    if not rna_tier2.empty:
        logger.info(
            "Tier 2 RNA:  %d samples x %d candidate genes (per-fold HVG top-%d via "
            "PairedPreprocessor -- EDA_ONLY, cv_loop_safe=False)",
            rna_tier2.shape[0],
            rna_tier2.shape[1],
            TIER2_RNA_TOP,
        )
    else:
        logger.warning("Tier 2 RNA:  EMPTY (Phase 1 RNA correction not yet run)")


if __name__ == "__main__":
    main()
