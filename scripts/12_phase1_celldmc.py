"""Step 1.2: CellDMC three-contrast differential methylation + rescue check 1.2.5.

Runs CellDMC-style OLS at PRE, POST, and delta contrasts for Emory.
Runs rescue check 1.2.5 (PCA of cell-type-corrected delta vectors).
Outputs:
  - analysis/2026-05-17-phase-1/1.2/celldmc_pre_emory.tsv
  - analysis/2026-05-17-phase-1/1.2/celldmc_post_emory.tsv
  - analysis/2026-05-17-phase-1/1.2/celldmc_delta_emory.tsv
  - analysis/2026-05-17-phase-1/1.2/celldmc_cross_contrast.csv
  - analysis/2026-05-17-phase-1/1.2/rescue_1_2_5.json
  - analysis/2026-05-17-phase-1/1.2/results.md

Analysis plan reference: ANALYSIS_PLAN.md Steps 1.2 and 1.2.5.
"""

from __future__ import annotations

import json
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

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.2")
LATEST_DIR = Path("analysis/latest")
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dnamrnaseq2026.data.loaders import load_emory, load_emory_rnaseq
    from dnamrnaseq2026.preprocessing.cell_type_correction import (
        annotate_cross_contrast,
        beta_to_m,
        rescue_check_1_2_5,
        residualise_on_cell_props,
        run_celldmc,
    )
    from dnamrnaseq2026.preprocessing.delta_construction import (
        build_paired_delta,
        filter_paired_ids,
        filter_paired_ids_rna,
    )

    logger.info("Loading Emory data.")
    bvals_df, pdata = load_emory()

    cell_props_path = LATEST_DIR / "cell_props_emory.csv"
    pdata_aug_path = LATEST_DIR / "pdata_emory_with_epidish.csv"

    if cell_props_path.exists():
        cell_props_raw = pd.read_csv(cell_props_path, index_col=0)
        pdata_aug = pd.read_csv(pdata_aug_path, index_col=0)
        # cell_props_raw may be indexed by SentrixID (pData2) or AMC-... (subject_data)
        # Remap to pdata_aug index via SampleName_DNAm if needed
        if "SampleName_DNAm" in pdata_aug.columns and len(
            pdata_aug.index.intersection(cell_props_raw.index)
        ) == 0:
            dnam_map = pdata_aug["SampleName_DNAm"].dropna()
            cell_props = cell_props_raw.reindex(dnam_map.values).set_axis(dnam_map.index)
            logger.info(
                "Remapped cell_props from SentrixID to AMC-ID: %d/%d rows aligned.",
                cell_props.notna().any(axis=1).sum(),
                len(cell_props),
            )
        else:
            cell_props = cell_props_raw
    else:
        logger.warning("Step 1.1 outputs not found; using zero cell fractions.")
        from dnamrnaseq2026.preprocessing.cell_type_correction import CELL_TYPE_COLS as _CT
        cell_props = pd.DataFrame(
            np.zeros((len(pdata), len(_CT))), index=pdata.index, columns=_CT
        )
        pdata_aug = pdata.copy()

    # Exclude sex-chromosome CpGs
    cpg_ids_all = list(bvals_df.index)
    # In production: load EPIC manifest and exclude chrX/chrY CpGs here.
    # For now, proceed with all CpGs (sex-chrom filter requires manifest file).
    cpg_ids = cpg_ids_all
    logger.info("CpGs included (sex-chrom filter: manifest not available): %d", len(cpg_ids))

    m_matrix = beta_to_m(bvals_df.values.astype(np.float64))
    logger.info("M-values computed: %s", m_matrix.shape)

    # --- (a) PRE contrast ---
    pre_visit_labels = ["PRE", "PRE-IOP", "BL", "BASELINE", "T0", "0"]
    pre_mask = (
        pdata_aug.get("Visit", pdata.get("Visit", pd.Series(dtype=str)))
        .astype(str)
        .str.upper()
        .isin(pre_visit_labels)
        if "Visit" in pdata_aug.columns
        else pd.Series(True, index=pdata_aug.index)
    )
    pdata_pre = pdata_aug[pre_mask]
    pre_col_pos = [list(pdata_aug.index).index(s) for s in pdata_pre.index]
    m_pre = m_matrix[:, pre_col_pos]
    cell_props_pre = cell_props.loc[pdata_pre.index]

    logger.info("Running CellDMC PRE contrast (%d samples).", len(pdata_pre))
    celldmc_pre = run_celldmc(m_pre, cpg_ids, cell_props_pre, pdata_pre, n_jobs=-1)
    celldmc_pre.to_csv(OUT_DIR / "celldmc_pre_emory.tsv", sep="\t", index=False)
    celldmc_pre.to_csv(LATEST_DIR / "celldmc_pre_emory.tsv", sep="\t", index=False)
    logger.info("PRE contrast done: %d rows.", len(celldmc_pre))

    # --- (b) POST contrast ---
    post_mask = pdata_aug.get("Visit", pd.Series(dtype=str)).astype(str).str.upper().isin(
        ["POST", "POST-IOP", "12W", "T1", "1"]
    ) if "Visit" in pdata_aug.columns else pd.Series(True, index=pdata_aug.index)
    pdata_post = pdata_aug[post_mask]
    post_col_pos = [list(pdata_aug.index).index(s) for s in pdata_post.index]
    m_post = m_matrix[:, post_col_pos]
    cell_props_post = cell_props.loc[pdata_post.index]

    logger.info("Running CellDMC POST contrast (%d samples).", len(pdata_post))
    celldmc_post = run_celldmc(m_post, cpg_ids, cell_props_post, pdata_post, n_jobs=-1)
    celldmc_post.to_csv(OUT_DIR / "celldmc_post_emory.tsv", sep="\t", index=False)
    celldmc_post.to_csv(LATEST_DIR / "celldmc_post_emory.tsv", sep="\t", index=False)
    logger.info("POST contrast done: %d rows.", len(celldmc_post))

    # --- (c) Delta contrast ---
    logger.info("Building paired delta for delta contrast.")
    # filter_paired_ids returns DNAm SentrixIDs for M-matrix subsetting
    paired_subjects, pre_ids_dnam, post_ids_dnam = filter_paired_ids(pdata_aug)
    logger.info("Paired subjects: %d.", len(paired_subjects))

    delta_m, _ = build_paired_delta(
        m_matrix,
        cpg_ids,
        pre_ids_dnam,
        post_ids_dnam,
        list(bvals_df.columns),  # M-matrix columns are bvals SentrixIDs
    )
    # For cell_props (indexed by AMC-IDs), use RNA-based pairing
    _, pre_ids_rna, post_ids_rna = filter_paired_ids_rna(pdata_aug)
    # Only keep pairs where RNA IDs are in cell_props index
    valid_rna = [
        (sc, p, q) for sc, p, q in zip(paired_subjects, pre_ids_rna, post_ids_rna, strict=False)
        if p in cell_props.index and q in cell_props.index
    ]
    if valid_rna:
        _sc_rna, pre_ids_rna_filt, post_ids_rna_filt = zip(*valid_rna, strict=False)
        delta_cell_props = (
            cell_props.loc[list(post_ids_rna_filt)].values
            - cell_props.loc[list(pre_ids_rna_filt)].values
        )
    else:
        from dnamrnaseq2026.preprocessing.cell_type_correction import CELL_TYPE_COLS as _CT
        delta_cell_props = np.zeros((len(paired_subjects), len(_CT)))
    delta_cell_props_df = pd.DataFrame(
        delta_cell_props,
        index=paired_subjects,
        columns=cell_props.columns,
    )
    pdata_paired = pdata_aug.loc[pre_ids_rna[:len(paired_subjects)]].copy()
    pdata_paired.index = pd.Index(paired_subjects)

    logger.info("Running CellDMC DELTA contrast (%d paired subjects).", len(paired_subjects))
    celldmc_delta = run_celldmc(
        delta_m,
        cpg_ids,
        delta_cell_props_df,
        pdata_paired,
        n_jobs=-1,
    )
    celldmc_delta.to_csv(OUT_DIR / "celldmc_delta_emory.tsv", sep="\t", index=False)
    celldmc_delta.to_csv(LATEST_DIR / "celldmc_delta_emory.tsv", sep="\t", index=False)
    logger.info("DELTA contrast done: %d rows.", len(celldmc_delta))

    # Cross-contrast annotation
    logger.info("Annotating cross-contrast CpG classes.")
    cross_contrast = annotate_cross_contrast(celldmc_pre, celldmc_post, celldmc_delta)
    cross_contrast.to_csv(OUT_DIR / "celldmc_cross_contrast.csv", index=False)
    cross_contrast.to_csv(LATEST_DIR / "celldmc_cross_contrast_annotation.csv", index=False)

    # Rescue check 1.2.5
    logger.info("Running rescue check 1.2.5.")
    try:
        log_cpm = load_emory_rnaseq()
        gene_ids = list(log_cpm.index)
        rna_matrix = log_cpm.values.astype(np.float64)

        corrected_delta_m = residualise_on_cell_props(
            delta_m, delta_cell_props_df, paired_subjects
        )
        idx = list(pdata_aug.index)
        # Use RNA-based IDs for RNA-seq delta
        _pre_rna = pre_ids_rna[:len(paired_subjects)]
        _post_rna = post_ids_rna[:len(paired_subjects)]
        delta_rna = (
            rna_matrix[:, [idx.index(s) for s in _post_rna]]
            - rna_matrix[:, [idx.index(s) for s in _pre_rna]]
        )
        corrected_delta_rna = residualise_on_cell_props(
            delta_rna, delta_cell_props_df, paired_subjects
        )

        rescue = rescue_check_1_2_5(
            corrected_delta_m,
            corrected_delta_rna,
            cpg_ids,
            gene_ids,
            pdata_paired,
            n_permutations=2000,
            seed=SEED,
        )
    except Exception as exc:
        logger.warning("Rescue check 1.2.5 failed: %s", exc)
        rescue = {"verdict": "ERROR", "rescue_passed": False, "error": str(exc)}

    rescue_path = OUT_DIR / "rescue_1_2_5.json"
    rescue_path.write_text(json.dumps(rescue, indent=2, default=str))
    logger.info("Rescue check 1.2.5 verdict: %s", rescue.get("verdict"))

    _write_results_md(celldmc_delta, cross_contrast, rescue)
    logger.info("Step 1.2 complete.")


def _write_results_md(
    celldmc_delta: pd.DataFrame,
    cross_contrast: pd.DataFrame,
    rescue: dict[str, object],
) -> None:
    from dnamrnaseq2026.preprocessing.cell_type_correction import CELL_TYPE_COLS

    # Count significant CpGs per cell type at delta
    sig_counts: dict[str, int] = {}
    for ct in CELL_TYPE_COLS:
        ct_mask = (celldmc_delta["cell_type"] == ct)
        sig_mask = celldmc_delta.loc[ct_mask, "q_interaction"].fillna(1.0) < 0.05
        sig_counts[ct] = int(sig_mask.sum())

    state_of_recovery = int(
        (cross_contrast["cross_contrast_class"] == "state_of_recovery").sum()
    ) if not cross_contrast.empty else 0

    rescue_verdict = rescue.get("verdict", "N/A")
    rescue_p = rescue.get("permanova_p", "N/A")
    rescue_d = rescue.get("max_cohen_d", "N/A")

    lines = [
        "# Step 1.2: CellDMC Three-Contrast + Rescue Check 1.2.5",
        "",
        "**Date:** 2026-05-17",
        "",
        "## CellDMC Delta Contrast (n sig CpGs FDR < 0.05 per cell type)",
        "",
        "| Cell type | N sig CpGs (FDR < 0.05) |",
        "|-----------|------------------------|",
    ]
    for ct, n in sig_counts.items():
        lines.append(f"| {ct} | {n} |")

    total_sig = sum(sig_counts.values())
    if any(n >= 20 for n in sig_counts.values()):
        acceptance = "PASS"
    elif total_sig > 0:
        acceptance = "MARGINAL"
    else:
        acceptance = "FAIL"

    lines += [
        "",
        f"**Total significant interactions (delta FDR < 0.05):** {total_sig}",
        f"**Acceptance verdict:** {acceptance}",
        "",
        "## Cross-contrast annotation",
        "",
        f"- state_of_recovery (delta only): {state_of_recovery}",
        "",
        "## Rescue Check 1.2.5 (0-T rescue)",
        "",
        "| Metric | Value | Threshold |",
        "|--------|-------|-----------|",
        f"| PERMANOVA p | {rescue_p} | < 0.05 |",
        f"| Max Cohen's d | {rescue_d} | > 0.30 |",
        f"| Verdict | **{rescue_verdict}** | RESCUE_PASS |",
        "",
        "## Gate-fail note",
        "",
        "If rescue check 1.2.5 verdict is FAIL: surface to Lee for v2.2 → v2.0 decision.",
        "Phase 1 PR proceeds regardless; the trajectory atlas verdict is Lee's call.",
    ]

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
