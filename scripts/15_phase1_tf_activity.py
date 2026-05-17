"""Step 1.5: decoupleR TF activity (CollecTRI) with NFAT/WNT priority TFs.

Outputs:
  - analysis/2026-05-17-phase-1/1.5/tf_activity_emory.parquet
  - analysis/2026-05-17-phase-1/1.5/tf_delta_emory.parquet
  - analysis/2026-05-17-phase-1/1.5/tf_response_test.tsv
  - analysis/2026-05-17-phase-1/1.5/priority_tf_table.tsv
  - analysis/2026-05-17-phase-1/1.5/results.md

Analysis plan reference: ANALYSIS_PLAN.md Step 1.5.
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

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.5")
LATEST_DIR = Path("analysis/latest")
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dnamrnaseq2026.data.loaders import load_emory, load_emory_rnaseq
    from dnamrnaseq2026.preprocessing.delta_construction import filter_paired_ids_rna
    from dnamrnaseq2026.preprocessing.tf_activity import (
        build_priority_tf_table,
        check_acceptance_criteria,
        compute_delta_tf_activity,
        get_collectri_net,
        run_tf_ulm,
        test_tf_response_association,
    )

    logger.info("Loading Emory RNA-seq.")
    log_cpm_df = load_emory_rnaseq()
    gene_ids = list(log_cpm_df.index)
    log_cpm = log_cpm_df.values.astype(np.float64)
    sample_ids = list(log_cpm_df.columns)

    _, pdata = load_emory()

    pdata_aug_path = LATEST_DIR / "pdata_emory_with_epidish.csv"
    pdata_aug = pd.read_csv(pdata_aug_path, index_col=0) if pdata_aug_path.exists() else pdata

    # Align RNA-seq samples to pdata
    shared_samples = [s for s in pdata_aug.index if s in log_cpm_df.columns]
    if not shared_samples:
        shared_samples = sample_ids
    col_pos = [sample_ids.index(s) for s in shared_samples]
    lc_shared = log_cpm[:, col_pos]

    # --- CollecTRI TF activity ---
    logger.info("Loading CollecTRI network.")
    collectri_net = get_collectri_net(organism="human", split_complexes=True, min_targets=5)

    logger.info("Running CollecTRI ULM.")
    tf_activity = run_tf_ulm(lc_shared, gene_ids, shared_samples, collectri_net)
    tf_activity.to_parquet(OUT_DIR / "tf_activity_emory.parquet")
    tf_activity.to_parquet(LATEST_DIR / "tf_activity_emory.parquet")
    logger.info("TF activity: %s shape, %d TFs.", tf_activity.shape, tf_activity.shape[1])

    # --- Paired delta ---
    paired_subjects, pre_ids, post_ids = filter_paired_ids_rna(pdata_aug)
    pre_in_rna = [s for s in pre_ids if s in shared_samples]
    post_in_rna = [s for s in post_ids if s in shared_samples]
    n_pairs = min(len(pre_in_rna), len(post_in_rna))
    pre_in_rna = pre_in_rna[:n_pairs]
    post_in_rna = post_in_rna[:n_pairs]
    paired_rna = list(paired_subjects[:n_pairs])

    logger.info("Computing delta TF activity (%d pairs).", n_pairs)
    tf_delta = compute_delta_tf_activity(tf_activity, pre_in_rna, post_in_rna, paired_rna)
    tf_delta.to_parquet(OUT_DIR / "tf_delta_emory.parquet")
    tf_delta.to_parquet(LATEST_DIR / "tf_delta_emory.parquet")

    # --- Response association test ---
    pdata_pre_paired = pdata_aug.loc[pre_in_rna].copy()
    pdata_pre_paired.index = pd.Index(paired_rna)

    logger.info("Testing TF-response association.")
    tf_response_test = test_tf_response_association(
        tf_delta,
        pdata_pre_paired,
        response_col="Response",
    )
    tf_response_test.to_csv(OUT_DIR / "tf_response_test.tsv", sep="\t", index=False)
    tf_response_test.to_csv(LATEST_DIR / "tf_response_test.tsv", sep="\t", index=False)

    # --- Priority TF table (NFAT + WNT) ---
    priority_tf = build_priority_tf_table(tf_response_test, fdr_threshold=0.10)
    priority_tf.to_csv(OUT_DIR / "priority_tf_table.tsv", sep="\t", index=False)
    priority_tf.to_csv(LATEST_DIR / "priority_tf_table.tsv", sep="\t", index=False)
    logger.info("Priority TF table: %d rows.", len(priority_tf))

    # --- Acceptance check ---
    acceptance = check_acceptance_criteria(tf_activity, tf_response_test)
    logger.info("Step 1.5 acceptance: %s", acceptance)

    _write_results_md(tf_response_test, priority_tf, acceptance)
    logger.info("Step 1.5 complete.")


def _write_results_md(
    tf_response_test: pd.DataFrame,
    priority_tf: pd.DataFrame,
    acceptance: dict[str, object],
) -> None:
    def n_sig(df: pd.DataFrame, threshold: float = 0.10) -> int:
        if df.empty or "q_response" not in df.columns:
            return 0
        return int((df["q_response"].fillna(1.0) < threshold).sum())

    n_total_tfs = len(tf_response_test) if not tf_response_test.empty else 0
    n_sig_tfs = n_sig(tf_response_test)
    n_nfat = int((priority_tf["priority_family"] == "NFAT").sum()) if not priority_tf.empty else 0
    n_wnt = int((priority_tf["priority_family"] == "WNT").sum()) if not priority_tf.empty else 0
    verdict = acceptance.get("verdict", "MARGINAL")

    lines = [
        "# Step 1.5: TF Activity (CollecTRI)",
        "",
        "**Date:** 2026-05-17",
        "",
        "## Summary",
        "",
        f"- Total TFs tested: {n_total_tfs}",
        f"- Significant TFs (FDR < 0.10): {n_sig_tfs}",
        f"- Priority NFAT family: {n_nfat}",
        f"- Priority WNT family: {n_wnt}",
        f"- **Acceptance:** {verdict}",
        "",
        "## Priority TF Table (NFAT + WNT)",
        "",
    ]

    if not priority_tf.empty:
        lines.append("| TF | Family | Beta | q |")
        lines.append("|----|--------|------|---|")
        for _, row in priority_tf.sort_values("q_response").head(20).iterrows():
            tf_name = row.get("tf", "N/A")
            fam = row.get("priority_family", "")
            tf_beta = row.get("beta", float("nan"))
            tf_q = row.get("q_response", float("nan"))
            lines.append(f"| {tf_name} | {fam} | {tf_beta:.3f} | {tf_q:.3g} |")
    else:
        lines.append("No priority TFs identified.")

    lines += [
        "",
        "## Top TFs overall (FDR < 0.10)",
        "",
    ]

    if not tf_response_test.empty and "q_response" in tf_response_test.columns:
        top = (
            tf_response_test[tf_response_test["q_response"].fillna(1.0) < 0.10]
            .sort_values("q_response")
            .head(10)
        )
        if not top.empty:
            lines.append("| TF | Beta | q |")
            lines.append("|-----|------|---|")
            for _, row in top.iterrows():
                lines.append(
                    f"| {row.get('tf', 'N/A')} | {row.get('beta', float('nan')):.3f} "
                    f"| {row.get('q_response', float('nan')):.3g} |"
                )
        else:
            lines.append("No TFs at FDR < 0.10.")
    else:
        lines.append("No result data available.")

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
