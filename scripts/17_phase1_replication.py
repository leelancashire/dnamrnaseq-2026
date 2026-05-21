"""Step 1.7: BEST replication of Emory CellDMC delta-significant CpGs.

Outputs:
  - analysis/2026-05-17-phase-1/1.7/replication_overall.tsv
  - analysis/2026-05-17-phase-1/1.7/replication_within_modality.tsv
  - analysis/2026-05-17-phase-1/1.7/replication_modality_interaction.tsv
  - analysis/2026-05-17-phase-1/1.7/replication_summary.json
  - analysis/2026-05-17-phase-1/1.7/results.md

Analysis plan reference: ANALYSIS_PLAN.md Step 1.7.
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

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.7")
LATEST_DIR = Path("analysis/latest")
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dnamrnaseq2026.data.loaders import load_best
    from dnamrnaseq2026.preprocessing.replication import (
        build_best_paired_ids,
        compute_best_delta_m,
        run_replication,
        summarise_replication,
    )

    # Load BEST data
    logger.info("Loading BEST cohort.")
    try:
        bvals_best_df, pdata_best = load_best()
    except Exception as exc:
        logger.error("Failed to load BEST cohort: %s", exc)
        _write_results_md({}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return

    # Load Emory CellDMC delta results from Step 1.2
    celldmc_path = LATEST_DIR / "celldmc_delta_emory.tsv"
    if celldmc_path.exists():
        celldmc_delta = pd.read_csv(celldmc_path, sep="\t")
        # Normalise column names for v5 CellDMC output (run_celldmc.R naming):
        #   cpg -> cpg_id; fdr -> q_interaction; coef -> beta_interaction
        rename_map: dict[str, str] = {}
        if "cpg" in celldmc_delta.columns and "cpg_id" not in celldmc_delta.columns:
            rename_map["cpg"] = "cpg_id"
        if "fdr" in celldmc_delta.columns and "q_interaction" not in celldmc_delta.columns:
            rename_map["fdr"] = "q_interaction"
        if "coef" in celldmc_delta.columns and "beta_interaction" not in celldmc_delta.columns:
            rename_map["coef"] = "beta_interaction"
        if rename_map:
            celldmc_delta = celldmc_delta.rename(columns=rename_map)
        logger.info("Loaded CellDMC delta: %d rows.", len(celldmc_delta))
    else:
        logger.warning("celldmc_delta_emory.tsv not found; using empty DataFrame.")
        celldmc_delta = pd.DataFrame(
            columns=["cpg_id", "cell_type", "q_interaction", "beta_interaction"]
        )

    # Extract Emory significant CpGs (FDR < 0.05)
    if not celldmc_delta.empty and "q_interaction" in celldmc_delta.columns:
        sig_mask = celldmc_delta["q_interaction"].fillna(1.0) < 0.05
        sig_cpg_ids = list(celldmc_delta.loc[sig_mask, "cpg_id"].unique())
        # Keep as DataFrame with 'cpg' and 'beta_interaction' columns for
        # run_replication() which expects emory_betas.groupby("cpg")["beta_interaction"]
        emory_betas = (
            celldmc_delta.loc[sig_mask, ["cpg_id", "beta_interaction"]]
            .drop_duplicates("cpg_id")
            .rename(columns={"cpg_id": "cpg"})
            .reset_index(drop=True)
        )
    else:
        sig_cpg_ids = []
        emory_betas = pd.Series(dtype=float)

    logger.info("Emory significant CpGs for replication: %d.", len(sig_cpg_ids))

    if not sig_cpg_ids:
        logger.warning("No significant CpGs to replicate; writing empty results.")
        summary: dict[str, object] = {
            "n_emory_sig_cpgs": 0,
            "verdict": "SKIPPED",
            "reason": "No significant CpGs from Step 1.2",
        }
        _write_results_md(summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return

    # Build BEST paired IDs
    logger.info("Building BEST paired sample IDs.")
    paired_subjects, pre_ids_best, post_ids_best = build_best_paired_ids(
        pdata_best,
        subcode_col="Subcode",
        visit_col="Visit",
        pre_label="BL",
        post_label="12W",
    )
    logger.info("BEST pairs: %d.", len(paired_subjects))

    if not paired_subjects:
        logger.warning("No paired BEST samples found.")
        summary = {
            "n_emory_sig_cpgs": len(sig_cpg_ids),
            "verdict": "SKIPPED",
            "reason": "No paired BEST samples found",
        }
        _write_results_md(summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        return

    # Compute BEST delta-M for the Emory sig CpG list
    logger.info("Computing BEST delta-M for %d Emory sig CpGs.", len(sig_cpg_ids))
    cpg_ids_best = list(bvals_best_df.index)
    delta_m_best, kept_cpg_ids = compute_best_delta_m(
        bvals_best_df.values.astype(np.float64),
        cpg_ids_best,
        pre_ids_best,
        post_ids_best,
        pdata_best,
        sig_cpg_ids,
    )
    logger.info("BEST delta-M shape: %s, kept CpGs: %d.", delta_m_best.shape, len(kept_cpg_ids))

    # Build paired pdata for BEST
    pdata_best_paired = pdata_best.loc[pre_ids_best].copy()
    pdata_best_paired.index = pd.Index(paired_subjects)

    # Log BEST cell props availability (used by run_replication internally if needed)
    cell_props_best_path = LATEST_DIR / "cell_props_best.csv"
    if cell_props_best_path.exists():
        logger.info("BEST cell props available at %s.", cell_props_best_path)
    else:
        logger.warning("cell_props_best.csv not found; Step 1.7 runs without cell-type covariate.")

    # Run replication
    logger.info("Running BEST replication OLS.")
    overall, within_modality, modality_interaction = run_replication(
        delta_m_best,
        kept_cpg_ids,
        emory_betas,
        pdata_best_paired,
        response_col="Response",
        therapy_col="Therapy",
        n_jobs=-1,
    )

    overall.to_csv(OUT_DIR / "replication_overall.tsv", sep="\t", index=False)
    within_modality.to_csv(OUT_DIR / "replication_within_modality.tsv", sep="\t", index=False)
    modality_interaction.to_csv(
        OUT_DIR / "replication_modality_interaction.tsv", sep="\t", index=False
    )
    overall.to_csv(LATEST_DIR / "replication_overall.tsv", sep="\t", index=False)

    # Summarise
    summary = summarise_replication(overall, uncorrected_p_threshold=0.05)
    summary["n_emory_sig_cpgs"] = len(sig_cpg_ids)
    summary["n_best_kept_cpgs"] = len(kept_cpg_ids)
    summary["n_best_pairs"] = len(paired_subjects)

    summary_path = OUT_DIR / "replication_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Replication summary: %s", summary)

    _write_results_md(summary, overall, within_modality, modality_interaction)
    logger.info("Step 1.7 complete.")


def _write_results_md(
    summary: dict[str, object],
    overall: pd.DataFrame,
    within_modality: pd.DataFrame,
    modality_interaction: pd.DataFrame,
) -> None:
    verdict = summary.get("verdict", "MARGINAL")
    pct_replicated = summary.get("pct_replicated", "N/A")
    spearman_rho = summary.get("spearman_rho", "N/A")
    n_emory = summary.get("n_emory_sig_cpgs", "N/A")
    n_kept = summary.get("n_best_kept_cpgs", "N/A")
    n_pairs = summary.get("n_best_pairs", "N/A")

    lines = [
        "# Step 1.7: BEST Replication",
        "",
        "**Date:** 2026-05-17",
        "",
        "## Summary",
        "",
        f"- Emory significant CpGs: {n_emory}",
        f"- BEST kept CpGs (overlap): {n_kept}",
        f"- BEST paired samples: {n_pairs}",
        f"- Percent replicated (p < 0.05 uncorrected, same sign): {pct_replicated}",
        f"- Spearman rho (Emory vs BEST beta): {spearman_rho}",
        f"- **Verdict:** {verdict}",
        "",
        "## Acceptance criteria",
        "",
        "- PASS: >= 40% replicated at p < 0.05 (uncorrected) with same sign",
        "- MARGINAL: 20-39% replicated",
        "- FAIL: < 20% replicated",
        "",
        "## Modality interaction note",
        "",
    ]

    if not modality_interaction.empty and "q_modality_interaction" in modality_interaction.columns:
        n_interaction = int(
            (modality_interaction["q_modality_interaction"].fillna(1.0) < 0.05).sum()
        )
        lines.append(
            f"CpGs with significant modality-by-response interaction (FDR < 0.05): {n_interaction}"
        )
        lines.append("These CpGs show therapy-specific effects and are flagged for follow-up.")
    else:
        lines.append("No modality interaction data available.")

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
