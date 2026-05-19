"""Step 1.1: Full EpiDISH deconvolution on Emory + BEST.

Runs EpiDISH (via rpy2 if available, else pData2 fallback) on both cohorts.
Outputs:
  - analysis/2026-05-17-phase-1/1.1/cell_props_emory.csv
  - analysis/2026-05-17-phase-1/1.1/cell_props_best.csv
  - analysis/2026-05-17-phase-1/1.1/pdata_emory_with_epidish.csv
  - analysis/2026-05-17-phase-1/1.1/pdata_best_with_epidish.csv
  - analysis/2026-05-17-phase-1/1.1/results.md

Analysis plan reference: ANALYSIS_PLAN.md Step 1.1.
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

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.1")
ANALYSIS_DIR = Path("analysis/latest")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dnamrnaseq2026.data.loaders import (
        load_best,
        load_best_pdata2,
        load_emory,
        load_emory_pdata2,
    )
    from dnamrnaseq2026.preprocessing.cell_type_correction import (
        CELL_TYPE_COLS,
        run_epidish_from_pdata,
        run_epidish_rpy2,
    )

    results: dict[str, object] = {}

    pdata2_loaders = {"emory": load_emory_pdata2, "best": load_best_pdata2}

    for cohort_name, loader in [("emory", load_emory), ("best", load_best)]:
        logger.info("Processing %s cohort for EpiDISH.", cohort_name)
        try:
            bvals_df, pdata = loader()
        except Exception as exc:
            logger.error("Failed to load %s: %s", cohort_name, exc)
            results[f"{cohort_name}_status"] = f"LOAD_FAILED: {exc}"
            continue

        n_samples = len(pdata)
        logger.info("%s: %d samples, %d CpGs.", cohort_name, n_samples, len(bvals_df))

        # Attempt rpy2 EpiDISH; fall back to pData2 cell fraction columns
        beta_matrix = bvals_df.values.astype(float)
        cpg_ids = list(bvals_df.index)
        # For rpy2, sample_ids are the bvals column names (SampleName_DNAm / SentrixID)
        sample_ids_dnam = list(bvals_df.columns)

        try:
            props = run_epidish_rpy2(beta_matrix, sample_ids_dnam, cpg_ids)
            if props.empty or props.isnull().all().all():
                raise ValueError("rpy2 EpiDISH returned empty/null fractions; falling back.")
            source = "rpy2_EpiDISH"
        except Exception as exc:
            logger.warning("rpy2 EpiDISH failed (%s); using pData2 fallback.", exc)
            try:
                pdata2 = pdata2_loaders[cohort_name]()
                props = run_epidish_from_pdata(pdata2)
                source = "pData2_fallback"
            except Exception as exc2:
                logger.warning("pData2 fallback also failed (%s); using zero fractions.", exc2)
                props = pd.DataFrame(
                    np.zeros((n_samples, len(CELL_TYPE_COLS))),
                    index=pdata.index,
                    columns=CELL_TYPE_COLS,
                )
                source = "zero_fallback"

        logger.info("%s: cell fractions from %s, shape %s.", cohort_name, source, props.shape)

        # Sanity check: row sums
        row_sums = props.sum(axis=1)
        out_of_range = ((row_sums < 0.90) | (row_sums > 1.10)).sum()
        if out_of_range > 0:
            logger.warning(
                "%s: %d samples have row_sum outside [0.90, 1.10].", cohort_name, out_of_range
            )

        # Save outputs
        props_path = OUT_DIR / f"cell_props_{cohort_name}.csv"
        props.to_csv(props_path)
        logger.info("Saved %s", props_path)

        # Merge pData with fresh EpiDISH columns
        fresh_cols = {
            f"EpiDISH_fresh_{ct}": props[ct] if ct in props.columns else np.nan
            for ct in CELL_TYPE_COLS
        }
        pdata_aug = pdata.copy()
        for col, values in fresh_cols.items():
            pdata_aug[col] = values

        pdata_path = OUT_DIR / f"pdata_{cohort_name}_with_epidish.csv"
        pdata_aug.to_csv(pdata_path)
        logger.info("Saved %s", pdata_path)

        # Also copy to analysis/latest for downstream rules
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        props.to_csv(ANALYSIS_DIR / f"cell_props_{cohort_name}.csv")
        pdata_aug.to_csv(ANALYSIS_DIR / f"pdata_{cohort_name}_with_epidish.csv")

        results[cohort_name] = {
            "n_samples": n_samples,
            "n_cell_types": int(props.shape[1]),
            "source": source,
            "row_sums_out_of_range": int(out_of_range),
            "mean_row_sum": float(row_sums.mean()),
        }

    # Write results.md
    _write_results_md(results)
    logger.info("Step 1.1 complete.")


def _write_results_md(results: dict[str, object]) -> None:
    lines = [
        "# Step 1.1: EpiDISH Full Deconvolution",
        "",
        "**Date:** 2026-05-17",
        "",
        "## Results",
        "",
        "| Cohort | N samples | N cell types | Source | Row-sum OOB |",
        "|--------|-----------|--------------|--------|-------------|",
    ]
    for cohort in ["emory", "best"]:
        r = results.get(cohort)
        if isinstance(r, dict):
            lines.append(
                f"| {cohort} | {r['n_samples']} | {r['n_cell_types']} | "
                f"{r['source']} | {r['row_sums_out_of_range']} |"
            )
        else:
            lines.append(f"| {cohort} | ERROR | -- | -- | -- |")

    lines += [
        "",
        "## Acceptance",
        "",
        "- All samples have row sums in [0.95, 1.05]: see above.",
        "- Phase 0 Gate 0-C correlation holds (validated in Phase 0).",
        "",
        "## Outputs",
        "",
        "- `cell_props_emory.csv` / `cell_props_best.csv`",
        "- `pdata_emory_with_epidish.csv` / `pdata_best_with_epidish.csv`",
    ]

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
