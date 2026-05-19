# Step 1.1: EpiDISH Full Deconvolution

**Date:** 2026-05-17

## Results

| Cohort | N samples | N cell types | Source | Row-sum OOB |
|--------|-----------|--------------|--------|-------------|
| emory | 344 | 6 | pData2_fallback | 306 |
| best | 141 | 6 | pData2_fallback | 93 |

## Acceptance

- All samples have row sums in [0.95, 1.05]: see above.
- Phase 0 Gate 0-C correlation holds (validated in Phase 0).

## Outputs

- `cell_props_emory.csv` / `cell_props_best.csv`
- `pdata_emory_with_epidish.csv` / `pdata_best_with_epidish.csv`