# Gate 0-T re-run: cell-type-corrected Δ matrices

**Date:** _to be filled when the script executes against real Emory data_
**Branch:** `phase-0/gate-t-rerun-cellDMC-corrected`
**Author:** Dr. Helen Zhao (biostatistics)
**Reviewer:** Kai (primary, CellDMC plumbing); Tobias (secondary, permutation invariance)
**Status:** template (script staged, awaiting real-data run)

## Rationale

Original Gate 0-T (raw paired-Δ PCA) returned MARGINAL on 2026-05-17:
PERMANOVA p=0.111, max Cohen's d=0.267. Hypothesis [INFERENCE, B]: PC1
(15.7% variance) is dominated by within-subject changes in blood cell
composition between PRE and POST, masking the treatment-response signal.
Re-running the gate on per-feature cell-type-corrected Δ matrices (CellDMC
residualisation on Δ-cell-fractions) should unmask the R vs NR signal if the
hypothesis holds.

See ``04-projects/dnamrnaseq/2026-05-17-phase-0-results.md`` and
ANALYSIS_PLAN.md Step 0-T MARGINAL recovery path.

## Inputs

- ``analysis/latest/cell_props_emory.csv`` (Phase 1 step 1.1 output)
- ``analysis/latest/pdata_emory_with_epidish.csv`` (Phase 1 step 1.1 output)
- Emory bVals + RNA-seq via ``dnamrnaseq2026.data.loaders``

## Method

1. Load EpiDISH cell fractions (step 1.1) and align to AMC subject IDs.
2. Compute Δ-cell-fractions = POST - PRE per paired subject.
3. Compute M-values from bVals, then per-CpG paired-Δ = POST - PRE.
4. Residualise each CpG's Δ-vector on Δ-cell-fractions (OLS, intercept + 6
   cell-type columns). Same step for per-gene log-CPM Δ-vectors.
5. Variance-filter: top 5000 CpGs + top 2000 genes by post-correction
   variance.
6. Build joint scaled (z-score per column) Δ matrix.
7. PCA (n_components=5).
8. PERMANOVA on PC scores (B=2000, seed=42), per-PC t-tests, per-PC
   Cohen's d, Hotelling's T-squared if pingouin available.
9. Apply canonical Gate 0-T thresholds: PASS p<0.05 AND max d>=0.30;
   MARGINAL 0.05<=p<0.15 OR p<0.05 with max d<0.30; FAIL p>=0.15.

## Verdict

_(to be filled by the script)_

## Metrics

| Metric | Value | Threshold |
|---|---|---|
| PERMANOVA p | _filled at run time_ | < 0.05 PASS |
| PERMANOVA F | _filled at run time_ | |
| Max Cohen's d (across PCs) | _filled at run time_ | >= 0.30 PASS |
| Hotelling T^2 p | _filled at run time_ | |
| n paired subjects (with R/NR + cell-props) | _filled at run time_ | |
| n features (CpGs + genes) | _filled at run time_ | |

## Per-PC breakdown

| PC | Explained variance | Cohen's d | t-test p |
|---|---|---|---|
| PC1 | _filled_ | _filled_ | _filled_ |
| PC2 | _filled_ | _filled_ | _filled_ |
| PC3 | _filled_ | _filled_ | _filled_ |
| PC4 | _filled_ | _filled_ | _filled_ |
| PC5 | _filled_ | _filled_ | _filled_ |

## Comparison to raw-Δ Gate 0-T

| Run | PERMANOVA p | Max Cohen's d | Verdict |
|---|---|---|---|
| Raw-Δ (2026-05-17) | 0.111 | 0.267 | MARGINAL |
| CellDMC-corrected Δ (this run) | _filled_ | _filled_ | _filled_ |

## Dependency

This template will be overwritten by
``scripts/01_phase0_gate_T_rerun_cellDMC.py`` once Phase 1 step 1.1 outputs
(``cell_props_emory.csv`` + ``pdata_emory_with_epidish.csv``) are present at
``analysis/latest/`` on a real-data run. Lee approves execution.

## Concepts discussed

- [[celldmc]]
- [[paired-design]]
- [[trajectory-atlas]]
- [[gate-zero-t]]
