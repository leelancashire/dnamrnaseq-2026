# Gate 0-T: Trajectory-structure visibility test

**Date:** 2026-05-17
**Verdict: MARGINAL**

## Method

PCA of paired delta-vectors (POST-PRE M-values / log-CPM) on Emory cohort.
Top 5000 CpGs (by within-subject delta variance) + top 2000 genes (same criterion).
Joint scaled (zero mean, unit variance per feature). Pseudo-F PERMANOVA with 2000
permutations (seed=42). Cohen's d for R vs NR centroid separation per PC.

## Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Paired subjects | 164 (R=103, NR=61) | -- | -- |
| Features | 7000 (5000 CpG + 2000 genes) | -- | -- |
| PERMANOVA F | 2.071 | -- | -- |
| PERMANOVA p | 0.1115 | < 0.05 (PASS) / < 0.15 (MARGINAL) | MARGINAL |
| Max Cohen's d | 0.267 (PC2) | >= 0.30 | FAIL |

Cohen's d per PC: PC1=0.242, PC2=0.267, PC3=0.239, PC4=0.117, PC5=0.115

PC1 explains 15.7% of variance (PC2: 2.3%, PC3: 1.8%).

Hotelling T^2: unavailable (pingouin import issue with jedi); per-PC t-test p-values:
PC1=0.139, PC2=0.102, PC3=0.143, PC4=0.473, PC5=0.479.

## Verdict rationale

PERMANOVA p=0.11 falls in the MARGINAL zone [0.05, 0.15]. Max Cohen's d=0.267 falls
just below the PASS threshold of 0.30. Both criteria are sub-threshold. Per the
ANALYSIS_PLAN.md: "MARGINAL: PERMANOVA p in [0.05, 0.15]. Proceed but flag in risk
register; reconsider after Phase 1 cell-type-corrected outputs land."

The weak separation is consistent with cell-type confounding dominating PC1 (15.7%
variance). Cell-type correction in Phase 1 (CellDMC) may unmask the treatment
response signal.

## Outputs

- `gate_0T_results.json` -- full statistics
- `gate_0T_loadings.csv` -- top 50 features per PC by loading
- `gate_0T_pca_arrows.png` / `.svg` -- arrow plot (gitignored)

## Known issues and limitations

- Hotelling T^2 unavailable due to pingouin/jedi conflict; per-PC t-tests substituted.
- Cohen's d computed on principal components, not raw feature space; PC1 variance
  dominated by cell-type proportions rather than treatment response.
- 2 subjects skipped for RNA-seq (BEST-325916, BEST-315639 -- BEST cohort irrelevant
  for 0-T; all 164 Emory paired subjects included).
