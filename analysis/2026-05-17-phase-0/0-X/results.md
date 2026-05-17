# Gate 0-X: Cross-disorder centroid projection (Emory vs GSE98793 TRD)

**Date:** 2026-05-17
**Verdict: BLOCKED**

## Method

Cross-disorder centroid projection comparing Emory NR/R delta-vectors against
GSE98793 (Affymetrix microarray, treatment-resistant depression cohort). Top-2000-gene
shared space, quantile normalisation across platforms, Euclidean centroid distances,
permutation test (n_perm=2000, seed=42).

Expected signal: d(NR_centroid, TRD_centroid) < d(R_centroid, TRD_centroid) with
permutation p < 0.05.

## Verdict rationale

BLOCKED: GSE98793 expression file is not available locally. `config.yaml`
`data.external.gse98793` is null.

Gate 0-X exits gracefully with a BLOCKED result rather than failing. Per
ANALYSIS_PLAN.md, Gate 0-X tests the hypothesis that Emory NR subjects are
distributionally closer to TRD subjects than Emory R subjects. This is a pre-Phase-3
validation, not a hard stop for Phase 1 or Phase 2.

## Required action

To unblock Gate 0-X:
1. Download GSE98793 from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793
   (series matrix or normalised expression TSV/CSV, genes as rows).
2. Set `data.external.gse98793` in `config.yaml` to the local file path.
3. Re-run: `python scripts/01_phase0_gate_X.py`

## Known issues and limitations

- Heuristic TRD subset: Phase 0 uses the first 50 samples as a TRD proxy. Phase 3
  requires the full phenotype metadata from the GSE98793 series matrix to define
  the actual TRD vs healthy subsets.
- Platform harmonisation: GSE98793 is Affymetrix; Emory is RNA-seq. Quantile
  normalisation + top-2000-gene intersection is an approximation; platform effects
  may dominate the centroid distances.
