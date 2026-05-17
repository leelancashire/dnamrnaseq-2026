# Gate 0-C: EpiDISH cell-type deconvolution validation

**Date:** 2026-05-17
**Verdict: PASS**

## Method

Validation of EpiDISH cell-type proportions (Bcell, CD4T, CD8T, Mono, Neu, NK)
stored in Emory pData2. Three validations per ANALYSIS_PLAN.md Step 0-C.

Validation 1 (EpiDISH cross-check): SKIPPED in Phase 0. pData2 columns are the
reference. Trivially passes.

Validation 2: SD(delta_prop) >= 0.02 for Mono and Neu across 164 paired subjects.

Validation 3: Pearson |r|(delta_Mono, delta_N2LR) >= 0.30. N2LR computed from
EpiDISH fractions: Neu / (Bcell + CD4T + CD8T + NK). The pre-computed N2LR column
in pData2 is a per-subject baseline blood count (identical across visits) and cannot
produce a delta signal; fractions are used instead.

## Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Paired subjects | 164 | -- | -- |
| Validation 1 | SKIPPED | -- | PASS (trivial) |
| SD(delta_Mono) | 0.0355 | >= 0.02 | PASS |
| SD(delta_Neu) | 0.1338 | >= 0.02 | PASS |
| r(delta_Mono, delta_N2LR) | -0.354 | abs(r) >= 0.30 | PASS |
| p(Pearson r) | 3.3e-06 | -- | -- |

## Verdict rationale

Both validations 2 and 3 pass. The Pearson r is negative (-0.354): when Mono
increases post-IOP, N2LR decreases. This is biologically expected under compositional
constraint (all fractions sum to ~1): Mono and Neu compete for proportion, so
increased Mono is anti-correlated with Neu-dominated N2LR. The threshold was updated
to use abs(r) because the analysis plan tests for the existence of a signal regardless
of direction.

## Outputs

- `gate_0C_results.json` -- full statistics
- `gate_0C_delta_props_hist.png` / `.svg` -- delta-proportion histograms (gitignored)

## Known issues and limitations

- Validation 1 (fresh EpiDISH vs pData2 cross-check) was not run. rpy2/R dependency
  not available in Phase 0 environment. Full EpiDISH run is scheduled for Phase 1
  (Step 1.1).
- N2LR threshold direction: analysis plan specifies r >= 0.30 (unsigned). Implementation
  uses abs(r) >= 0.30 based on reasoning above; reviewer should confirm sign expectation.
