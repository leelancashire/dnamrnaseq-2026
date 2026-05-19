# Step 1.2: CellDMC Three-Contrast + Rescue Check 1.2.5

**Date:** 2026-05-17

## CellDMC Delta Contrast (n sig CpGs FDR < 0.05 per cell type)

| Cell type | N sig CpGs (FDR < 0.05) |
|-----------|------------------------|
| Bcell | 0 |
| CD4T | 0 |
| CD8T | 0 |
| Mono | 0 |
| Neu | 0 |
| NK | 0 |

**Total significant interactions (delta FDR < 0.05):** 0
**Acceptance verdict:** FAIL

## Cross-contrast annotation

- state_of_recovery (delta only): 0

## Rescue Check 1.2.5 (0-T rescue)

| Metric | Value | Threshold |
|--------|-------|-----------|
| PERMANOVA p | 0.107 | < 0.05 |
| Max Cohen's d | 0.271943994047415 | > 0.30 |
| Verdict | **MARGINAL** | RESCUE_PASS |

## Gate-fail note

If rescue check 1.2.5 verdict is FAIL: surface to Lee for v2.2 → v2.0 decision.
Phase 1 PR proceeds regardless; the trajectory atlas verdict is Lee's call.