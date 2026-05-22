# STUDY OUTCOME: dnamrnaseq-2026

**Status:** Concluded — negative result (2026-05-22)
**Study:** PTSD treatment-response trajectory atlas, Emory + BEST cohorts
**Repo:** `leelancashire/dnamrnaseq-2026` (private)

---

## Summary

This study tested whether joint DNAm + RNA-seq multi-omics data from paired
pre/post-treatment samples could support a modellable PTSD treatment-response
recovery trajectory (the "trajectory atlas" hypothesis) and, as a pre-registered
fallback, whether CellDMC interaction testing would identify a cell-type-resolved
monocyte-WNT treatment-response mechanism.

Both hypotheses are unsupported at the available sample size (n=164 paired Emory,
n=50 paired BEST).

---

## Key results

### Gate 0-T: trajectory visibility (go/no-go)

- PERMANOVA p=0.154 (pre vs post, Emory, PCA embedding) — soft FAIL
- Max inter-group distance d=0.267, no responder/non-responder separation
- Decision: proceed to Phase 1 cell-type correction; reassess after CellDMC
- Result file: `analysis/2026-05-17-phase-0/0-T/results.md`

### Phase 1: CellDMC interaction tests (Emory)

- EpiDISH cell-type deconvolution: PASS (abs(r(dMono, dN2LR))=0.354, p<0.0001)
- CellDMC delta FDR<0.05: 555 hits across 6 cell types (Emory)
- These 555 hits were taken forward to verification

### Phase 2: 3-arm trajectory-atlas embedding leaderboard

All three arms (Arm A: foundation model, Arm B: MOFA+, Arm C: contrastive
learning) trained on the leakage-clean Phase 1 feature matrices (n=164 paired).

| Metric | Arm A (FM) | Arm B (MOFA+) | Arm C (contrastive) |
|--------|-----------|---------------|---------------------|
| (i) Trajectory consistency | FAIL (med=0.014) | PASS (med=0.998) | FAIL (med=-0.005) |
| (ii) Trait-state disentanglement | FAIL (trait=31, state=0) | FAIL (trait=9, state=4, rho=0.33) | FAIL (trait=0, state=0) |
| (iii) LOSO reconstruction (Delta-PCL MAE) | 14.828 | 14.129 | 14.029 |
| (iv) Conformal coverage | not run | not run | not run |
| (v) Biological coherence | pending Phase 1 re-run | pending Phase 1 re-run | pending Phase 1 re-run |
| (vi) Archetype clusterability | FAIL (ARI=0.586) | FAIL (ARI=0.269) | FAIL (ARI=0.615) |

- No arm predicts Delta-PCL (MAE ~14 units; PCL range ~0-80; chance ~15)
- Neural arms (A, C) overfit at n=164: trajectory consistency FAIL
- ICC-continuum classification (decision record 2026-05-22): Arm B's 20 MOFA+
  factors classify as 9 trait / 4 state / 7 mixed (primary; JAK-STAT sensitivity
  7/3/10). Metric (ii) Part (a) — interpretable trait and state structure —
  passes; Part (b) — CCA cross-subspace independence — fails (rho_max=0.33,
  above the 0.30 threshold), so the metric fails overall: the trait and state
  subspaces are not cleanly separable. Arms A/C have no state factors at all
- Full leaderboard: `analysis/latest/phase2_leaderboard.csv`

### CellDMC verification (decisive negative result)

- Per-cell-type enrichment (Step 1.8): 555 Emory hits show no hypothesis-relevant
  GO/KEGG/Reactome enrichment in any cell type (no monocyte-WNT, inflammatory, or
  stress-pathway signal)
- BEST replication (Step 1.9): pi1=0.000 — the 555 hits are false positives; sign
  concordance and effect-size correlation both null; 0 hits replicated at FDR<0.05
  in the independent BEST cohort (n=50 paired)
- The 555 CellDMC delta FDR<0.05 hits are false positives driven by
  multiple-testing artefact at this sample size
- Verification results: `analysis/latest/celldmc_enrichment_per_celltype.tsv`,
  `analysis/latest/replication_555hits.tsv`,
  `analysis/latest/concordance_stats.json`

---

## Conclusion

At n=164 paired (Emory) and n=50 paired (BEST) no detectable treatment-response
signal exists in the DNAm + RNA-seq data. This is consistent with either no
effect or an effect too small to detect at this sample size. The study cannot
distinguish those two explanations.

**What survives:** A leakage-safe multi-omics pipeline (Phase 0 gates, Phase 1
EpiDISH + CellDMC, Phase 2 three-arm embedding leaderboard, Phase 3 projection
scaffold) with full verification infrastructure. The pipeline is reproducible from
this repo and can be applied to a future better-powered study.

---

## Reproduction steps

A reader who clones `main` can reproduce the null result as follows. All steps
assume the Emory and BEST data are accessible at the OneDrive path in
`config.yaml`.

### 0. Prerequisites

```bash
# Python env (conda)
conda env create -f environment.yml
conda activate dnamrnaseq2026
pip install -e .

# R-Bioconductor env (Phase 1 only)
conda env create -f envs/r-bioc.yml
```

Copy and edit `config.yaml.example` to `config.yaml` with your OneDrive mount
path (default: `/mnt/d/lee/onedrive/work/nicol healthtech/cvb/emory-dnam`).

### 1. Phase 0 gates

```bash
python scripts/00_load_data.py                  # verify data access
python scripts/01_phase0_gate_T.py              # 0-T: trajectory visibility (soft FAIL p=0.154)
python scripts/01_phase0_gate_C.py              # 0-C: EpiDISH validation (PASS)
python scripts/01_phase0_gate_S.py              # 0-S: source-domain shift (PASS)
```

Results land in `analysis/2026-05-17-phase-0/<gate>/results.md`.

### 2. Phase 1: EpiDISH + CellDMC

```bash
snakemake --use-conda --cores 8 step_1_1_epidish_emory
snakemake --use-conda --cores 8 step_1_2_celldmc_delta_emory
snakemake --use-conda --cores 8 step_1_1_epidish_best
```

Outputs: `analysis/latest/celldmc_delta_emory.tsv` (555 hits), cell proportion
files, feature matrices.

### 3. Phase 1 verification (CellDMC false-positive test)

```bash
python scripts/18_celldmc_enrichment_per_celltype.py   # per-cell-type enrichment
python scripts/19_best_celldmc_and_replication.py       # BEST replication (pi1=0.000)
```

Outputs: `analysis/latest/celldmc_enrichment_per_celltype.tsv`,
`analysis/latest/replication_555hits.tsv`,
`analysis/latest/concordance_stats.json`.

### 4. Phase 2: embedding training + leaderboard

```bash
python scripts/22_phase2_build_feature_matrices.py      # build leakage-clean inputs
python scripts/23_phase2_arm_b_run.py                   # Arm B: MOFA+ (CPU, ~2h)
python scripts/20_phase2_train_embedding.py --arm all   # Arms A/C (GPU required)
python scripts/21_phase2_leaderboard.py                 # assemble leaderboard
```

Output: `analysis/latest/phase2_leaderboard.csv` (all arms FAIL on Delta-PCL
prediction, MAE ~14).

### 5. Verify CI green

```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/ scripts/
python -m mypy src/
```

All checks pass on the `main` branch at the time this file was written
(2026-05-22).

---

## Analysis plan

Full step-by-step executable specification: `docs/ANALYSIS_PLAN.md`

---

## Phase 3 and beyond (infrastructure preserved)

PRs #18 and #19 landed the Phase 3 proximity-test scaffold and external-cohort
projection scaffold on `main` as unexecuted infrastructure. These were written
before the Phase 2 null was confirmed. They are preserved for a future
better-powered study and are CI-green as scaffolds. They are not part of the
study record; no Phase 3 analysis was run.
