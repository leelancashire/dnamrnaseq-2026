# ANALYSIS_PLAN.md

**Project:** dnamrnaseq-2026 (PTSD treatment-response trajectory atlas, Emory + BEST joint DNAm + RNA-seq)
**Version:** 1.0 (2026-05-17)
**Status:** Draft, pending review (Helen, Kai, Tobias)
**Strategic backstop:** v2.2 in companion vault at `04-projects/dnamrnaseq/2026-05-17-integrated-analysis-plan-v2.md` (path on Lee's machine: `/home/llanc/claude-code/04-projects/dnamrnaseq/2026-05-17-integrated-analysis-plan-v2.md`)
**Author:** Dr. Aria Patel

---

## 0. Project Context

This repo implements the trajectory-atlas headline thesis: **treatment-response trajectories in PTSD trace toward healthy-state biology and away from the treatment-resistant-depression-inflammatory state**. The trajectory atlas (a 2D projection of per-subject Δ-vectors in a joint DNAm + RNA-seq embedding, with external psychiatric and healthy reference clouds projected in) is the load-bearing artefact. Everything else (Phase 0 gates, Phase 1 cell-type-resolved biology, Phase 2 embedding-and-conformal infrastructure, Phase 3 archetype and translational layers, Phase 4 manuscripts and library) feeds it.

Strategic v2.2 (vault) tells *why* each step exists and how it differentiates from the prior mmVAE work in this cohort. This document, ANALYSIS_PLAN.md, tells *how* each step runs: inputs, methods at package + function granularity, outputs at file-path granularity, acceptance criteria, risks, and Snakefile rule names where applicable. A team member opening this doc at the start of Phase 0 / 1 / 2.A / 2.B / 3 / 4 should be able to pick a step, read its section, and start running without further deliberation. v2.2 is the strategic backstop; ANALYSIS_PLAN.md is the executable contract.

### Three deliverables (anchor the work)

1. **Mechanism paper** (Aria first-author), *Nature Medicine* primary target. The trajectory atlas with biological annotation of the recovery axis, archetype clustering, external terminus projection, mediation as mechanism for trajectory direction, LINCS reversal on the recovery axis.
2. **Methods paper** (Tobias + Helen co-first-authorship), *Nature Methods* primary target. Head-to-head comparison of three trajectory-tuned embedding architectures at small-N + Mondrian weighted-conformal prediction sets for per-subject trajectory direction.
3. **Clinical artefact**, `trajectory-atlas-ptsd` Python library released on PyPI + GitHub Release. Allows external groups to project their own cohorts into the recovery axis.

### Phase overview

| Phase | Weeks | Owner | Headline output |
|---|---|---|---|
| 0 | 1 | Aria (0-T, 0-X), Kai (0-C), Helen (0-S) | Four parallel go/no-go gate decisions |
| 1 | 2-6 | Kai | Cell-type-resolved Δ-mediator catalogue + cross-cohort replication |
| 2.A | 4-10 | Tobias | Embedding-architecture leaderboard, winner declared |
| 2.B | 4-10 | Helen | Conformal prediction sets for trajectory direction, coverage-validated |
| 3 | 6-12 | Aria | Trajectory atlas with external reference clouds (the headline figure) + target nomination |
| 4 | 10-18 | Aria / Tobias / Helen | Mechanism paper, methods paper, library release, OSF pre-registration |

### How to use this document

1. **Find your step** by phase. Within a phase, steps are numbered X.Y. Phase 0 gates are 0-T / 0-C / 0-S / 0-X.
2. **Read top to bottom** of the step. Every step has the same template: Phase, Owner, Estimated duration, Depends on, Strategic justification, Objective, Inputs, Method, Outputs, Acceptance criteria, Risks / gotchas, Notes for the runner.
3. **Check dependencies.** If your step depends on Step X.Y, confirm X.Y is at status `done` in the project tracker before starting. Phase 0 gates have no dependencies. Phase 1 depends on Phase 0 passing. Phase 2.A depends on Phase 1 cell-type-corrected outputs. Phase 2.B depends on Phase 2.A winner being declared. Phase 3 depends on Phase 2.A winner + Phase 1 outputs. Phase 4 depends on Phase 3 main outputs.
4. **Run.** All file paths are repo-relative unless absolute. Snakefile rule names are given where the step has one; invoke via `snakemake --use-conda --cores N <rule_name>`.
5. **Tick the acceptance criterion** before declaring the step done. If the acceptance criterion fails, do not proceed: escalate to Lee with the failure diagnostic in the step's "Risks / gotchas" section.
6. **Log progress** to the daily note (`01-daily/YYYY-MM-DD.md` in the vault) as a one-line bullet under `## Agent activity log` with a wikilink to the detail artefact.

### File-path conventions

- **Source code:** `src/dnamrnaseq2026/<module>/<file>.py` for importable library code.
- **Runnable entry-points:** `scripts/NN_<phase>_<step>.py` (numbered) for one-off runners, `scripts/snakemake/<rule>.py` for Snakemake script targets.
- **Snakemake rule modules:** `workflow/rules/<module>.smk`. Rule naming convention: `rule step_<phase>_<step>_<short_name>:` (e.g. `rule step_1_2_celldmc_delta_emory:`).
- **Analysis outputs:** `analysis/latest/<artefact>.<ext>` for the current run; `analysis/<YYYY-MM-DD-slug>/<artefact>.<ext>` for archived runs. Symlink `analysis/latest` to the most recent dated run after each Phase milestone.
- **Manuscript figures:** `manuscript/figures/fig<N>_<slug>.<ext>`, `manuscript/supplementary/<table_or_figure>.<ext>`. Code-generated only.
- **Tests:** `tests/test_<module>.py` for unit tests; `tests/fixtures/` for synthetic data fixtures.

### Owner key

- **Aria:** Dr. Aria Patel (research scientist, trajectory atlas + Phase 3 + Phase 0-T / 0-X + Mechanism paper)
- **Kai:** Kai Nakamura (data engineer, Phase 1 cell-type-resolved biology, Phase 0-C)
- **Tobias:** Tobias Lindqvist (ML engineer, Phase 2.A embedding architectures, Methods paper Track A)
- **Helen:** Dr. Helen Zhao (biostatistician, Phase 2.B conformal, Phase 0-S, Methods paper Track B)
- **Lee:** Lee Lancashire (principal, overall accountability, data access, manuscript senior author)

---

# Phase 0: Four parallel go/no-go gates (Week 1)

Four gates run in parallel across the team in Week 1. Each is a 1-2 day analysis with a sharp pass/fail criterion. If any gate fails, the corresponding piece of the v2.2 plan is removed or pivoted before Phase 1 starts. Phase 0 gates have no dependencies on each other and no prior steps.


## Step 0-T: Trajectory-structure visibility test (PCA of Δ)

**Phase:** 0
**Owner:** Aria
**Estimated duration:** 1-2 days
**Depends on:** none (Day-0 verification `scripts/00_load_data.py` must have run successfully)
**Strategic justification:** v2.2 Section 5.0, Gate 0-T (the load-bearing v2.1 gate)

### Objective

Determine whether the simplest possible representation of within-subject change, a PCA of paired (POST minus PRE) difference vectors across Emory paired subjects, shows visible separation between responder and non-responder trajectories. If no separation exists at PCA level, the trajectory framing fails and v2.1 collapses to v2.0 (mediation-as-headline).

### Inputs

- `analysis/latest/data_emory.parquet` (output of Snakemake rule `load_emory`); Emory bVals + pData2 harmonised, ~292k CpGs by 388 samples plus covariates.
- Emory RNA-seq log-CPM matrix (loaded via `dnamrnaseq2026.data.loaders.load_emory_rnaseq`, to be added if missing; Kai 2026-05-17 Day-0 note confirms RNA-seq present in OneDrive mount).
- Paired-subject ID list (paired subjects = those with both PRE-IOP and POST-IOP samples). Approximately 164 paired subjects in Emory.

### Method

Implementation lives at `src/dnamrnaseq2026/preprocessing/delta_construction.py` (new file) for the Δ-construction helper, and `scripts/01_phase0_gate_T.py` for the gate-runner entry-point.

1. Subset to paired subjects with both PRE-IOP and POST-IOP samples in Emory. Validate via pData2 `Visit` and `SampleName` columns (Kai Day-0 verification).
2. Construct Δ-feature matrix: for each paired subject `i`, compute `Δ_M_CpG = M_POST - M_PRE` (M-values, not betas, via `numpy.log2(beta/(1-beta))`) and `Δ_logCPM_gene = logCPM_POST - logCPM_PRE`.
3. Variance-filter: top 5,000 Δ-CpGs and top 2,000 Δ-genes by within-paired-subject variance. Concatenate into a 7,000-column Δ matrix indexed by paired subject (164 rows by 7,000 cols).
4. Centre and scale per-feature (`sklearn.preprocessing.StandardScaler`).
5. PCA via `sklearn.decomposition.PCA(n_components=5)`. Retain first 5 PCs.
6. Visualise PC1 vs PC2 with `matplotlib`: arrows from origin to (PC1_i, PC2_i) per subject, coloured by Response (R vs NR). Save as `analysis/latest/figures/gate_0T_pca_arrows.png`.
7. Statistical tests:
   - PERMANOVA on 5-PC scores by Response, subject-clustered nonparametric bootstrap, B=2000 (via `skbio.stats.distance.permanova` with subject as cluster). Seed: `config["run"]["seed"]`.
   - Hotelling's T² test on centroid difference in 5-PC space (`scipy.stats.ttest_ind` on each PC plus combined T² via `pingouin.multivariate_ttest`).
8. Report: PERMANOVA p, centroid Cohen's d per PC, T² p.

### Outputs

- `analysis/latest/gate_0T_results.json`; PERMANOVA p, Cohen's d per PC, Hotelling's T² p, n paired subjects, decision tag (PASS / MARGINAL / FAIL).
- `analysis/latest/figures/gate_0T_pca_arrows.png`; the PCA-of-Δ arrow plot, coloured by Response.
- `analysis/latest/gate_0T_loadings.csv`; top 50 features by absolute loading on PC1 and PC2 for inspection.
- `analysis/latest/logs/gate_0T.log`; full numerical log.

### Acceptance criteria

- **PASS:** PERMANOVA p < 0.05 AND centroid-separation Cohen's d ≥ 0.3 on at least one of the first 5 PCs. Proceed with full trajectory-atlas build (Phase 1, Phase 2, Phase 3 unblocked).
- **MARGINAL:** PERMANOVA p in [0.05, 0.15]. Proceed but flag in risk register; reconsider after Phase 1 cell-type-corrected outputs land.
- **FAIL:** PERMANOVA p ≥ 0.15. Trajectory framing collapses. Pivot to v2.0 mediation-as-headline. The trajectory atlas demotes to a supplementary figure. Escalate to Lee for re-scoping meeting.

### Risks / gotchas

- M-value vs beta-value choice matters: use M-values for the PCA. Betas are bounded [0,1] and have heteroscedastic variance.
- Paired-subject identification: pData2 `SampleName` and `Visit` columns are the only reliable join keys. Do not use array position. Confirm against Kai's Day-0 verification.
- Outlier subjects with single-CpG huge Δ values can dominate the variance filter: pre-filter Δ values to [-3, 3] in M-value space; flag anything beyond that for inspection but do not include in the PCA.

### Notes for the runner

The arrow plot is the visual you show Lee on Week 1 Friday. Even before the formal test passes, the arrow plot tells you whether there's a trajectory signal. If the responder arrows visibly point in a different direction from non-responder arrows, you have something. Make the plot first, run the stats second.

---

## Step 0-C: EpiDISH 7-cell deconvolution validation

**Phase:** 0
**Owner:** Kai
**Estimated duration:** 1-2 days
**Depends on:** none
**Strategic justification:** v2.2 Section 5.0, Gate 0-K (EpiDISH deconvolution quality + Δ-cell-fraction)

### Objective

Validate that EpiDISH run fresh on the architecture-subset bVals reproduces the cell-type fractions pre-computed in pData2 (which derive from the mmVAE pipeline), and that within-subject Δ-cell-fractions have enough variance to use as predictors in CellDMC(Δ).

### Inputs

- `analysis/latest/data_emory.parquet` (output of `load_emory`). 292,674 CpGs by 388 samples.
- pData2 cell-fraction columns: `Bcell`, `CD4T`, `CD8T`, `Mono`, `Neu`, `NK`, `Eos` (7-cell IDOL reference). Confirmed present per Kai 2026-05-17 Day-0 note.
- Paired-subject ID list.
- N2LR (neutrophil-to-lymphocyte ratio) proxy column from pData2 if present, else compute as `Neu / (Bcell + CD4T + CD8T + NK)`.

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/cell_type_deconv.py` (new file) wrapping the R EpiDISH package via `rpy2`. Snakefile rule: `rule epidish_emory:` (already stubbed in `workflow/rules/preprocessing.smk`; this step implements it).

1. Load bVals via `loaders.load_emory_bvals()`. Cast to numpy 2D array, CpGs x samples.
2. Via `rpy2`, call `EpiDISH::epidish(beta.m = emory_bvals, ref.m = centDHSbloodDMC.m, method = "RPC")` using the IDOL-optimised 7-cell reference (`centDHSbloodDMC.m` comes with the EpiDISH package). Returns the cell-fraction matrix.
3. Save fresh estimates as `analysis/latest/cell_props_emory_fresh.csv` (samples x 7 cell types).
4. Validation 1 (cross-check against pData2): per-cell-type Pearson correlation between fresh estimates and pData2 columns. Pass per cell type if r ≥ 0.85. Save as `analysis/latest/gate_0C_correlation.csv`.
5. Validation 2 (Δ-cell-fraction stability): for each paired subject, compute `Δprop_c = prop_POST_c - prop_PRE_c` per cell type c. Histogram per cell type via `matplotlib`. Pass if SD(Δprop) ≥ 0.02 for Mono and Neu (the two cells the mmVAE NLR finding implicates).
6. Validation 3 (N2LR cross-check): Pearson correlation between Δ_Mono and Δ_N2LR across paired subjects. Pass if r ≥ 0.30.
7. Repeat steps 1-3 for BEST cohort via `rule epidish_best:`.

### Outputs

- `analysis/latest/cell_props_emory_fresh.csv`; fresh EpiDISH cell fractions, 388 samples by 7 cell types.
- `analysis/latest/cell_props_best_fresh.csv`; fresh EpiDISH cell fractions, 141 samples by 7 cell types.
- `analysis/latest/gate_0C_correlation.csv`; per-cell-type Pearson r between fresh and pData2 estimates.
- `analysis/latest/figures/gate_0C_delta_props_hist.png`; Δ-cell-fraction histograms per cell type, Emory.
- `analysis/latest/gate_0C_results.json`; decision tag plus all three validation statistics.
- `analysis/latest/logs/gate_0C.log`.

### Acceptance criteria

- **PASS:** all three validations pass (cell-type r ≥ 0.85 for all 7 cells; SD(Δprop) ≥ 0.02 for Mono and Neu; Δ_Mono × Δ_N2LR r ≥ 0.30). Proceed with CellDMC(Δ) in Phase 1.
- **FAIL on Validation 1 (r < 0.85 for any cell type):** investigate reference panel mismatch. Possible cause: pData2 fractions used a different reference (Houseman 6-cell?) or a custom Emory reference. Escalate to Lee with cell-type breakdown; do not silently use fresh estimates.
- **FAIL on Validation 2 (SD(Δprop) too small):** Δ-cell-fraction is too stable to use as a CellDMC predictor. Pivot to PRE-only and POST-only CellDMC in Phase 1, drop the Δ-level CellDMC interaction.
- **FAIL on Validation 3 (Δ_Mono × Δ_N2LR weak):** investigate whether N2LR proxy is being computed correctly. This is a sanity check, not a hard stop.

### Risks / gotchas

- rpy2 + EpiDISH conda env: `environment.yml` pins `bioconductor-epidish>=2.16`. If `rpy2` complains about R version mismatch, recreate the env with `r-base=4.3` pin re-enforced.
- IDOL reference (`centDHSbloodDMC.m`) ships with EpiDISH; do not download a separate reference. If `centDHSbloodDMC.m` is missing after `BiocManager::install("EpiDISH")`, run `library(EpiDISH); data(centDHSbloodDMC.m)` to load.
- Cell-fraction estimates can be near zero or near one for some cells (e.g. CD8T sometimes near zero); per-cell Pearson is sensitive to range. Report Spearman alongside Pearson as a secondary metric for robustness.

### Notes for the runner

If r < 0.85 for one cell type but ≥ 0.95 for all others, the issue is almost certainly reference panel mismatch on that one cell. Try the `centEpiFibIC.m` reference as a cross-check for monocytes specifically (it has higher monocyte specificity in some panels). Document whatever you find in `analysis/latest/logs/gate_0C.log` for Lee.

---

## Step 0-S: Source-domain classifier (Emory vs BEST covariate shift)

**Phase:** 0
**Owner:** Helen
**Estimated duration:** 1-2 days
**Depends on:** none
**Strategic justification:** v2.2 Section 5.0, Gate 0-H (Covariate-shift severity in Δ-space)

### Objective

Quantify the covariate shift between Emory paired-Δ feature space and BEST paired-Δ feature space by training a source-domain classifier (Emory vs BEST) on the Δ-feature matrix. The AUC of this classifier is the covariate-shift severity metric. Determines whether weighted Mondrian conformal in Phase 2.B is feasible or whether Phase 2.B must restrict to Emory-only.

### Inputs

- Emory paired-Δ feature matrix (164 paired subjects by ~7,000 Δ-features) constructed in Step 0-T.
- BEST paired-Δ feature matrix (~48 paired subjects by ~7,000 Δ-features) constructed the same way from `analysis/latest/data_best.parquet` and BEST RNA-seq log-CPM matrix.
- Harmonisation: feature columns must match. Use intersection of top-variance Δ-CpGs and Δ-genes across the two cohorts. Document the intersection size; if intersection drops below 3,000 features, expand the variance filter on the union before intersecting.

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/covariate_shift.py` (new file). Entry-point: `scripts/01_phase0_gate_S.py`.

1. Load both Δ-feature matrices. Label rows: 0 for Emory, 1 for BEST. Concatenate into one matrix `X` (212 rows by intersection-of-features cols) with binary label `y`.
2. Stratified train/test split via `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=config["run"]["seed"])`.
3. Train logistic regression (`sklearn.linear_model.LogisticRegression` with L2 penalty, `C=1.0`) and random forest (`sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=seed)`) classifiers. Two models to triangulate; logistic regression catches linear shift, random forest catches non-linear shift.
4. Compute 5-fold cross-validated AUC for each model. Report mean and 95% CI (cluster bootstrap on paired subject, B=2000).
5. Feature importance: top 20 features driving the classifier (logistic coefficients ranked by absolute value; RF feature importances). Save for inspection; these are the features most shifted between cohorts and need the most aggressive weighting in Phase 2.B Mondrian conformal.
6. Importance-weight construction (if PASS): logistic regression `Pr(source=BEST | x) / Pr(source=Emory | x)` per Emory sample becomes the importance weight for that sample when calibrating against BEST. Truncate weights at 99th percentile to control variance.

### Outputs

- `analysis/latest/gate_0S_classifier.json`; mean AUC, 95% CI, per-fold AUC, n features used, n samples per cohort, decision tag.
- `analysis/latest/gate_0S_top_shifted_features.csv`; top 20 features ranked by classifier importance, with their per-cohort mean and SD.
- `analysis/latest/gate_0S_importance_weights.csv`; per-Emory-subject importance weights (only if PASS).
- `analysis/latest/figures/gate_0S_auc_roc.png`; ROC curve.
- `analysis/latest/logs/gate_0S.log`.

### Acceptance criteria

- **PASS (AUC < 0.75):** moderate covariate shift, weighted Mondrian conformal is feasible. Phase 2.B Step 2.B.3 uses standard importance weighting.
- **MARGINAL (AUC 0.75-0.85):** truncate importance weights at the 99th percentile to control variance. Phase 2.B documents this as a methodological caveat and reports coverage under both truncated and untruncated weights.
- **FAIL (AUC > 0.85):** covariate shift is too severe for reliable importance weighting. Pivot: Phase 2.B reports coverage on Emory-held-out only; BEST becomes a documented covariate-shift case study in the methods paper rather than a calibration cohort.

### Risks / gotchas

- Sample sizes are imbalanced (Emory 164, BEST 48). Use `class_weight="balanced"` in the logistic regression to avoid the classifier just predicting the majority class.
- The feature intersection between Emory and BEST can shrink dramatically if variance filters are applied per-cohort first. Apply variance filter on the *union* of paired-Δ matrices, then intersect.
- Beware of trivial features that distinguish cohorts (e.g. batch-effect CpGs near the array control probes). Inspect the top-20 shifted features for biological plausibility. If the top features are all control probes or known batch markers, the gate is detecting a technical artefact, not biological shift; reduce to a biology-only feature set and rerun.

### Notes for the runner

Helen's prior wearable validation work (TPV) used cluster-bootstrap for CIs as a default. Same pattern here. Subject is the cluster; B=2000 is the default.

---

## Step 0-X: Cross-disorder centroid projection

**Phase:** 0
**Owner:** Aria
**Estimated duration:** 1-2 days
**Depends on:** none (but uses Emory non-responder subset, so Day-0 verification must have completed)
**Strategic justification:** v2.2 Section 5.0 (cross-disorder context for Phase 3.3 headline figure), Section 5.3 Step 3.4

### Objective

Test whether Emory non-responder centroid is closer to a TRD-inflammatory centroid (from GSE98793 antidepressant-non-responder MDD samples) than to Emory responder centroid, in a simple Δ-feature or baseline-feature space. If yes, the cross-disorder geometry needed for the Phase 3.3 headline figure exists. If no, the headline frame ("PTSD non-response trajectories trace toward TRD-inflammatory state") loses its quantitative anchor and the mechanism paper reframes around responder-vs-non-responder trajectory bundles alone.

### Inputs

- Emory baseline RNA-seq log-CPM matrix (PRE-IOP samples only), paired-subject subset.
- pData2 Response column (R / NR / partial) to label Emory samples.
- GSE98793 expression matrix (whole-blood microarray, 192 MDD + 64 controls) downloaded via `GEOquery` or NCBI FTP. If not yet locally available, this step blocks on that download. Confirm `config.yaml` has `data.external.gse98793` set.
- TRD-inflammatory subset definition: GSE98793 MDD samples with `non-responder` antidepressant status as labelled in the GSE supplementary, plus high-inflammation marker (CRP or NLR if available; else top-quartile of an inflammation-gene-set GSVA score on the GSE expression matrix).

### Method

Implementation: `src/dnamrnaseq2026/external_projection/cross_disorder_centroid.py` (new file). Entry-point: `scripts/01_phase0_gate_X.py`.

1. Load Emory baseline RNA-seq log-CPM and GSE98793 expression matrix. Harmonise on Ensembl gene IDs (use `pyensembl` or `biomart-client` for ID mapping); restrict to the intersection (~14k-18k genes typical).
2. Quantile-normalise across cohorts: combine into one matrix, apply `sklearn.preprocessing.QuantileTransformer(output_distribution='normal')`, then split back. This is a crude harmonisation; document it as the load-bearing caveat for cross-cohort comparison.
3. Variance-filter to top 2,000 genes by combined-cohort variance.
4. Identify TRD-inflammatory subset in GSE98793 (n ≈ 30-50 samples expected).
5. Compute centroids in the 2,000-gene space:
   - Emory R centroid (responder PRE-IOP, mean across responders).
   - Emory NR centroid (non-responder PRE-IOP, mean across non-responders).
   - GSE98793 TRD-inflammatory centroid (mean across TRD-inflammatory).
   - GSE98793 controls centroid (mean across GSE control samples).
6. Distance tests (Euclidean in 2,000-gene space):
   - d(Emory NR, GSE TRD-inflammatory) vs d(Emory R, GSE TRD-inflammatory). Test via permutation (B=2000) on subject labels in Emory.
   - d(Emory R, GSE controls) vs d(Emory NR, GSE controls). Same permutation test.
7. Project all four point sets to PC1 vs PC2 via PCA fit on Emory + GSE union. Visualise as scatter with centroid markers.

### Outputs

- `analysis/latest/gate_0X_centroids.json`; distance values, permutation p-values, decision tag.
- `analysis/latest/figures/gate_0X_centroid_projection.png`; 2D scatter with cohort and response status overlaid plus centroid markers.
- `analysis/latest/gate_0X_genes_used.csv`; the 2,000-gene set used, with mean expression per cohort/condition.
- `analysis/latest/logs/gate_0X.log`.

### Acceptance criteria

- **PASS:** d(Emory NR, GSE TRD-inflammatory) < d(Emory R, GSE TRD-inflammatory) at permutation p < 0.05. The cross-disorder anchor for the headline figure exists.
- **MARGINAL:** correct direction but p in [0.05, 0.15]. Phase 3.3 headline figure can still be built; document the marginal result honestly.
- **FAIL:** wrong direction (Emory R closer to TRD-inflammatory than Emory NR), or non-significant in either direction. Headline figure reframes around responder/non-responder trajectory bundles only; cross-disorder anchor demoted.

### Risks / gotchas

- The GSE98793 cohort uses microarray (Affymetrix Human Genome U133 Plus 2.0), Emory uses RNA-seq. Quantile normalisation across platforms is crude. A more principled harmonisation (ComBat, COCONUT) might be needed before the Phase 3.3 headline figure; for the Phase 0 gate, quantile-norm is sufficient as a sanity check.
- TRD-inflammatory subset depends on labelling. The GSE98793 supplementary distinguishes "non-responder" from "controls" but not always "TRD-inflammatory". Use the high-inflammation criterion as a conservative subset definition.
- If GSE98793 expression matrix has not yet been downloaded, this gate is delayed until the download completes. Trigger the download in parallel with Day-0 verification so it does not gate.

### Notes for the runner

This is the gate that decides whether the trajectory atlas headline frame is quantitative ("trajectories trace toward TRD-inflammatory state") or qualitative ("responder and non-responder trajectories diverge"). A FAIL here doesn't kill the project; it changes the manuscript framing. Document the result clearly.

---

# Phase 1: Biological annotation infrastructure (Weeks 2-6, Kai-led)

Phase 1 produces the cell-type-resolved Δ-mediator catalogue and the per-axis biological annotations that anchor the recovery axis in the trajectory atlas. All Phase 1 steps depend on Phase 0 passing (specifically Gate 0-C for EpiDISH).

---

## Step 1.1: EpiDISH on both cohorts (full deconvolution)

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 1 day
**Depends on:** Step 0-C (validation gate passed)
**Strategic justification:** v2.2 Section 5.1, Step 1.1

### Objective

Produce full-cohort EpiDISH cell-fraction estimates for Emory (388 samples) and BEST (141 samples), suitable for downstream CellDMC and cell-type-corrected RNA-seq differential expression.

### Inputs

- `analysis/latest/data_emory.parquet`, `analysis/latest/data_best.parquet` (output of `load_emory`, `load_best`).
- IDOL-optimised 7-cell reference (`centDHSbloodDMC.m`, ships with `EpiDISH`).

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/cell_type_deconv.py` (extended from Step 0-C). Snakefile rules: `rule epidish_emory:`, `rule epidish_best:` (the existing stubs in `workflow/rules/preprocessing.smk`, now implemented).

1. Run EpiDISH RPC method on the full Emory bVals matrix (architecture-subset CpGs by all 388 samples).
2. Same for BEST (141 samples).
3. Sanity-check: row sums should be approximately 1 per sample (cell fractions). Allow tolerance of 0.05 due to noise.
4. Append fresh estimates to pData2 as additional columns prefixed `EpiDISH_fresh_<celltype>`. Write to a merged pData with both pData2 originals and fresh EpiDISH.

### Outputs

- `analysis/latest/cell_props_emory.csv`; Emory fresh EpiDISH, 388 samples by 7 cell types. Replaces the stub.
- `analysis/latest/cell_props_best.csv`; BEST fresh EpiDISH, 141 samples by 7 cell types.
- `analysis/latest/pdata_emory_with_epidish.csv`; merged pData with fresh EpiDISH columns appended.
- `analysis/latest/pdata_best_with_epidish.csv`; same for BEST.

### Acceptance criteria

- All 388 (Emory) + 141 (BEST) samples have cell-fraction estimates with row sums in [0.95, 1.05].
- Phase 0 Gate 0-C correlation against pData2 holds (per-cell-type Pearson r ≥ 0.85 on the cross-cohort fresh-vs-pData2 comparison).

### Risks / gotchas

- Some samples may have extreme cell-fraction outliers (e.g. Mono > 0.4 in a treatment-history-positive sample). Flag in the log but do not exclude.
- BEST cohort may use a slightly different array version (EPIC v2 vs EPIC v1). EpiDISH supports both via the same reference, but check the `pyreadr` loaded shape (292,973 CpGs for BEST vs 292,674 for Emory) to confirm CpG coverage overlaps the reference panel.

### Notes for the runner

This step finalises the EpiDISH stub rules already in `workflow/rules/preprocessing.smk`. Once it's done, all Phase 1 downstream rules can consume `cell_props_emory.csv` and `cell_props_best.csv` directly.

---

## Step 1.2: CellDMC at three contrast levels (PRE, POST, Δ)

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 5-7 days (6-18 hours runtime on 16-core CPU, plus QC + iteration)
**Depends on:** Step 1.1
**Strategic justification:** v2.2 Section 5.1, Step 1.2 (THE differentiating step from mmVAE)

### Objective

Run CellDMC at three contrast levels (PRE-IOP only, POST-IOP only, within-subject Δ) on Emory to identify CpGs where the Response × cell-fraction interaction is significant. The Δ-level interaction is the load-bearing test: it asks whether response-discriminating methylation arises within a specific cell type's expansion/contraction, or represents methylation change within a cell type independent of fraction change.

### Inputs

- `analysis/latest/data_emory.parquet` (M-values, transform from beta via `M = log2(beta / (1 - beta))` if not already M).
- `analysis/latest/cell_props_emory.csv` (Step 1.1).
- `analysis/latest/pdata_emory_with_epidish.csv`; pData2 with covariates (Age, sex, ancestry PC1-6, smokingScore, Response, Visit).
- Paired-subject ID list (for Δ contrast).

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/cell_type_correction.py` (new file) wrapping `EpiDISH::CellDMC` via `rpy2`. Snakefile rules: `rule step_1_2_celldmc_pre_emory:`, `rule step_1_2_celldmc_post_emory:`, `rule step_1_2_celldmc_delta_emory:` (renaming the existing stubs `celldmc_pre_emory`, `celldmc_post_emory`, `celldmc_delta_emory` to follow the step-numbered convention).

For each of the three contrasts:

**(a) PRE-IOP only (Response x cell-fraction interaction at baseline):**
Model: `M_baseline ~ Response * cell_frac + Age + sex + ancestry_pca_PC1:6 + smokingScore`. Use `EpiDISH::CellDMC(beta.m, frac.m, cov.mod, mc.cores=16)`. Output: per-cell-type interaction tables (CpG, cell type, beta, p, q). Save as `analysis/latest/celldmc_pre_emory.tsv`.

**(b) POST-IOP only (Response x cell-fraction interaction post-treatment):**
Same model, samples restricted to POST-IOP. Output: `analysis/latest/celldmc_post_emory.tsv`.

**(c) Δ-level CellDMC (the load-bearing test):**
Model: `Δ_M ~ Response * Δ_cellfrac + Δ_Age (or visit-spacing) + sex + ancestry_pca_PC1:6 + smokingScore_PRE`. This requires custom code: CellDMC was designed for cross-sectional, not paired-Δ. Implementation strategy:
1. Construct paired-Δ matrices for M-values, cell fractions, and covariates per paired subject (164 rows by ~290k CpGs for M; 164 rows by 7 cells for fractions).
2. For each CpG, fit `Δ_M ~ Response * Δ_cellfrac_c + covariates` for each cell type c, using `statsmodels.OLS`. Extract the Response × Δ_cellfrac interaction beta and p.
3. FDR control via Benjamini-Hochberg per cell type (`statsmodels.stats.multitest.multipletests`).
4. Parallelise via `joblib.Parallel(n_jobs=16)` over CpGs.

Save as `analysis/latest/celldmc_delta_emory.tsv`.

Cross-contrast annotation: features that appear in (c) only are state-of-recovery markers; features in (a) and (c) are baseline-discriminating AND change with treatment; features in (a) only are trait markers.

### Outputs

- `analysis/latest/celldmc_pre_emory.tsv`; per-cell-type interaction table at PRE. Columns: CpG, cell_type, beta, p, q.
- `analysis/latest/celldmc_post_emory.tsv`; same at POST.
- `analysis/latest/celldmc_delta_emory.tsv`; per-cell-type Δ-level interaction table. Columns: CpG, cell_type, beta_interaction, p_interaction, q_interaction.
- `analysis/latest/celldmc_cross_contrast_annotation.csv`; CpG by classification (state-of-recovery / baseline-and-recovery / trait-stable) by cell type.
- `analysis/latest/logs/celldmc_*.log`.

### Acceptance criteria

- At least one cell type has ≥ 20 significant CpGs at FDR < 0.05 in the Δ contrast.
- PRE and POST contrasts each yield interpretable per-cell-type tables (≥ 10 significant CpGs at FDR < 0.10 per major cell type as a sanity check).
- Cross-contrast annotation produces a non-trivial partition (state-of-recovery class has > 5 members in at least one cell type).
- **FAIL:** Δ contrast yields zero significant CpGs at FDR < 0.10 across all cell types after the standard model. Escalate: re-check Δ-cell-fraction stability (Gate 0-C Validation 2), consider that the within-subject Δ-cell-fraction signal is below the detection floor for CellDMC, drop Δ-level CellDMC and report PRE + POST only in the mechanism paper.

### Risks / gotchas

- CellDMC at Δ-level is not a published method; implement it ourselves and document carefully. The standard CellDMC is cross-sectional. The Δ-level extension is justified by the same logic (interaction between exposure and cell fraction reveals within-cell effect controlling for fraction shifts) but applied to within-subject differences. Pre-register the model on OSF before running (see Step 4.4).
- Parallelisation: `BiocParallel` from R or `joblib` from Python. The Python OLS implementation is easier to maintain in this Python-first repo; the R EpiDISH::CellDMC is the published reference for PRE and POST. Use both and cross-check on PRE contrast that they agree.
- Sex chromosome CpGs: exclude chrX and chrY before fitting to avoid sex-confounded effects on Δ. Document the CpG count after exclusion.

### Notes for the runner

This is the longest single computational step in Phase 1. Plan for it to take a full week of wall-clock time (compute + iteration). Reserve 16-core machine time. If running on Lee's box, queue overnight.

---

## Step 1.3: Cell-type-corrected RNA-seq differential expression at PRE, POST, Δ

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 3-4 days
**Depends on:** Step 1.1
**Strategic justification:** v2.2 Section 5.1, Step 1.3 (decoupleR + extended for cell-type correction)

### Objective

Identify genes differentially expressed between Response groups at PRE, POST, and Δ contrast levels, with cell-type-fraction interactions modelled (analogous to CellDMC but for RNA-seq). This feeds the Phase 1.4 pathway/TF inference and the Phase 3.1 recovery-axis annotation.

### Inputs

- Emory RNA-seq raw counts matrix (genes by samples), loader: `src/dnamrnaseq2026/data/loaders.py::load_emory_rnaseq` (to be added if missing).
- `analysis/latest/cell_props_emory.csv` (Step 1.1).
- `analysis/latest/pdata_emory_with_epidish.csv` (covariates).
- Paired-subject IDs.

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/rnaseq_differential.py` (new file). Snakefile rule: `rule step_1_3_rnaseq_de_emory:` (new rule in `workflow/rules/preprocessing.smk`).

Uses `limma-voom` via `rpy2` (preferred) or `pydeseq2` (Python-native fallback). Recommend `limma-voom` for compatibility with the CellDMC modelling conventions.

For each contrast:

**(a) PRE-IOP:**
1. Subset to PRE-IOP samples.
2. Filter low-count genes (`edgeR::filterByExpr`).
3. `voom` for variance-stabilising transformation.
4. Design matrix: `~Response * cell_frac_summary + Age + sex + ancestry_PC1:6 + smokingScore`. `cell_frac_summary` is a summary of cell composition (e.g. PC1 of cell fractions, or NLR proxy) since fully crossing all 7 cell types blows up the design matrix at n=200.
5. `lmFit` + `eBayes` + `topTable` for the Response coefficient and the Response × cell_frac interaction.

**(b) POST-IOP:** same on POST samples.

**(c) Δ contrast:**
1. Construct per-paired-subject Δ-logCPM matrix.
2. Fit per-gene `Δ_logCPM ~ Response * Δ_cell_frac_summary + covariates` via `statsmodels.OLS` parallelised over genes.
3. FDR via BH per coefficient.

### Outputs

- `analysis/latest/de_pre_emory.tsv`; gene-level DE at PRE. Columns: gene, log2FC, p, q, n.
- `analysis/latest/de_post_emory.tsv`; same at POST.
- `analysis/latest/de_delta_emory.tsv`; Δ-level DE. Columns: gene, beta_Response, beta_interaction, p_Response, p_interaction, q_Response, q_interaction.
- `analysis/latest/logs/rnaseq_de_*.log`.

### Acceptance criteria

- Δ contrast yields ≥ 50 genes at FDR < 0.10 for the Response coefficient.
- PRE and POST contrasts each yield interpretable DE tables (≥ 100 genes at FDR < 0.10 as a sanity check; this is a low bar; if it fails, something is wrong with the model).
- **FAIL:** Δ contrast yields zero genes at FDR < 0.10 with the standard model. Drop the Δ-RNA contrast from the recovery-axis annotation and use PRE-only DE.

### Risks / gotchas

- Cell-fraction summary choice (PC1 vs NLR vs sum-of-myeloid) matters. Run the model with each as a sensitivity analysis; report all three in the supplementary.
- Library-size effects: `voom` handles them via the design matrix offset, but check sample-wise library-size distribution; samples with library size below 5M reads should be flagged.
- Δ-cell-fraction summary requires careful pairing: subjects must have both PRE and POST RNA-seq, not only PRE. Confirm against `data_emory.parquet` paired-subject metadata.

### Notes for the runner

If `limma-voom` via `rpy2` is unreliable in this conda env, fall back to `pydeseq2` for PRE and POST. The Δ contrast is custom OLS in either path.

---

## Step 1.4: Pathway activity inference (decoupleR + PROGENy + GSVA)

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 2-3 days
**Depends on:** Step 1.3
**Strategic justification:** v2.2 Section 5.1, Step 1.3 + Phase 3.1 (recovery-axis pathway annotation)

### Objective

Infer pathway activity scores per sample using PROGENy (14 cancer-and-inflammation pathways) and GSVA on MetaBase / KEGG / Reactome gene sets, on cell-type-corrected RNA-seq matrices. Feeds Phase 3.1 recovery-axis annotation.

### Inputs

- Emory cell-type-corrected RNA-seq matrix (residualised on cell fractions): produced as a side output of Step 1.3 via `removeBatchEffect(voom_logCPM, covariates=cell_props)` or via `decoupleR::run_ulm` directly on raw logCPM with cell fractions as covariates.
- PROGENy regulon database (loaded via `decoupler-py`).
- MetaBase pathway gene sets (must match mmVAE's pathway universe for apples-to-apples comparison; loaded via `dnamrnaseq2026.data.config.load_metabase_pathways`).

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/pathway_activity.py` (new file). Snakefile rule: `rule step_1_4_pathway_activity:`.

1. Load decoupleR Python wrapper (`pip install decoupler`).
2. Load PROGENy regulons: `decoupler.get_progeny(organism='human', top=100)`.
3. Run ULM (univariate linear model) per sample: `decoupler.run_ulm(mat=emory_rnaseq_corrected, net=progeny)`. Output: sample-by-pathway activity matrix (388 samples by 14 PROGENy pathways).
4. Repeat for BEST cohort.
5. Run GSVA on MetaBase gene sets via `gseapy.gsva()` or `GSVA::gsva()` via rpy2. Output: sample by gene-set activity matrix (388 samples by ~3,000 MetaBase pathways).
6. Compute Δ-pathway-activity per paired subject for both PROGENy and GSVA outputs.
7. Test Response association at each contrast (PRE, POST, Δ) via OLS.

### Outputs

- `analysis/latest/pathway_progeny_emory.csv`; PROGENy activity matrix Emory.
- `analysis/latest/pathway_progeny_best.csv`; same BEST.
- `analysis/latest/pathway_gsva_emory.csv`; GSVA on MetaBase Emory.
- `analysis/latest/pathway_response_test.csv`; per-pathway Response association at PRE / POST / Δ with p, q.
- `analysis/latest/logs/pathway_activity.log`.

### Acceptance criteria

- All samples have non-null pathway activity scores for at least 13/14 PROGENy pathways.
- At least 3 PROGENy pathways show Response association at FDR < 0.10 in the Δ contrast.
- GSVA matrix matches mmVAE pathway universe (verify pathway names overlap by ≥ 90%).

### Risks / gotchas

- Pathway database versioning: pin PROGENy to a specific version in `environment.yml` for reproducibility.
- GSVA can be slow on large gene-set databases; if MetaBase has > 5,000 sets, restrict to mmVAE's exact subset list.

### Notes for the runner

PROGENy is the priority output. GSVA on MetaBase is for cross-comparison with mmVAE.

---

## Step 1.5: TF activity inference (CollecTRI regulons via decoupleR)

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 2 days
**Depends on:** Step 1.3
**Strategic justification:** v2.2 Section 5.1, Step 1.3

### Objective

Infer transcription factor activity per sample using CollecTRI (the curated TF-target regulon database) via decoupleR's ULM method. Flag NFAT family (NFATC1-5) and WNT pathway TFs (TCF7L2, LEF1) given the mmVAE supplementary priors.

### Inputs

- Cell-type-corrected RNA-seq matrix (same as Step 1.4).
- CollecTRI regulon network loaded via `decoupler.get_collectri(organism='human', split_complexes=False)`.

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/tf_activity.py` (new file). Snakefile rule: `rule step_1_5_tf_activity:`.

1. Load CollecTRI: `net = decoupler.get_collectri()`. Confirm version pin.
2. Run ULM: `decoupler.run_ulm(mat=rnaseq, net=net, source='source', target='target', weight='weight')`. Output: sample-by-TF activity matrix (388 samples by ~1,200 TFs).
3. Repeat for BEST.
4. Δ-TF-activity per paired subject.
5. Per-TF Response test: `Δ_TF_activity_k ~ Response + Age + sex + ancestry_PC1:6 + smokingScore_PRE`. OLS with BH FDR.
6. Flag NFAT family and WNT TFs in output table.

### Outputs

- `analysis/latest/tf_activity_emory.csv`; sample by TF activity.
- `analysis/latest/tf_activity_best.csv`; same BEST.
- `analysis/latest/tf_response_test.csv`; per-TF Response association at PRE / POST / Δ.
- `analysis/latest/tf_priority_flags.csv`; NFAT + WNT family TFs with their test results highlighted.
- `analysis/latest/logs/tf_activity.log`.

### Acceptance criteria

- At least 1 TF family member (NFAT or WNT) with FDR < 0.10 Response association at Δ contrast.
- TF activity matrices have non-null values for ≥ 1,000 TFs per sample.

### Risks / gotchas

- CollecTRI is large (~1,200 TFs, ~50k interactions). ULM scales well but may need ~30 min runtime per cohort.
- Some TF activities are unstable when the regulon has few targets in the gene panel. Drop TFs with < 5 targets after intersection with the RNA-seq panel.

### Notes for the runner

The NFAT / WNT priority flags are specifically because the mmVAE supplementary highlighted these as candidates. Don't bury them in a 1,200-TF table; surface them in `tf_priority_flags.csv` for Lee.

---

## Step 1.6: ENCODE / EpiMap regulatory enrichment on top DMP hits

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 2 days
**Depends on:** Step 1.2
**Strategic justification:** v2.2 Section 5.1 (Phase 1 differentiation: per-CpG CellDMC vs mmVAE bulk EpiMap enrichment), Phase 3.1 (recovery-axis annotation)

### Objective

Test whether top Δ-CellDMC CpGs are enriched in ENCODE / EpiMap regulatory annotations (TFBS, enhancer marks, open chromatin). This is a region-set enrichment analysis on the CellDMC outputs; distinct from the per-CpG within-cell-type effects, and complementary to mmVAE's bulk EpiMap enrichment in that ours is on cell-type-resolved CellDMC hits.

### Inputs

- `analysis/latest/celldmc_delta_emory.tsv` (Step 1.2 output).
- ENCODE TF ChIP-seq tracks (download via `genomic-features-api` or ENCODE REST; restrict to blood cell types: K562, GM12878, monocyte-derived).
- EpiMap chromatin state segmentation (downloaded as BED files per cell type, ~833 EpiMap epigenomes; restrict to blood subset).
- Reference genome assembly: hg38. Convert from hg19 if needed via `pyliftover`.

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/regulatory_enrichment.py` (new file). Snakefile rule: `rule step_1_6_regulatory_enrichment:`.

1. Convert significant CellDMC(Δ) CpGs at FDR < 0.05 to BED format (1 bp intervals per CpG).
2. For each cell type c in CellDMC output, extract the cell-type-specific significant CpG list.
3. For each ENCODE TFBS track t and each EpiMap state s:
   - Count overlapping CpGs between (CpG list for c) and (regulatory feature t / s).
   - Compute expected overlap by chance via genome-wide CpG-array background.
   - Enrichment: observed / expected, with hypergeometric p-value.
4. FDR via BH across all (cell-type, regulatory-feature) pairs.
5. Use `pybedtools` for interval operations; `LOLA` via rpy2 as a secondary cross-check on a subset.

### Outputs

- `analysis/latest/regulatory_enrichment_emory.tsv`; per (cell-type, regulatory-feature) enrichment p, q, observed, expected.
- `analysis/latest/regulatory_top_hits.csv`; top 30 enriched (cell-type, regulatory-feature) pairs.
- `analysis/latest/logs/regulatory_enrichment.log`.

### Acceptance criteria

- At least 5 (cell-type, regulatory-feature) pairs enriched at FDR < 0.10.
- Enrichment cell types align with expectation (e.g. Mono-specific CpGs enriched in monocyte EpiMap states).

### Risks / gotchas

- hg19 vs hg38 mismatch: confirm coordinate system upfront. Most EPIC array annotations ship in hg19; ENCODE / EpiMap data ship in hg38. Use `pyliftover` to convert array coordinates if needed; document direction.
- Background CpG set must match the array: use the 850k EPIC v1 background, not genome-wide CpG distribution.

### Notes for the runner

Run this after Step 1.2 stabilises. Re-running it is cheap once the BED conversion is in place.

---

## Step 1.7: BEST replication of Emory top hits

**Phase:** 1
**Owner:** Kai
**Estimated duration:** 2 days
**Depends on:** Step 1.2, Step 1.3
**Strategic justification:** v2.2 Section 5.1, Step 1.5 (BEST replication within Therapy_type strata)

### Objective

Apply the Emory CellDMC(Δ) significant CpG list to BEST paired subjects, stratified by Therapy_type (CPT, PE, None). Test direction of effect and within-modality interaction. This is the cross-cohort replication for the mechanism paper.

### Inputs

- `analysis/latest/celldmc_delta_emory.tsv` (significant CpG list at FDR < 0.05).
- `analysis/latest/data_best.parquet` (BEST bVals + pData2).
- `analysis/latest/cell_props_best.csv` (Step 1.1).
- BEST paired-subject IDs stratified by Therapy_type (CPT n ≈ 30, PE n ≈ 10, None n ≈ 8; final counts confirmed in Week 2).

### Method

Implementation: `src/dnamrnaseq2026/preprocessing/replication.py` (new file). Snakefile rule: `rule step_1_7_replication:`.

1. Extract Emory-significant CpG list (FDR < 0.05 in Δ CellDMC per cell type, union across cell types).
2. For each CpG, compute paired-Δ-M in BEST per paired subject.
3. **(a) Overall replication:** test direction of effect (sign of beta_Response in BEST vs Emory) per CpG. Replication criterion: ≥ 40% same direction at uncorrected p < 0.10. Bonferroni-correct across the tested CpG set (not genome-wide).
4. **(b) Within-CPT:** restrict to BEST CPT-stratum paired subjects; same test.
5. **(c) Within-PE:** restrict to BEST PE-stratum paired subjects; same test.
6. **(d) Modality interaction:** for each replicated CpG, test `Δ_M ~ Response * Therapy_type + covariates` in BEST. Significant Therapy_type interaction (FDR < 0.10) flags modality-specific mediators.
7. Cross-cohort scatter plot: Emory beta vs BEST beta per CpG, coloured by within-modality significance. Save as `analysis/latest/figures/celldmc_replication_scatter.png`.

### Outputs

- `analysis/latest/replication_overall.tsv`; per-CpG: Emory beta, BEST beta, same-direction-flag, BEST p, BEST q.
- `analysis/latest/replication_within_modality.tsv`; same per modality stratum.
- `analysis/latest/replication_modality_interaction.tsv`; per-CpG Therapy_type interaction test.
- `analysis/latest/figures/celldmc_replication_scatter.png`.
- `analysis/latest/logs/replication.log`.

### Acceptance criteria

- Overall replication: ≥ 40% of Emory-significant CpGs same direction in BEST at uncorrected p < 0.10 (Bonferroni passes at FDR < 0.10 across the tested set).
- At least one of CPT or PE strata replicates the overall pattern (i.e. ≥ 40% same direction within that modality).
- **FAIL on overall replication < 40%:** investigate cohort-level differences (CAPS5 vs PCL outcome, different demographic mix). Document honestly. Mechanism paper reports Emory results with BEST as a limited replication case.

### Risks / gotchas

- BEST modality strata are small (PE n ≈ 10 paired). Within-modality tests will be statistically underpowered; report effect sizes alongside p-values.
- CAPS5 vs PCL: outcome metrics differ between cohorts. The Response definition (R / NR / partial) is what aligns across cohorts; verify the Response coding matches in `pData_best_with_epidish.csv`.

### Notes for the runner

The cross-cohort scatter plot is the visual that goes into Figure 2 of the mechanism paper.

---

# Phase 2.A: Embedding architecture leaderboard (Weeks 4-10, Tobias-led)

Phase 2.A produces the methods-paper headline: head-to-head comparison of three trajectory-tuned embedding architectures at small N. The winner becomes the v2.2 trajectory-atlas embedding consumed by Phase 3. Phase 2.A depends on Phase 1 cell-type-corrected RNA-seq outputs (Step 1.3) and EpiDISH outputs (Step 1.1) being available.

---

## Step 2.A.1: Architecture 1: FM + linear projection

**Phase:** 2.A
**Owner:** Tobias
**Estimated duration:** 7-10 days
**Depends on:** Step 1.1, Step 1.3
**Strategic justification:** v2.2 Section 5.2.A, Architecture 1

### Objective

Train the FM-based arm of the embedding leaderboard: Geneformer (or scGPT) bulk RNA-seq embeddings paired with a methylation-side encoder, joined via a shallow MLP head trained for trajectory consistency (within-subject Δ-cosine similarity across 10 seeds).

### Inputs

- Emory cell-type-corrected RNA-seq matrix (Step 1.3 side output, residualised on cell fractions).
- Emory M-value matrix (EpiDISH-cell-fraction-adjusted via residualisation): produced by `src/dnamrnaseq2026/preprocessing/cell_type_correction.py::residualise_on_cell_props`.
- Geneformer pretrained weights (HuggingFace `ctheodoris/Geneformer`, ~5GB). Pin to a specific commit hash for reproducibility.
- Paired-subject metadata (Visit, Response).

### Method

Implementation: `src/dnamrnaseq2026/embedding/fm_arm.py` (new file). Snakefile rule: `rule step_2_A_1_fm_embedding:`.

1. **RNA encoder:** Load Geneformer via `transformers.AutoModel.from_pretrained('ctheodoris/Geneformer')`. Tokenise bulk RNA-seq via the Geneformer tokeniser (rank-based gene tokens). Extract per-sample 256-D embedding (CLS-like pooling per the Geneformer convention).
2. **DNAm encoder:** Variance-filter M-values to top 500 features per Phase 0 PCA loadings. Pass through a learned linear projection to 256-D: `torch.nn.Linear(500, 256)`. Initialise from `sklearn.decomposition.PCA(n_components=256)` weights as a warm start.
3. **Joint projection head:** Concatenate the 256-D RNA + 256-D DNAm embeddings to 512-D. Pass through a 2-layer MLP (`Linear(512, 256) → ReLU → Linear(256, 128)`) to produce the joint 128-D latent. Final 128-D is the per-sample embedding.
4. **Trajectory-consistency loss:** For each paired subject `i`, compute `z_i_PRE` and `z_i_POST`, then `Δz_i = z_i_POST - z_i_PRE`. Loss per minibatch: maximise cosine similarity of `Δz_i` to itself under data augmentation (dropout on input features as a regulariser) and minimise variance of `Δz_i` across the 10 training seeds (computed batch-wise).
5. **Training:** Adam optimiser, lr=1e-4, batch size 16 paired subjects (32 samples), 100 epochs. 10 seeds (`config["run"]["embedding_seeds"]`). Save per-seed checkpoints.
6. **Output embeddings:** For each subject, compute the per-sample 128-D embedding via the trained encoder (RNA + DNAm → joint head). Save as a tensor.

### Outputs

- `analysis/latest/embedding_fm.pt`; PyTorch tensor: 388 + 141 samples by 128-D joint latent. Replaces existing stub.
- `analysis/latest/embedding_fm_per_seed.pt`; per-seed embeddings (10 seeds × samples × 128-D) for trajectory-consistency analysis.
- `analysis/latest/embedding_fm_metadata.csv`; sample ID, cohort, Visit, Response.
- `analysis/latest/embedding_fm_training.log`; training curves.

### Acceptance criteria

- Training completes without NaN loss on all 10 seeds.
- Within-subject trajectory consistency: mean cosine similarity of Δz across 10 seeds ≥ 0.5 per paired subject (averaged across subjects).
- Embeddings differ meaningfully between PRE and POST (paired t-test on per-subject Δ-latent magnitude p < 0.05).

### Risks / gotchas

- Geneformer expects single-cell tokenisation. For bulk RNA-seq we pass it as a "pseudo-single-cell" with rank-encoded genes. Document this as a methodological caveat in the methods paper.
- GPU required (Lee's machine has RTX 5090). On CPU, Geneformer inference is impractical. Add `--gpu` flag check to the script.
- Geneformer pretraining bias: it was pretrained on >30M single cells. Bulk PBMC samples may project poorly. Sanity-check by clustering the resulting embeddings by cell-type-composition PC1; if it's just a cell-composition embedder, that's a real finding but contraindicates this architecture.

### Notes for the runner

Tobias has the GPU + PyTorch expertise here. Reserve 5090 time for training. Document the Geneformer commit hash in `analysis/latest/embedding_fm_metadata.csv` for reproducibility.

---

## Step 2.A.2: Architecture 2: MOFA+ with explicit trait-state factor decomposition

**Phase:** 2.A
**Owner:** Tobias
**Estimated duration:** 5-7 days
**Depends on:** Step 1.1, Step 1.3
**Strategic justification:** v2.2 Section 5.2.A, Architecture 2

### Objective

Train the MOFA+ matrix-factorisation arm with subject as a random effect at the factor level, structured so some factors are baseline-invariant (trait factors) and others are state-driving (drive within-subject Δ-PCL). Linear, interpretable, sparse by construction.

### Inputs

- Emory cell-type-corrected RNA-seq (voom log-CPM, residualised on cell fractions).
- Emory cell-type-corrected M-values.
- Paired-subject metadata (Visit, Response, PCL_total at PRE and POST).

### Method

Implementation: `src/dnamrnaseq2026/embedding/mofa_arm.py` (new file). Snakefile rule: `rule step_2_A_2_mofa_embedding:`.

1. Format data for MOFA+ via `mofapy2`: per-modality matrices (samples by features), with sample-IDs aligned via paired-subject structure.
2. Two views: DNAm and RNA. Each view contributes features.
3. Group structure: subject ID as a group; visits within subject grouped.
4. Configure MOFA+ to learn 20 factors. Train via `mofapy2.run.entry_point()`.
5. Post-hoc: classify each factor as trait, state, or hybrid based on within-subject Δ-variance / between-subject baseline-variance ratio. Trait: ratio < 0.10. State: ratio > 0.50.
6. The 20-factor embedding is the per-sample MOFA+ latent. Save.

### Outputs

- `analysis/latest/embedding_mofa.h5`; MOFA+ model and per-sample 20-factor embedding. Replaces stub.
- `analysis/latest/embedding_mofa_factor_classification.csv`; per-factor trait/state/hybrid label with the variance ratio.
- `analysis/latest/embedding_mofa_metadata.csv`; sample ID, cohort, Visit, Response.
- `analysis/latest/embedding_mofa_training.log`.

### Acceptance criteria

- MOFA+ training converges (ELBO stabilises in last 100 iterations).
- At least 3 trait factors and 3 state factors identified post-hoc.
- Variance explained by the top 20 factors ≥ 30% of total per modality.

### Risks / gotchas

- MOFA+ requires sample-IDs aligned exactly across views; double-check the sample-ID matching.
- Hyperparameter sensitivity: number of factors (20 is a starting point; if convergence is poor or factor classification is unstable, sweep over {15, 20, 25, 30}).
- MOFA+ can underperform when feature counts vastly exceed sample counts. The voom log-CPM matrix is ~18k genes × 388 samples; consider top-2000 variance-filtering per modality.

### Notes for the runner

Run this in parallel with Step 2.A.1. MOFA+ does not need GPU; it'll run on Lee's box CPU.

---

## Step 2.A.3: Architecture 3: Contrastive trajectory-aware embedding (triplet loss)

**Phase:** 2.A
**Owner:** Tobias
**Estimated duration:** 7-10 days
**Depends on:** Step 1.1, Step 1.3
**Strategic justification:** v2.2 Section 5.2.A, Architecture 3 (the strongest "not mmVAE-with-arrows" defence)

### Objective

Train a bespoke contrastive embedding from scratch with the within-subject trajectory consistency baked into the loss function: (subject_i, PRE) and (subject_i, POST) embed near each other relative to (subject_j, *) for j ≠ i, while preserving response-discriminative direction.

### Inputs

- Same as Step 2.A.1.

### Method

Implementation: `src/dnamrnaseq2026/embedding/contrastive_arm.py` (new file). Snakefile rule: `rule step_2_A_3_contrastive_embedding:`.

1. Encoder: 3-layer MLP per modality: `Linear(input_dim, 512) → ReLU → Linear(512, 256) → ReLU → Linear(256, 128)`. Two encoders (DNAm-side and RNA-side), then concatenate to 256 and pass through a final `Linear(256, 128)` joint head.
2. Triplet loss via `pytorch_metric_learning.losses.TripletMarginLoss(margin=0.5)`:
   - Anchor: `z_i_PRE`.
   - Positive: `z_i_POST` (same subject, different visit).
   - Negative: `z_j_*` (different subject, sampled randomly).
3. Within-subject pairs are constructed in each minibatch via a `TripletMarginMiner` from `pytorch_metric_learning`. Mine 5 negatives per anchor.
4. Optional: add an auxiliary InfoNCE term over (subject_i, *) similarity to enforce per-subject identity.
5. Training: Adam, lr=1e-4, batch size 32 subjects (64 samples), 200 epochs. 10 seeds.
6. Save per-seed embeddings as in Step 2.A.1.

### Outputs

- `analysis/latest/embedding_contrastive.pt`; joint 128-D contrastive embedding per sample. Replaces stub.
- `analysis/latest/embedding_contrastive_per_seed.pt`; 10-seed per-sample embeddings.
- `analysis/latest/embedding_contrastive_metadata.csv`.
- `analysis/latest/embedding_contrastive_training.log`.

### Acceptance criteria

- Training converges (triplet loss decreases over epochs and stabilises in last 50 epochs).
- Within-subject trajectory consistency: mean cosine similarity of Δz across 10 seeds ≥ 0.7 per paired subject (higher threshold than FM arm; contrastive should explicitly produce this).
- Negative-positive separation: mean d(anchor, positive) < mean d(anchor, negative) on validation fold.

### Risks / gotchas

- Triplet loss can collapse to a constant embedding if the margin is too small or the miner is too aggressive. Monitor for embedding collapse (variance of embeddings across samples decreasing each epoch).
- Subject IDs as the contrastive signal mean the encoder learns subject identity as much as biology. This is partially desired (within-subject coherence) but at extreme can overfit. Regularise via dropout (p=0.2 on inputs).
- Small N (164 paired): minibatch construction needs careful sampling to avoid the same subject appearing twice in one batch.

### Notes for the runner

The contrastive arm has the highest variance in expected performance (could win cleanly, could collapse). Plan for at least one re-training cycle after the first pass.

---

## Step 2.A.4: Head-to-head comparison and winner declaration

**Phase:** 2.A
**Owner:** Tobias
**Estimated duration:** 3-4 days
**Depends on:** Step 2.A.1, Step 2.A.2, Step 2.A.3
**Strategic justification:** v2.2 Section 5.2.A (scoring criteria, leaderboard, methods-paper headline)

### Objective

Score the three trained embeddings on six criteria (within-subject trajectory consistency, reference-cloud separability, biological coherence, Δ-symptom predictive validity, archetype clusterability, trait-state decomposition quality). Declare the winner. Publish the leaderboard. Pass the winning embedding to Phase 2.B and Phase 3.

### Inputs

- `analysis/latest/embedding_fm.pt`, `analysis/latest/embedding_mofa.h5`, `analysis/latest/embedding_contrastive.pt`.
- Per-seed embeddings for trajectory consistency.
- Phase 1 outputs for biological coherence enrichment.

### Method

Implementation: `src/dnamrnaseq2026/embedding/leaderboard.py` (new file). Snakefile rule: `rule step_2_A_4_leaderboard:`.

For each architecture A in {FM, MOFA, contrastive}:

1. **Within-subject trajectory consistency:** mean cosine similarity of Δz across 10 seeds per paired subject; aggregate across subjects.
2. **Reference-cloud separability:** project Phase 0 Gate 0-X external clouds (GSE98793, GTEx) into the embedding via the RNA-side encoder; PERMANOVA F statistic on Emory R + Emory NR + GSE-TRD-inflam + GTEx cloud labels.
3. **Biological coherence:** for each latent axis (PC1, PC2 of the embedding), top-loading features (200 by absolute loading); enrichment via Fisher exact on cell-type / pathway / TF / TFBS annotations from Phase 1. Count axes passing FDR < 0.10 on at least one annotation channel.
4. **Δ-symptom predictive validity:** regression `Δ_PCL ~ Δ_latent_direction`, 5-fold nested CV R² with cluster-bootstrap CI (B=2000).
5. **Archetype clusterability:** Gaussian mixture on Δz, k=3, 10 seeds; mean silhouette and adjusted Rand index across seeds.
6. **Trait-state decomposition quality (MOFA explicit, FM and contrastive post-hoc via factor classification):** fraction of factors with within-subject Δ-variance / between-subject baseline-variance ratio < 0.10 (trait-like) vs > 0.50 (state-like).

Aggregate into leaderboard table. Each criterion is reported with mean and 95% CI. Winner = highest mean rank across criteria (Borda count).

### Outputs

- `analysis/latest/embedding_leaderboard.csv`; three architectures by six criteria, with means and CIs.
- `analysis/latest/embedding_leaderboard.json`; full per-seed numerical results.
- `analysis/latest/embedding_winner.txt`; single-line file naming the winning architecture (consumed by downstream Phase 2.B and Phase 3 rules).
- `analysis/latest/figures/embedding_leaderboard_radar.png`; radar chart per architecture per criterion.
- `analysis/latest/logs/leaderboard.log`.

### Acceptance criteria

- Leaderboard has three architectures with six criteria each, all reported with CIs.
- Winner declared with clear margin (Borda count difference ≥ 2 between winner and runner-up) OR explicit tie noted (in which case the methods paper presents both as candidates and the simpler model wins by Occam).
- Winner's within-subject trajectory consistency ≥ 0.5 (the floor for downstream archetype clustering to be reliable).

### Risks / gotchas

- Reading the FM-arm verdict honestly: if FM scores worst across all criteria, that IS the methods-paper headline (Geneformer doesn't transfer to small-N bulk treatment-trajectory visualisation). Don't bury it.
- Borda count can mask important asymmetries: if one architecture wins on 3 criteria by tiny margins and loses on 3 by huge margins, that's not the winner. Show effect sizes alongside ranks.
- The winner being declared via Borda count requires committing to the criterion weights in advance. Pre-register the criteria + weighting on OSF before computing (see Step 4.4).

### Notes for the runner

This step writes `embedding_winner.txt`, which Phase 2.B and Phase 3 Snakefile rules read to determine which embedding to consume. Make sure the format is a single line with the architecture name (`fm`, `mofa`, or `contrastive`); downstream rules use string matching.

---

# Phase 2.B: Conformal trajectory direction prediction (Weeks 4-10, Helen-led)

Phase 2.B builds the conformal prediction infrastructure on the winning embedding from Phase 2.A. Phase 2.B steps depend on Step 2.A.4 producing `embedding_winner.txt`.

---

## Step 2.B.1: Conformalised quantile regression on Δ-symptom score

**Phase:** 2.B
**Owner:** Helen
**Estimated duration:** 5-7 days
**Depends on:** Step 2.A.4
**Strategic justification:** v2.2 Section 5.2.B (Phase 2.B headline: continuous Δ-symptom prediction with finite-sample-valid bounds)

### Objective

Train a conformalised quantile regression (CQR) predictor of Δ-PCL_total (Emory) and Δ-CAPS5 (BEST) on the winning embedding's Δ-latent. Coverage target: α = 0.10, so 90% prediction intervals. This is the methods-paper Track B headline.

### Inputs

- Winning embedding `embedding_<winner>.pt` (read from `embedding_winner.txt`).
- Per-subject Δ-latent: `Δz_i = z_i_POST - z_i_PRE`.
- Per-subject Δ-symptom: `Δ_PCL_i = PCL_POST - PCL_PRE` (Emory), `Δ_CAPS5_i` (BEST).
- Covariates: Age, sex, ancestry PC1-6, baseline PCL.

### Method

Implementation: `src/dnamrnaseq2026/conformal/cqr.py` (new file). Snakefile rule: `rule step_2_B_1_cqr:`.

1. Quantile regressor: `sklearn.ensemble.GradientBoostingRegressor(loss='quantile', alpha=0.05)` for lower and `alpha=0.95` for upper. Two models. Inputs: Δ-latent + covariates.
2. Conformal calibration via `mapie.regression.MapieQuantileRegressor(alpha=0.10)`.
3. Split-conformal: 70% train, 15% calibration, 15% test on Emory paired subjects, with subject as the unit of splitting (no leakage).
4. Repeated splits: 10 random seeds; aggregate coverage and interval width.
5. Report: empirical coverage on test, mean interval width, conditional coverage by Response category.

### Outputs

- `analysis/latest/cqr_predictor.pkl`; fitted MapieQuantileRegressor.
- `analysis/latest/cqr_coverage_emory.csv`; empirical coverage per fold, mean interval width.
- `analysis/latest/cqr_predictions_emory.csv`; per-subject lower bound, upper bound, point prediction, observed Δ-PCL.
- `analysis/latest/figures/cqr_calibration_plot.png`; predicted-vs-observed with intervals.
- `analysis/latest/logs/cqr.log`.

### Acceptance criteria

- Empirical coverage on Emory test fold ≥ 88% (90% target with ±2% tolerance).
- Mean interval width is informative (not the entire range of observed Δ-PCL; aim for width < 0.6 × observed range).
- Conditional coverage by Response category does not deviate from marginal by > 5%.

### Risks / gotchas

- Subject as splitting unit is critical. Leaking PRE and POST of the same subject across train/test inflates coverage spuriously.
- Quantile regression on small N (164 paired) can be unstable. Try MAPIE's `MapieRegressor` with absolute-residual conformity score as a fallback.

### Notes for the runner

Helen's TPV cluster-bootstrap defaults apply. B=2000, subject-clustered.

---

## Step 2.B.2: Trajectory direction prediction sets

**Phase:** 2.B
**Owner:** Helen
**Estimated duration:** 5-7 days
**Depends on:** Step 2.A.4
**Strategic justification:** v2.2 Section 5.2.B (per-subject conformal bounds on the recovery-axis projection; the methodological innovation)

### Objective

Produce per-subject conformal prediction sets on the recovery-axis projection of Δz. Instead of predicting Δ-symptom score, predict the *direction* of trajectory along the recovery axis with finite-sample-valid prediction sets.

### Inputs

- Per-subject Δ-latent (winning embedding).
- Recovery axis (defined in Phase 3 Step 3.0; for Phase 2.B development, use Phase 0 Gate 0-T PC1 as a placeholder; refresh after Phase 3 Step 3.0).
- Δ-symptom (for outcome).

### Method

Implementation: `src/dnamrnaseq2026/conformal/direction.py` (new file). Snakefile rule: `rule step_2_B_2_direction_conformal:`.

1. Project Δz onto the recovery axis: `Δ_recovery_i = Δz_i · v_recovery / ||v_recovery||`.
2. The outcome is the signed scalar Δ_recovery_i. Discretise into 3 ordered categories: strongly-negative (move away from healthy), neutral, strongly-positive (move toward healthy), based on tertiles.
3. Conformal classifier: split-conformal on a multinomial classifier (`sklearn.ensemble.RandomForestClassifier`). Wrap in `mapie.classification.MapieClassifier(method='lac', alpha=0.10)`.
4. Output: per-subject prediction set (singleton, doubleton, or all-three) at α=0.10.
5. Conditional coverage by Response category and by Therapy_type (BEST).

### Outputs

- `analysis/latest/direction_predictor.pkl`; fitted MapieClassifier.
- `analysis/latest/direction_predictions.csv`; per-subject prediction set, observed direction category, set size.
- `analysis/latest/direction_coverage.csv`; marginal and conditional coverage.
- `analysis/latest/logs/direction_conformal.log`.

### Acceptance criteria

- Marginal coverage ≥ 88% on Emory test fold.
- Mean prediction set size < 2 (the predictor is informative).
- Conditional coverage by Response category within ±5% of marginal.

### Risks / gotchas

- The recovery-axis definition is forward-referencing Phase 3 Step 3.0. For Phase 2.B development cycle, use PC1 of Δz as a placeholder; refresh once Phase 3 Step 3.0 lands the formal recovery axis.
- Tertile discretisation is sensitive to N. With 164 paired Emory subjects, tertiles have ~55 subjects each. Cross-check coverage on quintiles as a sensitivity analysis.

### Notes for the runner

The interesting case is the "doubleton" prediction set: subjects whose trajectory direction is genuinely uncertain at 90% coverage. These are the candidates for adjunctive intervention (LINCS reversal, Step 3.5).

---

## Step 2.B.3: Coverage validation on BEST + Mondrian stratification

**Phase:** 2.B
**Owner:** Helen
**Estimated duration:** 3-5 days
**Depends on:** Step 2.B.1, Step 2.B.2, Step 0-S
**Strategic justification:** v2.2 Section 5.2.B (Mondrian stratified by Therapy_type, weighted conformal under Emory→BEST shift)

### Objective

Test whether CQR (Step 2.B.1) and direction predictor (Step 2.B.2) maintain coverage when calibrated on Emory and evaluated on BEST. Use importance weights from Step 0-S. Apply Mondrian stratification by Therapy_type (CPT, PE, None).

### Inputs

- `analysis/latest/cqr_predictor.pkl`, `analysis/latest/direction_predictor.pkl`.
- BEST Δ-latent (winning embedding applied to BEST).
- BEST Δ-CAPS5 + Therapy_type per paired subject.
- `analysis/latest/gate_0S_importance_weights.csv` (Step 0-S; only valid if Gate 0-S passed).

### Method

Implementation: `src/dnamrnaseq2026/conformal/mondrian.py` (new file). Snakefile rule: `rule step_2_B_3_mondrian:`.

1. Apply the Emory-trained CQR predictor to BEST Δ-latent. Compute empirical coverage on BEST overall.
2. Apply importance weights (from Gate 0-S) to recalibrate the conformity threshold; recompute coverage.
3. Mondrian stratification: per Therapy_type stratum, compute conformity threshold separately. Recompute coverage per stratum.
4. Coverage-gap statistic per stratum (`|empirical_coverage - target_coverage|`).
5. Same for direction predictor.

### Outputs

- `analysis/latest/cqr_coverage_best.csv`; BEST overall + Mondrian-stratified coverage.
- `analysis/latest/direction_coverage_best.csv`; same.
- `analysis/latest/conformal_coverage_summary.json`; combined summary for the methods paper.
- `analysis/latest/figures/coverage_by_stratum.png`; coverage per stratum bar chart.
- `analysis/latest/logs/mondrian.log`.

### Acceptance criteria

- BEST overall coverage with importance weights ≥ 85% (allowing 5% drop from Emory due to shift).
- Mondrian-stratified coverage ≥ 85% in CPT and PE strata (None stratum may be too small to evaluate; report anyway).
- Coverage-gap statistic < 0.10 in CPT and PE strata.

### Risks / gotchas

- If Gate 0-S failed (AUC > 0.85), importance weighting is unreliable. Report coverage without weighting as the primary; weighting as a sensitivity.
- PE stratum n ≈ 10 is too small for meaningful coverage estimates. Report with wide CI and a caveat.
- The Mondrian stratification requires calibration samples in each stratum. If BEST None stratum has < 5 paired subjects, drop it from Mondrian and report as "insufficient calibration samples".

### Notes for the runner

This is the punchline of the methods paper: the conformal predictor maintains coverage under cross-cohort shift via Mondrian + weighted calibration. If coverage drops below 85% in any stratum, dig into why before declaring failure.

---

# Phase 3: Trajectory atlas + translational layer (Weeks 6-12, Aria-led)

Phase 3 produces the headline figure (trajectory atlas with external reference clouds) and the translational outputs (target nomination, LINCS reversal, mediation as mechanism). Phase 3 depends on Step 2.A.4 (winning embedding) and Phase 1 outputs.

---

## Step 3.0: Compute trajectory vectors per subject

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 1-2 days
**Depends on:** Step 2.A.4
**Strategic justification:** v2.2 Section 5.3, Step 3.0

### Objective

For every paired subject in Emory and BEST, compute the per-subject trajectory vector `Δz_i = z_i_POST - z_i_PRE` in the winning embedding space, decompose into magnitude and unit direction, and identify the recovery axis as the first principal direction of the responder-only Δz cloud (or discriminant direction between responder and non-responder Δz clouds, per Gate 0-T result).

### Inputs

- `analysis/latest/embedding_<winner>.pt` (read winner from `embedding_winner.txt`).
- `analysis/latest/embedding_<winner>_metadata.csv` (sample IDs, Visit, Response).
- Paired-subject IDs (Emory 164, BEST 48).

### Method

Implementation: `src/dnamrnaseq2026/trajectory/vectors.py` (new file). Snakefile rule: `rule step_3_0_trajectory_vectors:`.

1. Load winner embedding tensor. Per paired subject, compute `z_i_PRE` and `z_i_POST` (look up by sample ID and Visit).
2. Compute `Δz_i = z_i_POST - z_i_PRE` per subject.
3. Decompose into magnitude `||Δz_i||` and unit direction `Δẑ_i = Δz_i / ||Δz_i||`.
4. **Recovery axis identification:**
   - Option A (preferred if Gate 0-T PERMANOVA p < 0.05): first principal direction of responder-only Δz cloud via PCA.
   - Option B (fallback if Gate 0-T was marginal): discriminant direction between responder Δz and non-responder Δz via linear discriminant analysis (`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`).
5. Save: per-subject Δz, magnitude, unit direction, projection on recovery axis.

### Outputs

- `analysis/latest/trajectory_vectors_emory.csv`; per paired subject: subject_id, Δz_1..Δz_128, magnitude, unit_direction (128-D), recovery_projection.
- `analysis/latest/trajectory_vectors_best.csv`; same BEST.
- `analysis/latest/recovery_axis.csv`; 128-D recovery axis vector, plus method (PCA or LDA).
- `analysis/latest/logs/trajectory_vectors.log`.

### Acceptance criteria

- All 164 Emory + 48 BEST paired subjects have a valid Δz (non-NaN, non-zero magnitude).
- Recovery axis vector has ||v_recovery|| ≈ 1 (unit vector).
- Recovery projection separates responders from non-responders at p < 0.05 (paired t-test on recovery_projection grouped by Response).

### Risks / gotchas

- If the winning embedding's latent dimension differs from 128 (e.g. MOFA+ uses 20 factors), the recovery axis vector matches that dimension. Document.
- LDA on small N can overfit; if used, report cross-validated discriminant projection alongside the in-sample.

### Notes for the runner

This step is fast but load-bearing. Everything downstream in Phase 3 consumes `trajectory_vectors_*.csv` and `recovery_axis.csv`.

---

## Step 3.1: Trajectory archetype clustering

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 3-4 days
**Depends on:** Step 3.0
**Strategic justification:** v2.2 Section 5.3, Step 3.2

### Objective

Cluster paired-subject Δz vectors via Gaussian mixture (and spectral fallback) to identify archetypes: reversers (Δz moves toward healthy), partial responders, non-responders, possibly reverser-inflammatory-flare. Each archetype is named by its dominant direction and Phase 1 biological annotation.

### Inputs

- `analysis/latest/trajectory_vectors_emory.csv`, `analysis/latest/trajectory_vectors_best.csv`.
- `analysis/latest/recovery_axis.csv`.

### Method

Implementation: `src/dnamrnaseq2026/trajectory/archetypes.py` (new file). Snakefile rule: `rule step_3_1_archetypes:` (renaming the existing `cluster_archetypes` stub).

1. Concatenate Emory + BEST Δz vectors (212 subjects by latent dim).
2. Gaussian mixture model via `sklearn.mixture.GaussianMixture(n_components=k, random_state=seed)` for `k ∈ {2, 3, 4, 5}`. 10 seeds per k.
3. BIC per k; bootstrap ARI stability across seeds.
4. Select k by BIC + bootstrap ARI > 0.7.
5. Spectral clustering on cosine-similarity graph (`sklearn.cluster.SpectralClustering(affinity='precomputed', n_clusters=k_selected)`) as a robustness check.
6. Per archetype, compute centroid Δz, dominant unit direction, recovery projection, and Response/Therapy_type composition.
7. Name archetypes by direction and biology.

### Outputs

- `analysis/latest/archetypes.csv`; replaces stub. Columns: subject_id, archetype_id, archetype_name, posterior probability, recovery_projection.
- `analysis/latest/archetype_centroids.csv`; per archetype: centroid Δz, dominant direction.
- `analysis/latest/archetype_response_table.csv`; archetype by Response by Therapy_type counts.
- `analysis/latest/figures/archetype_silhouette.png`.
- `analysis/latest/logs/archetypes.log`.

### Acceptance criteria

- Selected k has BIC minimum and bootstrap ARI > 0.7 across seeds.
- At least 3 distinct archetypes (reverser, non-responder, partial) identifiable from cluster centroids.
- Archetype-by-Response chi-squared p < 0.05.

### Risks / gotchas

- GMM can overfit at high k; cap k=5.
- Cosine vs Euclidean similarity for clustering: Euclidean on Δz (magnitude matters), cosine on Δẑ (direction only). Run both as sensitivity; report the one consistent with BIC.

### Notes for the runner

Archetype naming is the place to inject domain knowledge from Phase 1. "Reverser-monocyte-quiescence" is named because the cluster's centroid loads on the monocyte-quiescence axis from Phase 1.4.

---

## Step 3.2: Recovery-axis biological annotation

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 4-5 days
**Depends on:** Step 3.0, Phase 1 outputs (Step 1.2, 1.3, 1.4, 1.5, 1.6)
**Strategic justification:** v2.2 Section 5.3, Step 3.1 (recovery-axis biological annotation)

### Objective

For the recovery axis and each of the next 2-4 top latent directions, identify top-loading features (CpGs and genes), run cell-type / pathway / TF / TFBS enrichment, and annotate each axis with its biological meaning.

### Inputs

- `analysis/latest/recovery_axis.csv` and other top latent direction vectors.
- Phase 1 outputs: CellDMC tables, decoupleR TF activity, PROGENy/GSVA pathway tables, ENCODE/EpiMap enrichment.

### Method

Implementation: `src/dnamrnaseq2026/trajectory/axis_annotation.py` (new file). Snakefile rule: `rule step_3_2_axis_annotation:` (renaming `annotate_recovery_axis` stub).

1. For each axis (recovery axis + top 4 axes):
   - Top 200 features by absolute loading.
   - Map features back to CpGs / genes via the embedding's input feature index.
2. Cell-type enrichment: Fisher exact on (cell-type-specific CellDMC FDR<0.05 lists from Step 1.2) ∩ (top-loading features).
3. Pathway enrichment: decoupleR ULM on top-loading genes against PROGENy pathways.
4. TF enrichment: decoupleR ULM on top-loading genes against CollecTRI regulons.
5. TFBS enrichment: LOLA or hypergeometric test on top-loading CpGs against ENCODE TF ChIP-seq.
6. Aggregate annotations per axis. Write narrative: "axis k encodes a shift from {cell-type-X} to {cell-type-Y} driven by {pathway-Z} and {TF-W}".

### Outputs

- `analysis/latest/recovery_axis_annotation.csv`; replaces stub. Columns: axis_id, top_loading_genes, top_loading_cpgs, enriched_celltypes, enriched_pathways, enriched_tfs, enriched_tfbs, narrative.
- `analysis/latest/axis_loading_heatmap.png`; top-loading features by axis.
- `analysis/latest/logs/axis_annotation.log`.

### Acceptance criteria

- Recovery axis has at least one enriched cell type (FDR < 0.10), one enriched pathway, one enriched TF.
- The narrative for the recovery axis is a single sentence that the manuscript can quote directly.

### Risks / gotchas

- Loadings can be unstable across seeds for the FM and contrastive arms. Use seed-averaged loadings (mean absolute loading per feature across 10 seeds).
- "Top 200" is a heuristic; sweep over {100, 200, 500} as sensitivity.

### Notes for the runner

This is what makes the latent space mechanistic instead of a black box. Spend time on the narrative; it goes into Figure 1 caption.

---

## Step 3.3: External-cohort terminus projection (THE HEADLINE FIGURE)

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 5-7 days
**Depends on:** Step 3.0, Step 3.1
**Strategic justification:** v2.2 Section 5.3, Step 3.3 (the headline figure)

### Objective

Project external cohorts (GSE98793 TRD, GTEx healthy whole blood, AURORA trauma-exposed if accessible) into the winning embedding space as static reference clouds. Compute per-Emory-and-BEST-subject terminus location (`z_i_POST`) relative to those clouds. Test whether responder termini cluster closer to healthy clouds and non-responder termini closer to TRD-inflammatory clouds. **THIS IS FIGURE 1 OF THE MECHANISM PAPER.**

### Inputs

- Winning embedding's RNA-side encoder (extracted from `embedding_<winner>.pt`).
- GSE98793 expression matrix (downloaded; see Gate 0-X).
- GTEx whole-blood expression matrix (downloaded from GTEx v8 portal; restrict to whole-blood tissue).
- AURORA cohort if accessible (TBD; flag in config).
- `analysis/latest/trajectory_vectors_emory.csv`, `analysis/latest/trajectory_vectors_best.csv`.

### Method

Implementation: `src/dnamrnaseq2026/external_projection/terminus.py` (new file). Snakefile rules: `rule step_3_3_project_gse98793:`, `rule step_3_3_project_gtex:`, `rule step_3_3_terminus_test:` (renaming the existing stubs `project_gse98793`, `project_gtex`, `terminus_permanova`).

1. For each external cohort, harmonise expression to Emory RNA-seq feature space: match Ensembl gene IDs, quantile-normalise across cohorts, restrict to features used by the winning embedding's RNA encoder.
2. If the winning embedding requires DNAm (FM and contrastive arms), train an auxiliary RNA-only projection: regression of Emory RNA → full-embedding latent, fit on Emory PRE-IOP samples. Project externals via this auxiliary. **Document this caveat: cross-cohort projection assumes the RNA-only auxiliary preserves the relevant latent structure.**
3. If the winning embedding is MOFA+, project externals via the RNA-view weights only (MOFA+ supports this natively).
4. Compute external sample positions in the embedding. Save.
5. Compute terminus location per Emory + BEST paired subject: `z_i_POST`.
6. Compute distances: per Emory + BEST subject, Mahalanobis distance to each external cloud (GSE98793 TRD-inflammatory, GSE98793 controls, GTEx healthy, AURORA if available).
7. Test: responder vs non-responder Emory termini distance to GTEx (PERMANOVA). Responder vs non-responder Emory termini distance to GSE98793 TRD-inflammatory (PERMANOVA).
8. **Headline figure:** 2D UMAP or first-two-PC projection of the embedding, showing:
   - Per-subject arrows from `z_i_PRE` to `z_i_POST`, coloured by Response.
   - External reference clouds as static scatter overlays.
   - Recovery axis annotated with its biological narrative.
   - Archetype clusters as background shading.

### Outputs

- `analysis/latest/terminus_gse98793.csv`; replaces stub. Per GSE98793 sample: latent coordinates, group label (TRD-inflam, control, other).
- `analysis/latest/terminus_gtex.csv`; replaces stub. Per GTEx sample: latent coordinates.
- `analysis/latest/terminus_aurora.csv`; if available; otherwise empty placeholder.
- `analysis/latest/terminus_test.csv`; replaces stub. PERMANOVA F, p; Mahalanobis distances per Emory + BEST subject to each cloud.
- `manuscript/figures/fig1_trajectory_atlas.png`; replaces stub. THE headline figure.
- `analysis/latest/logs/terminus_*.log`.

### Acceptance criteria

- All external cohorts projected with valid latent coordinates (non-NaN) for ≥ 80% of samples.
- PERMANOVA on responder-vs-non-responder Emory termini distance to GTEx p < 0.05 with responders closer.
- PERMANOVA on responder-vs-non-responder Emory termini distance to GSE98793 TRD-inflammatory p < 0.05 with non-responders closer.
- Figure renders cleanly at 300 dpi, < 30 cm width, with all labels readable at print size.

### Risks / gotchas

- Cross-cohort RNA-seq vs microarray (GSE98793) harmonisation is crude under quantile-norm. ComBat-Seq or COCONUT is more principled; if Gate 0-X passed under quantile-norm, document the choice. If Gate 0-X was marginal, try ComBat-Seq as a sensitivity.
- The RNA-only auxiliary projection (for FM/contrastive arms) is a methodological caveat. Make it explicit in the methods section.
- GSE98793 has known batch effects within the dataset; restrict to samples passing GSE98793's own QC.

### Notes for the runner

THIS IS THE FIGURE. Spend time on it. Iterate with Lee until it looks right. The caption is provisional ("Treatment response trajectories in PTSD trace toward healthy-state biology and away from TRD-inflammatory state"); refine after the figure is built.

---

## Step 3.4: Conditional mQTL discovery arm

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 5-7 days (conditional)
**Depends on:** Step 1.2, plus Lee confirming raw GSA genotype access
**Strategic justification:** v2.2 Section 5.3, Step 3.7 (conditional mQTL; only if genotypes accessible)

### Objective

If raw GSA genotype data is accessible, run cis-mQTL discovery on the CellDMC(Δ)-significant CpGs and cis-eQTL on the corresponding cis-genes. Apply Mendelian-randomisation-style sensitivity to strengthen the causal interpretation of the Δ-mediator → Δ-outcome relationship.

### Inputs

- Emory GSA genotype data (CONDITIONAL on Lee confirming access).
- `analysis/latest/celldmc_delta_emory.tsv` (CellDMC significant CpGs).
- `analysis/latest/de_delta_emory.tsv` (RNA-seq DE).

### Method

Implementation: `src/dnamrnaseq2026/external_projection/mqtl.py` (new file). Snakefile rule: `rule step_3_4_mqtl:`.

1. Genotype QC: PLINK QC (call rate > 0.95, HWE p > 1e-6, MAF > 0.01).
2. Cis-mQTL per CellDMC(Δ)-significant CpG: `Δ_M ~ SNP + Δ_cell_frac + ancestry_PCs` for each SNP within ±1 Mb. Use `MatrixEQTL` via rpy2 or `tensorqtl`. FDR per CpG.
3. Cis-eQTL on Δ-gene-expression for cis-genes.
4. MR-style sensitivity: SNPs with significant mQTL or eQTL effects become IVs for the Δ-mediator → Δ-outcome relationship. Compute MR effect via 2-sample MR (`MendelianRandomization` package).

### Outputs

- `analysis/latest/mqtl_results.csv`; per CpG-SNP cis-mQTL test.
- `analysis/latest/eqtl_results.csv`; per gene-SNP cis-eQTL test.
- `analysis/latest/mr_sensitivity.csv`; MR effect per IV-mediator-outcome triplet.
- `analysis/latest/logs/mqtl.log`.

### Acceptance criteria

- At least 5 cis-mQTL significant at FDR < 0.10.
- At least 1 MR sensitivity analysis producing a coherent effect estimate.
- **FAIL gracefully if genotypes inaccessible:** Skip step; document as "MR arm not run due to data access" in mechanism paper limitations.

### Risks / gotchas

- Genotype access is the blocker. Resolve in Week 2-3.
- MR with single-cohort observational design is fragile; report as exploratory.
- Multiple-testing across many CpG-SNP pairs requires careful FDR control.

### Notes for the runner

If genotypes are inaccessible, mark this step as cancelled in the task ledger and proceed without it. Mechanism paper notes the limitation in the Discussion.

---

## Step 3.5: LINCS L1000 signature reversal on the recovery axis

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 4-5 days
**Depends on:** Step 3.0, Step 3.2
**Strategic justification:** v2.2 Section 5.3, Step 3.6 (target nomination + LINCS reversal on recovery axis)

### Objective

Query LINCS L1000 for compounds whose expression signature reverses the non-responder-archetype centroid direction toward the responder-archetype centroid direction. Output: ranked list of candidate adjunctive interventions for PTSD non-responders.

### Inputs

- `analysis/latest/recovery_axis_annotation.csv` (top-loading genes per axis).
- `analysis/latest/archetype_centroids.csv` (archetype centroids).
- LINCS L1000 Level 5 signatures (downloaded via clue.io API or local mirror; cell-line-matched to monocyte / T-cell / mixed PBMC lines where available).

### Method

Implementation: `src/dnamrnaseq2026/external_projection/lincs_reversal.py` (new file). Snakefile rule: `rule step_3_5_lincs:`.

1. Define query signature: the non-responder-to-responder direction in gene-expression space, derived by projecting archetype centroid differences back through the embedding's RNA encoder.
2. Restrict LINCS signatures to PBMC / monocyte / T-cell lines per CTRP / DepMap annotations.
3. Connectivity Score per compound: cosine similarity between (negative of query signature) and (compound signature). Negative similarity = reversal.
4. Rank compounds by reversal score. Filter by mechanism class (anti-inflammatory, immunomodulator, etc.).
5. Cross-reference with Open Targets / Pharos / DGIdb for druggability and clinical-stage status.
6. Output: ranked compound table for the mechanism paper's translational section.

### Outputs

- `analysis/latest/lincs_reversal_ranked.csv`; per compound: reversal score, mechanism class, druggability, clinical-stage.
- `analysis/latest/lincs_top_candidates.csv`; top 20 candidates with Open Targets / Pharos / DGIdb annotations.
- `analysis/latest/logs/lincs.log`.

### Acceptance criteria

- At least 50 LINCS compounds with reversal score < -0.3 (clear reversal direction).
- Top 10 candidates have mechanism classes consistent with PTSD recovery biology (inflammation reversal, glucocorticoid signalling, etc.) at least 50% of the time.

### Risks / gotchas

- LINCS API rate limits; cache aggressively.
- PBMC LINCS coverage is limited compared to cancer cell lines. Document the cell-line panel used.

### Notes for the runner

The actionable output of the mechanism paper. The top candidate is a manuscript-level talking point.

---

## Step 3.6: Target nomination synthesis

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 3-4 days
**Depends on:** Step 3.2, Step 3.5
**Strategic justification:** v2.2 Section 5.3, Step 3.6

### Objective

Synthesise target nominations across Open Targets (genetic + tractability), Pharos (TDL score), DGIdb (existing chemistry), ChEMBL (compound matter), and LINCS reversal candidates from Step 3.5. Produce a ranked target table for the mechanism paper.

### Inputs

- `analysis/latest/recovery_axis_annotation.csv` (top-loading genes).
- Open Targets via `opentargets-py` API.
- Pharos via TCRD API.
- DGIdb via REST API.
- ChEMBL via `chembl_webresource_client`.
- `analysis/latest/lincs_top_candidates.csv`.

### Method

Implementation: `src/dnamrnaseq2026/external_projection/target_nomination.py` (new file). Snakefile rule: `rule step_3_6_targets:`.

1. For each top-loading gene on the recovery axis: query Open Targets for genetic association scores, Pharos for Target Development Level (TDL), DGIdb for known drug-gene interactions, ChEMBL for compound bioactivity.
2. Cross-reference with Step 3.5 LINCS compounds: any LINCS compound targeting one of the top genes is flagged.
3. Composite ranking: weighted combination of genetic association, tractability (TDL Tclin > Tchem > Tbio > Tdark), and reversal signal.
4. Output: ranked target table.

### Outputs

- `analysis/latest/target_nomination.csv`; gene, OT genetic score, Pharos TDL, DGIdb drugs, ChEMBL compounds, LINCS reversal flag, composite rank.
- `analysis/latest/logs/target_nomination.log`.

### Acceptance criteria

- Top 20 targets cover ≥ 3 cell types from the recovery-axis annotation.
- At least 5 of the top 20 targets are Tclin or Tchem (existing approved or chemistry).

### Risks / gotchas

- API failures: cache, retry, log. Cron-job-style retries in the implementation.
- TDL is a heuristic; do not over-weight it.

### Notes for the runner

This synthesis is what gives the mechanism paper its translational claim.

---

## Step 3.7: Mediation as mechanism for trajectory direction (HIMA + BAMA + E-values)

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 5-7 days
**Depends on:** Step 1.2, Step 1.3, Step 1.4
**Strategic justification:** v2.2 Section 5.3, Step 3.5 (mediation now framed as mechanism for trajectory direction, not standalone)

### Objective

Run high-dimensional mediation analysis on `Δ-DNAm → Δ-mRNA → Δ-PCL` using HIMA and BAMA for cross-validation. E-value sensitivity to bound unmeasured confounding. The output is a Δ-mediator catalogue: which CpG-gene pairs mediate trajectory direction along the recovery axis.

### Inputs

- `analysis/latest/celldmc_delta_emory.tsv` (mediator candidates: significant Δ-CpGs).
- `analysis/latest/de_delta_emory.tsv` (RNA-side mediator candidates: significant Δ-genes).
- Cis-pair list (CpG within ±100 kb of a gene): from Step 1.4 convergent table.
- Δ-PCL_total per paired subject (outcome).
- Covariates: Age, sex, ancestry PCs, smokingScore_PRE, baseline PCL, EpiDISH cell fractions at PRE.

### Method

Implementation: `src/dnamrnaseq2026/mediation/hima_bama.py` (new file). Snakefile rules: `rule hima_delta:` (already stubbed), `rule bama_delta:` (already stubbed), `rule evalue_sensitivity:` (already stubbed). Renamed in `workflow/rules/mediation.smk` to `rule step_3_7_hima:`, `rule step_3_7_bama:`, `rule step_3_7_evalue:`.

For each cis-pair (CpG_j, gene_g(j)) from the convergent table:

1. **HIMA:** via `HIMA::hima()` in R through rpy2. Inputs: exposure = Δ_treatment (R/NR), mediator = Δ_M_CpG, outcome = Δ_PCL, covariates. HIMA returns α, β, indirect effect, joint-significance p, FDR-q per mediator.
2. **BAMA:** via `bama::bama()` similarly. Bayesian framework, Markov-chain credible intervals on indirect effect.
3. **E-value:** per significant mediator, compute E-value via `EValue::evalues.OLS()` per VanderWeele & Ding 2017. E-value ≥ 1.5 = moderate robustness.
4. Output: Δ-mediator catalogue with HIMA + BAMA estimates and E-value.

### Outputs

- `analysis/latest/hima_results.csv`; replaces stub. Per cis-pair: HIMA α, β, indirect, p, q.
- `analysis/latest/bama_results.csv`; replaces stub. Per cis-pair: BAMA estimate, credible interval.
- `analysis/latest/mediation_evalues.csv`; replaces stub. Per significant mediator: E-value.
- `analysis/latest/mediation_catalogue.csv`; combined catalogue with mediator class (from Step 1.4 convergent annotation) appended.
- `analysis/latest/logs/mediation_*.log`.

### Acceptance criteria

- At least 5 significant mediators at HIMA FDR < 0.10 with BAMA credible interval excluding zero.
- E-value ≥ 1.5 for at least 2 significant mediators.

### Risks / gotchas

- HIMA assumes sequential ignorability: a strong assumption that unmeasured confounders do not affect both mediator and outcome. E-value bounds the sensitivity but does not eliminate the concern.
- Cis-pair list size: if Step 1.4 yields > 1,000 cis-pairs, HIMA runtime balloons. Sub-sample to top 500 by convergent ranking.

### Notes for the runner

Mediation is secondary, not headline. Frame in the mechanism paper as evidence that the recovery axis direction is *causally driven by* specific cell-type-resolved Δ-CpG → Δ-gene mediators.

---

## Step 3.8: Treatment-modality stratification in BEST

**Phase:** 3
**Owner:** Aria
**Estimated duration:** 3-4 days
**Depends on:** Step 1.7, Step 3.0, Step 3.1
**Strategic justification:** v2.2 Section 5.3 (treatment-modality stratification in BEST: CPT vs PE molecular response signatures)

### Objective

In BEST cohort, test whether CPT-treated paired subjects and PE-treated paired subjects show different molecular trajectory signatures along the recovery axis. This addresses whether the recovery-axis biology is therapy-agnostic or therapy-specific.

### Inputs

- `analysis/latest/trajectory_vectors_best.csv` (Δz per BEST paired subject).
- BEST Therapy_type column (CPT, PE, None).
- `analysis/latest/recovery_axis.csv` (from Step 3.0).
- `analysis/latest/archetypes.csv` (from Step 3.1, archetype assignment per BEST subject).

### Method

Implementation: `src/dnamrnaseq2026/trajectory/modality_stratification.py` (new file). Snakefile rule: `rule step_3_8_modality:`.

1. Stratify BEST paired subjects by Therapy_type.
2. Per stratum, compute mean recovery projection, mean Δz magnitude, archetype enrichment.
3. Test: `recovery_projection ~ Therapy_type + Response + interaction` via OLS.
4. Per-modality recovery-axis loadings: do CPT and PE modalities load on different top features?

### Outputs

- `analysis/latest/modality_stratification.csv`; per stratum: n, mean recovery projection, archetype distribution.
- `analysis/latest/modality_recovery_axis_loadings.csv`; per-modality top-loading features.
- `analysis/latest/figures/modality_stratification.png`.
- `analysis/latest/logs/modality.log`.

### Acceptance criteria

- At least one Therapy_type × Response interaction significant at p < 0.10.
- If interaction is non-significant, the recovery axis is reported as therapy-agnostic (a finding in itself).

### Risks / gotchas

- Small n per modality (PE n ≈ 10 paired). Cluster-bootstrap CIs are critical.
- Confounding: Therapy_type may correlate with Response by design (some patients self-select). Document.

### Notes for the runner

This is a Phase 3 substep that feeds the mechanism paper's "BEST replication" figure.

---

# Phase 4: Manuscripts + clinical artefact (Weeks 10-18)

Phase 4 produces three deliverables: mechanism paper, methods paper, library release. OSF pre-registration timing is also decided here.

---

## Step 4.1: Mechanism paper draft

**Phase:** 4
**Owner:** Aria (first author)
**Estimated duration:** 4-6 weeks
**Depends on:** Step 3.3 (headline figure), Step 3.7, Step 3.8, all Phase 1 outputs
**Strategic justification:** v2.2 Section 5.4.8.1, Mechanism paper structure

### Objective

Draft the mechanism paper for *Nature Medicine* (primary) or *Lancet Psychiatry* (secondary). Headline: "Treatment response trajectories in PTSD trace toward healthy-state biology and away from treatment-resistant-depression-inflammatory state: a multi-omics trajectory atlas with cell-type-resolved recovery-axis annotation."

### Inputs

- All Phase 3 outputs.
- Phase 1 cell-type-resolved Δ-mediator catalogue.
- Companion vault notes: `04-projects/dnamrnaseq/2026-05-17-integrated-analysis-plan-v2.md` (v2.2), `04-projects/dnamrnaseq/2026-05-17-mechanism-paper-outline.md` (to be written).

### Method

Manuscript drafting follows the vault convention: prose drafts live in `06-writing/dnamrnaseq/`; figures in `manuscript/figures/`; supplementary tables in `manuscript/supplementary/`. Code that generates the figures lives in `src/dnamrnaseq2026/viz/`.

Sections per v2.2 §5.4.8.1:

1. Introduction; trajectory-atlas question framing.
2. Phase 0 gate results (one paragraph synthesis).
3. Phase 2.A: joint embedding architecture choice; winner reported.
4. **Headline figure: the trajectory atlas with external reference clouds.**
5. Phase 3.2 biological annotation of the recovery axis.
6. Phase 3.1 archetype clustering.
7. Phase 3.3 external terminus projection.
8. Phase 1 cell-type-resolved CellDMC.
9. Phase 3.7 mediation catalogue.
10. Phase 1.7 BEST replication.
11. Phase 3.5 LINCS reversal.
12. Discussion; explicit differentiation from mmVAE.

Methods section requirements per v2.2 §5.4.8.1: joint-latent-space framing with CVB Parkinson's precedent, Δ-construction, embedding architecture comparison protocol, recovery-axis identification, archetype clustering reproducibility, external cohort projection protocol (with the RNA-only auxiliary caveat), CellDMC at three contrasts, HIMA/BAMA + E-values, cluster-bootstrap parameters, LINCS connectivity protocol.

### Outputs

- `06-writing/dnamrnaseq/mechanism-paper-draft.md` (in companion vault); manuscript draft.
- `manuscript/figures/fig1_trajectory_atlas.png` through `fig8_*`; code-generated figures.
- `manuscript/supplementary/sup_table_*.csv`; supplementary tables.

### Acceptance criteria

- Draft is reviewable by Lee and co-authors (Tobias, Helen, Kai).
- All 6+ figures are code-generated and reproducible from the analysis pipeline.
- Methods section reproduces the v2.2 protocols.

### Risks / gotchas

- Submission target may shift (Nature Medicine vs Lancet Psychiatry vs lower-tier first); make the draft target-agnostic until decision.
- Figure count discipline: target paper allows ~6 main figures + 8-10 supplementary. Trim.

### Notes for the runner

Lee is the senior author. Decisions on framing, scope, and target journal go to him.

---

## Step 4.2: Methods paper draft

**Phase:** 4
**Owner:** Tobias + Helen (co-first authors)
**Estimated duration:** 4-6 weeks
**Depends on:** Step 2.A.4, Step 2.B.3
**Strategic justification:** v2.2 Section 5.4.8.2, Methods paper structure

### Objective

Draft the methods paper for *Nature Methods* (primary) or *NeurIPS Datasets & Benchmarks* / *NeurIPS ML4H* (secondary). Headline: "Joint multi-omics latent spaces for treatment-response trajectory visualisation at small N: foundation-model embeddings, matrix factorisation with trait-state structure, and contrastive learning compared."

### Inputs

- All Phase 2.A and 2.B outputs.

### Method

Two-track structure per v2.2 §5.4.8.2:

**Track A (Tobias):** three trajectory-tuned architectures head-to-head. Scoring criteria. Trajectory-consistency benchmark. Result: what works at n ≈ 200 paired multi-omics. Likely null for FM arm; positive null is still a methods contribution.

**Track B (Helen):** Mondrian weighted-conformal prediction sets for per-subject trajectory direction under covariate shift. Three-class Mondrian for BEST partial-response. Conformity-score ablation.

### Outputs

- `06-writing/dnamrnaseq/methods-paper-draft.md` (in companion vault); manuscript draft.
- `manuscript/figures/methods_fig*.png`; figures for the methods paper.

### Acceptance criteria

- Draft is reviewable by Lee and co-authors.
- All figures are code-generated and reproducible.

### Risks / gotchas

- Methods paper publication timing: ideally co-submitted or shortly after mechanism paper. Pre-print on bioRxiv at the same time as mechanism paper submission to establish methods priority.

### Notes for the runner

Tobias and Helen co-first; Aria and Lee are co-authors.

---

## Step 4.3: `trajectory-atlas-ptsd` library release

**Phase:** 4
**Owner:** Tobias (lead) + Aria, Helen, Kai (contributors)
**Estimated duration:** 2-3 weeks
**Depends on:** Step 2.A.4, Step 3.0, Step 3.3
**Strategic justification:** v2.2 Section 5.4.8.3 (clinical artefact / library)

### Objective

Release a `trajectory-atlas-ptsd` Python library on PyPI + GitHub Release that lets external groups project their own cohorts into the recovery-axis space, score archetype membership, and compute terminus position relative to GSE98793 and GTEx references.

### Inputs

- `analysis/latest/embedding_<winner>.pt` (trained encoder).
- `analysis/latest/recovery_axis.csv`.
- `analysis/latest/archetype_centroids.csv`.
- `analysis/latest/terminus_gse98793.csv`, `analysis/latest/terminus_gtex.csv`.

### Method

Refactor the analysis library into a release package:

1. Public API in `src/dnamrnaseq2026/trajectory_atlas/__init__.py`:
   - `load_atlas()`; load trained encoder + reference clouds + recovery axis.
   - `project_cohort(rnaseq_matrix, dnam_matrix=None)`; project a new cohort, return latent coordinates and archetype assignments.
   - `score_terminus(coords)`; distance to each reference cloud per subject.
2. CLI entry-point: `trajectory-atlas project --rnaseq input.csv --dnam input2.csv --output results.csv`.
3. Package metadata + versioning per `pyproject.toml`.
4. Documentation: README, tutorial Jupyter notebook in `notebooks/`, ReadTheDocs.
5. CI: pytest on synthetic fixtures.
6. Release: tag in git, push to PyPI via `twine`, GitHub Release with bundled trained encoder weights (or HuggingFace Hub upload for the weights if too large).

### Outputs

- Tagged release at `github.com/leelancashire/dnamrnaseq-2026/releases/v1.0.0`.
- PyPI package `trajectory-atlas-ptsd==1.0.0`.
- Trained encoder weights on HuggingFace Hub (or GitHub Release artefact).
- Tutorial notebook in `notebooks/01_external_cohort_projection.ipynb`.

### Acceptance criteria

- `pip install trajectory-atlas-ptsd` works in a fresh conda env.
- Tutorial notebook runs end-to-end on the bundled synthetic fixture.
- README explains use cases, limitations, citation.

### Risks / gotchas

- Trained encoder weights may be larger than PyPI permits. Use GitHub Release or HuggingFace Hub.
- License: MIT for the code, but the embedding was trained on Emory + BEST data; verify with Lee whether weights distribution requires CVB / Emory IRB sign-off.

### Notes for the runner

This is the most likely high-impact deliverable. External groups can run their own cohort against the trajectory atlas immediately.

---

## Step 4.4: OSF pre-registration timing decision

**Phase:** 4
**Owner:** Aria + Lee
**Estimated duration:** 1-2 days
**Depends on:** Phase 0 results, Step 2.A.4 (pre-registration of leaderboard criteria), Step 3.7 (pre-registration of mediation model)
**Strategic justification:** v2.2 Section 5.4 (pre-registration timing relative to mmVAE publication)

### Objective

Decide when to OSF pre-register the analysis plan: before mmVAE publication (preserves priority) or after (avoids signalling the analysis to mmVAE team prematurely). Recommended: pre-register the methods criteria and mediation model upfront in Phase 0-1; pre-register the trajectory-atlas headline framing closer to Phase 3 completion.

### Inputs

- v2.2 strategic plan.
- mmVAE publication timeline (from CVB).
- Current PR + decision record history.

### Method

1. Draft OSF pre-registration document covering:
   - Phase 0 gates (already specified in v2.2 §5.0).
   - Phase 2.A leaderboard scoring criteria (pre-registered before computing the leaderboard).
   - Phase 3.7 mediation model specification.
2. Upload to OSF via `osfclient` or web UI.
3. Lock the registration once Phase 0 + Phase 1 are running.

### Outputs

- OSF pre-registration entry with DOI.
- Vault decision record `99-admin/decisions/<date>-osf-prereg-dnamrnaseq.md`.

### Acceptance criteria

- OSF entry locked before Phase 2.A leaderboard is computed.
- DOI cited in mechanism + methods papers.

### Risks / gotchas

- Pre-registration that is too detailed becomes hard to deviate from. Pre-register the *acceptance criteria* and *high-level approach*, not every parameter.

### Notes for the runner

Discuss with Lee in Week 1 or Week 2. The conservative path is to pre-register Phase 0 and Phase 1 in Week 1 and refine the Phase 2 / 3 pre-registration when the embedding leaderboard is ready to run.

---

## Appendix A: Snakefile rule index

| Step | Rule name | Module |
|---|---|---|
| 0-T | (no rule; runs as `scripts/01_phase0_gate_T.py`) | n/a |
| 0-C | `epidish_emory`, `epidish_best` | `workflow/rules/preprocessing.smk` |
| 0-S | (no rule; runs as `scripts/01_phase0_gate_S.py`) | n/a |
| 0-X | (no rule; runs as `scripts/01_phase0_gate_X.py`) | n/a |
| 1.1 | `epidish_emory`, `epidish_best` (final implementation) | `workflow/rules/preprocessing.smk` |
| 1.2 | `step_1_2_celldmc_pre_emory`, `step_1_2_celldmc_post_emory`, `step_1_2_celldmc_delta_emory` | `workflow/rules/preprocessing.smk` |
| 1.3 | `step_1_3_rnaseq_de_emory` | `workflow/rules/preprocessing.smk` |
| 1.4 | `step_1_4_pathway_activity` | `workflow/rules/preprocessing.smk` |
| 1.5 | `step_1_5_tf_activity` | `workflow/rules/preprocessing.smk` |
| 1.6 | `step_1_6_regulatory_enrichment` | `workflow/rules/preprocessing.smk` |
| 1.7 | `step_1_7_replication` | `workflow/rules/preprocessing.smk` |
| 2.A.1 | `step_2_A_1_fm_embedding` | `workflow/rules/embedding.smk` |
| 2.A.2 | `step_2_A_2_mofa_embedding` | `workflow/rules/embedding.smk` |
| 2.A.3 | `step_2_A_3_contrastive_embedding` | `workflow/rules/embedding.smk` |
| 2.A.4 | `step_2_A_4_leaderboard` | `workflow/rules/embedding.smk` |
| 2.B.1 | `step_2_B_1_cqr` | `workflow/rules/embedding.smk` (or new `conformal.smk`) |
| 2.B.2 | `step_2_B_2_direction_conformal` | `workflow/rules/embedding.smk` |
| 2.B.3 | `step_2_B_3_mondrian` | `workflow/rules/embedding.smk` |
| 3.0 | `step_3_0_trajectory_vectors` | `workflow/rules/trajectory.smk` |
| 3.1 | `step_3_1_archetypes` (replaces `cluster_archetypes`) | `workflow/rules/trajectory.smk` |
| 3.2 | `step_3_2_axis_annotation` (replaces `annotate_recovery_axis`) | `workflow/rules/trajectory.smk` |
| 3.3 | `step_3_3_project_gse98793`, `step_3_3_project_gtex`, `step_3_3_terminus_test` | `workflow/rules/external_projection.smk` |
| 3.4 | `step_3_4_mqtl` (conditional) | `workflow/rules/external_projection.smk` |
| 3.5 | `step_3_5_lincs` | `workflow/rules/external_projection.smk` |
| 3.6 | `step_3_6_targets` | `workflow/rules/external_projection.smk` |
| 3.7 | `step_3_7_hima`, `step_3_7_bama`, `step_3_7_evalue` (rename existing stubs) | `workflow/rules/mediation.smk` |
| 3.8 | `step_3_8_modality` | `workflow/rules/trajectory.smk` |
| 4.1 | `rule render_atlas_figure` + manuscript build outside Snakemake | `workflow/rules/manuscript_figures.smk` |
| 4.2 | (manuscript build outside Snakemake) | n/a |
| 4.3 | (release build outside Snakemake) | n/a |
| 4.4 | (admin) | n/a |

## Appendix B: Dependency DAG

```
Phase 0 gates: 0-T, 0-C, 0-S, 0-X (no dependencies)
              |
              v (all gates pass)
Phase 1: 1.1
       |
       v
       1.2 (1.3, 1.4, 1.5, 1.6 in parallel after 1.1)
       1.3
       1.4 (depends on 1.3)
       1.5 (depends on 1.3)
       1.6 (depends on 1.2)
       1.7 (depends on 1.2, 1.3)
              |
              v
Phase 2.A: 2.A.1, 2.A.2, 2.A.3 (parallel; each depends on 1.1, 1.3)
                       |
                       v
                       2.A.4 (depends on 2.A.1, 2.A.2, 2.A.3)
                       |
                       v
Phase 2.B: 2.B.1 (depends on 2.A.4)
          2.B.2 (depends on 2.A.4)
          2.B.3 (depends on 2.B.1, 2.B.2, 0-S)

Phase 3: 3.0 (depends on 2.A.4)
        3.1 (depends on 3.0)
        3.2 (depends on 3.0, Phase 1)
        3.3 (depends on 3.0, 3.1)
        3.4 (conditional; depends on 1.2, genotype access)
        3.5 (depends on 3.0, 3.2)
        3.6 (depends on 3.2, 3.5)
        3.7 (depends on 1.2, 1.3, 1.4)
        3.8 (depends on 1.7, 3.0, 3.1)

Phase 4: 4.1 (depends on 3.3, 3.7, 3.8, Phase 1)
        4.2 (depends on 2.A.4, 2.B.3)
        4.3 (depends on 2.A.4, 3.0, 3.3)
        4.4 (depends on Phase 0 results, 2.A.4, 3.7)
```

## Appendix C: File-path quick index for the runner

| Need | Path |
|---|---|
| Raw Emory bVals | `analysis/latest/data_emory.parquet` |
| Raw BEST bVals | `analysis/latest/data_best.parquet` |
| Fresh cell fractions Emory | `analysis/latest/cell_props_emory.csv` |
| Fresh cell fractions BEST | `analysis/latest/cell_props_best.csv` |
| Gate 0-T result | `analysis/latest/gate_0T_results.json` |
| Gate 0-C result | `analysis/latest/gate_0C_results.json` |
| Gate 0-S result | `analysis/latest/gate_0S_classifier.json` |
| Gate 0-X result | `analysis/latest/gate_0X_centroids.json` |
| CellDMC Δ | `analysis/latest/celldmc_delta_emory.tsv` |
| RNA-seq DE Δ | `analysis/latest/de_delta_emory.tsv` |
| PROGENy / GSVA | `analysis/latest/pathway_progeny_emory.csv`, `analysis/latest/pathway_gsva_emory.csv` |
| TF activity | `analysis/latest/tf_activity_emory.csv` |
| Regulatory enrichment | `analysis/latest/regulatory_enrichment_emory.tsv` |
| BEST replication | `analysis/latest/replication_overall.tsv` |
| Winning embedding | `analysis/latest/embedding_<winner>.pt` (winner from `embedding_winner.txt`) |
| Leaderboard | `analysis/latest/embedding_leaderboard.csv` |
| Trajectory vectors | `analysis/latest/trajectory_vectors_emory.csv`, `analysis/latest/trajectory_vectors_best.csv` |
| Recovery axis | `analysis/latest/recovery_axis.csv` |
| Archetypes | `analysis/latest/archetypes.csv` |
| Recovery-axis annotation | `analysis/latest/recovery_axis_annotation.csv` |
| Terminus projection | `analysis/latest/terminus_gse98793.csv`, `analysis/latest/terminus_gtex.csv`, `analysis/latest/terminus_test.csv` |
| Mediation catalogue | `analysis/latest/mediation_catalogue.csv` |
| LINCS reversal | `analysis/latest/lincs_top_candidates.csv` |
| Target nominations | `analysis/latest/target_nomination.csv` |
| Headline figure | `manuscript/figures/fig1_trajectory_atlas.png` |
| CQR coverage | `analysis/latest/cqr_coverage_emory.csv`, `analysis/latest/cqr_coverage_best.csv` |
| Direction predictor coverage | `analysis/latest/direction_coverage.csv`, `analysis/latest/direction_coverage_best.csv` |

## Appendix D: How to declare a step done

For each step:

1. Run the step (script or Snakemake rule).
2. Inspect outputs against the "Outputs" section: every listed file should exist.
3. Run the acceptance check: every listed criterion should pass.
4. Log a one-line entry in the daily note (`01-daily/YYYY-MM-DD.md` in the companion vault) under `## Agent activity log`: `- [HH:MM] Step X.Y <name> complete (<owner>) [[detail-artefact]]`.
5. Update the project tracker (currently the relevant task ledger in `08-system/tasks/`) with `status: done` and a one-line outcome.
6. If the step depended on a forward reference (e.g. Step 2.B.2 referencing Step 3.0 recovery axis), refresh the forward-referenced output and re-run the dependent step.

If a step fails its acceptance criterion: do not silently proceed. Document the failure in the step's `Risks / gotchas` follow-up section, escalate to Lee with the failure diagnostic, and decide whether to re-scope or pivot before continuing downstream.

---

## End of ANALYSIS_PLAN.md

Strategic backstop in companion vault: `04-projects/dnamrnaseq/2026-05-17-integrated-analysis-plan-v2.md`. Project overview: `04-projects/dnamrnaseq/PROJECT-OVERVIEW.md`. Repo scaffold record: `04-projects/dnamrnaseq/2026-05-17-repo-init-record.md`.
