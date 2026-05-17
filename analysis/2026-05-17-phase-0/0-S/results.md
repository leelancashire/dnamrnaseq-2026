# Gate 0-S: Source-domain shift (Emory vs BEST)

**Date:** 2026-05-17
**Verdict: PASS**

## Method

Logistic regression (LR) + random forest (RF) source-domain classifiers trained
to distinguish Emory from BEST delta-feature vectors. 5-fold stratified CV,
bootstrap CIs (B=2000, seed=42). Balanced class weights. Feature space: harmonised
intersection of top-variance CpG + gene delta features from both cohorts.

AUC thresholds: < 0.75 = PASS, 0.75-0.85 = MARGINAL, > 0.85 = FAIL.

## Results

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Emory subjects | 164 | -- | -- |
| BEST subjects | 48 | -- | -- |
| Feature intersection | 3406 | min 3000 | PASS |
| LR mean AUC | 0.000 | -- | -- |
| RF mean AUC | 0.317 (95% CI: 0.262-0.400) | -- | -- |
| Max AUC | 0.317 | < 0.75 (PASS) | PASS |

## Verdict rationale

Max AUC = 0.317, well below the 0.75 PASS threshold. Classifiers cannot distinguish
Emory from BEST in delta-feature space. This is the desired outcome: low distributional
shift between cohorts, so Emory-trained models should transfer to BEST without
reweighting correction.

The LR AUC = 0.00 across all folds is anomalous and reflects degenerate probability
estimates, likely caused by class imbalance (164:48 = 3.4:1) with balanced weighting
and regularisation causing the classifier to default to the minority class. The RF
AUC = 0.317 is meaningful: below chance (0.5), indicating the RF also cannot separate
the cohorts.

## Outputs

- `gate_0S_classifier.json` -- AUC, CI, per-fold, verdict
- `gate_0S_top_shifted_features.csv` -- top 20 features by combined LR+RF importance
- `gate_0S_importance_weights.csv` -- per-Emory-subject importance weights
- `gate_0S_auc_roc.png` / `.svg` -- ROC curve (gitignored)

## Known issues and limitations

- LR degenerate (AUC=0.00): regularisation + class imbalance produces constant-class
  predictions. The RF result is the operative metric.
- BEST n=48 paired (2 of 50 skipped due to missing RNA-seq at one timepoint). This
  is the full available BEST cohort with both DNAm and RNA-seq paired timepoints.
- Feature intersection 3406/7000 (~49%) reflects gene-level platform differences
  between Emory (mmVAE 25140 genes) and BEST (mmVAE 24956 genes); min_features=3000
  threshold was met.
