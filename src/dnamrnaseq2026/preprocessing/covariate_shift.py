"""Source-domain classifier for Gate 0-S: Emory vs BEST covariate-shift audit.

Trains a binary classifier on the joint delta-feature space to predict
whether a subject is from Emory or BEST. The cross-validated AUC quantifies
covariate-shift severity.

Acceptance thresholds (ANALYSIS_PLAN.md Step 0-S):
  - PASS: AUC < 0.75 (tractable shift, standard importance weighting)
  - MARGINAL: 0.75 <= AUC <= 0.85 (truncate weights at 99th pct)
  - FAIL: AUC > 0.85 (shift too severe; Phase 2.B reverts to Emory-only)

Analysis plan reference: ANALYSIS_PLAN.md Step 0-S.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)

# Thresholds (ANALYSIS_PLAN.md Step 0-S acceptance criteria)
AUC_PASS_THRESHOLD = 0.75
AUC_MARGINAL_THRESHOLD = 0.85

# Bootstrap parameters (TPV cluster-bootstrap default, ANALYSIS_PLAN.md note)
BOOTSTRAP_N = 2000
IMPORTANCE_WEIGHT_PERCENTILE = 99.0


def harmonise_feature_sets(
    emory_delta: pd.DataFrame,
    best_delta: pd.DataFrame,
    min_features: int = 3000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Intersect Emory and BEST delta feature columns.

    Per ANALYSIS_PLAN.md Step 0-S Method: use intersection of features;
    if intersection < min_features, expand the variance filter (warning only).

    Parameters
    ----------
    emory_delta:
        Emory paired-delta matrix, shape (n_emory, n_features_emory).
    best_delta:
        BEST paired-delta matrix, shape (n_best, n_features_best).
    min_features:
        Minimum acceptable intersection size.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Both matrices restricted to their column intersection.
    """
    shared_cols = emory_delta.columns.intersection(best_delta.columns)
    n_shared = len(shared_cols)
    logger.info(
        "Feature intersection: %d (Emory=%d, BEST=%d)",
        n_shared,
        len(emory_delta.columns),
        len(best_delta.columns),
    )
    if n_shared < min_features:
        logger.warning(
            "Feature intersection (%d) < min_features (%d). "
            "Consider expanding variance filters before intersecting.",
            n_shared,
            min_features,
        )
    return emory_delta[shared_cols], best_delta[shared_cols]


def train_source_domain_classifier(
    emory_delta: pd.DataFrame,
    best_delta: pd.DataFrame,
    seed: int = 42,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Train source-domain classifiers and compute cross-validated AUC.

    Two models per ANALYSIS_PLAN.md: logistic regression (linear shift) +
    random forest (non-linear shift). Reports per-fold AUC and bootstrap CI.

    Parameters
    ----------
    emory_delta:
        Emory delta-feature matrix (already harmonised to BEST features).
    best_delta:
        BEST delta-feature matrix (already harmonised to Emory features).
    seed:
        Random seed for reproducibility.
    n_jobs:
        Parallel workers for CV and RF.

    Returns
    -------
    dict
        Keys: 'lr_mean_auc', 'rf_mean_auc', 'lr_per_fold_auc',
        'rf_per_fold_auc', 'lr_ci_95', 'rf_ci_95', 'n_features',
        'n_emory', 'n_best', 'lr_coef', 'rf_importances'.
    """
    # Label: 0 = Emory, 1 = BEST (lowercase per N806 convention)
    feature_matrix = pd.concat([emory_delta, best_delta], axis=0)
    labels = np.array([0] * len(emory_delta) + [1] * len(best_delta))

    logger.info(
        "Source-domain classifier: n=%d (Emory=%d, BEST=%d), features=%d",
        len(feature_matrix),
        len(emory_delta),
        len(best_delta),
        feature_matrix.shape[1],
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Logistic regression with balanced class weight
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    lr_proba = cross_val_predict(
        lr, feature_matrix.values, labels, cv=cv, method="predict_proba", n_jobs=n_jobs
    )
    lr_per_fold_auc = []
    for _, test_idx in cv.split(feature_matrix.values, labels):
        fold_auc = roc_auc_score(labels[test_idx], lr_proba[test_idx, 1])
        lr_per_fold_auc.append(float(fold_auc))
    lr_mean_auc = float(np.mean(lr_per_fold_auc))

    # Random forest
    rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=n_jobs
    )
    rf_proba = cross_val_predict(
        rf, feature_matrix.values, labels, cv=cv, method="predict_proba", n_jobs=n_jobs
    )
    rf_per_fold_auc = []
    for _, test_idx in cv.split(feature_matrix.values, labels):
        fold_auc = roc_auc_score(labels[test_idx], rf_proba[test_idx, 1])
        rf_per_fold_auc.append(float(fold_auc))
    rf_mean_auc = float(np.mean(rf_per_fold_auc))

    logger.info("LR mean AUC: %.4f, RF mean AUC: %.4f", lr_mean_auc, rf_mean_auc)

    # Bootstrap CIs on fold AUCs
    rng = np.random.default_rng(seed)
    lr_boot = [
        float(np.mean(rng.choice(lr_per_fold_auc, size=5, replace=True)))
        for _ in range(BOOTSTRAP_N)
    ]
    rf_boot = [
        float(np.mean(rng.choice(rf_per_fold_auc, size=5, replace=True)))
        for _ in range(BOOTSTRAP_N)
    ]
    lr_ci = (float(np.percentile(lr_boot, 2.5)), float(np.percentile(lr_boot, 97.5)))
    rf_ci = (float(np.percentile(rf_boot, 2.5)), float(np.percentile(rf_boot, 97.5)))

    # Fit final LR for coefficients + importance weights
    lr.fit(feature_matrix.values, labels)
    lr_coef = pd.Series(lr.coef_[0], index=feature_matrix.columns)

    # Fit final RF for feature importances
    rf.fit(feature_matrix.values, labels)
    rf_importances = pd.Series(rf.feature_importances_, index=feature_matrix.columns)

    return {
        "lr_mean_auc": lr_mean_auc,
        "rf_mean_auc": rf_mean_auc,
        "lr_per_fold_auc": lr_per_fold_auc,
        "rf_per_fold_auc": rf_per_fold_auc,
        "lr_ci_95": lr_ci,
        "rf_ci_95": rf_ci,
        "n_features": int(feature_matrix.shape[1]),
        "n_emory": len(emory_delta),
        "n_best": len(best_delta),
        "lr_coef": lr_coef,
        "rf_importances": rf_importances,
        "fitted_lr": lr,
        "feature_matrix": feature_matrix,
        "labels": labels,
    }


def compute_importance_weights(
    classifier_results: dict[str, Any],
    percentile_cap: float = IMPORTANCE_WEIGHT_PERCENTILE,
) -> pd.Series:
    """Compute per-Emory-subject importance weights for conformal calibration.

    Importance weight = Pr(source=BEST|x) / Pr(source=Emory|x) per
    ANALYSIS_PLAN.md Step 0-S Method step 6. Weights are truncated at
    percentile_cap to control variance.

    Parameters
    ----------
    classifier_results:
        Output of train_source_domain_classifier.
    percentile_cap:
        Percentile for weight truncation.

    Returns
    -------
    pd.Series
        Importance weights for each Emory subject (indexed by subject).
    """
    lr = classifier_results["fitted_lr"]
    feature_matrix: pd.DataFrame = classifier_results["feature_matrix"]
    n_emory = classifier_results["n_emory"]

    proba = lr.predict_proba(feature_matrix.values[:n_emory])
    # Classes: 0=Emory, 1=BEST. Index 1 = Pr(BEST), index 0 = Pr(Emory)
    pr_best = proba[:, 1]
    pr_emory = proba[:, 0]
    weights = pr_best / np.maximum(pr_emory, 1e-8)

    # Truncate at percentile_cap
    cap = float(np.percentile(weights, percentile_cap))
    weights_capped = np.minimum(weights, cap)
    logger.info(
        "Importance weights: mean=%.3f, max_pre_cap=%.3f, cap(p%.0f)=%.3f, max_post_cap=%.3f",
        float(weights.mean()),
        float(weights.max()),
        percentile_cap,
        cap,
        float(weights_capped.max()),
    )

    emory_index = feature_matrix.index[:n_emory]
    return pd.Series(weights_capped, index=emory_index, name="importance_weight")


def determine_gate_0s_verdict(mean_auc: float) -> str:
    """Return PASS, MARGINAL, or FAIL verdict for Gate 0-S.

    Uses the maximum of LR and RF AUC to be conservative.

    Parameters
    ----------
    mean_auc:
        Mean cross-validated AUC (use max of LR and RF).

    Returns
    -------
    str
        'PASS', 'MARGINAL', or 'FAIL'.
    """
    if mean_auc < AUC_PASS_THRESHOLD:
        return "PASS"
    if mean_auc <= AUC_MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "FAIL"
