#!/usr/bin/env python
"""Gate 0-S entry-point: source-domain classifier (Emory vs BEST shift).

Reads config.yaml, loads both cohorts' delta-feature matrices, trains
logistic regression + random forest classifiers, outputs AUC and verdict.

Outputs (to analysis/2026-05-17-phase-0/0-S/):
  gate_0S_classifier.json           -- AUC, CI, per-fold, verdict
  gate_0S_top_shifted_features.csv  -- top 20 features by classifier importance
  gate_0S_importance_weights.csv    -- per-Emory-subject importance weights (if PASS/MARGINAL)
  gate_0S_auc_roc.png               -- ROC curve
  gate_0S_auc_roc.svg

Usage:
    python scripts/01_phase0_gate_S.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

OUT_DIR = _REPO_ROOT / "analysis/2026-05-17-phase-0/0-S"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Gate 0-S starting.")

    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import (
        load_best_bvals,
        load_best_pdata2,
        load_best_rnaseq,
        load_emory_bvals,
        load_emory_rnaseq,
        load_emory_subject_data,
    )
    from dnamrnaseq2026.preprocessing.covariate_shift import (
        compute_importance_weights,
        determine_gate_0s_verdict,
        harmonise_feature_sets,
        train_source_domain_classifier,
    )
    from dnamrnaseq2026.preprocessing.delta_construction import (
        build_dnam_delta_matrix,
        build_joint_delta_matrix,
        build_rnaseq_delta_matrix,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]
    n_jobs = cfg["run"].get("n_jobs", 4)
    logger.info("Seed=%d, n_jobs=%d", seed, n_jobs)

    # Load Emory data
    logger.info("Loading Emory data...")
    emory_bvals = load_emory_bvals()
    emory_rnaseq = load_emory_rnaseq()
    emory_subject_data = load_emory_subject_data()

    # Load BEST data
    logger.info("Loading BEST data...")
    best_bvals = load_best_bvals()
    best_rnaseq = load_best_rnaseq()
    best_pdata = load_best_pdata2()

    # Build BEST subject data from pData2 (BL/12W -> PRE/POST mapping)
    import pandas as pd

    # BEST pData2 has Visit 'BL' (baseline) and '12W' (12 weeks) per Day-0 check.
    # Map to PRE-IOP / POST-IOP equivalent for delta construction.
    # Build a synthetic subject_data frame for BEST mimicking the Emory format.
    # Identify paired BEST subjects (those with both BL and 12W)
    if "Visit" not in best_pdata.columns:
        logger.warning("BEST pData2 has no Visit column; cannot construct BEST delta matrix.")
        logger.warning("Gate 0-S requires both cohorts. Skipping.")
        sys.exit(1)

    best_subject_rows = []
    # BL = baseline (PRE), 12W = 12-week follow-up (POST)
    visit_map = {"BL": "PRE-IOP", "12W": "POST-IOP"}
    # Map numeric Response coding: 1.0=R, 2.0=NR, 3.0=NA
    response_map = {"1.0": "R", "2.0": "NR", "3.0": "NA", "1": "R", "2": "NR", "3": "NA"}

    for sample_name, row in best_pdata.iterrows():
        raw_visit = str(row.get("Visit", ""))
        visit_mapped = visit_map.get(raw_visit, raw_visit)
        # pData2 has a Subcode column; use it directly rather than parsing the
        # array barcode (207944480128_R02C01) which contains no visit suffix.
        subcode = str(row.get("Subcode", sample_name))
        # BEST DNAm bVals columns use the array barcode (pData2 index).
        dnam_sample = str(sample_name)
        # BEST RNA-seq columns use {Subcode}-{BL|12W} format,
        # e.g. BEST-307964-BL (Subcode already contains the BEST prefix).
        rnaseq_sample = f"{subcode}-{raw_visit}"
        best_subject_rows.append(
            {
                "Subcode": subcode,
                "Visit": visit_mapped,
                "Response": response_map.get(str(row.get("Response", "NA")), "NA"),
                "SampleName_DNAm": dnam_sample,
                "SampleName_RNASeq": rnaseq_sample,
            }
        )
    best_subject_data = pd.DataFrame(best_subject_rows)

    # Check BEST RNA-seq column format matches subject data
    logger.info("BEST RNA-seq columns sample: %s", list(best_rnaseq.columns[:3]))
    logger.info(
        "BEST subject data SampleName_RNASeq sample: %s",
        list(best_subject_data["SampleName_RNASeq"][:3]),
    )

    # Build Emory delta matrix
    emory_dnam_delta = build_dnam_delta_matrix(emory_bvals, emory_subject_data, top_n_cpgs=5000)
    emory_rna_delta = build_rnaseq_delta_matrix(emory_rnaseq, emory_subject_data, top_n_genes=2000)
    emory_joint = build_joint_delta_matrix(emory_dnam_delta, emory_rna_delta, scale=True)
    logger.info("Emory joint delta: %s", emory_joint.shape)

    # Build BEST delta matrix
    best_dnam_delta = build_dnam_delta_matrix(best_bvals, best_subject_data, top_n_cpgs=5000)
    best_rna_delta = build_rnaseq_delta_matrix(best_rnaseq, best_subject_data, top_n_genes=2000)
    best_joint = build_joint_delta_matrix(best_dnam_delta, best_rna_delta, scale=True)
    logger.info("BEST joint delta: %s", best_joint.shape)

    if len(best_joint) == 0:
        logger.error(
            "BEST delta matrix is empty. Possible causes: "
            "RNA-seq column names do not match SampleName_RNASeq in subject data, "
            "or no paired subjects found. "
            "Inspect best_rnaseq.columns vs best_subject_data.SampleName_RNASeq."
        )
        logger.info("BEST rnaseq cols: %s", list(best_rnaseq.columns[:10]))
        logger.info(
            "BEST subject SampleName_RNASeq: %s",
            list(best_subject_data["SampleName_RNASeq"].unique()[:10]),
        )
        sys.exit(1)

    # Harmonise feature intersection
    emory_h, best_h = harmonise_feature_sets(emory_joint, best_joint, min_features=3000)
    logger.info("After harmonisation: Emory=%s, BEST=%s", emory_h.shape, best_h.shape)

    # Train classifiers
    clf_results = train_source_domain_classifier(emory_h, best_h, seed=seed, n_jobs=n_jobs)
    max_auc = max(clf_results["lr_mean_auc"], clf_results["rf_mean_auc"])
    verdict = determine_gate_0s_verdict(max_auc)

    logger.info(
        "Gate 0-S verdict: %s (LR AUC=%.4f, RF AUC=%.4f)",
        verdict,
        clf_results["lr_mean_auc"],
        clf_results["rf_mean_auc"],
    )

    # Compute importance weights if pass/marginal
    importance_weights = None
    if verdict in {"PASS", "MARGINAL"}:
        importance_weights = compute_importance_weights(clf_results)

    # Write results JSON
    out = {
        "gate": "0-S",
        "verdict": verdict,
        "lr_mean_auc": float(clf_results["lr_mean_auc"]),
        "rf_mean_auc": float(clf_results["rf_mean_auc"]),
        "max_auc": float(max_auc),
        "lr_per_fold_auc": clf_results["lr_per_fold_auc"],
        "rf_per_fold_auc": clf_results["rf_per_fold_auc"],
        "lr_ci_95": list(clf_results["lr_ci_95"]),
        "rf_ci_95": list(clf_results["rf_ci_95"]),
        "n_features": clf_results["n_features"],
        "n_emory": clf_results["n_emory"],
        "n_best": clf_results["n_best"],
        "seed": seed,
    }
    with (OUT_DIR / "gate_0S_classifier.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    # Top 20 shifted features (max of LR + RF importance)
    combined_importance = (
        clf_results["lr_coef"].abs() / clf_results["lr_coef"].abs().sum()
        + clf_results["rf_importances"] / clf_results["rf_importances"].sum()
    )
    top20 = combined_importance.nlargest(20)
    top20_df = pd.DataFrame(
        {
            "feature": top20.index,
            "combined_importance": top20.values,
            "lr_coef": clf_results["lr_coef"][top20.index].values,
            "rf_importance": clf_results["rf_importances"][top20.index].values,
        }
    )
    top20_df.to_csv(str(OUT_DIR / "gate_0S_top_shifted_features.csv"), index=False)

    # Importance weights CSV
    if importance_weights is not None:
        importance_weights.to_csv(str(OUT_DIR / "gate_0S_importance_weights.csv"), header=True)

    # ROC curve plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    feature_matrix = clf_results["feature_matrix"]
    labels = clf_results["labels"]
    lr = clf_results["fitted_lr"]
    proba_lr = lr.predict_proba(feature_matrix.values)[:, 1]

    fpr, tpr, _ = roc_curve(labels, proba_lr)
    auc_val = clf_results["lr_mean_auc"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"LR (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        f"Gate 0-S: Source-domain classifier ROC\n"
        f"(Emory=0, BEST=1)\nVerdict: {verdict} (max AUC={max_auc:.3f})"
    )
    ax.legend(loc="lower right")
    fig.savefig(str(OUT_DIR / "gate_0S_auc_roc.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(OUT_DIR / "gate_0S_auc_roc.svg"), bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print()
    print("=" * 60)
    print("Gate 0-S: Source-domain shift (Emory vs BEST)")
    print("=" * 60)
    print(f"Emory subjects: {out['n_emory']}, BEST subjects: {out['n_best']}")
    print(f"Feature intersection: {out['n_features']}")
    lr_lo, lr_hi = out["lr_ci_95"]
    rf_lo, rf_hi = out["rf_ci_95"]
    print(f"LR mean AUC: {out['lr_mean_auc']:.4f} (95% CI: {lr_lo:.3f}-{lr_hi:.3f})")
    print(f"RF mean AUC: {out['rf_mean_auc']:.4f} (95% CI: {rf_lo:.3f}-{rf_hi:.3f})")
    print(f"Max AUC: {out['max_auc']:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
