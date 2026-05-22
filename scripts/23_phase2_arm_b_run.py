"""Phase 2 Arm B: run MOFA+ trait-state decomposition on the real multi-omics data.

Arm B is the interpretable, linear, statistically identifiable embedding baseline
(design Section 2.2). This script runs it end to end on the genuine Phase 1
feature matrices in ``analysis/latest/`` and writes Arm B's leaderboard entry.

CPU ONLY. MOFA+ does not use the GPU; the GPU is reserved for the MedFict LLM
panel. This script never imports torch and never touches CUDA.

Pipeline (see dnamrnaseq2026.embedding.arm_b_run for detail):
  1. Load Tier 1 DNAm (CV-loop-safe) + Tier 1 RNA candidate matrix. The RNA
     TF panel is selected by variance per MOFA+ fit on the fit's training rows,
     never cohort-wide (leakage fix, Helen Zhao 2026-05-22).
  2. Intersect on SentrixID -> 344 sample-visits, 164 paired subjects.
  3. Residualise sex + age + ancestry PCs out of both views.
  4. Fit MOFA+ (CPU), classify each factor trait/state/mixed via LMM-LRT.
  5. JAK-STAT sensitivity fit (15 outlier sample-visits excluded).
  6. Score Arm B on the leaderboard. Metric (iii) uses the leakage-clean LOSO
     that refits MOFA+ (and re-selects the TF panel) per held-out subject.

Outputs:
  - analysis/latest/arm_b_factor_classification.csv      (primary)
  - analysis/latest/arm_b_factor_classification_sens.csv (JAK-STAT sensitivity)
  - analysis/latest/arm_b_mofa_factor_scores.csv         (per sample-visit)
  - analysis/latest/arm_b_leaderboard_entry.csv          (6-metric column)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dnamrnaseq2026.embedding.arm_b_mofa import MOFAFactors
from dnamrnaseq2026.embedding.arm_b_run import leakage_clean_loso_mae, run_arm_b
from dnamrnaseq2026.embedding.leaderboard import build_leaderboard, score_arm
from dnamrnaseq2026.embedding.real_data import Phase1ArtefactError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LATEST_DIR = Path("analysis/latest")
N_FACTORS = 20
N_BOOTSTRAP = 2000
SEED = 42
# Multiple seeds for the across-seed trajectory-consistency metric (metric i).
SEEDS = (42, 43, 44, 45, 46)


def _state_subspace_delta(
    factors: MOFAFactors,
    classification: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Project subjects onto the state-factor subspace and return (delta_z, subjects).

    Design Section 2.2: the Arm B trajectory unit lives in the state-factor
    subspace. delta_z_i = z_POST_state(i) - z_PRE_state(i). When no factor is
    classified ``state``, falls back to the full factor set so the leaderboard
    metrics still have a subspace to score (logged).
    """
    state_idx = classification.index[classification["classification"] == "state"].tolist()
    if not state_idx:
        logger.warning(
            "Arm B: no state-classified factors; trajectory metrics fall back to "
            "the full %d-factor space",
            factors.n_factors,
        )
        state_idx = list(range(factors.n_factors))

    subjects = np.unique(factors.subject_ids)
    delta = np.zeros((len(subjects), len(state_idx)))
    for si, subj in enumerate(subjects):
        rows = factors.subject_ids == subj
        pre = factors.scores[rows & (factors.visit == 0)][:, state_idx]
        post = factors.scores[rows & (factors.visit == 1)][:, state_idx]
        if pre.size and post.size:
            delta[si] = post.mean(axis=0) - pre.mean(axis=0)
    return delta, subjects


def _responder_and_pcl(
    factors: MOFAFactors,
    subjects: np.ndarray,
    pdata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-subject responder mask + delta-PCL aligned to ``subjects``."""
    by_subject = pdata.groupby("Subcode")
    responder = np.zeros(len(subjects), dtype=bool)
    delta_pcl = np.full(len(subjects), np.nan)
    for si, subj in enumerate(subjects):
        if subj not in by_subject.groups:
            continue
        grp = by_subject.get_group(subj)
        resp = str(grp["Response"].iloc[0]).upper()
        responder[si] = resp in {"R", "1", "RESPONDER"}
        visit_norm = grp["Visit"].astype(str).str.upper()
        pre = grp.loc[visit_norm.str.startswith("PRE"), "PCL_total"]
        post = grp.loc[visit_norm.str.startswith("POST"), "PCL_total"]
        if not pre.empty and not post.empty:
            pcl_pre = pd.to_numeric(pre.iloc[0], errors="coerce")
            pcl_post = pd.to_numeric(post.iloc[0], errors="coerce")
            delta_pcl[si] = float(pcl_post) - float(pcl_pre)
    return responder, delta_pcl


def main() -> None:
    """Run Arm B end to end and write the four output artefacts."""
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    try:
        result = run_arm_b(n_factors=N_FACTORS, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    except Phase1ArtefactError as exc:
        logger.error("Arm B cannot run: %s", exc)
        sys.exit(1)

    factors = result.factors
    classification = result.classification

    # --- Persist factor scores + classification tables --------------------
    score_df = pd.DataFrame(
        factors.scores,
        columns=[f"factor_{k}" for k in range(factors.n_factors)],
    )
    score_df.insert(0, "sentrix_id", result.data.sentrix_ids)
    score_df.insert(1, "subject_id", factors.subject_ids)
    score_df.insert(2, "visit", np.where(factors.visit == 0, "PRE", "POST"))
    score_df.to_csv(LATEST_DIR / "arm_b_mofa_factor_scores.csv", index=False)
    classification.to_csv(LATEST_DIR / "arm_b_factor_classification.csv", index=False)
    result.classification_sensitivity.to_csv(
        LATEST_DIR / "arm_b_factor_classification_sens.csv", index=False
    )

    n_trait = int((classification["classification"] == "trait").sum())
    n_state = int((classification["classification"] == "state").sum())
    n_mixed = int((classification["classification"] == "mixed").sum())
    logger.info(
        "Arm B factor classification (primary): %d trait, %d state, %d mixed",
        n_trait,
        n_state,
        n_mixed,
    )

    # --- Trajectory metrics over the state-factor subspace ----------------
    delta_z, subjects = _state_subspace_delta(factors, classification)

    # Across-seed consistency: re-fit MOFA+ under additional seeds, project each
    # onto the same primary state subspace, collect delta_z per seed.
    delta_z_by_seed: list[np.ndarray] = [delta_z]
    for extra_seed in SEEDS[1:]:
        seed_result = run_arm_b(n_factors=N_FACTORS, n_bootstrap=200, seed=extra_seed)
        seed_delta, _ = _state_subspace_delta(seed_result.factors, classification)
        delta_z_by_seed.append(seed_delta)

    pdata = pd.read_csv(LATEST_DIR / "pdata_emory_with_epidish.csv")
    responder, delta_pcl = _responder_and_pcl(factors, subjects, pdata)

    arm_score = score_arm(
        "arm_b_mofa",
        delta_z=delta_z,
        responder_mask=responder,
        delta_z_by_seed=delta_z_by_seed,
        factors=factors,
        delta_pcl=delta_pcl,
        conformal_result=None,  # metric (iv): downstream calibration step, not in Arm B run
        latent_loadings=None,  # metric (v): gated on Phase 1 enrichment artefacts
        artefact_dir=LATEST_DIR,
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
    )

    # Metric (iii) override: score_arm's LOSO runs LeaveOneOut on a delta_z that
    # came from one cohort-wide MOFA+ fit, so the TF panel saw every held-out
    # subject. Replace it with the leakage-clean LOSO that refits MOFA+ and
    # re-selects the TF panel per held-out subject (Helen Zhao 2026-05-22). This
    # is the only CV-evaluated leaderboard metric and the only one materially
    # exposed to the Tier 1 RNA TF-selection leak.
    clean_loso = leakage_clean_loso_mae(
        result.data, delta_pcl, subjects, n_factors=N_FACTORS, seed=SEED
    )
    contaminated_mae = arm_score.metrics["iii_loso_reconstruction"].get("loso_mae")
    arm_score.metrics["iii_loso_reconstruction"] = dict(clean_loso)
    arm_score.metrics["iii_loso_reconstruction"]["selection"] = "leakage_clean_per_fold_mofa_refit"
    logger.info(
        "Metric (iii) leakage-clean LOSO: MAE %.3f over %d subjects "
        "(contaminated single-fit LOSO was MAE %s)",
        clean_loso["loso_mae"],
        clean_loso["n_subjects"],
        f"{contaminated_mae:.3f}" if isinstance(contaminated_mae, float) else "n/a",
    )

    leaderboard = build_leaderboard([arm_score])
    leaderboard.to_csv(LATEST_DIR / "arm_b_leaderboard_entry.csv")

    logger.info("Arm B leaderboard entry:\n%s", leaderboard.to_string())
    logger.info("Arm B run complete. Artefacts in %s", LATEST_DIR)


if __name__ == "__main__":
    main()
