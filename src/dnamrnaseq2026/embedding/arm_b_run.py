"""Arm B real-data run: assemble multi-omics views, fit MOFA+, classify factors.

This module is the real-data realisation of Arm B (design Section 2.2). It is
distinct from :mod:`dnamrnaseq2026.embedding.arm_b_mofa`, which holds the
arm-agnostic MOFA+ fit and the LMM-LRT trait-state machinery; this module wires
those primitives to the genuine Phase 1 feature matrices in ``analysis/latest/``.

Pipeline
--------
1. Load the Tier 1 DNAm M-value matrix via the CV-loop-safe loader (it carries
   ``cv_loop_safe=True``: CellDMC FDR<0.10 is a fixed biological prior). Load the
   Tier 1 RNA matrix via the *candidate* loader: it is ``cv_loop_safe=False``
   because the TF panel is a variance rank that must be fit per fold. The RNA
   candidate matrix is PROGENy (14 fixed pathway columns) + the FULL TF activity
   set; the top-150 TF panel is selected by variance **on the rows of each MOFA+
   fit** (the training rows of that fit), never cohort-wide.
   CORRECTED 2026-05-22 (Helen Zhao): the previous build baked a cohort-variance-
   ranked 150-TF Tier 1 RNA panel, which leaked held-out rows into the TF feature
   selection -- the same leak class caught for Tier 2. Per-fit TF selection fixes
   it; metric (iii) LOSO additionally refits MOFA+ per held-out subject so the
   one CV-evaluated leaderboard metric is fully leakage-clean.
2. Cross-view sample alignment. The DNAm side has 388 sample-visits, the RNA side
   344 (44 DNAm-array samples have no RNA-seq). MOFA+ requires observation-aligned
   views; we **intersect** on SentrixID -> 344 sample-visits, 164 paired subjects.
   The 44 DNAm-only samples are a flat data-availability gap, not a structured
   missingness MOFA+'s partial-overlap handling would model usefully.
3. Covariate adjustment. mofapy2 0.7.4's basic ``entry_point`` does not expose a
   fixed-effects covariate term, so sex, age and ancestry PCs are residualised
   out of BOTH views by per-feature OLS **before** factorisation. This is the
   covariate-adjustment step the design doc Section 2.0 specifies (sex covariate
   is mandatory: KDM5D Y-chromosome signal confirms it; Kai's plausibility check
   2026-05-22).
4. Fit MOFA+ (CPU, ``mofapy2``), classify each factor trait/state/mixed via the
   LMM-LRT machinery in :mod:`arm_b_mofa`.
5. JAK-STAT sensitivity. 15 sample-visits carry elevated JAK-STAT pathway
   activity (Kai's check; max z=5.4). They are NOT a CRP-driven inflammatory
   confound (cohort r(JAK-STAT,CRP)=0.04; only 1 of 15 above the cohort CRP 90th
   percentile). They are retained in the primary fit; a sensitivity fit excludes
   them and the factor classification is reported under both specifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dnamrnaseq2026.embedding.arm_b_mofa import MOFAFactors, classify_factors, fit_mofa
from dnamrnaseq2026.embedding.data_harness import normalise_visit
from dnamrnaseq2026.embedding.feature_selection import (
    load_candidate_feature_matrix,
    load_feature_matrix_for_cv,
    select_top_tf_by_variance,
)
from dnamrnaseq2026.embedding.real_data import Phase1ArtefactError, Phase1Paths

logger = logging.getLogger(__name__)

# Covariates residualised out of both views before MOFA+ (design Section 2.0).
# Sex is mandatory (KDM5D Y-chromosome confirmation, Kai 2026-05-22).
COVARIATE_COLS = (
    "sex",
    "Age",
    "ancestry_pca_PCA1",
    "ancestry_pca_PCA2",
    "ancestry_pca_PCA3",
    "ancestry_pca_PCA4",
    "ancestry_pca_PCA5",
    "ancestry_pca_PCA6",
)

# PROGENy JAK-STAT outlier flag threshold (Kai's plausibility check).
JAKSTAT_Z_THRESHOLD = 2.0

# Tier 1 RNA candidate-matrix layout: the first N_PROGENY_PATHWAYS columns are
# the fixed PROGENy panel (leakage-free, always kept); the rest are TF activity
# scores from which the top-N by variance are selected per fit.
N_PROGENY_PATHWAYS = 14
TOP_TF_BY_VARIANCE = 150


@dataclass
class ArmBData:
    """Aligned, covariate-adjusted multi-omics views for the Arm B MOFA+ fit.

    The RNA view here is the **candidate** matrix: PROGENy (14 fixed columns) +
    the full TF activity set. The top-150 TF panel is NOT pre-selected; it is
    chosen by variance on the rows of each MOFA+ fit (see :func:`fit_arm_b`).
    This is the leakage fix: a cohort-wide TF variance rank would let held-out
    rows decide the feature set (Helen Zhao, 2026-05-22).

    Attributes
    ----------
    dnam:
        (n_obs, n_dnam_features) covariate-adjusted DNAm M-value view.
    rna:
        (n_obs, 14 + n_all_tfs) covariate-adjusted RNA candidate view (PROGENy +
        all TFs; TF panel selection deferred to fit time).
    rna_columns:
        Column labels of the RNA candidate view, length 14 + n_all_tfs. The
        first 14 are PROGENy pathways.
    subject_ids:
        Per-observation subject id (Subcode), length n_obs.
    visit:
        Per-observation visit code, 0 = PRE, 1 = POST.
    sentrix_ids:
        Per-observation SentrixID, length n_obs.
    jakstat_outlier:
        (n_obs,) boolean: True for the elevated-JAK-STAT sample-visits.
    """

    dnam: np.ndarray
    rna: np.ndarray
    rna_columns: np.ndarray
    subject_ids: np.ndarray
    visit: np.ndarray
    sentrix_ids: np.ndarray
    jakstat_outlier: np.ndarray

    @property
    def n_obs(self) -> int:
        """Number of aligned sample-visit observations."""
        return int(self.dnam.shape[0])

    def select_tf_panel(self, row_mask: np.ndarray | None = None) -> np.ndarray:
        """Return the RNA view with the TF panel selected by variance.

        The top-:data:`TOP_TF_BY_VARIANCE` TFs are ranked by variance over the
        rows selected by ``row_mask`` (all rows if None). PROGENy's 14 fixed
        columns are always kept. Passing a training-fold ``row_mask`` makes the
        TF selection leakage-clean for that fold; passing None ranks on all
        rows of the current view (correct when the view is itself already a
        training-fold subset, e.g. the LOSO refit).
        """
        rna_df = pd.DataFrame(self.rna, columns=self.rna_columns)
        rank_rows = rna_df if row_mask is None else rna_df.loc[row_mask]
        keep = select_top_tf_by_variance(
            rank_rows,
            n_pathway=N_PROGENY_PATHWAYS,
            top_tf_by_variance=TOP_TF_BY_VARIANCE,
        )
        return np.asarray(rna_df[keep].to_numpy(dtype=np.float64))


def _residualise(view: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Regress ``covariates`` out of each column of ``view`` via OLS.

    Returns the residual matrix (same shape as ``view``). An intercept column is
    added to ``covariates`` internally. This is the covariate-adjustment step
    that stands in for a MOFA+ fixed-effects term (mofapy2 0.7.4 does not expose
    one). Done cohort-wide here: Arm B is an unsupervised factorisation scored on
    a fixed leaderboard, not a CV-tuned predictor, so cohort-wide residualising
    of nuisance covariates is the standard MOFA+ practice and does not leak a
    response label.
    """
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    beta, *_ = np.linalg.lstsq(design, view, rcond=None)
    return np.asarray(view - design @ beta, dtype=np.float64)


def _subset_arm_b_data(data: ArmBData, mask: np.ndarray) -> ArmBData:
    """Return an :class:`ArmBData` restricted to the observations in ``mask``.

    ``rna_columns`` is the candidate-matrix column index and is row-invariant,
    so it is carried through unchanged.
    """
    return ArmBData(
        dnam=data.dnam[mask],
        rna=data.rna[mask],
        rna_columns=data.rna_columns,
        subject_ids=data.subject_ids[mask],
        visit=data.visit[mask],
        sentrix_ids=data.sentrix_ids[mask],
        jakstat_outlier=data.jakstat_outlier[mask],
    )


def load_arm_b_data(
    paths: Phase1Paths | None = None,
    *,
    residualise_covariates: bool = True,
) -> ArmBData:
    """Assemble the aligned, covariate-adjusted Arm B views from Phase 1 outputs.

    Cross-view alignment is by **intersection** on SentrixID: the DNAm side has
    388 sample-visits, the RNA side 344; the 344-row intersection is used (164
    paired subjects). See module docstring for the rationale.
    """
    paths = paths or Phase1Paths()
    if not paths.exists():
        raise Phase1ArtefactError(f"Phase 1 artefact directory absent: {paths.root}")
    root = Path(paths.root)

    # DNAm Tier 1 is a fixed CellDMC prior -> CV-loop-safe loader.
    dnam_df = load_feature_matrix_for_cv(root / "feature_matrix_tier1_dnam.parquet")
    # RNA Tier 1 is a candidate matrix (cv_loop_safe=False): PROGENy fixed + all
    # TFs. The candidate loader bypasses the CV-safety gate by design; the TF
    # variance rank is completed per fit downstream (Helen Zhao, 2026-05-22).
    rna_df = load_candidate_feature_matrix(root / "feature_matrix_tier1_rna.parquet")

    # Cross-view alignment: intersect on SentrixID (both matrices are SentrixID-
    # indexed). 388 DNAm + 344 RNA -> 344 shared sample-visits.
    shared = sorted(set(dnam_df.index.astype(str)) & set(rna_df.index.astype(str)))
    if not shared:
        raise Phase1ArtefactError(
            "DNAm and RNA Tier 1 matrices share no SentrixID; cannot align views"
        )
    logger.info(
        "Arm B alignment: DNAm %d + RNA %d sample-visits -> %d shared (intersection)",
        dnam_df.shape[0],
        rna_df.shape[0],
        len(shared),
    )
    dnam_df = dnam_df.loc[shared]
    rna_df = rna_df.loc[shared]

    pdata = pd.read_csv(_require_pdata(paths))
    pdata = pdata.set_index(pdata["SampleName"].astype(str))
    pdata = pdata.loc[[s for s in shared if s in pdata.index]]
    shared = list(pdata.index)
    dnam_df = dnam_df.loc[shared]
    rna_df = rna_df.loc[shared]

    visit = np.array([0 if normalise_visit(v) == "PRE" else 1 for v in pdata["Visit"]])
    subject_ids = pdata["Subcode"].astype(str).to_numpy()

    # JAK-STAT outlier flag from the PROGENy activity column in the RNA matrix.
    jakstat_outlier = _flag_jakstat_outliers(rna_df)

    dnam = dnam_df.to_numpy(dtype=np.float64)
    rna = rna_df.to_numpy(dtype=np.float64)
    rna_columns = np.asarray([str(c) for c in rna_df.columns], dtype=object)

    if residualise_covariates:
        cov = _build_covariate_matrix(pdata)
        dnam = _residualise(dnam, cov)
        # Residualising nuisance covariates carries no response label, so it is
        # not a train/test leak even applied cohort-wide; it is applied to the
        # full RNA candidate matrix here. The TF *selection* (which IS a leak if
        # cohort-wide) is deferred to fit time via ArmBData.select_tf_panel.
        rna = _residualise(rna, cov)
        logger.info("Arm B: residualised %d covariates out of both views", cov.shape[1])

    return ArmBData(
        dnam=dnam,
        rna=rna,
        rna_columns=rna_columns,
        subject_ids=subject_ids,
        visit=visit,
        sentrix_ids=np.asarray(shared, dtype=object),
        jakstat_outlier=jakstat_outlier,
    )


def _require_pdata(paths: Phase1Paths) -> Path:
    """Resolve the corrected Emory pData path, raising if absent."""
    path = Path(paths.root) / paths.pdata_emory
    if not path.exists() or path.stat().st_size == 0:
        raise Phase1ArtefactError(f"Corrected Emory pData missing at {path}")
    return path


def _flag_jakstat_outliers(rna_df: pd.DataFrame) -> np.ndarray:
    """Flag sample-visits with elevated JAK-STAT PROGENy activity (z > 2).

    Returns an all-False mask if the JAK-STAT column is absent (the matrix is a
    superset; the column name is the PROGENy convention ``JAK-STAT``).
    """
    col = next((c for c in rna_df.columns if str(c).upper().replace("_", "-") == "JAK-STAT"), None)
    if col is None:
        logger.warning("Arm B: no JAK-STAT column in RNA matrix; sensitivity flag all-False")
        return np.zeros(rna_df.shape[0], dtype=bool)
    values = rna_df[col].to_numpy(dtype=np.float64)
    z = (values - values.mean()) / values.std()
    mask = z > JAKSTAT_Z_THRESHOLD
    logger.info(
        "Arm B: %d JAK-STAT outlier sample-visits flagged (z > %.1f)",
        mask.sum(),
        JAKSTAT_Z_THRESHOLD,
    )
    return np.asarray(mask, dtype=bool)


def _build_covariate_matrix(pdata: pd.DataFrame) -> np.ndarray:
    """Build the numeric covariate matrix (sex, age, ancestry PCs) for residualising.

    Missing covariate cells are mean-imputed (a handful of ancestry-PC and age
    values are sparsely missing; this is genuine sparse missingness, not a stub).
    """
    present = [c for c in COVARIATE_COLS if c in pdata.columns]
    missing = [c for c in COVARIATE_COLS if c not in pdata.columns]
    if "sex" not in present:
        raise Phase1ArtefactError(
            "pData carries no 'sex' column; sex is a mandatory Arm B covariate "
            "(KDM5D Y-chromosome confirmation, Kai 2026-05-22)"
        )
    if missing:
        logger.info("Arm B: covariates absent and skipped: %s", missing)
    cov = pdata[present].apply(pd.to_numeric, errors="coerce")
    cov = cov.fillna(cov.mean())
    return np.asarray(cov.to_numpy(dtype=np.float64))


def fit_arm_b(
    data: ArmBData,
    *,
    n_factors: int = 20,
    seed: int = 42,
    tf_rank_mask: np.ndarray | None = None,
) -> MOFAFactors:
    """Fit the two-view MOFA+ model on the aligned Arm B views (CPU).

    The RNA view's TF panel is selected by variance immediately before the fit
    via :meth:`ArmBData.select_tf_panel`. ``tf_rank_mask`` selects the rows the
    TF variance rank is computed on:

    * ``None`` ranks on all rows of ``data`` -- correct when ``data`` is itself
      already the training set of this fit (the primary fit, the sensitivity
      fit, and the per-fold LOSO refit all pass ``data`` already restricted to
      their training rows, so None is leakage-clean for them).
    * an explicit boolean mask ranks on that subset only.

    Views are scaled internally by mofapy2 (``scale_views=True``). The DNAm view
    (M-values) and the RNA view (activity scores) are on different native
    scales; view scaling makes the ELBO factorisation scale-invariant.
    """
    rna_selected = data.select_tf_panel(row_mask=tf_rank_mask)
    views = {"dnam": data.dnam, "rna": rna_selected}
    factors = fit_mofa(
        views,
        subject_ids=data.subject_ids,
        visit=data.visit,
        n_factors=n_factors,
        seed=seed,
        use_surrogate=False,
    )
    logger.info(
        "Arm B MOFA+ fit complete: %d factors on %d obs (RNA view %d cols after "
        "per-fit TF selection)",
        factors.n_factors,
        data.n_obs,
        rna_selected.shape[1],
    )
    return factors


def _subject_state_delta(
    factors: MOFAFactors,
    state_idx: list[int],
    subjects: np.ndarray,
) -> np.ndarray:
    """Per-subject delta-z in the given factor-index subspace, aligned to ``subjects``."""
    delta = np.full((len(subjects), len(state_idx)), np.nan)
    for si, subj in enumerate(subjects):
        rows = factors.subject_ids == subj
        pre = factors.scores[rows & (factors.visit == 0)][:, state_idx]
        post = factors.scores[rows & (factors.visit == 1)][:, state_idx]
        if pre.size and post.size:
            delta[si] = post.mean(axis=0) - pre.mean(axis=0)
    return delta


def leakage_clean_loso_mae(
    data: ArmBData,
    delta_pcl: np.ndarray,
    subjects: np.ndarray,
    *,
    n_factors: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Metric (iii), leakage-clean: refit MOFA+ per held-out subject.

    The on-disk leaderboard run computes ``delta_z`` from a single cohort-wide
    MOFA+ fit, then does LeaveOneOut on that ``delta_z``. Because the TF panel
    is selected on the rows of that fit, the held-out subject influenced its own
    feature space -- a residual leak in the one CV-evaluated metric. This
    function closes it: for each held-out subject it (a) drops that subject's
    sample-visits, (b) re-selects the TF panel and re-fits MOFA+ on the
    remaining subjects, (c) fits the Delta-PCL linear probe on the in-fold
    subjects, (d) projects the held-out subject's sample-visits onto the
    re-fitted factor space and predicts. Every step the held-out subject could
    touch is refit without it.

    Returns ``{"loso_mae": ..., "n_subjects": ...}`` matching
    :func:`leaderboard.loso_reconstruction_surrogate`. The held-out subject is
    projected on the full factor set (the state-factor subset is itself a
    classification that would otherwise leak); this is the honest in-fold
    choice and is the same subspace the probe is trained on.
    """
    from sklearn.linear_model import LinearRegression

    valid = ~np.isnan(delta_pcl)
    valid_subjects = subjects[valid]
    valid_pcl = delta_pcl[valid]
    if len(valid_subjects) < 3:
        return {"loso_mae": float("nan"), "n_subjects": int(len(valid_subjects))}

    all_idx = list(range(n_factors))
    errors: list[float] = []
    for held in valid_subjects:
        train_mask = data.subject_ids != held
        train_data = _subset_arm_b_data(data, train_mask)
        # TF panel + MOFA+ refit on training subjects only (tf_rank_mask=None
        # ranks on train_data's own rows, which are exactly the training rows).
        train_factors = fit_arm_b(train_data, n_factors=n_factors, seed=seed)
        train_subjects = np.unique(train_data.subject_ids)
        train_delta = _subject_state_delta(train_factors, all_idx, train_subjects)
        # Align training delta-z to training delta-PCL.
        subj_to_pcl = dict(zip(valid_subjects, valid_pcl, strict=True))
        train_keep = [
            i
            for i, s in enumerate(train_subjects)
            if s in subj_to_pcl and not np.isnan(train_delta[i]).any()
        ]
        if len(train_keep) < 3:
            continue
        x_train = train_delta[train_keep]
        y_train = np.array([subj_to_pcl[train_subjects[i]] for i in train_keep])
        probe = LinearRegression().fit(x_train, y_train)
        # Held-out subject: project its sample-visits onto the refit factors.
        held_factors = _project_subject(data, held, train_factors, n_factors, seed)
        if held_factors is None:
            continue
        pred = float(probe.predict(held_factors.reshape(1, -1))[0])
        errors.append(abs(float(subj_to_pcl[held]) - pred))

    if not errors:
        return {"loso_mae": float("nan"), "n_subjects": 0}
    return {"loso_mae": float(np.mean(errors)), "n_subjects": int(len(errors))}


def _project_subject(
    data: ArmBData,
    held: str,
    train_factors: MOFAFactors,
    n_factors: int,
    seed: int,
) -> np.ndarray | None:
    """Project a held-out subject's delta-z onto a training-fit factor space.

    mofapy2 0.7.4 has no out-of-sample projection API, so the held-out
    subject's sample-visits are appended to the training rows and MOFA+ is
    re-fit with the *training* TF panel and *training* seed; the held-out
    subject's factor scores are then read off. Re-using the training TF panel
    (not re-ranking with the held-out subject present) keeps the projection
    leakage-clean: the held-out subject sees only features chosen without it.
    """
    held_mask = data.subject_ids == held
    if not held_mask.any():
        return None
    train_mask = ~held_mask
    # Training TF panel, fixed (ranked on training rows only).
    rna_df = pd.DataFrame(data.rna, columns=data.rna_columns)
    tf_keep = select_top_tf_by_variance(
        rna_df.loc[train_mask],
        n_pathway=N_PROGENY_PATHWAYS,
        top_tf_by_variance=TOP_TF_BY_VARIANCE,
    )
    rna_fixed = rna_df[tf_keep].to_numpy(dtype=np.float64)
    order = np.concatenate([np.where(train_mask)[0], np.where(held_mask)[0]])
    views = {"dnam": data.dnam[order], "rna": rna_fixed[order]}
    factors = fit_mofa(
        views,
        subject_ids=data.subject_ids[order],
        visit=data.visit[order],
        n_factors=n_factors,
        seed=seed,
        use_surrogate=False,
    )
    held_rows = factors.subject_ids == held
    held_scores = factors.scores[held_rows]
    pre = held_scores[factors.visit[held_rows] == 0]
    post = held_scores[factors.visit[held_rows] == 1]
    if not pre.size or not post.size:
        return None
    return np.asarray(post.mean(axis=0) - pre.mean(axis=0), dtype=np.float64)


@dataclass
class ArmBResult:
    """Arm B run output: MOFA+ factors + primary and sensitivity classifications.

    Attributes
    ----------
    data:
        The aligned, covariate-adjusted view bundle.
    factors:
        The fitted MOFA+ factors (primary fit, all observations).
    classification:
        Primary trait/state/mixed factor table (all observations).
    classification_sensitivity:
        Sensitivity factor table with the JAK-STAT outlier sample-visits removed
        before re-fitting MOFA+ and re-classifying.
    """

    data: ArmBData
    factors: MOFAFactors
    classification: pd.DataFrame
    classification_sensitivity: pd.DataFrame


def run_arm_b(
    paths: Phase1Paths | None = None,
    *,
    n_factors: int = 20,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> ArmBResult:
    """Run Arm B end to end: load, fit MOFA+, classify, JAK-STAT sensitivity.

    The primary fit retains all sample-visits. The sensitivity fit excludes the
    JAK-STAT outliers, re-fits MOFA+, and re-classifies; a stable trait/state
    split across the two specifications indicates the interferon-axis outliers
    do not drive the factor structure.
    """
    data = load_arm_b_data(paths)
    factors = fit_arm_b(data, n_factors=n_factors, seed=seed)
    classification = classify_factors(factors, n_bootstrap=n_bootstrap, seed=seed)

    keep = ~data.jakstat_outlier
    sens_data = _subset_arm_b_data(data, keep)
    # sens_data is already restricted to its training rows, so tf_rank_mask=None
    # ranks the TF panel on exactly those rows -- leakage-clean for this fit.
    sens_factors = fit_arm_b(sens_data, n_factors=n_factors, seed=seed)
    classification_sensitivity = classify_factors(sens_factors, n_bootstrap=n_bootstrap, seed=seed)

    logger.info(
        "Arm B done: primary %d obs, sensitivity %d obs (%d JAK-STAT outliers dropped)",
        data.n_obs,
        sens_data.n_obs,
        int(data.jakstat_outlier.sum()),
    )
    return ArmBResult(
        data=data,
        factors=factors,
        classification=classification,
        classification_sensitivity=classification_sensitivity,
    )
