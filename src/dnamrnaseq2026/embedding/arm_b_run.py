"""Arm B real-data run: assemble multi-omics views, fit MOFA+, classify factors.

This module is the real-data realisation of Arm B (design Section 2.2). It is
distinct from :mod:`dnamrnaseq2026.embedding.arm_b_mofa`, which holds the
arm-agnostic MOFA+ fit and the LMM-LRT trait-state machinery; this module wires
those primitives to the genuine Phase 1 feature matrices in ``analysis/latest/``.

Pipeline
--------
1. Load the Tier 1 DNAm M-value matrix and the Tier 1 RNA activity matrix via the
   CV-loop-safe loader (both carry ``cv_loop_safe=True`` provenance sidecars; the
   Tier 1 features are a fixed biological prior, not a cohort-data-driven
   selection, so they are admissible directly).
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
from dnamrnaseq2026.embedding.feature_selection import load_feature_matrix_for_cv
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


@dataclass
class ArmBData:
    """Aligned, covariate-adjusted multi-omics views for the Arm B MOFA+ fit.

    Attributes
    ----------
    dnam:
        (n_obs, n_dnam_features) covariate-adjusted DNAm M-value view.
    rna:
        (n_obs, n_rna_features) covariate-adjusted RNA activity view.
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
    subject_ids: np.ndarray
    visit: np.ndarray
    sentrix_ids: np.ndarray
    jakstat_outlier: np.ndarray

    @property
    def n_obs(self) -> int:
        """Number of aligned sample-visit observations."""
        return int(self.dnam.shape[0])


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

    dnam_df = load_feature_matrix_for_cv(root / "feature_matrix_tier1_dnam.parquet")
    rna_df = load_feature_matrix_for_cv(root / "feature_matrix_tier1_rna.parquet")

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

    if residualise_covariates:
        cov = _build_covariate_matrix(pdata)
        dnam = _residualise(dnam, cov)
        rna = _residualise(rna, cov)
        logger.info("Arm B: residualised %d covariates out of both views", cov.shape[1])

    return ArmBData(
        dnam=dnam,
        rna=rna,
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
) -> MOFAFactors:
    """Fit the two-view MOFA+ model on the aligned Arm B views (CPU).

    Views are scaled internally by mofapy2 (``scale_views=True``). The DNAm view
    (M-values) and the RNA view (activity scores) are on different native
    scales; view scaling makes the ELBO factorisation scale-invariant across
    views.
    """
    views = {"dnam": data.dnam, "rna": data.rna}
    factors = fit_mofa(
        views,
        subject_ids=data.subject_ids,
        visit=data.visit,
        n_factors=n_factors,
        seed=seed,
        use_surrogate=False,
    )
    logger.info("Arm B MOFA+ fit complete: %d factors on %d obs", factors.n_factors, data.n_obs)
    return factors


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
    sens_data = ArmBData(
        dnam=data.dnam[keep],
        rna=data.rna[keep],
        subject_ids=data.subject_ids[keep],
        visit=data.visit[keep],
        sentrix_ids=data.sentrix_ids[keep],
        jakstat_outlier=data.jakstat_outlier[keep],
    )
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
