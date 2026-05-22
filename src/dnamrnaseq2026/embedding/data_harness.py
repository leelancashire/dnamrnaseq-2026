"""Phase 2 data harness: paired (subject, visit) tensors + subject-level splits.

Implements the common-input layer shared by all three embedding arms
(Phase 2 design doc Section 2.0 / 2.4 / 4.1 / 4.2).

Responsibilities
----------------
- Assemble per-(subject, visit) feature vectors from the DNAm M-value matrix,
  the RNA-side PROGENy/decoupleR activity scores, and clinical covariates.
- Enforce the two-tier feature-subsetting scheme (Section 2.0): Tier 1
  CellDMC-prioritised feature list when Phase 1 step 1.2 returns non-null
  interaction CpGs; Tier 2 biology-informed variance filter as the documented
  fallback.
- Build the per-subject paired trajectory unit ``(x_PRE, x_POST)`` plus the
  explicit delta tuple ``delta_x``.
- Provide a subject-level ``GroupKFold`` splitter so both visits of a held-out
  subject land in the same fold (Section 4.1 / 4.2 hard rule).

This module is data-source agnostic: it operates on already-loaded DataFrames.
The Phase 1 -> Phase 2 artefact interface is documented in design-doc Section
3 (v); see ``feature_selection.py`` for the concrete file readers.

All training-fold-only preprocessing (variance filtering, HVG selection,
normalisation) is fit inside the outer CV loop via :class:`PairedPreprocessor`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

VisitLabel = Literal["PRE", "POST"]
PRE: VisitLabel = "PRE"
POST: VisitLabel = "POST"

# Visit-string normalisation: real pData2 uses PRE_IOP / POST_IOP, BEST uses
# BL / 12W; synthetic fixtures use PRE / POST. Map everything to PRE / POST.
_PRE_ALIASES = {"PRE", "PRE_IOP", "BL", "BASELINE", "PRE-IOP"}
_POST_ALIASES = {"POST", "POST_IOP", "12W", "WEEK12", "POST-IOP"}


def normalise_visit(raw: str) -> VisitLabel:
    """Map a raw visit string onto the canonical PRE / POST label."""
    token = str(raw).strip().upper()
    if token in _PRE_ALIASES:
        return PRE
    if token in _POST_ALIASES:
        return POST
    raise ValueError(f"Unrecognised visit label: {raw!r}")


@dataclass
class PairedDataset:
    """Container for per-(subject, visit) feature matrices.

    Attributes
    ----------
    x_pre:
        (n_subjects, n_features) matrix of PRE-visit feature vectors.
    x_post:
        (n_subjects, n_features) matrix of POST-visit feature vectors.
    subject_ids:
        Subject identifiers, aligned row-wise with ``x_pre`` / ``x_post``.
    response:
        Per-subject response label (binary for Emory, 3-class for BEST).
    feature_names:
        Column names for the feature axis.
    delta_pcl:
        Per-subject change in PCL score (the LOSO surrogate target, metric iii).
        NaN where unavailable.
    cohort:
        Cohort tag per subject ("Emory" / "BEST").
    """

    x_pre: np.ndarray
    x_post: np.ndarray
    subject_ids: np.ndarray
    response: np.ndarray
    feature_names: list[str]
    delta_pcl: np.ndarray = field(default_factory=lambda: np.array([]))
    cohort: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        n = self.x_pre.shape[0]
        if self.x_post.shape[0] != n:
            raise ValueError("x_pre and x_post must have the same number of subjects")
        if self.x_pre.shape[1] != self.x_post.shape[1]:
            raise ValueError("x_pre and x_post must have the same feature dimension")
        if len(self.subject_ids) != n:
            raise ValueError("subject_ids length must match subject count")
        if len(self.feature_names) != self.x_pre.shape[1]:
            raise ValueError("feature_names length must match feature dimension")

    @property
    def n_subjects(self) -> int:
        """Number of paired subjects."""
        return int(self.x_pre.shape[0])

    @property
    def n_features(self) -> int:
        """Feature-vector dimension."""
        return int(self.x_pre.shape[1])

    @property
    def delta_x(self) -> np.ndarray:
        """Per-subject input-space delta ``x_POST - x_PRE`` (n_subjects, n_features)."""
        return np.asarray(self.x_post - self.x_pre, dtype=np.float64)

    def stacked(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, subject_index, visit_code) stacked at the observation level.

        visit_code is 0 for PRE and 1 for POST. Row order is all PRE then all
        POST, so row ``i`` and row ``i + n_subjects`` are the same subject.
        """
        x = np.vstack([self.x_pre, self.x_post])
        subj = np.concatenate([np.arange(self.n_subjects), np.arange(self.n_subjects)])
        visit = np.concatenate(
            [np.zeros(self.n_subjects, dtype=int), np.ones(self.n_subjects, dtype=int)]
        )
        return x, subj, visit


def build_paired_dataset(
    features: pd.DataFrame,
    pdata: pd.DataFrame,
    *,
    subject_col: str = "Subcode",
    visit_col: str = "Visit",
    response_col: str = "Response",
    pcl_col: str | None = "PCL_total",
    cohort_label: str = "Emory",
) -> PairedDataset:
    """Assemble a :class:`PairedDataset` from a sample-level feature matrix.

    Parameters
    ----------
    features:
        (n_samples, n_features) sample-level feature matrix, indexed by sample id.
    pdata:
        Sample-level phenotype table, indexed by the same sample ids.
    subject_col, visit_col, response_col:
        pData column names for subject id, visit, response.
    pcl_col:
        pData column for the PCL score; None if not present.
    cohort_label:
        Cohort tag stored on the dataset.

    Only subjects with BOTH a PRE and a POST observation are retained;
    unpaired samples are dropped with a logged count.
    """
    aligned = pdata.reindex(features.index)
    visits = aligned[visit_col].map(normalise_visit)

    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    subj_ids: list[str] = []
    responses: list[object] = []
    delta_pcl: list[float] = []

    feat_values = features.to_numpy(dtype=np.float64)
    row_index = {sid: i for i, sid in enumerate(features.index)}

    dropped = 0
    for subject, group in aligned.groupby(subject_col, sort=True):
        gvisits = visits.reindex(group.index)
        pre_idx = gvisits.index[gvisits == PRE]
        post_idx = gvisits.index[gvisits == POST]
        if len(pre_idx) != 1 or len(post_idx) != 1:
            dropped += 1
            continue
        pre_sid, post_sid = pre_idx[0], post_idx[0]
        pre_rows.append(feat_values[row_index[pre_sid]])
        post_rows.append(feat_values[row_index[post_sid]])
        subj_ids.append(str(subject))
        responses.append(group[response_col].iloc[0])
        if pcl_col is not None and pcl_col in group.columns:
            d = float(group.loc[post_sid, pcl_col]) - float(group.loc[pre_sid, pcl_col])
        else:
            d = np.nan
        delta_pcl.append(d)

    if dropped:
        logger.info("build_paired_dataset: dropped %d unpaired subject(s)", dropped)
    if not subj_ids:
        raise ValueError("No paired subjects found; check subject/visit columns")

    return PairedDataset(
        x_pre=np.asarray(pre_rows, dtype=np.float64),
        x_post=np.asarray(post_rows, dtype=np.float64),
        subject_ids=np.asarray(subj_ids, dtype=object),
        response=np.asarray(responses, dtype=object),
        feature_names=list(features.columns),
        delta_pcl=np.asarray(delta_pcl, dtype=np.float64),
        cohort=np.asarray([cohort_label] * len(subj_ids), dtype=object),
    )


def subject_level_folds(
    dataset: PairedDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return subject-level train/test index pairs (Section 4.1 outer CV).

    Both visits of a held-out subject go to the same fold by construction:
    the splitter operates on subject indices, not observation rows.

    Parameters
    ----------
    dataset:
        The paired dataset to split.
    n_splits:
        Number of outer CV folds.
    seed:
        Seed; GroupKFold itself is deterministic but we permute subject order
        first so the fold assignment is reproducibly shuffled.

    Returns
    -------
    list of (train_subject_idx, test_subject_idx) arrays.
    """
    n = dataset.n_subjects
    if n_splits > n:
        raise ValueError(f"n_splits={n_splits} exceeds subject count {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    groups = perm  # each subject is its own group; permuted for shuffled folds
    splitter = GroupKFold(n_splits=n_splits)
    dummy_x = np.zeros((n, 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in splitter.split(dummy_x, groups=groups):
        folds.append((np.sort(train_idx), np.sort(test_idx)))
    return folds


@dataclass
class PairedPreprocessor:
    """Fold-internal Tier 2 feature selection + normalisation (design Section 4.2).

    Design Section 4.2 (hard rule): *all* preprocessing, including Tier 2
    variance filtering and HVG selection, is fit INSIDE the outer CV loop, on
    the training fold only, then applied to the held-out fold. Ranking variance
    or HVGs on the full cohort lets held-out test rows influence which features
    exist; that is a train/test leak that invalidates an embedding benchmark.

    This preprocessor is the fold-aware home of that selection. The intended
    per-fold usage is::

        prep = PairedPreprocessor(tier2_dnam_top=5000, tier2_rna_top=2000)
        prep.fit(train_dnam, train_rna)         # ranking on training rows only
        dnam_tr, rna_tr = prep.transform(train_dnam, train_rna)
        dnam_te, rna_te = prep.transform(test_dnam, test_rna)  # held-out fold

    The selected feature lists and the per-feature mean/std come from the
    training fold exclusively; ``transform`` only subsets and standardises with
    those frozen statistics. ``fit`` must never see test-fold rows.

    Tier 1 (CellDMC FDR<0.10) is a fixed biological prior, not a data-driven
    selection, and is resolved separately via ``feature_selection.resolve_feature_tier``;
    it is correctly pre-computable cohort-wide and is NOT handled here.

    Inputs are (n_samples, n_features) sample-major frames -- the orientation
    the loaders assemble -- so variance/HVG ranking is over ``axis=0``.
    """

    tier2_dnam_top: int = 5000
    tier2_rna_top: int = 2000
    standardise: bool = True
    _dnam_features: list[str] = field(default_factory=list)
    _rna_features: list[str] = field(default_factory=list)
    _dnam_mean: pd.Series | None = field(default=None, repr=False)
    _dnam_std: pd.Series | None = field(default=None, repr=False)
    _rna_mean: pd.Series | None = field(default=None, repr=False)
    _rna_std: pd.Series | None = field(default=None, repr=False)
    _fitted: bool = False

    def fit(self, dnam_train: pd.DataFrame, rna_train: pd.DataFrame) -> PairedPreprocessor:
        """Rank Tier 2 features on the TRAINING FOLD ONLY and freeze statistics.

        ``dnam_train`` / ``rna_train`` are (n_train_samples, n_features) frames
        restricted to the current outer-fold training subjects. Never pass
        held-out rows here: that is the leak Section 4.2 forbids.
        """
        dnam_var = dnam_train.var(axis=0).sort_values(ascending=False)
        self._dnam_features = [str(c) for c in dnam_var.head(self.tier2_dnam_top).index]
        rna_var = rna_train.var(axis=0).sort_values(ascending=False)
        self._rna_features = [str(c) for c in rna_var.head(self.tier2_rna_top).index]

        if self.standardise:
            dnam_sel = dnam_train[self._dnam_features]
            rna_sel = rna_train[self._rna_features]
            self._dnam_mean = dnam_sel.mean(axis=0)
            self._dnam_std = dnam_sel.std(axis=0).replace(0.0, 1.0)
            self._rna_mean = rna_sel.mean(axis=0)
            self._rna_std = rna_sel.std(axis=0).replace(0.0, 1.0)

        self._fitted = True
        logger.info(
            "PairedPreprocessor.fit: Tier 2 selection on %d training samples "
            "(%d DNAm, %d RNA features kept)",
            len(dnam_train),
            len(self._dnam_features),
            len(self._rna_features),
        )
        return self

    def transform(self, dnam: pd.DataFrame, rna: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Subset + standardise with the frozen training-fold selection.

        Applied identically to the training fold and the held-out fold; the
        held-out fold never influences which features are kept or the mean/std.
        """
        if not self._fitted:
            raise RuntimeError("PairedPreprocessor.transform called before fit")
        dnam_out = dnam.reindex(columns=self._dnam_features)
        rna_out = rna.reindex(columns=self._rna_features)
        if self.standardise:
            assert self._dnam_mean is not None and self._dnam_std is not None
            assert self._rna_mean is not None and self._rna_std is not None
            dnam_out = (dnam_out - self._dnam_mean) / self._dnam_std
            rna_out = (rna_out - self._rna_mean) / self._rna_std
        return dnam_out, rna_out

    @property
    def dnam_features(self) -> list[str]:
        """Tier 2 DNAm features selected on the training fold."""
        return list(self._dnam_features)

    @property
    def rna_features(self) -> list[str]:
        """Tier 2 RNA HVGs selected on the training fold."""
        return list(self._rna_features)


def inner_calibration_split(
    train_subject_idx: np.ndarray,
    calib_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an outer-fold training set into predictor-fit vs conformal-calibration.

    Subject-disjoint by construction (Section 4.1): a subject appears in either
    the predictor-fit set or the calibration set, never both.

    Returns
    -------
    (fit_idx, calib_idx) arrays of subject indices into the original dataset.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_subject_idx)
    n_calib = max(1, int(round(len(shuffled) * calib_fraction)))
    calib_idx = np.sort(shuffled[:n_calib])
    fit_idx = np.sort(shuffled[n_calib:])
    return fit_idx, calib_idx
