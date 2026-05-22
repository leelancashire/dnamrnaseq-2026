"""Real Phase 1 -> Phase 2 data wiring (design Section 2.0 / 2.1 / 3v).

This module replaces the synthetic fixtures on the *production* code path. It
reads the real v5 Phase 1 outputs from ``analysis/latest/`` and assembles them
into the :class:`~dnamrnaseq2026.embedding.data_harness.PairedDataset` and the
per-arm batch tensors the three embedding arms consume.

It is the concrete realisation of the Phase 1 -> Phase 2 artefact interface
documented in design-doc Section 3 (v). Two things to know:

1. **Paths are configurable.** The design doc names artefacts
   ``progeny_pathway_activity.parquet`` etc.; the actual v5 Phase 1 run writes
   ``progeny_activity_emory.parquet`` (and a TSV for CellDMC). :class:`Phase1Paths`
   defaults to the *real* v5 filenames and accepts the design-doc names as
   aliases, so the loaders snap to whatever Kai's schema/paths spec settles on
   without a code change -- only the dataclass defaults move.

2. **It degrades gracefully.** Any missing or stub artefact raises a precise
   :class:`Phase1ArtefactError` naming the file, rather than failing deep inside
   numpy. The real-data smoke tests rely on this to skip cleanly when
   ``analysis/latest/`` is absent (CI, fresh clone, no OneDrive).

The synthetic fixtures in ``tests/phase2_fixtures.py`` are retained for the unit
tests; this module is exercised by ``tests/test_phase2_real_data.py``, which is
skipped unless the real artefacts are present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dnamrnaseq2026.embedding.data_harness import PairedDataset, normalise_visit

logger = logging.getLogger(__name__)

# Default Phase 1 artefact directory. analysis/latest/ is a symlink to the
# active dated run; keep it relative so it resolves from the repo root.
DEFAULT_PHASE1_DIR = Path("analysis/latest")

# EpiDISH cell-fraction columns on the corrected pData (design Section 2.0).
EPIDISH_COLS = (
    "epidish_B",
    "epidish_NK",
    "epidish_CD4T",
    "epidish_CD8T",
    "epidish_Mono",
    "epidish_Neutro",
    "epidish_Eosino",
)

# Clinical covariates fed alongside the omics features (design Section 2.0).
# Only columns actually present on the corrected pData are used; the loader
# logs and drops any that are absent rather than failing.
CLINICAL_COLS = (
    "Age",
    "ancestry_pca_PCA1",
    "ancestry_pca_PCA2",
    "ancestry_pca_PCA3",
    "ancestry_pca_PCA4",
    "ancestry_pca_PCA5",
    "ancestry_pca_PCA6",
    "smokingScore",
)


class Phase1ArtefactError(RuntimeError):
    """Raised when a required Phase 1 artefact is missing or a stub.

    Carries the offending path so the real-data smoke tests can skip cleanly
    instead of failing inside pandas/numpy.
    """


@dataclass
class Phase1Paths:
    """Configurable Phase 1 artefact paths (design Section 3v interface).

    Defaults target the real v5 Phase 1 output filenames. The design doc names
    some artefacts differently (e.g. ``progeny_pathway_activity.parquet``); when
    Kai publishes the schema/paths spec note, only these defaults move -- the
    loader logic does not.

    All paths are resolved relative to ``root`` unless absolute.
    """

    root: Path = DEFAULT_PHASE1_DIR
    pdata_emory: str = "pdata_emory_with_epidish.csv"
    progeny_activity: str = "progeny_activity_emory.parquet"
    tf_activity: str = "tf_activity_emory.parquet"
    celldmc_interactions: str = "celldmc_delta_emory.tsv"
    rnaseq_corrected: str = "rnaseq_corrected_emory.parquet"
    # Design-doc alias filenames tried as a fallback when the primary is absent.
    _aliases: dict[str, str] = field(
        default_factory=lambda: {
            "progeny_activity": "progeny_pathway_activity.parquet",
            "tf_activity": "decoupler_tf_activity.parquet",
            "celldmc_interactions": "celldmc_interaction_results.parquet",
        }
    )

    def resolve(self, attr: str) -> Path:
        """Return the resolved path for ``attr``, trying the design-doc alias.

        The primary (v5) filename wins if it exists; otherwise the design-doc
        alias is tried. The primary path is returned (even if absent) when
        neither exists, so the caller's error message names the expected file.
        """
        root = Path(self.root)
        primary = root / str(getattr(self, attr))
        if primary.exists():
            return primary
        alias = self._aliases.get(attr)
        if alias is not None and (root / alias).exists():
            logger.info("Phase1Paths: %s resolved via design-doc alias %s", attr, alias)
            return root / alias
        return primary

    def exists(self) -> bool:
        """True if the Phase 1 root directory exists (cheap CI-skip predicate)."""
        return Path(self.root).is_dir()


def _require(path: Path, what: str) -> Path:
    """Return ``path`` if it exists and is non-empty, else raise Phase1ArtefactError."""
    if not path.exists():
        raise Phase1ArtefactError(f"{what} artefact missing at {path}")
    if path.stat().st_size == 0:
        raise Phase1ArtefactError(f"{what} artefact is a zero-byte stub at {path}")
    return path


def load_emory_pdata(paths: Phase1Paths | None = None) -> pd.DataFrame:
    """Load the EpiDISH-corrected Emory pData (the design Section 2.0 covariate table).

    Returns a sample-level frame indexed by ``Subcode`` + ``Visit`` (canonical
    PRE / POST), carrying Response, PCL_total, the seven EpiDISH cell fractions,
    and the clinical covariates.
    """
    paths = paths or Phase1Paths()
    path = _require(paths.resolve("pdata_emory"), "Emory corrected pData")
    pdata = pd.read_csv(path)
    if "Subcode" not in pdata.columns or "Visit" not in pdata.columns:
        raise Phase1ArtefactError(f"pData at {path} lacks Subcode/Visit columns")
    pdata = pdata.copy()
    pdata["Visit"] = pdata["Visit"].map(normalise_visit)
    logger.info("Loaded corrected Emory pData: %s", pdata.shape)
    return pdata


def _mean_impute(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Mean-impute sparse NaNs in covariate columns (harness-level handling).

    The real corrected pData carries a handful of missing ancestry-PC values.
    This is genuine sparse missingness, not a stub. Harness-level mean-imputation
    keeps those subjects in the paired set; the fold-safe variant (impute on the
    training fold only) is applied inside the CV loop by ``PairedPreprocessor``.
    A column that is entirely NaN is left untouched and surfaces downstream.
    """
    out = frame.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        n_missing = int(series.isna().sum())
        if 0 < n_missing < len(series):
            out[col] = series.fillna(series.mean())
            logger.info("real_data: mean-imputed %d NaN(s) in covariate %s", n_missing, col)
        else:
            out[col] = series
    return out


def _read_activity(path: Path, what: str) -> pd.DataFrame:
    """Read a PROGENy/TF activity parquet keyed by ``{Subcode}-{Visit}`` sample id.

    Raises :class:`Phase1ArtefactError` when the artefact is a stub: zero
    feature columns, zero rows, or an all-NaN value block. The in-progress
    Phase 1 step 1.4/1.5 re-run leaves a column-and-index-populated but
    all-NaN parquet; that is a stub, not data, and must not reach the encoders.
    """
    _require(path, what)
    df = pd.read_parquet(path)
    if df.shape[1] == 0:
        raise Phase1ArtefactError(f"{what} artefact at {path} has zero feature columns (stub)")
    if df.shape[0] == 0:
        raise Phase1ArtefactError(f"{what} artefact at {path} has zero rows (stub)")
    if not np.isfinite(df.to_numpy(dtype=np.float64)).any():
        raise Phase1ArtefactError(
            f"{what} artefact at {path} is entirely non-finite (all-NaN stub; "
            "pending the Phase 1 step 1.4/1.5 re-run on real EpiDISH cell fractions)"
        )
    return df


def _sample_key(subcode: str, visit: str) -> str:
    """Build the ``{Subcode}-{Visit}`` activity-matrix key from canonical visit."""
    raw_visit = "PRE-IOP" if visit == "PRE" else "POST-IOP"
    return f"{subcode}-{raw_visit}"


def _assert_block_finite(x: np.ndarray, feature_names: list[str], visit: str, what: str) -> None:
    """Raise Phase1ArtefactError if ``x`` carries any non-finite value.

    ``np.isfinite(...).all()`` catches *partial* NaN contamination: a single
    NaN cell anywhere in the assembled feature block fails the check, where the
    upstream ``_read_activity`` all-NaN guard (``.any()``) would let it through.
    The in-progress Phase 1 step 1.4/1.5 re-run is exactly the regime where a
    partially populated activity parquet (some TFs converged, some did not) is
    plausible, so this guard must be value-level, not artefact-level.
    """
    finite = np.isfinite(x)
    if finite.all():
        return
    bad_rows, bad_cols = np.where(~finite)
    n_bad = int(bad_rows.size)
    first_col = int(bad_cols[0])
    feat = feature_names[first_col] if first_col < len(feature_names) else f"col_{first_col}"
    raise Phase1ArtefactError(
        f"{what} ({visit} block) carries {n_bad} non-finite value(s); first at "
        f"sample row {int(bad_rows[0])}, feature '{feat}'. A partially populated "
        "activity artefact (pending the Phase 1 step 1.4/1.5 re-run) reaches here "
        "with NaNs that the all-NaN stub guard does not catch; NaNs must not "
        "flow into the encoders."
    )


def build_rna_activity_matrix(
    pdata: pd.DataFrame,
    paths: Phase1Paths | None = None,
    *,
    top_tf_by_variance: int = 150,
    tf_rank_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the Arm A RNA-side input: PROGENy + top-variance TF activity.

    Output is a sample-level frame indexed by ``{Subcode}-{Visit}`` activity
    keys (design Section 2.1: ~14 PROGENy + up to ``top_tf_by_variance`` TF
    scores). Raises :class:`Phase1ArtefactError` if both artefacts are stubs.

    Leakage-safe TF selection (design Section 4.2)
    ----------------------------------------------
    The TF panel is a data-driven variance rank: ranking it over the whole
    cohort lets held-out test sample-visits influence which TFs exist, the
    train/test leak PRs #14-#16 closed for Arm B. ``tf_rank_keys`` restricts the
    variance rank to the named sample-visit keys (the training rows of the
    current outer fold / LOSO fit). All rows of the returned frame carry the
    full PROGENy panel plus the TF panel ranked on the training rows only.
    Passing ``None`` ranks over every row, which is correct ONLY when the caller
    is producing a cohort-wide descriptive (non-CV-evaluated) matrix; any
    CV / LOSO path MUST pass a training-row key list.
    """
    paths = paths or Phase1Paths()
    progeny = _read_activity(paths.resolve("progeny_activity"), "PROGENy pathway activity")
    tf = _read_activity(paths.resolve("tf_activity"), "decoupleR TF activity")

    if tf_rank_keys is not None:
        rank_keys = [k for k in tf_rank_keys if k in tf.index]
        if len(rank_keys) < 2:
            raise Phase1ArtefactError(
                f"TF variance rank needs >=2 training sample-visits; only "
                f"{len(rank_keys)} of {len(tf_rank_keys)} requested keys are in "
                "the TF activity index. The fold-training key set does not match "
                "the activity-matrix index format."
            )
        tf_rank_block = tf.loc[rank_keys]
    else:
        tf_rank_block = tf
    tf_var = tf_rank_block.var(axis=0).sort_values(ascending=False)
    top_tfs = tf_var.head(min(top_tf_by_variance, tf.shape[1])).index
    combined = pd.concat([progeny, tf[top_tfs]], axis=1)
    logger.info(
        "RNA activity matrix: %d PROGENy + %d TF = %s (TF rank on %d %s rows)",
        progeny.shape[1],
        len(top_tfs),
        combined.shape,
        len(tf_rank_block),
        "training-fold" if tf_rank_keys is not None else "cohort-wide",
    )
    return combined


@dataclass
class ArmInputs:
    """Per-arm real-data inputs assembled from Phase 1 outputs.

    Attributes
    ----------
    paired:
        The :class:`PairedDataset` (concatenated DNAm-side cell fractions +
        RNA activity + clinical covariates) used by Arm C and the harness.
    rna_pre, rna_post:
        (n_subjects, d_rna_in) RNA pathway/TF activity, the Arm A RNA path.
    dnam_pre, dnam_post:
        (n_subjects, d_dnam_in) DNAm-side features (EpiDISH cell fractions in
        the wired-now configuration; swaps to the Tier 1 CellDMC CpG matrix once
        Kai's feature matrix lands -- see :func:`build_arm_inputs`).
    clin_pre, clin_post:
        (n_subjects, d_clinical_in) clinical covariates.
    responder_mask:
        (n_subjects,) boolean responder flag.
    """

    paired: PairedDataset
    rna_pre: np.ndarray
    rna_post: np.ndarray
    dnam_pre: np.ndarray
    dnam_post: np.ndarray
    clin_pre: np.ndarray
    clin_post: np.ndarray
    responder_mask: np.ndarray

    @property
    def d_rna_in(self) -> int:
        """RNA activity feature count."""
        return int(self.rna_pre.shape[1])

    @property
    def d_dnam_in(self) -> int:
        """DNAm-side feature count."""
        return int(self.dnam_pre.shape[1])

    @property
    def d_clinical_in(self) -> int:
        """Clinical covariate count."""
        return int(self.clin_pre.shape[1])


def _stack_paired(
    pdata: pd.DataFrame,
    feature_lookup: dict[str, np.ndarray],
    *,
    dnam_cols: list[str],
    rna_cols: list[str],
    clin_cols: list[str],
) -> ArmInputs:
    """Assemble per-subject paired arrays from a per-sample feature lookup.

    ``feature_lookup`` maps a ``{Subcode}-{Visit}`` key to its full feature
    vector; ``*_cols`` give the slice boundaries inside that vector.
    """
    n_dnam, n_rna = len(dnam_cols), len(rna_cols)
    pre_rows: list[np.ndarray] = []
    post_rows: list[np.ndarray] = []
    subj_ids: list[str] = []
    responses: list[object] = []
    delta_pcl: list[float] = []
    responder: list[bool] = []

    dropped = 0
    for subcode, group in pdata.groupby("Subcode", sort=True):
        visits = group.set_index("Visit")
        if "PRE" not in visits.index or "POST" not in visits.index:
            dropped += 1
            continue
        pre_key = _sample_key(str(subcode), "PRE")
        post_key = _sample_key(str(subcode), "POST")
        if pre_key not in feature_lookup or post_key not in feature_lookup:
            dropped += 1
            continue
        pre_rows.append(feature_lookup[pre_key])
        post_rows.append(feature_lookup[post_key])
        subj_ids.append(str(subcode))
        resp = visits.loc["PRE", "Response"]
        responses.append(resp)
        responder.append(str(resp).upper() in {"R", "1", "RESPONDER"})
        pcl_pre = pd.to_numeric(visits.loc["PRE", "PCL_total"], errors="coerce")
        pcl_post = pd.to_numeric(visits.loc["POST", "PCL_total"], errors="coerce")
        delta_pcl.append(float(pcl_post) - float(pcl_pre))

    if dropped:
        logger.info(
            "real_data: dropped %d subject(s) lacking a complete paired feature row", dropped
        )
    if not subj_ids:
        raise Phase1ArtefactError("No paired subjects with complete feature rows assembled")

    x_pre = np.asarray(pre_rows, dtype=np.float64)
    x_post = np.asarray(post_rows, dtype=np.float64)
    feature_names = dnam_cols + rna_cols + clin_cols

    # Fail loud on partial-NaN contamination before the ArmInputs is returned.
    # The upstream all-NaN guard only rejects an entirely non-finite artefact;
    # a parquet with one finite cell passes it and seeds NaNs into x_pre/x_post.
    _assert_block_finite(x_pre, feature_names, "PRE", "Assembled paired feature block")
    _assert_block_finite(x_post, feature_names, "POST", "Assembled paired feature block")

    # Responder-mask sanity: a silently inverted or degenerate Response coding
    # (e.g. Y/N or 0/1-with-responder-0) collapses the mask to all-True or
    # all-False and flips the Arm A/C contrastive labels. Assert a plausible
    # fraction rather than trusting the string coding blindly.
    responder_arr = np.asarray(responder, dtype=bool)
    responder_frac = float(responder_arr.mean())
    if not 0.05 <= responder_frac <= 0.95:
        raise Phase1ArtefactError(
            f"Responder fraction {responder_frac:.3f} is degenerate "
            f"({int(responder_arr.sum())} of {len(responder_arr)} subjects flagged "
            "responders). The Emory 'Response' column coding likely does not match "
            "the expected {R, 1, RESPONDER} token set; the responder mask, and the "
            "Arm A/C contrastive labels derived from it, would be wrong."
        )

    paired = PairedDataset(
        x_pre=x_pre,
        x_post=x_post,
        subject_ids=np.asarray(subj_ids, dtype=object),
        response=np.asarray(responses, dtype=object),
        feature_names=feature_names,
        delta_pcl=np.asarray(delta_pcl, dtype=np.float64),
        cohort=np.asarray(["Emory"] * len(subj_ids), dtype=object),
    )
    return ArmInputs(
        paired=paired,
        dnam_pre=x_pre[:, :n_dnam],
        dnam_post=x_post[:, :n_dnam],
        rna_pre=x_pre[:, n_dnam : n_dnam + n_rna],
        rna_post=x_post[:, n_dnam : n_dnam + n_rna],
        clin_pre=x_pre[:, n_dnam + n_rna :],
        clin_post=x_post[:, n_dnam + n_rna :],
        responder_mask=responder_arr,
    )


def build_arm_inputs(
    paths: Phase1Paths | None = None,
    *,
    tf_rank_keys: list[str] | None = None,
) -> ArmInputs:
    """Assemble the real-data per-arm inputs from the Phase 1 outputs.

    ``tf_rank_keys`` is forwarded to :func:`build_rna_activity_matrix`: pass the
    training-fold sample-visit keys so the TF variance panel is selected on
    training rows only (design Section 4.2 leakage rule). ``None`` ranks the TF
    panel cohort-wide, which is correct only for a descriptive, non-CV-evaluated
    embedding; the leakage-clean LOSO path passes per-fold key lists.

    Wired-now configuration (design Section 2.1, Tier 2 fallback active):

    - RNA side: PROGENy + decoupleR/CollecTRI TF activity (the real v5
      ``progeny_activity_emory.parquet`` / ``tf_activity_emory.parquet``).
    - DNAm side: EpiDISH cell fractions from the corrected pData. This is the
      Tier 2-compatible standing input; it swaps to the Tier 1 CellDMC-
      prioritised CpG matrix once Kai publishes that feature matrix, at which
      point only the ``dnam_*`` block changes (the slice arithmetic and the
      paired-construction logic are tier-agnostic).
    - Clinical side: Age, ancestry PCs, smoking score from the corrected pData.

    Raises :class:`Phase1ArtefactError` (naming the file) on any missing/stub
    artefact, so callers and the smoke tests can degrade cleanly.
    """
    paths = paths or Phase1Paths()
    if not paths.exists():
        raise Phase1ArtefactError(f"Phase 1 artefact directory absent: {paths.root}")

    pdata = load_emory_pdata(paths)
    rna = build_rna_activity_matrix(pdata, paths, tf_rank_keys=tf_rank_keys)

    dnam_cols = [c for c in EPIDISH_COLS if c in pdata.columns]
    if not dnam_cols:
        raise Phase1ArtefactError("Corrected pData carries no epidish_* cell-fraction columns")
    clin_cols = [c for c in CLINICAL_COLS if c in pdata.columns]
    missing_clin = [c for c in CLINICAL_COLS if c not in pdata.columns]
    if missing_clin:
        logger.info("real_data: clinical covariates absent and skipped: %s", missing_clin)
    rna_cols = list(rna.columns)

    # Mean-impute sparse NaNs in the cell-fraction + clinical covariate block.
    pdata = _mean_impute(pdata, list(dnam_cols) + list(clin_cols))

    # Build the per-sample feature lookup keyed by {Subcode}-{Visit}.
    pdata_keyed = pdata.copy()
    pdata_keyed["_key"] = [
        _sample_key(str(s), str(v))
        for s, v in zip(pdata_keyed["Subcode"], pdata_keyed["Visit"], strict=True)
    ]
    cov = pdata_keyed.set_index("_key")[list(dnam_cols) + list(clin_cols)]

    feature_lookup: dict[str, np.ndarray] = {}
    for key in rna.index:
        if key not in cov.index:
            continue
        dnam_vec = cov.loc[key, dnam_cols].to_numpy(dtype=np.float64)
        clin_vec = cov.loc[key, clin_cols].to_numpy(dtype=np.float64)
        rna_vec = rna.loc[key].to_numpy(dtype=np.float64)
        feature_lookup[str(key)] = np.concatenate([dnam_vec, rna_vec, clin_vec])

    # Join-key sanity. The {Subcode}-{Visit} key is built from hard-coded
    # PRE-IOP / POST-IOP raw visit strings; if the activity parquet ever keys on
    # a different raw form (PRE / Pre-IOP / whitespace) every membership test
    # fails silently and the loader assembles 0 subjects. Log the match rate and
    # fail loud with a key-format-specific message rather than a generic error.
    n_matched = len(feature_lookup)
    n_rna_keys = int(len(rna.index))
    logger.info(
        "build_arm_inputs: %d of %d RNA activity keys matched a covariate row",
        n_matched,
        n_rna_keys,
    )
    if n_matched == 0:
        raise Phase1ArtefactError(
            f"None of the {n_rna_keys} RNA activity keys matched a pData covariate "
            f"row. Sample keys are built as '{{Subcode}}-PRE-IOP'/'-POST-IOP'; the "
            f"activity index likely uses a different raw visit form. "
            f"Example RNA key: {rna.index[0]!r}; example covariate key: "
            f"{cov.index[0]!r}."
        )

    inputs = _stack_paired(
        pdata,
        feature_lookup,
        dnam_cols=list(dnam_cols),
        rna_cols=rna_cols,
        clin_cols=list(clin_cols),
    )
    logger.info(
        "build_arm_inputs: %d paired subjects, d=(dnam %d, rna %d, clin %d)",
        inputs.paired.n_subjects,
        inputs.d_dnam_in,
        inputs.d_rna_in,
        inputs.d_clinical_in,
    )
    return inputs
