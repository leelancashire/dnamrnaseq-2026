"""Phase 3 external-cohort projection + two-anchor recovery-axis construction.

Design reference: 2026-05-17-integrated-analysis-plan-v2.md Section 5.3 Step 3.3;
2026-05-19-headline-framing-rationale.md.

This module is scaffolded ahead of Phase 2 completion. It operates against
synthetic fixtures now and against real atlas artifacts once the Phase 2
leaderboard picks the winning arm. **No real-data paths are hard-coded.**

Two-anchor recovery axis
------------------------
The framing-rationale doc establishes why the recovery axis needs two anchors:

1. **GTEx whole-blood cloud (healthy anchor)** -- the destination for PTSD
   responder trajectories; samples were Emory-aligned in Phase 2 Wave 1
   (``gtex_whole_blood_emory_aligned.parquet`` produced by Kai's GTEx prep).

2. **GSE98793 TRD-inflammatory cluster (pathological anchor)** -- the
   predicted terminus for PTSD non-responders; group centroids (TRD vs
   antidepressant-responder) were precomputed in Phase 2 Wave 1
   (``gse98793_group_centroids.parquet``).

The recovery axis is defined in the atlas latent space as::

    axis = unit( gtex_cloud_centroid - gse_trd_centroid )

That is, the axis *points from the TRD-inflammatory region toward the
healthy region*. Positive projection scores indicate movement toward health;
negative scores indicate movement toward TRD-inflammatory biology. The sign
convention is preserved regardless of embedding dimensionality.

Output interface
----------------
The public output of this module is consumed by Helen's Phase 3.3 proximity
test (``tests/test_phase3_proximity.py``). The contract is:

- :class:`ProjectionResult` -- the single output dataclass.
- Fields consumed by the proximity test:

    * ``terminus_latent`` -- ``(n_ptsd_subjects, d_latent)`` POST latent coords
      for each PTSD subject (Emory + BEST).
    * ``gtex_latent`` -- ``(n_gtex, d_latent)`` GTEx reference cloud.
    * ``gse_trd_latent`` -- ``(n_gse_trd, d_latent)`` GSE98793 TRD cluster.
    * ``recovery_axis`` -- ``(d_latent,)`` unit vector pointing toward healthy.
    * ``terminus_recovery_score`` -- ``(n_ptsd_subjects,)`` projection of each
      terminus onto the recovery axis (scalar; higher == more healthy-like).
    * ``subject_ids`` -- ``(n_ptsd_subjects,)`` string identifiers.
    * ``response`` -- ``(n_ptsd_subjects,)`` response labels ('R', 'NR', or
      partial-response integers 1/2/3).

The proximity test imports :class:`ProjectionResult` directly and must not
depend on any implementation details inside this module.

Atlas embedding model interface
--------------------------------
The winning Phase 2 arm is any callable implementing :class:`AtlasEncoder`:

    ``encode(rna_matrix: np.ndarray) -> np.ndarray``

where ``rna_matrix`` has shape ``(n_samples, n_rna_features)`` and the output
has shape ``(n_samples, d_latent)``. External cohorts (GTEx, GSE98793) are
RNA-only; the encoder is the RNA-side projection of the winning arm's full
multi-omics encoder. If the winning arm requires DNAm for the RNA encoder to
function, an RNA-only auxiliary projection trained on Emory PRE-IOP
RNA-to-latent regression is used instead (document as load-bearing caveat in
the methods section).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum cloud sizes for meaningful axis estimation
_MIN_GTEX_SAMPLES = 10
_MIN_GSE_TRD_SAMPLES = 5
_MIN_PTSD_SUBJECTS = 10


# ---------------------------------------------------------------------------
# Atlas encoder protocol (the interface any Phase 2 winning arm must satisfy)
# ---------------------------------------------------------------------------


class AtlasEncoder(Protocol):
    """RNA-only encoder extracted from the Phase 2 winning embedding arm.

    Any callable that maps ``(n_samples, n_rna_features) -> (n_samples, d_latent)``
    satisfies this protocol. The actual implementation is produced by Phase 2
    once the winning arm is chosen (Arm A FM, Arm B MOFA+, or Arm C contrastive).

    For external cohorts that lack matched DNAm, only the RNA encoder path is
    used; this is the load-bearing methodological caveat noted in the design doc
    (Section 5.3 Step 3.3).
    """

    def encode(self, rna_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Embed RNA expression into the atlas latent space.

        Parameters
        ----------
        rna_matrix:
            (n_samples, n_rna_features) expression matrix (log-CPM or equivalent
            scale expected by the trained encoder). Gene ordering must match the
            training feature set.

        Returns
        -------
        (n_samples, d_latent) float64 array of latent coordinates.
        """
        ...


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExternalCohortData:
    """Prepared external-cohort matrices ready for atlas projection.

    Both cohorts were prepared in Phase 2 Wave 1 (Kai):
    - GTEx v10 whole blood, Emory-gene-panel aligned.
    - GSE98793 preprocessed, group centroids computed.

    Fields
    ------
    gtex_rna:
        (n_gtex, n_rna_features) GTEx whole-blood expression, genes aligned
        to the atlas training feature set.
    gtex_sample_ids:
        (n_gtex,) string identifiers (GTEx donor IDs or sample IDs).
    gse_trd_rna:
        (n_gse_trd, n_rna_features) GSE98793 TRD-inflammatory sample expression,
        restricted to TRD-labelled subjects, genes aligned to the atlas panel.
    gse_trd_sample_ids:
        (n_gse_trd,) string identifiers.
    gse_trd_centroid_rna:
        (n_rna_features,) precomputed TRD group centroid in expression space
        (mean across TRD samples). Provided as a convenience; the atlas-latent
        centroid is computed post-projection.
    feature_names:
        (n_rna_features,) gene names / Ensembl IDs; must match the atlas
        training feature order exactly.
    """

    gtex_rna: np.ndarray
    gtex_sample_ids: np.ndarray
    gse_trd_rna: np.ndarray
    gse_trd_sample_ids: np.ndarray
    gse_trd_centroid_rna: np.ndarray
    feature_names: np.ndarray


@dataclass
class PtsdAtlasData:
    """PTSD subjects (Emory + BEST) with POST-IOP latent coordinates.

    These come from the Phase 2 atlas build step (Step 3.0 of the design):
    ``z_i_POST = E*(x_i_POST)`` for each Emory/BEST paired subject.

    Fields
    ------
    terminus_latent:
        (n_subjects, d_latent) POST-IOP latent coordinates per subject.
    subject_ids:
        (n_subjects,) string identifiers.
    response:
        (n_subjects,) response labels; 'R'/'NR' strings or 1/2/3 integers
        (BEST 3-class). Helen's proximity test accepts both conventions.
    cohort:
        (n_subjects,) cohort tag; 'Emory' or 'BEST' per subject.
    """

    terminus_latent: np.ndarray
    subject_ids: np.ndarray
    response: np.ndarray
    cohort: np.ndarray


@dataclass
class ProjectionResult:
    """Full Phase 3 projection output consumed by Helen's proximity test.

    This is the output interface contract. Helen's
    ``test_phase3_proximity.py`` imports this dataclass and runs on real or
    synthetic instances of it. Do not change field names or types without
    coordinating with the proximity-test module.

    Fields
    ------
    terminus_latent:
        (n_ptsd_subjects, d_latent) POST-IOP latent coords for Emory + BEST.
    gtex_latent:
        (n_gtex, d_latent) GTEx healthy reference cloud in atlas space.
    gse_trd_latent:
        (n_gse_trd, d_latent) GSE98793 TRD-inflammatory cluster in atlas space.
    recovery_axis:
        (d_latent,) unit vector pointing from TRD-inflammatory toward healthy.
        Computed as ``unit(gtex_centroid - gse_trd_centroid)`` in latent space.
    terminus_recovery_score:
        (n_ptsd_subjects,) projection of each terminus onto ``recovery_axis``.
        Higher == more healthy-like, lower == more TRD-inflammatory-like.
        The proximity test's primary scalar input.
    subject_ids:
        (n_ptsd_subjects,) string identifiers, aligned with terminus_latent rows.
    response:
        (n_ptsd_subjects,) response labels, aligned with terminus_latent rows.
    cohort:
        (n_ptsd_subjects,) cohort tag ('Emory' / 'BEST').
    gtex_centroid_latent:
        (d_latent,) centroid of the GTEx cloud in atlas space.
    gse_trd_centroid_latent:
        (d_latent,) centroid of the GSE98793 TRD cluster in atlas space.
    n_gtex:
        Number of GTEx samples projected.
    n_gse_trd:
        Number of GSE98793 TRD samples projected.
    provenance:
        Dict with projection metadata: encoder class name, feature count,
        latent dim, gtex_source, gse_source, axis_computation_method.
    """

    terminus_latent: np.ndarray
    gtex_latent: np.ndarray
    gse_trd_latent: np.ndarray
    recovery_axis: np.ndarray
    terminus_recovery_score: np.ndarray
    subject_ids: np.ndarray
    response: np.ndarray
    cohort: np.ndarray
    gtex_centroid_latent: np.ndarray
    gse_trd_centroid_latent: np.ndarray
    n_gtex: int
    n_gse_trd: int
    provenance: dict[str, Any]


# ---------------------------------------------------------------------------
# Recovery axis construction
# ---------------------------------------------------------------------------


def build_two_anchor_recovery_axis(
    gtex_latent: np.ndarray,
    gse_trd_latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the two-anchor recovery axis from projected reference clouds.

    The axis is defined as::

        axis = unit( centroid(GTEx) - centroid(GSE_TRD) )

    i.e., it points from the TRD-inflammatory region toward the healthy region.
    Projection scores on this axis are positive for health-like and negative for
    disease-like coordinates.

    Rationale (framing-rationale doc Section 1): two anchors give the recovery
    axis a *destination*, not just a gradient. One anchor alone is consistent
    with regression to the mean; two anchors anchor the biological direction of
    recovery in the latent space.

    Parameters
    ----------
    gtex_latent:
        (n_gtex, d_latent) GTEx healthy reference cloud in atlas space.
    gse_trd_latent:
        (n_gse_trd, d_latent) GSE98793 TRD cluster in atlas space.

    Returns
    -------
    axis:
        (d_latent,) unit vector pointing toward health.
    gtex_centroid:
        (d_latent,) mean of the GTEx cloud.
    gse_trd_centroid:
        (d_latent,) mean of the TRD cloud.

    Raises
    ------
    ValueError:
        If either cloud has fewer than the required minimum samples, or if
        the two centroids are degenerate (distance < 1e-12).
    """
    if gtex_latent.shape[0] < _MIN_GTEX_SAMPLES:
        raise ValueError(
            f"GTEx cloud has only {gtex_latent.shape[0]} samples; "
            f"need >= {_MIN_GTEX_SAMPLES} for reliable centroid estimation."
        )
    if gse_trd_latent.shape[0] < _MIN_GSE_TRD_SAMPLES:
        raise ValueError(
            f"GSE98793 TRD cloud has only {gse_trd_latent.shape[0]} samples; "
            f"need >= {_MIN_GSE_TRD_SAMPLES} for reliable centroid estimation."
        )

    gtex_centroid = gtex_latent.mean(axis=0)
    gse_trd_centroid = gse_trd_latent.mean(axis=0)

    raw_axis = gtex_centroid - gse_trd_centroid
    axis_norm = float(np.linalg.norm(raw_axis))
    if axis_norm < 1e-12:
        raise ValueError(
            "GTEx centroid and GSE98793 TRD centroid are degenerate "
            "(distance < 1e-12) in atlas latent space. "
            "Check that the encoder was applied to both cohorts correctly."
        )

    axis = (raw_axis / axis_norm).astype(np.float64)
    logger.info(
        "Two-anchor recovery axis constructed: "
        "||gtex_centroid - gse_trd_centroid|| = %.4f, d_latent = %d",
        axis_norm,
        axis.shape[0],
    )
    return axis, gtex_centroid.astype(np.float64), gse_trd_centroid.astype(np.float64)


def project_onto_recovery_axis(
    terminus_latent: npt.NDArray[np.float64],
    recovery_axis_vec: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Project PTSD trajectory termini onto the two-anchor recovery axis.

    Parameters
    ----------
    terminus_latent:
        (n_subjects, d_latent) POST-IOP latent coordinates.
    recovery_axis_vec:
        (d_latent,) unit vector (output of :func:`build_two_anchor_recovery_axis`).

    Returns
    -------
    (n_subjects,) float64 recovery scores. Positive == healthy-like,
    negative == TRD-inflammatory-like.
    """
    if terminus_latent.shape[1] != recovery_axis_vec.shape[0]:
        raise ValueError(
            f"Dimension mismatch: terminus_latent has d={terminus_latent.shape[1]}, "
            f"recovery_axis has d={recovery_axis_vec.shape[0]}."
        )
    scores: npt.NDArray[np.float64] = np.asarray(
        terminus_latent @ recovery_axis_vec, dtype=np.float64
    )
    logger.debug(
        "Recovery scores: mean=%.4f, sd=%.4f, min=%.4f, max=%.4f",
        float(scores.mean()),
        float(scores.std()),
        float(scores.min()),
        float(scores.max()),
    )
    return scores


# ---------------------------------------------------------------------------
# Loading helpers for Phase 2 Wave 1 prepared external-cohort files
# ---------------------------------------------------------------------------


def load_external_cohorts_from_parquet(
    gtex_path: Path,
    gse_centroid_path: Path,
    gse_sample_path: Path | None = None,
) -> ExternalCohortData:
    """Load Phase 2 Wave 1 prepared external-cohort files.

    Both inputs were produced by Kai's Phase 2 Wave 1 external-cohort prep:
    - GTEx v10 whole blood, Emory-gene-panel aligned (``gtex_path``).
    - GSE98793 TRD group centroid (``gse_centroid_path``).
    - Optionally: individual-sample GSE98793 TRD expression (``gse_sample_path``);
      if absent, the centroid is used as a single representative sample.

    Parameters
    ----------
    gtex_path:
        Path to ``gtex_whole_blood_emory_aligned.parquet``. Columns = gene names;
        rows = samples (index = GTEx sample IDs).
    gse_centroid_path:
        Path to ``gse98793_group_centroids.parquet``. Must contain a row with
        index == 'TRD' (the TRD-inflammatory group centroid).
    gse_sample_path:
        Optional path to individual GSE98793 TRD-sample expression parquet
        (columns = gene names, rows = TRD samples). If provided, the axis
        computation uses the full cloud rather than the precomputed centroid.

    Returns
    -------
    ExternalCohortData
    """
    # GTEx
    if not gtex_path.exists():
        raise FileNotFoundError(
            f"GTEx prepared file not found: {gtex_path}\n"
            "Expected: Phase 2 Wave 1 output from Kai's external-cohort prep. "
            "Run the Phase 2 Wave 1 external-cohort preparation scripts first."
        )
    gtex_df = pd.read_parquet(gtex_path)
    gtex_rna = gtex_df.values.astype(np.float64)
    gtex_sample_ids = np.array(gtex_df.index.tolist(), dtype=object)
    feature_names = np.array(gtex_df.columns.tolist(), dtype=object)
    logger.info("Loaded GTEx: %d samples, %d features", *gtex_rna.shape)

    # GSE98793 centroids
    if not gse_centroid_path.exists():
        raise FileNotFoundError(
            f"GSE98793 centroid file not found: {gse_centroid_path}\n"
            "Expected: Phase 2 Wave 1 output from Kai's external-cohort prep."
        )
    centroid_df = pd.read_parquet(gse_centroid_path)
    if "TRD" not in centroid_df.index:
        raise ValueError(
            f"GSE98793 centroid file {gse_centroid_path} has no row with index 'TRD'. "
            f"Available rows: {list(centroid_df.index)}"
        )
    gse_trd_centroid_rna: npt.NDArray[np.float64] = np.asarray(
        centroid_df.loc["TRD"].to_numpy(), dtype=np.float64
    )

    # GSE98793 individual TRD samples (optional)
    if gse_sample_path is not None and gse_sample_path.exists():
        gse_df = pd.read_parquet(gse_sample_path)
        gse_trd_rna: npt.NDArray[np.float64] = np.asarray(gse_df.to_numpy(), dtype=np.float64)
        gse_trd_sample_ids = np.array(gse_df.index.tolist(), dtype=object)
        logger.info("Loaded GSE98793 TRD samples: %d samples", gse_trd_rna.shape[0])
    else:
        # Fall back to centroid as single representative row
        logger.warning(
            "GSE98793 TRD sample file not provided or not found. "
            "Using precomputed centroid as a single representative sample. "
            "The recovery axis will be less precise than a full-cloud estimate."
        )
        gse_trd_rna = gse_trd_centroid_rna.reshape(1, -1)
        gse_trd_sample_ids = np.array(["GSE98793_TRD_centroid"], dtype=object)

    return ExternalCohortData(
        gtex_rna=gtex_rna,
        gtex_sample_ids=gtex_sample_ids,
        gse_trd_rna=gse_trd_rna,
        gse_trd_sample_ids=gse_trd_sample_ids,
        gse_trd_centroid_rna=gse_trd_centroid_rna,
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# Main projection pipeline
# ---------------------------------------------------------------------------


def project_external_cohorts(
    encoder: AtlasEncoder,
    external_data: ExternalCohortData,
    ptsd_data: PtsdAtlasData,
    *,
    encoder_name: str = "unknown",
    gtex_source: str = "gtex_v10_whole_blood_emory_aligned",
    gse_source: str = "gse98793_trd_inflammatory",
) -> ProjectionResult:
    """Project GTEx + GSE98793 into the atlas latent space and build the recovery axis.

    This is the Phase 3 external-cohort projection entry point. It:

    1. Projects GTEx whole-blood samples through the RNA encoder.
    2. Projects GSE98793 TRD samples through the RNA encoder.
    3. Builds the two-anchor recovery axis from the projected reference clouds.
    4. Projects each PTSD trajectory terminus onto the recovery axis.
    5. Returns the :class:`ProjectionResult` consumed by Helen's proximity test.

    Parameters
    ----------
    encoder:
        The Phase 2 winning arm RNA encoder (must satisfy :class:`AtlasEncoder`).
        At scaffold time this is a synthetic fixture. At execution time it is
        loaded from the Phase 2 checkpoint directory.
    external_data:
        Phase 2 Wave 1 prepared external-cohort matrices.
    ptsd_data:
        PTSD subjects' POST-IOP latent coords from Phase 2 Step 3.0.
    encoder_name:
        Human-readable name for the encoder (e.g. ``'arm_c_contrastive'``).
        Recorded in provenance.
    gtex_source:
        Source tag for provenance.
    gse_source:
        Source tag for provenance.

    Returns
    -------
    ProjectionResult
    """
    n_rna = external_data.gtex_rna.shape[1]
    logger.info(
        "Projecting external cohorts: n_gtex=%d, n_gse_trd=%d, n_rna_features=%d",
        external_data.gtex_rna.shape[0],
        external_data.gse_trd_rna.shape[0],
        n_rna,
    )

    # Step 1: project GTEx
    gtex_latent = encoder.encode(external_data.gtex_rna).astype(np.float64)
    d_latent = gtex_latent.shape[1]
    logger.info("GTEx projected: %d samples -> latent dim %d", gtex_latent.shape[0], d_latent)

    # Step 2: project GSE98793 TRD
    gse_trd_latent = encoder.encode(external_data.gse_trd_rna).astype(np.float64)
    logger.info("GSE98793 TRD projected: %d samples", gse_trd_latent.shape[0])

    # Validate PTSD terminus dimensionality
    if ptsd_data.terminus_latent.shape[1] != d_latent:
        raise ValueError(
            f"PTSD terminus latent dim {ptsd_data.terminus_latent.shape[1]} "
            f"does not match external-cohort latent dim {d_latent}. "
            "Ensure all data passed through the same encoder."
        )

    if ptsd_data.terminus_latent.shape[0] < _MIN_PTSD_SUBJECTS:
        raise ValueError(
            f"Only {ptsd_data.terminus_latent.shape[0]} PTSD subjects; "
            f"need >= {_MIN_PTSD_SUBJECTS}."
        )

    # Step 3: build two-anchor recovery axis
    axis, gtex_centroid, gse_trd_centroid = build_two_anchor_recovery_axis(
        gtex_latent, gse_trd_latent
    )

    # Step 4: project PTSD termini
    scores = project_onto_recovery_axis(ptsd_data.terminus_latent, axis)

    provenance: dict[str, Any] = {
        "encoder": encoder_name,
        "n_rna_features": n_rna,
        "d_latent": d_latent,
        "gtex_source": gtex_source,
        "gse_source": gse_source,
        "n_gtex": gtex_latent.shape[0],
        "n_gse_trd": gse_trd_latent.shape[0],
        "n_ptsd_subjects": ptsd_data.terminus_latent.shape[0],
        "axis_computation_method": "unit(centroid(gtex) - centroid(gse_trd))",
    }

    logger.info(
        "Phase 3 projection complete. Recovery axis built. Mean recovery score: R=%.3f, NR=%.3f",
        float(scores[ptsd_data.response == "R"].mean())
        if np.any(ptsd_data.response == "R")
        else float("nan"),
        float(scores[ptsd_data.response == "NR"].mean())
        if np.any(ptsd_data.response == "NR")
        else float("nan"),
    )

    return ProjectionResult(
        terminus_latent=ptsd_data.terminus_latent.astype(np.float64),
        gtex_latent=gtex_latent,
        gse_trd_latent=gse_trd_latent,
        recovery_axis=axis,
        terminus_recovery_score=scores,
        subject_ids=ptsd_data.subject_ids,
        response=ptsd_data.response,
        cohort=ptsd_data.cohort,
        gtex_centroid_latent=gtex_centroid,
        gse_trd_centroid_latent=gse_trd_centroid,
        n_gtex=gtex_latent.shape[0],
        n_gse_trd=gse_trd_latent.shape[0],
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def save_projection_result(result: ProjectionResult, output_dir: Path) -> None:
    """Serialise ProjectionResult arrays to parquet + JSON for downstream steps.

    Writes the following files to ``output_dir``:

    - ``terminus_coords.parquet`` -- subject_id, response, cohort,
      recovery_score, latent_z0 ... latent_z{d-1}.
    - ``reference_clouds.parquet`` -- GTEx + TRD cloud latent coords with
      a ``cloud`` column ('gtex_healthy' or 'gse_trd_inflammatory').
    - ``recovery_axis.npy`` -- the (d_latent,) axis vector.
    - ``provenance.json`` -- metadata dict.

    These are the files Helen's proximity test and the visualisation notebook
    will consume.
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    d = result.terminus_latent.shape[1]

    # Terminus frame
    terminus_df = pd.DataFrame(
        result.terminus_latent,
        columns=[f"latent_z{i}" for i in range(d)],
    )
    terminus_df.insert(0, "subject_id", result.subject_ids)
    terminus_df.insert(1, "response", result.response)
    terminus_df.insert(2, "cohort", result.cohort)
    terminus_df.insert(3, "recovery_score", result.terminus_recovery_score)
    terminus_df.to_parquet(output_dir / "terminus_coords.parquet", index=False)

    # Reference cloud frame
    gtex_df = pd.DataFrame(result.gtex_latent, columns=[f"latent_z{i}" for i in range(d)])
    gtex_df["cloud"] = "gtex_healthy"
    gse_df = pd.DataFrame(result.gse_trd_latent, columns=[f"latent_z{i}" for i in range(d)])
    gse_df["cloud"] = "gse_trd_inflammatory"
    ref_df = pd.concat([gtex_df, gse_df], ignore_index=True)
    ref_df.to_parquet(output_dir / "reference_clouds.parquet", index=False)

    # Recovery axis
    np.save(output_dir / "recovery_axis.npy", result.recovery_axis)

    # Provenance
    prov = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in result.provenance.items()
    }
    prov["gtex_centroid_latent"] = result.gtex_centroid_latent.tolist()
    prov["gse_trd_centroid_latent"] = result.gse_trd_centroid_latent.tolist()
    with open(output_dir / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2)

    logger.info("ProjectionResult saved to %s", output_dir)


def load_projection_result(output_dir: Path) -> ProjectionResult:
    """Load a previously saved ProjectionResult from parquet + JSON files.

    Inverse of :func:`save_projection_result`. Used by Helen's proximity test
    and the visualisation notebook to load the Phase 3 projection outputs without
    depending on the encoder or external-cohort files.
    """
    import json

    terminus_df = pd.read_parquet(output_dir / "terminus_coords.parquet")
    ref_df = pd.read_parquet(output_dir / "reference_clouds.parquet")
    axis = np.load(output_dir / "recovery_axis.npy")

    d = axis.shape[0]
    latent_cols = [f"latent_z{i}" for i in range(d)]

    gtex_df = ref_df[ref_df["cloud"] == "gtex_healthy"]
    gse_df = ref_df[ref_df["cloud"] == "gse_trd_inflammatory"]

    with open(output_dir / "provenance.json") as f:
        prov_raw: dict[str, Any] = json.load(f)

    gtex_centroid = np.array(prov_raw.pop("gtex_centroid_latent"), dtype=np.float64)
    gse_trd_centroid = np.array(prov_raw.pop("gse_trd_centroid_latent"), dtype=np.float64)

    return ProjectionResult(
        terminus_latent=np.asarray(terminus_df[latent_cols].to_numpy(), dtype=np.float64),
        gtex_latent=np.asarray(gtex_df[latent_cols].to_numpy(), dtype=np.float64),
        gse_trd_latent=np.asarray(gse_df[latent_cols].to_numpy(), dtype=np.float64),
        recovery_axis=axis.astype(np.float64),
        terminus_recovery_score=np.asarray(
            terminus_df["recovery_score"].to_numpy(), dtype=np.float64
        ),
        subject_ids=np.asarray(terminus_df["subject_id"].to_numpy(), dtype=object),
        response=np.asarray(terminus_df["response"].to_numpy(), dtype=object),
        cohort=np.asarray(terminus_df["cohort"].to_numpy(), dtype=object),
        gtex_centroid_latent=gtex_centroid,
        gse_trd_centroid_latent=gse_trd_centroid,
        n_gtex=len(gtex_df),
        n_gse_trd=len(gse_df),
        provenance=prov_raw,
    )
