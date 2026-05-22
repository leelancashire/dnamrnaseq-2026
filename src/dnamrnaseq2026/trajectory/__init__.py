"""Trajectory archetype clustering, recovery-axis annotation, and external projection.

Implements design-doc Sections 3 (i), (vi) geometry: recovery-axis estimation,
trajectory-consistency scoring, GMM archetype clustering, and Phase 3
external-cohort projection into the atlas latent space.
"""

from dnamrnaseq2026.trajectory.external_projection import (
    AtlasEncoder,
    ExternalCohortData,
    ProjectionResult,
    PtsdAtlasData,
    build_two_anchor_recovery_axis,
    load_external_cohorts_from_parquet,
    load_projection_result,
    project_external_cohorts,
    project_onto_recovery_axis,
    save_projection_result,
)
from dnamrnaseq2026.trajectory.geometry import (
    ArchetypeResult,
    ConsistencyScore,
    across_seed_consistency,
    cluster_archetypes,
    recovery_axis,
    trajectory_consistency,
)

__all__ = [
    # geometry
    "ArchetypeResult",
    "ConsistencyScore",
    "across_seed_consistency",
    "cluster_archetypes",
    "recovery_axis",
    "trajectory_consistency",
    # external projection (Phase 3)
    "AtlasEncoder",
    "ExternalCohortData",
    "ProjectionResult",
    "PtsdAtlasData",
    "build_two_anchor_recovery_axis",
    "load_external_cohorts_from_parquet",
    "load_projection_result",
    "project_external_cohorts",
    "project_onto_recovery_axis",
    "save_projection_result",
]
