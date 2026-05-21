"""Trajectory archetype clustering and recovery-axis annotation.

Implements design-doc Section 3 (i) and (vi) geometry: recovery-axis estimation,
trajectory-consistency scoring, and GMM archetype clustering.
"""

from dnamrnaseq2026.trajectory.geometry import (
    ArchetypeResult,
    ConsistencyScore,
    across_seed_consistency,
    cluster_archetypes,
    recovery_axis,
    trajectory_consistency,
)

__all__ = [
    "ArchetypeResult",
    "ConsistencyScore",
    "across_seed_consistency",
    "cluster_archetypes",
    "recovery_axis",
    "trajectory_consistency",
]
