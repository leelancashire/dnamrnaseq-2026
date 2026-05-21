"""Three-arm embedding: foundation-model, MOFA+, contrastive trajectory.

Phase 2 of the v2.2 analysis plan. See ``04-projects/dnamrnaseq/`` design doc
``2026-05-19-phase-2-design.md`` for the architecture rationale.

Public surface:

- ``data_harness`` — paired (subject, visit) tensors + subject-level splits.
- ``feature_selection`` — two-tier feature subsetting + Phase 1 artefact readers.
- ``arm_a_fm`` — Arm A: pathway-activity encoder + trajectory-consistency head.
- ``arm_b_mofa`` — Arm B: MOFA+ + trait-state LMM-LRT classification.
- ``arm_c_contrastive`` — Arm C: contrastive within-subject encoder.
- ``leaderboard`` — six-metric leaderboard aggregation.
"""

from dnamrnaseq2026.embedding.data_harness import (
    PairedDataset,
    PairedPreprocessor,
    build_paired_dataset,
    inner_calibration_split,
    subject_level_folds,
)
from dnamrnaseq2026.embedding.feature_selection import (
    FeatureTier,
    resolve_feature_tier,
)

__all__ = [
    "PairedDataset",
    "PairedPreprocessor",
    "build_paired_dataset",
    "subject_level_folds",
    "inner_calibration_split",
    "FeatureTier",
    "resolve_feature_tier",
]
