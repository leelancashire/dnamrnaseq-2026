"""Embedding rules: FM arm, MOFA+ arm, contrastive arm.

Phase 2.A of the v2.2 analysis plan. Design doc:
04-projects/dnamrnaseq/2026-05-19-phase-2-design.md (v1.1, approved).

Implementation status (2026-05-20, phase-2/implementation-scaffold):
  - Arm modules (arm_a_fm, arm_b_mofa, arm_c_contrastive), the data harness,
    feature selection, the conformal metric, and the six-metric leaderboard
    are all implemented in src/dnamrnaseq2026/ and synthetic-fixture tested.
  - The train_embedding rule below invokes 20_phase2_train_embedding.py, a thin
    orchestrator. Real-data training is GATED on the Phase 1 re-run producing
    cell-type-corrected feature inputs (Tier 1 CellDMC-prioritised set).
"""


rule train_embedding:
    """Train all three embedding arms (Arm A FM, Arm B MOFA+, Arm C contrastive).

    Gated on Phase 1 cell-type-corrected outputs; see the script docstring.
    """
    input:
        emory   = "analysis/latest/data_emory.parquet",
        best    = "analysis/latest/data_best.parquet",
    output:
        fm      = "analysis/latest/embedding_fm.pt",
        mofa    = "analysis/latest/embedding_mofa.h5",
        contra  = "analysis/latest/embedding_contrastive.pt",
    log:
        "analysis/latest/phase2_train_embedding.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/20_phase2_train_embedding.py > {log} 2>&1"


rule phase2_leaderboard:
    """Build the six-metric embedding-architecture leaderboard (design Section 3)."""
    input:
        fm      = "analysis/latest/embedding_fm.pt",
        mofa    = "analysis/latest/embedding_mofa.h5",
        contra  = "analysis/latest/embedding_contrastive.pt",
    output:
        board   = "analysis/latest/phase2_leaderboard.csv",
    log:
        "analysis/latest/phase2_leaderboard.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/21_phase2_leaderboard.py > {log} 2>&1"
