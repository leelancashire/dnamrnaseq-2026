"""Embedding rules: FM arm, MOFA+ arm, contrastive arm.

Phase 2.A of the v2.2 analysis plan. All stubs — implemented after
Phase 0 gates confirm the signal exists.
"""


rule train_embedding:
    """Train all three embedding arms (stub)."""
    input:
        emory   = "analysis/latest/data_emory.parquet",
        best    = "analysis/latest/data_best.parquet",
    output:
        fm      = "analysis/latest/embedding_fm.pt",
        mofa    = "analysis/latest/embedding_mofa.h5",
        contra  = "analysis/latest/embedding_contrastive.pt",
    shell:
        "echo 'TODO: implement embedding training (Phase 2.A)'"
