"""Trajectory atlas rules: archetype clustering, recovery-axis annotation.

Phase 2.B of the v2.2 analysis plan. All stubs.
"""


rule build_trajectory_atlas:
    """Build trajectory atlas from embedding outputs (stub)."""
    input:
        fm      = "analysis/latest/embedding_fm.pt",
        mofa    = "analysis/latest/embedding_mofa.h5",
        contra  = "analysis/latest/embedding_contrastive.pt",
    output:
        atlas   = "analysis/latest/trajectory_atlas.csv",
    shell:
        "echo 'TODO: implement trajectory atlas (Phase 2.B)'"


rule annotate_recovery_axis:
    """Annotate recovery axis with cell-type / pathway / TF biology (stub)."""
    input:
        atlas   = "analysis/latest/trajectory_atlas.csv",
    output:
        annot   = "analysis/latest/recovery_axis_annotation.csv",
    shell:
        "echo 'TODO: implement recovery axis annotation'"


rule cluster_archetypes:
    """Gaussian mixture / spectral clustering on delta-z (stub)."""
    input:
        atlas   = "analysis/latest/trajectory_atlas.csv",
    output:
        archs   = "analysis/latest/archetypes.csv",
    shell:
        "echo 'TODO: implement archetype clustering'"
