"""External cohort projection rules: GSE98793, GTEx, AURORA.

Phase 0 Gate 0-X and Phase 4 of the v2.2 analysis plan.
"""


# ---------------------------------------------------------------------------
# Phase 0: Gate 0-X cross-disorder centroid projection
# ---------------------------------------------------------------------------

rule step_0_X_cross_disorder_centroid:
    """Gate 0-X: cross-disorder centroid projection (Emory vs GSE98793 TRD)."""
    output:
        results   = "analysis/2026-05-17-phase-0/0-X/gate_0X_centroids.json",
        genes     = "analysis/2026-05-17-phase-0/0-X/gate_0X_genes_used.csv",
        fig_png   = "analysis/2026-05-17-phase-0/0-X/gate_0X_centroid_projection.png",
    log:
        "analysis/2026-05-17-phase-0/0-X/gate_0X.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/01_phase0_gate_X.py > {log} 2>&1"


# ---------------------------------------------------------------------------
# Phase 4 stubs
# ---------------------------------------------------------------------------

rule project_gse98793:
    """Project GSE98793 TRD cohort into trajectory space (stub)."""
    input:
        atlas   = "analysis/latest/trajectory_atlas.csv",
    output:
        terminus = "analysis/latest/terminus_gse98793.csv",
    shell:
        "echo 'TODO: implement GSE98793 projection'"


rule project_gtex:
    """Project GTEx whole-blood healthy reference into trajectory space (stub)."""
    input:
        atlas   = "analysis/latest/trajectory_atlas.csv",
    output:
        terminus = "analysis/latest/terminus_gtex.csv",
    shell:
        "echo 'TODO: implement GTEx projection'"


rule terminus_permanova:
    """PERMANOVA test on cohort terminus clusters (stub)."""
    input:
        gse     = "analysis/latest/terminus_gse98793.csv",
        gtex    = "analysis/latest/terminus_gtex.csv",
    output:
        test    = "analysis/latest/terminus_test.csv",
    shell:
        "echo 'TODO: implement PERMANOVA terminus test'"
