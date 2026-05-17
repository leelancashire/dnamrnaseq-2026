"""External cohort projection rules: GSE98793, GTEx, AURORA.

Phase 4 of the v2.2 analysis plan. All stubs.
"""


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
