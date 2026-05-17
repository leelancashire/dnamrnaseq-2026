"""Mediation rules: HIMA, BAMA, E-value sensitivity.

Phase 3 of the v2.2 analysis plan. All stubs.
"""


rule hima_delta:
    """HIMA high-dimensional mediation on within-subject deltas (stub)."""
    input:
        data    = "analysis/latest/trajectory_atlas.csv",
    output:
        results = "analysis/latest/hima_results.csv",
    shell:
        "echo 'TODO: implement HIMA mediation'"


rule bama_delta:
    """BAMA Bayesian mediation on within-subject deltas (stub)."""
    input:
        data    = "analysis/latest/trajectory_atlas.csv",
    output:
        results = "analysis/latest/bama_results.csv",
    shell:
        "echo 'TODO: implement BAMA mediation'"


rule evalue_sensitivity:
    """E-value sensitivity analysis for mediation estimates (stub)."""
    input:
        hima    = "analysis/latest/hima_results.csv",
        bama    = "analysis/latest/bama_results.csv",
    output:
        evalues = "analysis/latest/mediation_evalues.csv",
    shell:
        "echo 'TODO: implement E-value sensitivity'"
