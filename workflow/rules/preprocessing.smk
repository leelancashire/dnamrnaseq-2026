"""Preprocessing rules: data loading, cell-type deconvolution (EpiDISH), CellDMC.

Phase 0 gates and Phase 1 of the v2.2 analysis plan.

Rules implemented:
  Phase 0 gates:
  - step_0_T_pca_delta: Gate 0-T PCA of Emory paired delta-vectors
  - step_0_C_epidish_validation: Gate 0-C cell-type deconvolution validation
  - step_0_S_source_domain: Gate 0-S source-domain classifier (Emory vs BEST)
  Phase 1 data loading:
  - load_emory: RData -> parquet + pdata CSV (functional)
  - load_best: RData -> parquet (functional)
  Phase 1 steps (implemented, Python-wrapper path via python-scientific env):
  - step_1_1_epidish_emory: Step 1.1 EpiDISH for Emory (full deconvolution)
  - step_1_1_epidish_best: Step 1.1 EpiDISH for BEST (full deconvolution)
  - step_1_2_celldmc_pre_emory: Step 1.2a CellDMC PRE contrast
  - step_1_2_celldmc_post_emory: Step 1.2b CellDMC POST contrast
  - step_1_2_celldmc_delta_emory: Step 1.2c CellDMC delta + rescue 1.2.5
  - step_1_3_rnaseq_de_emory: Step 1.3 RNA-seq DE + rescue 1.3.5
  - step_1_4_pathway_activity: Step 1.4 PROGENy + Reactome/GSVA
  - step_1_5_tf_activity: Step 1.5 CollecTRI TF activity
  - step_1_6_regulatory_enrichment: Step 1.6 ENCODE TFBS / EpiMap enrichment
  - step_1_7_replication: Step 1.7 BEST replication
  R-direct rules (Snakemake --use-conda r-bioconductor.yaml, write to analysis/latest/):
  - epidish_emory, epidish_best: EpiDISH via Rscript (centEpicV2, RPC method)
  - celldmc_pre_emory, celldmc_post_emory, celldmc_delta_emory: CellDMC via Rscript
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Rule: load_emory — RData -> parquet
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 0 gate rules
# ---------------------------------------------------------------------------

rule step_0_T_pca_delta:
    """Gate 0-T: PCA of Emory paired delta-vectors with PERMANOVA."""
    output:
        results   = "analysis/2026-05-17-phase-0/0-T/gate_0T_results.json",
        loadings  = "analysis/2026-05-17-phase-0/0-T/gate_0T_loadings.csv",
        fig_png   = "analysis/2026-05-17-phase-0/0-T/gate_0T_pca_arrows.png",
    log:
        "analysis/2026-05-17-phase-0/0-T/gate_0T.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/01_phase0_gate_T.py > {log} 2>&1"


rule step_0_C_epidish_validation:
    """Gate 0-C: EpiDISH cell-type deconvolution validation."""
    output:
        results   = "analysis/2026-05-17-phase-0/0-C/gate_0C_results.json",
        fig_png   = "analysis/2026-05-17-phase-0/0-C/gate_0C_delta_props_hist.png",
    log:
        "analysis/2026-05-17-phase-0/0-C/gate_0C.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/01_phase0_gate_C.py > {log} 2>&1"


rule step_0_S_source_domain:
    """Gate 0-S: source-domain classifier (Emory vs BEST covariate shift)."""
    output:
        results   = "analysis/2026-05-17-phase-0/0-S/gate_0S_classifier.json",
        features  = "analysis/2026-05-17-phase-0/0-S/gate_0S_top_shifted_features.csv",
        fig_png   = "analysis/2026-05-17-phase-0/0-S/gate_0S_auc_roc.png",
    log:
        "analysis/2026-05-17-phase-0/0-S/gate_0S.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/01_phase0_gate_S.py > {log} 2>&1"


# ---------------------------------------------------------------------------
# Phase 1 data loading rules
# ---------------------------------------------------------------------------

rule load_emory:
    """Load Emory bVals + pData2, validate sample alignment, write parquet + pdata CSV.

    Outputs both the beta-value parquet (CpG x sample) and the pData CSV so that
    downstream CellDMC rules (celldmc_pre_emory, celldmc_post_emory,
    celldmc_delta_emory) have a single producing rule for pdata_emory.csv.
    CSV includes all pData2 columns; at minimum: Visit, Response, Age, Sex.

    Uses workflow/scripts/load_cohort.R (R-native RData reader) under the
    r-bioconductor conda env. Replaces the pyreadr-based load_cohort.py path
    which cannot handle R matrix objects (PyreadrError on bVals architecture
    matrix, 292674 CpGs x 388 samples). Fix applied 2026-05-21.

    Parquet orientation: CpG rows x sample columns (first col = "cpg").
    run_epidish.R and run_celldmc.R both expect this orientation.
    """
    input:
        bvals   = str(Path(config["data"]["emory_dnam_dir"]) / "emory.bVals.architecture.RData"),
        pdata   = str(Path(config["data"]["emory_dnam_dir"]) / "emory_pData2.RData"),
    output:
        data    = "analysis/latest/data_emory.parquet",
        pdata   = "analysis/latest/pdata_emory.csv",
    log:
        "analysis/latest/logs/load_emory.log",
    conda:
        "../envs/r-bioconductor.yaml"
    shell:
        "Rscript workflow/scripts/load_cohort.R"
        " --bvals    {input.bvals}"
        " --pdata    {input.pdata}"
        " --out_data {output.data}"
        " --out_pdata {output.pdata}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rule: load_best — RData -> parquet
# ---------------------------------------------------------------------------

rule load_best:
    """Load BEST bVals + pData2, validate sample alignment, write parquet.

    Uses workflow/scripts/load_cohort.R (R-native RData reader) under the
    r-bioconductor conda env. See load_emory for rationale.
    BEST does not currently need a separate pData CSV for CellDMC
    (no celldmc_best rules in Phase 1); out_pdata is omitted.
    """
    input:
        bvals   = str(Path(config["data"]["emory_dnam_dir"]) / "best.bVals.architecture.RData"),
        pdata   = str(Path(config["data"]["emory_dnam_dir"]) / "best_pData2.RData"),
    output:
        data    = "analysis/latest/data_best.parquet",
    log:
        "analysis/latest/logs/load_best.log",
    conda:
        "../envs/r-bioconductor.yaml"
    shell:
        "Rscript workflow/scripts/load_cohort.R"
        " --bvals    {input.bvals}"
        " --pdata    {input.pdata}"
        " --out_data {output.data}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Phase 1 rules
# ---------------------------------------------------------------------------

# Step 1.1: EpiDISH full deconvolution

rule step_1_1_epidish_emory:
    """Step 1.1: EpiDISH cell-fraction estimation for Emory cohort."""
    output:
        props       = "analysis/2026-05-17-phase-1/1.1/cell_props_emory.csv",
        pdata_aug   = "analysis/2026-05-17-phase-1/1.1/pdata_emory_with_epidish.csv",
        results_md  = "analysis/2026-05-17-phase-1/1.1/results.md",
    log:
        "analysis/2026-05-17-phase-1/1.1/step_1_1_emory.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/11_phase1_epidish.py > {log} 2>&1"


rule step_1_1_epidish_best:
    """Step 1.1: EpiDISH cell-fraction estimation for BEST cohort."""
    output:
        props       = "analysis/2026-05-17-phase-1/1.1/cell_props_best.csv",
        pdata_aug   = "analysis/2026-05-17-phase-1/1.1/pdata_best_with_epidish.csv",
    log:
        "analysis/2026-05-17-phase-1/1.1/step_1_1_best.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/11_phase1_epidish.py > {log} 2>&1"


# Step 1.2: CellDMC three-contrast + rescue check 1.2.5

rule step_1_2_celldmc_pre_emory:
    """Step 1.2a: CellDMC at PRE-IOP for Emory."""
    input:
        props   = "analysis/2026-05-17-phase-1/1.1/cell_props_emory.csv",
        pdata   = "analysis/2026-05-17-phase-1/1.1/pdata_emory_with_epidish.csv",
    output:
        results = "analysis/2026-05-17-phase-1/1.2/celldmc_pre_emory.tsv",
    log:
        "analysis/2026-05-17-phase-1/1.2/step_1_2_pre.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/12_phase1_celldmc.py > {log} 2>&1"


rule step_1_2_celldmc_post_emory:
    """Step 1.2b: CellDMC at POST-IOP for Emory."""
    input:
        props   = "analysis/2026-05-17-phase-1/1.1/cell_props_emory.csv",
        pdata   = "analysis/2026-05-17-phase-1/1.1/pdata_emory_with_epidish.csv",
    output:
        results = "analysis/2026-05-17-phase-1/1.2/celldmc_post_emory.tsv",
    log:
        "analysis/2026-05-17-phase-1/1.2/step_1_2_post.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/12_phase1_celldmc.py > {log} 2>&1"


rule step_1_2_celldmc_delta_emory:
    """Step 1.2c: CellDMC delta + rescue check 1.2.5 for Emory."""
    input:
        pre     = "analysis/2026-05-17-phase-1/1.2/celldmc_pre_emory.tsv",
        post    = "analysis/2026-05-17-phase-1/1.2/celldmc_post_emory.tsv",
    output:
        results     = "analysis/2026-05-17-phase-1/1.2/celldmc_delta_emory.tsv",
        rescue      = "analysis/2026-05-17-phase-1/1.2/rescue_1_2_5.json",
        results_md  = "analysis/2026-05-17-phase-1/1.2/results.md",
    log:
        "analysis/2026-05-17-phase-1/1.2/step_1_2_delta.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/12_phase1_celldmc.py > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rule: merge_pdata_epidish_emory
# Joins pdata_emory.csv (from load_emory) with cell_props_emory.csv (from
# epidish_emory) to produce the augmented pdata used by steps 1.3-1.5 and
# the gate-0-T re-run script.  Pure Python, no R env needed.
# Replaces the pdata_emory_with_epidish.csv that the old Python step_1_1_*
# rules wrote (those rules fall back to null pData2 and are superseded).
# ---------------------------------------------------------------------------

rule merge_pdata_epidish_emory:
    """Merge pdata_emory.csv + EpiDISH cell fractions into augmented pdata.

    Produces analysis/latest/pdata_emory_with_epidish.csv: all pData2 columns
    plus the 7 EpiDISH cell-type fraction columns (Bcell, CD4T, CD8T, Gran,
    Mono, NK, nRBC).  The join key is the shared sample index (SampleName /
    AMC-ID).  Samples present in pdata but absent from cell_props get NaN
    cell-fraction columns; a warning is logged.
    """
    input:
        pdata   = "analysis/latest/pdata_emory.csv",
        props   = "analysis/latest/cell_props_emory.csv",
    output:
        pdata_aug = "analysis/latest/pdata_emory_with_epidish.csv",
    log:
        "analysis/latest/logs/merge_pdata_epidish_emory.log",
    conda:
        "../envs/python-scientific.yaml"
    run:
        import logging
        import pandas as pd

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log[0]), logging.StreamHandler()],
        )
        logger = logging.getLogger("merge_pdata_epidish_emory")

        pdata = pd.read_csv(input.pdata, index_col=0)
        props = pd.read_csv(input.props, index_col=0)

        logger.info("pdata: %d rows x %d cols", len(pdata), len(pdata.columns))
        logger.info("cell_props: %d rows x %d cols", len(props), len(props.columns))

        overlap = pdata.index.intersection(props.index)
        missing = pdata.index.difference(props.index)
        if len(missing) > 0:
            logger.warning(
                "%d pdata samples absent from cell_props (will be NaN): %s",
                len(missing),
                list(missing)[:10],
            )
        logger.info("%d samples aligned between pdata and cell_props.", len(overlap))

        pdata_aug = pdata.join(props, how="left")
        pdata_aug.to_csv(output.pdata_aug, index=True)
        logger.info(
            "Written: %s (%d rows x %d cols)",
            output.pdata_aug,
            len(pdata_aug),
            len(pdata_aug.columns),
        )


# Step 1.3: Cell-type-corrected RNA-seq DE + rescue check 1.3.5

rule step_1_3_rnaseq_de_emory:
    """Step 1.3: Cell-type-corrected RNA-seq DE at PRE, POST, delta + rescue 1.3.5."""
    input:
        props   = "analysis/latest/cell_props_emory.csv",
        pdata   = "analysis/latest/pdata_emory_with_epidish.csv",
    output:
        de_pre      = "analysis/2026-05-17-phase-1/1.3/de_pre_emory.tsv",
        de_post     = "analysis/2026-05-17-phase-1/1.3/de_post_emory.tsv",
        de_delta    = "analysis/2026-05-17-phase-1/1.3/de_delta_emory.tsv",
        rescue      = "analysis/2026-05-17-phase-1/1.3/rescue_1_3_5.json",
        results_md  = "analysis/2026-05-17-phase-1/1.3/results.md",
    log:
        "analysis/2026-05-17-phase-1/1.3/step_1_3.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/13_phase1_rnaseq_de.py > {log} 2>&1"


# Step 1.4: Pathway activity (PROGENy + Reactome/GSVA)

rule step_1_4_pathway_activity:
    """Step 1.4: decoupleR pathway activity (PROGENy + Reactome/GSVA)."""
    input:
        pdata   = "analysis/latest/pdata_emory_with_epidish.csv",
    output:
        results_md  = "analysis/2026-05-17-phase-1/1.4/results.md",
        test_tsv    = "analysis/2026-05-17-phase-1/1.4/pathway_response_test.tsv",
    log:
        "analysis/2026-05-17-phase-1/1.4/step_1_4.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/14_phase1_pathway_activity.py > {log} 2>&1"


# Step 1.5: TF activity (CollecTRI)

rule step_1_5_tf_activity:
    """Step 1.5: decoupleR TF activity (CollecTRI)."""
    input:
        pdata   = "analysis/latest/pdata_emory_with_epidish.csv",
    output:
        results_md  = "analysis/2026-05-17-phase-1/1.5/results.md",
        test_tsv    = "analysis/2026-05-17-phase-1/1.5/tf_response_test.tsv",
        priority    = "analysis/2026-05-17-phase-1/1.5/priority_tf_table.tsv",
    log:
        "analysis/2026-05-17-phase-1/1.5/step_1_5.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/15_phase1_tf_activity.py > {log} 2>&1"


# Step 1.6: Regulatory enrichment (ENCODE TFBS / EpiMap)

rule step_1_6_regulatory_enrichment:
    """Step 1.6: ENCODE TFBS / EpiMap enrichment on CellDMC delta CpGs."""
    input:
        celldmc = "analysis/latest/celldmc_delta_emory.tsv",
    output:
        enrichment  = "analysis/2026-05-17-phase-1/1.6/regulatory_enrichment.tsv",
        results_md  = "analysis/2026-05-17-phase-1/1.6/results.md",
    log:
        "analysis/2026-05-17-phase-1/1.6/step_1_6.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/16_phase1_regulatory_enrichment.py > {log} 2>&1"


# Step 1.7: BEST replication

rule step_1_7_replication:
    """Step 1.7: BEST replication of Emory CellDMC delta-significant CpGs."""
    input:
        celldmc     = "analysis/latest/celldmc_delta_emory.tsv",
        props_best  = "analysis/latest/cell_props_best.csv",
    output:
        overall         = "analysis/2026-05-17-phase-1/1.7/replication_overall.tsv",
        summary_json    = "analysis/2026-05-17-phase-1/1.7/replication_summary.json",
        results_md      = "analysis/2026-05-17-phase-1/1.7/results.md",
    log:
        "analysis/2026-05-17-phase-1/1.7/step_1_7.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/17_phase1_replication.py > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rule: epidish_emory -- EpiDISH cell-fraction estimation (Phase 1 Step 1.1)
# Wired to the dnamrnaseq2026-r-bioc conda env via --use-conda.
# R script writes CSV; Python rules read it. rpy2 is NOT used.
# ---------------------------------------------------------------------------

rule epidish_emory:
    """EpiDISH cell-fraction estimation for Emory cohort.

    Calls workflow/scripts/run_epidish.R with the centEpicV2 reference panel
    (IDOL-optimised, ships with the EpiDISH package -- no external download).
    Output: sample x cell-type CSV (7 cell types: Bcell, CD4T, CD8T, Gran,
    Mono, NK, nRBC). Row sums ~1. Variance check included in the R script.

    Architecture: Option 2 (Snakemake --use-conda R env, R writes CSV,
    Python reads CSV). Chosen over rpy2 for stability.
    """
    input:
        bvals   = "analysis/latest/data_emory.parquet",
    output:
        props   = "analysis/latest/cell_props_emory.csv",
    log:
        "analysis/latest/logs/epidish_emory.log",
    conda:
        "../envs/r-bioconductor.yaml"
    shell:
        "Rscript workflow/scripts/run_epidish.R"
        " --input {input.bvals}"
        " --output {output.props}"
        " --ref centEpicV2"
        " --method RPC"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rule: epidish_best -- EpiDISH cell-fraction estimation (Phase 1 Step 1.1)
# ---------------------------------------------------------------------------

rule epidish_best:
    """EpiDISH cell-fraction estimation for BEST cohort.

    Same reference panel (centEpicV2) and method (RPC) as Emory for
    cross-cohort comparability. See epidish_emory for architecture notes.
    """
    input:
        bvals   = "analysis/latest/data_best.parquet",
    output:
        props   = "analysis/latest/cell_props_best.csv",
    log:
        "analysis/latest/logs/epidish_best.log",
    conda:
        "../envs/r-bioconductor.yaml"
    shell:
        "Rscript workflow/scripts/run_epidish.R"
        " --input {input.bvals}"
        " --output {output.props}"
        " --ref centEpicV2"
        " --method RPC"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rules: CellDMC (Phase 1 Step 1.2)
# CellDMC() is exported from EpiDISH -- no separate package needed.
# Three contrasts: PRE-IOP, POST-IOP, and within-subject delta.
# pdata_emory.csv is produced by the load_emory rule (output.pdata).
# The load_emory rule is the single authoritative producer of this file.
# ---------------------------------------------------------------------------

rule celldmc_pre_emory:
    """CellDMC response x cell-type interaction at PRE-IOP for Emory.

    Fits: M_cpg ~ Response + cell_type_j + Response:cell_type_j + Age + Sex
    Interaction term identifies cell-type-specific DMPs for Treatment Response.
    Uses real EpiDISH cell fractions from epidish_emory (not pData2 fallback).
    """
    input:
        bvals   = "analysis/latest/data_emory.parquet",
        props   = "analysis/latest/cell_props_emory.csv",
        pdata   = "analysis/latest/pdata_emory.csv",
    output:
        results = "analysis/latest/celldmc_pre_emory.tsv",
    log:
        "analysis/latest/logs/celldmc_pre_emory.log",
    conda:
        "../envs/r-bioconductor.yaml"
    threads: 4
    shell:
        "Rscript workflow/scripts/run_celldmc.R"
        " --bvals  {input.bvals}"
        " --fracs  {input.props}"
        " --pdata  {input.pdata}"
        " --pheno  Response"
        " --visit  PRE-IOP"
        " --covars Age,sex"
        " --output {output.results}"
        " --fdr    0.05"
        " --ncore  {threads}"
        " > {log} 2>&1"


rule celldmc_post_emory:
    """CellDMC response x cell-type interaction at POST-IOP for Emory."""
    input:
        bvals   = "analysis/latest/data_emory.parquet",
        props   = "analysis/latest/cell_props_emory.csv",
        pdata   = "analysis/latest/pdata_emory.csv",
    output:
        results = "analysis/latest/celldmc_post_emory.tsv",
    log:
        "analysis/latest/logs/celldmc_post_emory.log",
    conda:
        "../envs/r-bioconductor.yaml"
    threads: 4
    shell:
        "Rscript workflow/scripts/run_celldmc.R"
        " --bvals  {input.bvals}"
        " --fracs  {input.props}"
        " --pdata  {input.pdata}"
        " --pheno  Response"
        " --visit  POST-IOP"
        " --covars Age,sex"
        " --output {output.results}"
        " --fdr    0.05"
        " --ncore  {threads}"
        " > {log} 2>&1"


rule celldmc_delta_emory:
    """CellDMC response x cell-type on within-subject delta (POST - PRE) for Emory.

    Uses all visits (--visit ALL) because the delta is a per-subject single row.
    The R script subsets to paired subjects; the pdata must include a column
    that enables paired matching (Subject_ID or equivalent).
    """
    input:
        bvals   = "analysis/latest/data_emory.parquet",
        props   = "analysis/latest/cell_props_emory.csv",
        pdata   = "analysis/latest/pdata_emory.csv",
    output:
        results = "analysis/latest/celldmc_delta_emory.tsv",
    log:
        "analysis/latest/logs/celldmc_delta_emory.log",
    conda:
        "../envs/r-bioconductor.yaml"
    threads: 4
    shell:
        "Rscript workflow/scripts/run_celldmc.R"
        " --bvals  {input.bvals}"
        " --fracs  {input.props}"
        " --pdata  {input.pdata}"
        " --pheno  Response"
        " --visit  ALL"
        " --covars Age,sex"
        " --output {output.results}"
        " --fdr    0.05"
        " --ncore  {threads}"
        " > {log} 2>&1"


# ---------------------------------------------------------------------------
# Rule: gate_0T_rerun_celldmc
# Re-runs Gate 0-T using cell-type-corrected paired-delta matrices.  Wires
# the full dependency chain: load_emory -> epidish_emory ->
# merge_pdata_epidish_emory -> celldmc_delta_emory -> gate_0T_rerun_celldmc.
# Fires automatically once step 1.2 (celldmc_delta_emory) completes.
# ---------------------------------------------------------------------------

rule gate_0T_rerun_celldmc:
    """Gate 0-T re-run: PERMANOVA + Cohen's d on CellDMC-corrected paired-Δ PCA.

    Re-runs Gate 0-T (raw verdict: MARGINAL, PERMANOVA p=0.111, max d=0.267)
    using cell-type-corrected paired-delta matrices after real EpiDISH cell
    fractions and CellDMC interaction-term outputs are available.

    Inputs consumed:
      - analysis/latest/cell_props_emory.csv  (from epidish_emory)
      - analysis/latest/pdata_emory_with_epidish.csv  (from merge_pdata_epidish_emory)
      - analysis/latest/celldmc_delta_emory.tsv  (from celldmc_delta_emory)

    The gate script also loads Emory bVals + RNA-seq via loaders -- those
    are not declared here because they are raw data files (not Snakemake
    outputs), but declaring the three artefact inputs is sufficient for
    correct DAG ordering.
    """
    input:
        cell_props  = "analysis/latest/cell_props_emory.csv",
        pdata_aug   = "analysis/latest/pdata_emory_with_epidish.csv",
        celldmc     = "analysis/latest/celldmc_delta_emory.tsv",
    output:
        results_json    = "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/gate_0T_rerun_results.json",
        loadings_csv    = "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/gate_0T_rerun_loadings.csv",
        fig_png         = "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/gate_0T_rerun_pca_arrows.png",
        results_md      = "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/results.md",
    log:
        "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/gate_0T_rerun.log",
    conda:
        "../envs/python-scientific.yaml"
    shell:
        "python scripts/01_phase0_gate_t_rerun_celldmc.py > {log} 2>&1"
