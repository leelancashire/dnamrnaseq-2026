"""Preprocessing rules: data loading, cell-type deconvolution (EpiDISH), CellDMC.

Phase 1 of the v2.2 analysis plan.

Rules implemented:
  - load_emory: RData -> parquet (functional)
  - load_best: RData -> parquet (functional)
  - epidish_emory: EpiDISH cell-type fraction estimation (stub)
  - epidish_best: EpiDISH cell-type fraction estimation (stub)
  - celldmc_pre_emory: CellDMC at PRE-IOP (stub)
  - celldmc_post_emory: CellDMC at POST-IOP (stub)
  - celldmc_delta_emory: CellDMC on within-subject delta (stub)
"""

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Rule: load_emory — RData -> parquet
# ---------------------------------------------------------------------------

rule load_emory:
    """Load Emory bVals + pData2, validate sample alignment, write parquet."""
    input:
        bvals   = str(Path(config["data"]["emory_dnam_dir"]) / "emory.bVals.architecture.RData"),
        pdata   = str(Path(config["data"]["emory_dnam_dir"]) / "emory_pData2.RData"),
    output:
        data    = "analysis/latest/data_emory.parquet",
    log:
        "analysis/latest/logs/load_emory.log",
    conda:
        "../envs/python-scientific.yaml"
    script:
        "../../scripts/snakemake/load_cohort.py"


# ---------------------------------------------------------------------------
# Rule: load_best — RData -> parquet
# ---------------------------------------------------------------------------

rule load_best:
    """Load BEST bVals + pData2, validate sample alignment, write parquet."""
    input:
        bvals   = str(Path(config["data"]["emory_dnam_dir"]) / "best.bVals.architecture.RData"),
        pdata   = str(Path(config["data"]["emory_dnam_dir"]) / "best_pData2.RData"),
    output:
        data    = "analysis/latest/data_best.parquet",
    log:
        "analysis/latest/logs/load_best.log",
    conda:
        "../envs/python-scientific.yaml"
    script:
        "../../scripts/snakemake/load_cohort.py"


# ---------------------------------------------------------------------------
# Rule: epidish_emory — EpiDISH cell-fraction estimation
# TODO: implement in Phase 1
# ---------------------------------------------------------------------------

rule epidish_emory:
    """EpiDISH cell-fraction estimation for Emory cohort (stub)."""
    input:
        bvals   = "analysis/latest/data_emory.parquet",
    output:
        props   = "analysis/latest/cell_props_emory.csv",
    log:
        "analysis/latest/logs/epidish_emory.log",
    shell:
        "echo 'TODO: implement EpiDISH for Emory' && touch {output.props}"


# ---------------------------------------------------------------------------
# Rule: epidish_best — EpiDISH cell-fraction estimation
# TODO: implement in Phase 1
# ---------------------------------------------------------------------------

rule epidish_best:
    """EpiDISH cell-fraction estimation for BEST cohort (stub)."""
    input:
        bvals   = "analysis/latest/data_best.parquet",
    output:
        props   = "analysis/latest/cell_props_best.csv",
    log:
        "analysis/latest/logs/epidish_best.log",
    shell:
        "echo 'TODO: implement EpiDISH for BEST' && touch {output.props}"


# ---------------------------------------------------------------------------
# Stub rules: CellDMC (to be implemented Phase 1.2)
# ---------------------------------------------------------------------------

rule celldmc_pre_emory:
    """CellDMC at PRE-IOP for Emory (stub)."""
    input:
        bvals   = "analysis/latest/data_emory.parquet",
        props   = "analysis/latest/cell_props_emory.csv",
    output:
        results = "analysis/latest/celldmc_pre_emory.tsv",
    shell:
        "echo 'TODO: CellDMC PRE-IOP Emory' && touch {output.results}"


rule celldmc_post_emory:
    """CellDMC at POST-IOP for Emory (stub)."""
    input:
        bvals   = "analysis/latest/data_emory.parquet",
        props   = "analysis/latest/cell_props_emory.csv",
    output:
        results = "analysis/latest/celldmc_post_emory.tsv",
    shell:
        "echo 'TODO: CellDMC POST-IOP Emory' && touch {output.results}"


rule celldmc_delta_emory:
    """CellDMC on within-subject delta for Emory (stub)."""
    input:
        pre     = "analysis/latest/celldmc_pre_emory.tsv",
        post    = "analysis/latest/celldmc_post_emory.tsv",
    output:
        results = "analysis/latest/celldmc_delta_emory.tsv",
    shell:
        "echo 'TODO: CellDMC delta Emory' && touch {output.results}"
