"""Top-level Snakemake DAG for dnamrnaseq-2026.

Analysis plan v2.2, Section 13.5. See workflow/rules/ for rule modules.

Usage:
    # Dry run
    snakemake --use-conda --cores 4 -n

    # Run preprocessing phase
    snakemake --use-conda --cores 4 preprocess_emory preprocess_best

    # Full pipeline (when all rules implemented)
    snakemake --use-conda --cores 4 all

    # Visualise DAG
    snakemake --dag | dot -Tpng > dag.png
"""

import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path("config.yaml.example")

configfile: str(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Include rule modules
# ---------------------------------------------------------------------------

include: "workflow/rules/preprocessing.smk"
include: "workflow/rules/embedding.smk"
include: "workflow/rules/trajectory.smk"
include: "workflow/rules/mediation.smk"
include: "workflow/rules/external_projection.smk"
include: "workflow/rules/manuscript_figures.smk"


# ---------------------------------------------------------------------------
# All rule: full pipeline target
# ---------------------------------------------------------------------------

rule all:
    input:
        # Preprocessing outputs
        "analysis/latest/data_emory.parquet",
        "analysis/latest/data_best.parquet",
        "analysis/latest/cell_props_emory.csv",
        "analysis/latest/cell_props_best.csv",
        # Embedding outputs (TODO: implemented in Phase 2.A)
        # Trajectory outputs (TODO: implemented in Phase 2.B)
        # Manuscript figures (TODO: implemented when analysis is complete)
    default_target: True
