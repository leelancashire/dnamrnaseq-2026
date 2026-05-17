"""Generate tiny synthetic data for smoke-pipeline CI tests.

Creates minimal RData-like artefacts (as parquet, since the CI environment
does not have OneDrive data access) that mimic the shape of the real data.
Also used by conftest.py fixtures for unit tests.

Usage (standalone):
    python tests/fixtures/generate_synthetic.py --output-dir /tmp/dnamrnaseq-test

The smoke-pipeline workflow calls this before running Snakemake on synthetic data.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Synthetic dimensions (small enough for CI; mirrors real data shape)
N_CPGS_EMORY = 100
N_CPGS_BEST = 100
N_SAMPLES_EMORY = 10
N_SAMPLES_BEST = 8
N_COVARIATES = 12

EMORY_SEED = 42
BEST_SEED = 43


def make_bvals(n_cpgs: int, n_samples: int, seed: int, prefix: str) -> pd.DataFrame:
    """Return a (n_cpgs x n_samples) DataFrame of beta values in [0, 1]."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 1.0, size=(n_cpgs, n_samples))
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
    sample_ids = [f"{prefix}_sample_{i:03d}" for i in range(n_samples)]
    return pd.DataFrame(data, index=cpg_ids, columns=sample_ids)


def make_pdata(sample_ids: list[str], cohort: str, seed: int) -> pd.DataFrame:
    """Return a (n_samples x n_covariates) DataFrame with minimal clinical columns."""
    rng = np.random.default_rng(seed)
    n = len(sample_ids)
    return pd.DataFrame(
        {
            "Response": rng.choice(["R", "NR"], size=n),
            "Visit": rng.choice(["PRE_IOP", "POST_IOP"], size=n),
            "Therapy_type": rng.choice(["CPT", "PE", "None"], size=n),
            "Age": rng.integers(20, 70, size=n),
            "Sex": rng.choice(["M", "F"], size=n),
            "Cohort": [cohort] * n,
            "PCL_total": rng.integers(10, 80, size=n).astype(float),
            "CAPS5_total": rng.integers(10, 80, size=n).astype(float),
            "N2LR": rng.uniform(1.0, 5.0, size=n),
            "ancestry_pca_PCA1": rng.standard_normal(n),
            "ancestry_pca_PCA2": rng.standard_normal(n),
            "Batch": rng.choice(["B1", "B2"], size=n),
        },
        index=pd.Index(sample_ids, name="SampleName"),
    )


def generate(output_dir: Path) -> None:
    """Generate and write synthetic data files to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Emory
    emory_bvals = make_bvals(N_CPGS_EMORY, N_SAMPLES_EMORY, EMORY_SEED, "emory")
    emory_pdata = make_pdata(emory_bvals.columns.tolist(), "Emory", EMORY_SEED + 10)

    # BEST
    best_bvals = make_bvals(N_CPGS_BEST, N_SAMPLES_BEST, BEST_SEED, "best")
    best_pdata = make_pdata(best_bvals.columns.tolist(), "BEST", BEST_SEED + 10)

    # Write as parquet (matches Snakemake rule output format)
    # bVals are transposed (samples x CpGs) for downstream use
    emory_bvals.T.to_parquet(output_dir / "data_emory.parquet")
    best_bvals.T.to_parquet(output_dir / "data_best.parquet")

    # Write pData as CSV (for inspection)
    emory_pdata.to_csv(output_dir / "pdata_emory.csv")
    best_pdata.to_csv(output_dir / "pdata_best.csv")

    # Write stub CSV files to satisfy downstream rules
    for stub_file in [
        "cell_props_emory.csv",
        "cell_props_best.csv",
        "celldmc_pre_emory.tsv",
        "celldmc_post_emory.tsv",
        "celldmc_delta_emory.tsv",
        "embedding_fm.pt",
        "embedding_mofa.h5",
        "embedding_contrastive.pt",
        "trajectory_atlas.csv",
        "recovery_axis_annotation.csv",
        "archetypes.csv",
        "terminus_gse98793.csv",
        "terminus_gtex.csv",
        "terminus_test.csv",
        "hima_results.csv",
        "bama_results.csv",
        "mediation_evalues.csv",
    ]:
        (output_dir / stub_file).touch()

    print(f"Synthetic data written to {output_dir}")
    print(f"  Emory bVals: {emory_bvals.shape}  (CpGs x samples)")
    print(f"  Emory pData: {emory_pdata.shape}  (samples x covariates)")
    print(f"  BEST  bVals: {best_bvals.shape}  (CpGs x samples)")
    print(f"  BEST  pData: {best_pdata.shape}  (samples x covariates)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic test data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/latest"),
        help="Directory to write synthetic data (default: analysis/latest).",
    )
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
