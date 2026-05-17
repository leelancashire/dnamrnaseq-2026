"""Shared pytest fixtures for dnamrnaseq2026 tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Synthetic data dimensions (tiny: CI-safe, no OneDrive access needed)
N_CPGS = 100
N_GENES = 50
N_SAMPLES_EMORY = 10
N_SAMPLES_BEST = 8
N_COVARIATES = 5


@pytest.fixture
def synthetic_bvals_emory() -> pd.DataFrame:
    """10 synthetic Emory samples x 100 CpG sites (beta values in [0, 1])."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.0, 1.0, size=(N_CPGS, N_SAMPLES_EMORY))
    cpg_ids = [f"cg{i:08d}" for i in range(N_CPGS)]
    sample_ids = [f"emory_sample_{i:03d}" for i in range(N_SAMPLES_EMORY)]
    return pd.DataFrame(data, index=cpg_ids, columns=sample_ids)


@pytest.fixture
def synthetic_bvals_best() -> pd.DataFrame:
    """8 synthetic BEST samples x 100 CpG sites (beta values in [0, 1])."""
    rng = np.random.default_rng(43)
    data = rng.uniform(0.0, 1.0, size=(N_CPGS, N_SAMPLES_BEST))
    cpg_ids = [f"cg{i:08d}" for i in range(N_CPGS)]
    sample_ids = [f"best_sample_{i:03d}" for i in range(N_SAMPLES_BEST)]
    return pd.DataFrame(data, index=cpg_ids, columns=sample_ids)


@pytest.fixture
def synthetic_pdata_emory(synthetic_bvals_emory: pd.DataFrame) -> pd.DataFrame:
    """Synthetic Emory pData2: sample_ids as index, minimal covariate columns."""
    rng = np.random.default_rng(44)
    sample_ids = synthetic_bvals_emory.columns.tolist()
    return pd.DataFrame(
        {
            "Response": rng.choice(["R", "NR"], size=N_SAMPLES_EMORY),
            "Age": rng.integers(20, 70, size=N_SAMPLES_EMORY),
            "Sex": rng.choice(["M", "F"], size=N_SAMPLES_EMORY),
            "Visit": rng.choice(["PRE_IOP", "POST_IOP"], size=N_SAMPLES_EMORY),
            "Cohort": ["Emory"] * N_SAMPLES_EMORY,
        },
        index=sample_ids,
    )


@pytest.fixture
def synthetic_pdata_best(synthetic_bvals_best: pd.DataFrame) -> pd.DataFrame:
    """Synthetic BEST pData2: sample_ids as index, minimal covariate columns."""
    rng = np.random.default_rng(45)
    sample_ids = synthetic_bvals_best.columns.tolist()
    return pd.DataFrame(
        {
            "Response": rng.choice(["R", "NR"], size=N_SAMPLES_BEST),
            "Age": rng.integers(20, 70, size=N_SAMPLES_BEST),
            "Sex": rng.choice(["M", "F"], size=N_SAMPLES_BEST),
            "Visit": rng.choice(["PRE_IOP", "POST_IOP"], size=N_SAMPLES_BEST),
            "Cohort": ["BEST"] * N_SAMPLES_BEST,
        },
        index=sample_ids,
    )
