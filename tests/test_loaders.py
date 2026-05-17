"""Smoke tests for dnamrnaseq2026.data.loaders using synthetic fixtures.

These tests do NOT require OneDrive access and must pass in CI.
They test the loading logic, shape validation, and sample-ID alignment
against small synthetic DataFrames injected via conftest.py fixtures.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from dnamrnaseq2026.data.loaders import check_sample_alignment

# ---------------------------------------------------------------------------
# check_sample_alignment tests
# ---------------------------------------------------------------------------


class TestCheckSampleAlignment:
    """Tests for the sample-ID alignment checker."""

    def test_perfect_alignment(
        self,
        synthetic_bvals_emory: pd.DataFrame,
        synthetic_pdata_emory: pd.DataFrame,
    ) -> None:
        """No exception when all bVals columns are in pData2 index."""
        check_sample_alignment(synthetic_bvals_emory, synthetic_pdata_emory, cohort="Emory")

    def test_partial_alignment_raises(
        self,
        synthetic_bvals_emory: pd.DataFrame,
        synthetic_pdata_emory: pd.DataFrame,
    ) -> None:
        """ValueError when bVals has samples missing from pData2 index."""
        # bvals has CpG as index, samples as columns; build an orphan column properly.
        import numpy as np

        extra = pd.DataFrame(
            np.random.default_rng(99).uniform(0, 1, size=(len(synthetic_bvals_emory), 1)),
            index=synthetic_bvals_emory.index,
            columns=["orphan_sample"],
        )
        bvals_with_orphan = pd.concat([synthetic_bvals_emory, extra], axis=1)
        with pytest.raises(ValueError, match="alignment failure"):
            check_sample_alignment(bvals_with_orphan, synthetic_pdata_emory, cohort="Emory")

    def test_best_cohort_alignment(
        self,
        synthetic_bvals_best: pd.DataFrame,
        synthetic_pdata_best: pd.DataFrame,
    ) -> None:
        """No exception for BEST cohort synthetic fixtures."""
        check_sample_alignment(synthetic_bvals_best, synthetic_pdata_best, cohort="BEST")


# ---------------------------------------------------------------------------
# Loader function shape tests (mock the RData reads)
# ---------------------------------------------------------------------------


class TestLoaderShapes:
    """Tests that loaders return DataFrames of the expected shape.

    Uses unittest.mock to replace _load_rdata so we don't need real files.
    """

    def test_load_emory_bvals_shape(
        self, synthetic_bvals_emory: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_emory_bvals returns (n_cpg, n_sample) DataFrame."""
        # Create a fake file so the existence check passes
        fake_path = tmp_path / "emory.bVals.architecture.RData"
        fake_path.touch()

        with (
            patch(
                "dnamrnaseq2026.data.loaders.get_emory_dnam_dir",
                return_value=tmp_path,
            ),
            patch(
                "dnamrnaseq2026.data.loaders._load_rdata",
                return_value=synthetic_bvals_emory,
            ),
        ):
            from dnamrnaseq2026.data.loaders import load_emory_bvals

            df = load_emory_bvals()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == synthetic_bvals_emory.shape

    def test_load_emory_pdata2_shape(
        self, synthetic_pdata_emory: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_emory_pdata2 returns (n_sample, n_covariate) DataFrame."""
        fake_path = tmp_path / "emory_pData2.RData"
        fake_path.touch()

        with (
            patch(
                "dnamrnaseq2026.data.loaders.get_emory_dnam_dir",
                return_value=tmp_path,
            ),
            patch(
                "dnamrnaseq2026.data.loaders._load_rdata",
                return_value=synthetic_pdata_emory,
            ),
        ):
            from dnamrnaseq2026.data.loaders import load_emory_pdata2

            df = load_emory_pdata2()

        assert isinstance(df, pd.DataFrame)
        # Shape: n_samples x n_covariates (SampleName set as index)
        assert df.shape[0] == synthetic_pdata_emory.shape[0]

    def test_load_best_bvals_shape(
        self, synthetic_bvals_best: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_best_bvals returns (n_cpg, n_sample) DataFrame."""
        fake_path = tmp_path / "best.bVals.architecture.RData"
        fake_path.touch()

        with (
            patch(
                "dnamrnaseq2026.data.loaders.get_emory_dnam_dir",
                return_value=tmp_path,
            ),
            patch(
                "dnamrnaseq2026.data.loaders._load_rdata",
                return_value=synthetic_bvals_best,
            ),
        ):
            from dnamrnaseq2026.data.loaders import load_best_bvals

            df = load_best_bvals()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == synthetic_bvals_best.shape

    def test_load_best_pdata2_shape(
        self, synthetic_pdata_best: pd.DataFrame, tmp_path: Path
    ) -> None:
        """load_best_pdata2 returns (n_sample, n_covariate) DataFrame."""
        fake_path = tmp_path / "best_pData2.RData"
        fake_path.touch()

        with (
            patch(
                "dnamrnaseq2026.data.loaders.get_emory_dnam_dir",
                return_value=tmp_path,
            ),
            patch(
                "dnamrnaseq2026.data.loaders._load_rdata",
                return_value=synthetic_pdata_best,
            ),
        ):
            from dnamrnaseq2026.data.loaders import load_best_pdata2

            df = load_best_pdata2()

        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == synthetic_pdata_best.shape[0]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """load_emory_bvals raises FileNotFoundError when file is absent."""
        with patch(
            "dnamrnaseq2026.data.loaders.get_emory_dnam_dir",
            return_value=tmp_path,
        ):
            from dnamrnaseq2026.data.loaders import load_emory_bvals

            with pytest.raises(FileNotFoundError):
                load_emory_bvals()
