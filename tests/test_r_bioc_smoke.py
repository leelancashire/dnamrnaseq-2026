"""Smoke tests for the R-Bioconductor conda env (EpiDISH + CellDMC wrappers).

These tests require the dnamrnaseq2026-r-bioc conda env to be active and
Rscript to be on PATH. They are skipped in default CI (which does not install
the full Bioconductor stack) and run via the separate r-bioc.yml CI workflow
(weekly or manual trigger).

Marks:
    requires_r_bioc : tests that need R + EpiDISH + CellDMC installed.

Run:
    # Activate the R-bioc env first:
    conda activate dnamrnaseq2026-r-bioc

    # Then run the marked tests:
    pytest tests/test_r_bioc_smoke.py -m requires_r_bioc -v

All tests use synthetic fixtures only (no OneDrive data).
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mark: skip unless the r-bioc env is active (Rscript on PATH + EpiDISH loads)
# ---------------------------------------------------------------------------

requires_r_bioc = pytest.mark.skipif(
    shutil.which("Rscript") is None,
    reason="Rscript not found; activate dnamrnaseq2026-r-bioc conda env",
)


def _r_library_available(package: str) -> bool:
    """Return True if the named R package can be loaded in the active Rscript."""
    result = subprocess.run(
        ["Rscript", "-e", f"library({package}); cat('OK')"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "OK" in result.stdout


def _assert_r_bioc_env() -> None:
    """Raise if EpiDISH is not importable. Called inside each test."""
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not on PATH; activate dnamrnaseq2026-r-bioc env")
    if not _r_library_available("EpiDISH"):
        pytest.skip("EpiDISH not available in current Rscript; activate r-bioc env")


# ---------------------------------------------------------------------------
# Synthetic fixture helpers (no parquet dependency — write CSV for simplicity)
# ---------------------------------------------------------------------------

N_CPGS = 200
N_SAMPLES = 12
RNG_SEED = 42

CELL_TYPES = ["Bcell", "CD4T", "CD8T", "Gran", "Mono", "NK", "nRBC"]


def _make_beta_parquet(tmp_path: Path, n_cpgs: int = N_CPGS, n_samples: int = N_SAMPLES) -> Path:
    """Write a synthetic beta-value parquet: CpGs as index, samples as columns."""
    rng = np.random.default_rng(RNG_SEED)
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    data = rng.uniform(0.05, 0.95, size=(n_cpgs, n_samples))
    df = pd.DataFrame(data, index=cpg_ids, columns=sample_ids)
    out = tmp_path / "beta.parquet"
    df.to_parquet(out)
    return out


def _make_cell_fracs_csv(tmp_path: Path, n_samples: int = N_SAMPLES) -> Path:
    """Write a synthetic cell-fraction CSV: samples x cell-types (sum ~1 per row)."""
    rng = np.random.default_rng(RNG_SEED + 1)
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    raw = rng.dirichlet(alpha=[2.0] * len(CELL_TYPES), size=n_samples)
    df = pd.DataFrame(raw, index=sample_ids, columns=CELL_TYPES)
    out = tmp_path / "cell_fracs.csv"
    df.to_csv(out)
    return out


def _make_pdata_csv(tmp_path: Path, n_samples: int = N_SAMPLES) -> Path:
    """Write a synthetic pData CSV: samples x covariates."""
    rng = np.random.default_rng(RNG_SEED + 2)
    sample_ids = [f"sample_{i:03d}" for i in range(n_samples)]
    # Ensure balanced R/NR split so phenotype variance > 0
    response = ["R"] * (n_samples // 2) + ["NR"] * (n_samples - n_samples // 2)
    rng.shuffle(response)
    visit = rng.choice(["PRE_IOP", "POST_IOP"], size=n_samples).tolist()
    age = rng.integers(25, 65, size=n_samples).tolist()
    sex = rng.choice(["M", "F"], size=n_samples).tolist()
    df = pd.DataFrame(
        {"Response": response, "Visit": visit, "Age": age, "Sex": sex},
        index=pd.Index(sample_ids, name="SampleName"),
    )
    out = tmp_path / "pdata.csv"
    df.to_csv(out)
    return out


# ---------------------------------------------------------------------------
# Test: Rscript available + EpiDISH loads
# ---------------------------------------------------------------------------


@requires_r_bioc
class TestRBiocEnvSetup:
    """Verify the r-bioc conda env is correctly set up."""

    def test_rscript_on_path(self) -> None:
        """Rscript must be findable on PATH."""
        assert shutil.which("Rscript") is not None, "Rscript not on PATH"

    def test_epidish_loads(self) -> None:
        """EpiDISH loads without error."""
        result = subprocess.run(
            ["Rscript", "-e", "library(EpiDISH); cat('EpiDISH OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"EpiDISH load failed:\n{result.stderr}"
        assert "EpiDISH OK" in result.stdout

    def test_celldmc_exported(self) -> None:
        """CellDMC is an exported function in EpiDISH (not a separate package)."""
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                (
                    "library(EpiDISH); "
                    "stopifnot(existsMethod('CellDMC')||exists('CellDMC')); "
                    "cat('CellDMC OK')"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CellDMC not found in EpiDISH namespace:\n{result.stderr}"
        assert "CellDMC OK" in result.stdout

    def test_decoupler_loads(self) -> None:
        """bioconductor-decoupler loads without error."""
        result = subprocess.run(
            ["Rscript", "-e", "library(decoupleR); cat('decoupleR OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"decoupleR load failed:\n{result.stderr}"
        assert "decoupleR OK" in result.stdout

    def test_mcsea_loads(self) -> None:
        """mCSEA loads without error."""
        result = subprocess.run(
            ["Rscript", "-e", "library(mCSEA); cat('mCSEA OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"mCSEA load failed:\n{result.stderr}"
        assert "mCSEA OK" in result.stdout

    def test_methylclock_loads(self) -> None:
        """methylclock loads without error."""
        result = subprocess.run(
            ["Rscript", "-e", "library(methylclock); cat('methylclock OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"methylclock load failed:\n{result.stderr}"
        assert "methylclock OK" in result.stdout

    def test_epidish_reference_panels_ship_with_package(self) -> None:
        """centDHSbloodDMC.m (project default) and cent12CT.m are available via data().

        NOTE: centEpicV2 and centEpicV1 do NOT exist in EpiDISH 2.16.0. This test
        was updated from the original centEpicV2 check (which would always fail)
        to validate the actual reference panels present in this version.
        """
        r_script = textwrap.dedent("""
            library(EpiDISH)
            # centDHSbloodDMC.m: project default (333 CpGs, 7 types)
            data(centDHSbloodDMC.m, package='EpiDISH', envir=environment())
            stopifnot(exists('centDHSbloodDMC.m'))
            mat <- get('centDHSbloodDMC.m')
            cat(sprintf('centDHSbloodDMC.m %d CpGs x %d cell types\\n',
                        nrow(mat), ncol(mat)))
            # cent12CT.m: alternative high-resolution reference
            data(cent12CT.m, package='EpiDISH', envir=environment())
            stopifnot(exists('cent12CT.m'))
            mat12 <- get('cent12CT.m')
            cat(sprintf('cent12CT.m %d CpGs x %d cell types\\n',
                        nrow(mat12), ncol(mat12)))
            cat('PANELS OK\\n')
        """)
        result = subprocess.run(
            ["Rscript", "-e", r_script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Reference panel check failed:\n{result.stderr}"
        assert "PANELS OK" in result.stdout


# ---------------------------------------------------------------------------
# Test: run_epidish.R script against synthetic fixture
# ---------------------------------------------------------------------------


@requires_r_bioc
class TestRunEpiDISHScript:
    """End-to-end test of workflow/scripts/run_epidish.R on synthetic data."""

    def test_epidish_script_runs_and_produces_csv(self, tmp_path: Path) -> None:
        """run_epidish.R writes a well-formed cell-fraction CSV."""
        _assert_r_bioc_env()

        beta_path = _make_beta_parquet(tmp_path)
        out_path = tmp_path / "cell_props.csv"

        script = Path("workflow/scripts/run_epidish.R")
        assert script.exists(), f"Script not found: {script}"

        result = subprocess.run(
            [
                "Rscript",
                str(script),
                "--input",
                str(beta_path),
                "--output",
                str(out_path),
                "--ref",
                "centDHSbloodDMC.m",
                "--method",
                "RPC",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"run_epidish.R failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        assert out_path.exists(), "Output CSV not created by run_epidish.R"

        # Schema validation
        df = pd.read_csv(out_path, index_col=0)

        assert df.shape[0] == N_SAMPLES, f"Expected {N_SAMPLES} samples, got {df.shape[0]}"
        assert df.shape[1] >= 5, f"Expected >=5 cell-type columns, got {df.shape[1]}"

        # All values in [0, 1]
        assert (df.values >= 0.0).all(), "Negative cell fractions found"
        assert (df.values <= 1.0).all(), "Cell fractions > 1.0 found"

        # Row sums approx 1
        row_sums = df.sum(axis=1)
        assert (abs(row_sums - 1.0) < 0.05).all(), (
            f"Row sums deviate from 1.0: {row_sums.describe()}"
        )

        # No constant columns (the Phase 1 failure mode)
        col_vars = df.var(axis=0)
        assert (col_vars > 1e-8).all(), (
            f"Constant cell-type columns detected: {col_vars[col_vars <= 1e-8].index.tolist()}"
        )

    def test_epidish_rejects_non_beta_values(self, tmp_path: Path) -> None:
        """run_epidish.R exits non-zero when input contains M-values (range outside [0,1])."""
        _assert_r_bioc_env()

        # Write a parquet with M-values (range roughly [-5, 5])
        rng = np.random.default_rng(99)
        cpg_ids = [f"cg{i:08d}" for i in range(50)]
        sample_ids = [f"sample_{i:03d}" for i in range(8)]
        data = rng.normal(0, 2, size=(50, 8))  # M-values outside [0, 1]
        df = pd.DataFrame(data, index=cpg_ids, columns=sample_ids)
        bad_path = tmp_path / "m_values.parquet"
        df.to_parquet(bad_path)

        out_path = tmp_path / "cell_props_bad.csv"
        script = Path("workflow/scripts/run_epidish.R")

        result = subprocess.run(
            [
                "Rscript",
                str(script),
                "--input",
                str(bad_path),
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, (
            "run_epidish.R should have rejected M-values (out-of-range betas)"
        )


# ---------------------------------------------------------------------------
# Test: run_celldmc.R script against synthetic fixture
# ---------------------------------------------------------------------------


@requires_r_bioc
class TestRunCellDMCScript:
    """End-to-end test of workflow/scripts/run_celldmc.R on synthetic data."""

    def test_celldmc_script_produces_tsv(self, tmp_path: Path) -> None:
        """run_celldmc.R writes a well-formed per-cell-type DMP TSV."""
        _assert_r_bioc_env()

        beta_path = _make_beta_parquet(tmp_path)
        fracs_path = _make_cell_fracs_csv(tmp_path)
        pdata_path = _make_pdata_csv(tmp_path)
        out_path = tmp_path / "celldmc_out.tsv"

        script = Path("workflow/scripts/run_celldmc.R")
        assert script.exists(), f"Script not found: {script}"

        result = subprocess.run(
            [
                "Rscript",
                str(script),
                "--bvals",
                str(beta_path),
                "--fracs",
                str(fracs_path),
                "--pdata",
                str(pdata_path),
                "--pheno",
                "Response",
                "--visit",
                "ALL",
                "--covars",
                "Age,Sex",
                "--output",
                str(out_path),
                "--fdr",
                "0.05",
                "--ncore",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"run_celldmc.R failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        assert out_path.exists(), "Output TSV not created by run_celldmc.R"

        df = pd.read_csv(out_path, sep="\t")

        # Required columns
        required_cols = {"cpg", "cell_type", "coef", "se", "t_stat", "p_val", "fdr", "sig"}
        missing = required_cols - set(df.columns)
        assert not missing, f"Missing columns in CellDMC output: {missing}"

        assert len(df) > 0, "CellDMC output is empty"

        # One row per (CpG, cell-type) pair
        assert df["cell_type"].nunique() >= 5, (
            f"Expected >=5 cell types, got {df['cell_type'].nunique()}"
        )

        # p-values in [0, 1]
        assert (df["p_val"].dropna() >= 0).all(), "Negative p-values in CellDMC output"
        assert (df["p_val"].dropna() <= 1).all(), "p-values > 1 in CellDMC output"

    def test_celldmc_fails_on_constant_phenotype(self, tmp_path: Path) -> None:
        """run_celldmc.R exits non-zero when phenotype is constant (degenerate model)."""
        _assert_r_bioc_env()

        beta_path = _make_beta_parquet(tmp_path)
        fracs_path = _make_cell_fracs_csv(tmp_path)

        # Write pData with constant Response = all "R"
        sample_ids = [f"sample_{i:03d}" for i in range(N_SAMPLES)]
        pdata_df = pd.DataFrame(
            {
                "Response": ["R"] * N_SAMPLES,
                "Visit": ["PRE_IOP"] * N_SAMPLES,
                "Age": [40] * N_SAMPLES,
                "Sex": ["M"] * N_SAMPLES,
            },
            index=pd.Index(sample_ids, name="SampleName"),
        )
        bad_pdata = tmp_path / "pdata_constant.csv"
        pdata_df.to_csv(bad_pdata)

        out_path = tmp_path / "celldmc_constant.tsv"
        script = Path("workflow/scripts/run_celldmc.R")

        result = subprocess.run(
            [
                "Rscript",
                str(script),
                "--bvals",
                str(beta_path),
                "--fracs",
                str(fracs_path),
                "--pdata",
                str(bad_pdata),
                "--pheno",
                "Response",
                "--visit",
                "ALL",
                "--output",
                str(out_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, (
            "run_celldmc.R should have rejected a constant phenotype vector"
        )
