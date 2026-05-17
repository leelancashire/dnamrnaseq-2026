"""RData loaders for the Emory and BEST DNAm + pData2 files.

Strategy:
  1. Try pyreadr first (fast, C-backed, works on all four critical files).
  2. Fall back to rdata if pyreadr returns an unrecognised object or raises
     (pyreadr fails on pData ndarray-of-shape-(N,1) edge cases on some
     versions; rdata handles those cleanly).

All loaders return pandas DataFrames. Shapes match Kai's Day-0 verification
(2026-05-17 S3 data verification note):
  emory bVals:  292,674 x 388  (CpG sites x samples)
  emory pData2: 388 x 366      (samples x covariates)
  best  bVals:  292,973 x 141  (CpG sites x samples)
  best  pData2: 141 x 678      (samples x covariates)

The join key between bVals columns and pData2 rows is `SampleName`
(Sentrix ID format: {Barcode}_{Position}).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from dnamrnaseq2026.data.config import get_emory_dnam_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level RData loading helpers
# ---------------------------------------------------------------------------


def _load_rdata_pyreadr(path: Path) -> dict[str, pd.DataFrame]:
    """Load an RData file via pyreadr. Returns {object_name: DataFrame}."""
    import pyreadr

    result = pyreadr.read_r(str(path))
    return dict(result)


def _rdata_squeeze_dataframe_constructor(
    obj: Any,
    attrs: Mapping[str, Any],
) -> pd.DataFrame:
    """Custom rdata DataFrame constructor that squeezes (N,1) ndarray columns.

    Bioconductor pData2 files (emory_pData2.RData, best_pData2.RData) contain
    some columns stored as (N,1) 2-D arrays in R. rdata 1.0.0's default
    constructor passes these directly to pd.DataFrame, which raises ValueError
    ("Data must be 1-dimensional, got ndarray of shape (N,1)"). This constructor
    squeezes such columns before building the DataFrame.
    """
    import numpy as np

    squeezed: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == 1:
            squeezed[k] = v.squeeze(axis=1)
        else:
            squeezed[k] = v

    index = attrs.get("row.names") if attrs else None
    return pd.DataFrame(squeezed, index=index)


def _load_rdata_rdata(path: Path, object_name: str) -> pd.DataFrame:
    """Load a named object from an RData file via the rdata package.

    Uses a custom DataFrame constructor (_rdata_squeeze_dataframe_constructor)
    to handle the ndarray-(N,1) column issue in pData2 files from
    minfi/Bioconductor pipelines.
    """
    import numpy as np
    import rdata

    parsed = rdata.parser.parse_file(str(path))

    converter = rdata.conversion.SimpleConverter(
        constructor_dict={"data.frame": _rdata_squeeze_dataframe_constructor}
    )
    converted = converter.convert(parsed)

    if object_name not in converted:
        available = list(converted.keys())
        raise KeyError(
            f"Object '{object_name}' not found in {path.name}. " f"Available: {available}"
        )

    obj = converted[object_name]
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    raise TypeError(f"Object '{object_name}' loaded as {type(obj).__name__}, expected DataFrame.")


def _load_rdata(
    path: Path,
    object_name: str,
    transpose: bool = False,
) -> pd.DataFrame:
    """Try pyreadr first; fall back to rdata on failure.

    Parameters
    ----------
    path:
        Path to the .RData file.
    object_name:
        Name of the R object to extract.
    transpose:
        If True, transpose the resulting DataFrame.
        bVals matrices are stored as (CpG x sample) in R but pyreadr
        may load them transposed depending on the R object class.

    Returns
    -------
    pd.DataFrame
    """
    try:
        result = _load_rdata_pyreadr(path)
        if object_name not in result:
            available = list(result.keys())
            raise KeyError(
                f"Object '{object_name}' not found via pyreadr in {path.name}. "
                f"Available: {available}"
            )
        df = result[object_name]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"pyreadr returned {type(df).__name__}, expected DataFrame.")
        logger.debug("Loaded %s via pyreadr: shape %s", path.name, df.shape)
    except Exception as pyreadr_err:  # noqa: BLE001
        logger.warning(
            "pyreadr failed on %s (%s); trying rdata fallback.",
            path.name,
            pyreadr_err,
        )
        df = _load_rdata_rdata(path, object_name)
        logger.debug("Loaded %s via rdata: shape %s", path.name, df.shape)

    if transpose:
        df = df.T

    return df


# ---------------------------------------------------------------------------
# Public loaders — Emory cohort
# ---------------------------------------------------------------------------


def load_emory_bvals(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Emory DNAm beta values (architecture CpG subset).

    Returns a DataFrame of shape (n_cpgs, n_samples).
    Expected from Day-0 verification: (292,674 x 388).

    Parameters
    ----------
    data_dir:
        Override the data directory from config.yaml.

    Returns
    -------
    pd.DataFrame
        Index: CpG site IDs. Columns: sample IDs (SampleName / Sentrix format).
    """
    dnam_dir = data_dir or get_emory_dnam_dir()
    path = Path(dnam_dir) / "emory.bVals.architecture.RData"
    if not path.exists():
        raise FileNotFoundError(f"Emory bVals not found at {path}. Check config.yaml.")

    logger.info("Loading Emory DNAm bVals (architecture subset)...")
    df = _load_rdata(path, "emory.bVals.architecture")
    logger.info("  emory.bVals.architecture: %s  [CpG sites x samples]", df.shape)
    return df


def load_emory_pdata2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load Emory pData2 (sample metadata / covariates).

    Returns a DataFrame of shape (n_samples, n_covariates).
    Expected from Day-0 verification: (388 x 366).
    The index is `SampleName` (Sentrix ID), the join key to bVals columns.

    Parameters
    ----------
    data_dir:
        Override the data directory from config.yaml.

    Returns
    -------
    pd.DataFrame
        Index: SampleName. Columns: covariates.
    """
    dnam_dir = data_dir or get_emory_dnam_dir()
    path = Path(dnam_dir) / "emory_pData2.RData"
    if not path.exists():
        raise FileNotFoundError(f"Emory pData2 not found at {path}. Check config.yaml.")

    logger.info("Loading Emory pData2...")
    df = _load_rdata(path, "emory_pData2")

    # Set SampleName as index if it is a column (not already the index)
    if "SampleName" in df.columns and df.index.name != "SampleName":
        df = df.set_index("SampleName")

    logger.info("  emory_pData2: %s  [samples x covariates]", df.shape)
    return df


# ---------------------------------------------------------------------------
# Public loaders — BEST cohort
# ---------------------------------------------------------------------------


def load_best_bvals(data_dir: Path | None = None) -> pd.DataFrame:
    """Load BEST DNAm beta values (architecture CpG subset).

    Returns a DataFrame of shape (n_cpgs, n_samples).
    Expected from Day-0 verification: (292,973 x 141).

    Parameters
    ----------
    data_dir:
        Override the data directory from config.yaml.

    Returns
    -------
    pd.DataFrame
        Index: CpG site IDs. Columns: sample IDs.
    """
    dnam_dir = data_dir or get_emory_dnam_dir()
    path = Path(dnam_dir) / "best.bVals.architecture.RData"
    if not path.exists():
        raise FileNotFoundError(f"BEST bVals not found at {path}. Check config.yaml.")

    logger.info("Loading BEST DNAm bVals (architecture subset)...")
    df = _load_rdata(path, "best.bVals.architecture")
    logger.info("  best.bVals.architecture: %s  [CpG sites x samples]", df.shape)
    return df


def load_best_pdata2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load BEST pData2 (sample metadata / covariates).

    Returns a DataFrame of shape (n_samples, n_covariates).
    Expected from Day-0 verification: (141 x 678).
    The index is `SampleName` (Sentrix ID), the join key to bVals columns.

    Parameters
    ----------
    data_dir:
        Override the data directory from config.yaml.

    Returns
    -------
    pd.DataFrame
        Index: SampleName. Columns: covariates.
    """
    dnam_dir = data_dir or get_emory_dnam_dir()
    path = Path(dnam_dir) / "best_pData2.RData"
    if not path.exists():
        raise FileNotFoundError(f"BEST pData2 not found at {path}. Check config.yaml.")

    logger.info("Loading BEST pData2...")
    df = _load_rdata(path, "best_pData2")

    if "SampleName" in df.columns and df.index.name != "SampleName":
        df = df.set_index("SampleName")

    logger.info("  best_pData2: %s  [samples x covariates]", df.shape)
    return df


# ---------------------------------------------------------------------------
# Sample-ID alignment check
# ---------------------------------------------------------------------------


def check_sample_alignment(
    bvals: pd.DataFrame,
    pdata: pd.DataFrame,
    cohort: str = "",
) -> None:
    """Assert that bVals column names are present in pData2 index.

    Raises ValueError if alignment is incomplete.

    Parameters
    ----------
    bvals:
        DNAm beta values DataFrame (CpG x sample). Columns are sample IDs.
    pdata:
        pData2 DataFrame (sample x covariate). Index is SampleName.
    cohort:
        Label for logging (e.g. "Emory", "BEST").
    """
    bvals_samples = set(bvals.columns)
    pdata_samples = set(pdata.index)
    overlap = bvals_samples & pdata_samples
    missing_from_pdata = bvals_samples - pdata_samples

    label = f"[{cohort}] " if cohort else ""
    logger.info(
        "%sbVals cols in pData2 index: %d/%d",
        label,
        len(overlap),
        len(bvals_samples),
    )

    if missing_from_pdata:
        raise ValueError(
            f"{label}Sample-ID alignment failure: "
            f"{len(missing_from_pdata)} bVals samples not found in pData2 index. "
            f"First 5: {sorted(missing_from_pdata)[:5]}"
        )


# ---------------------------------------------------------------------------
# CLI entry point (called by `dnamrnaseq-load` console script)
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: load all four files and report shapes. Equivalent to 00_load_data.py."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        emory_bvals = load_emory_bvals()
        emory_pdata = load_emory_pdata2()
        best_bvals = load_best_bvals()
        best_pdata = load_best_pdata2()
    except FileNotFoundError as e:
        logger.error("Data load failed: %s", e)
        sys.exit(1)

    print("\nSample-ID alignment check:")
    check_sample_alignment(emory_bvals, emory_pdata, cohort="Emory")
    check_sample_alignment(best_bvals, best_pdata, cohort="BEST")

    if "Response" in emory_pdata.columns:
        print("\nResponse value counts (Emory):")
        print(emory_pdata["Response"].value_counts().to_string())

    if "Response" in best_pdata.columns:
        print("\nResponse value counts (BEST):")
        print(best_pdata["Response"].value_counts().to_string())

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
