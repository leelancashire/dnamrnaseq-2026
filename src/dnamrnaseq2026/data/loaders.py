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
from pathlib import Path
from typing import Optional

import pandas as pd

from dnamrnaseq2026.data.config import get_emory_dnam_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level RData loading helpers
# ---------------------------------------------------------------------------


def _load_rdata_pyreadr(path: Path) -> dict[str, pd.DataFrame]:
    """Load an RData file via pyreadr. Returns {object_name: DataFrame}."""
    import pyreadr  # type: ignore[import-untyped]

    result = pyreadr.read_r(str(path))
    return dict(result)


def _load_rdata_rdata(path: Path, object_name: str) -> pd.DataFrame:
    """Load a named object from an RData file via the rdata package.

    Handles the ndarray-of-shape-(N,1) bug in rdata's DataFrame constructor
    by squeezing (N,1) column arrays to 1-D before constructing the DataFrame.
    """
    import rdata  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]

    parsed = rdata.parser.parse_file(str(path))

    # Use a custom converter to intercept the ndarray-(N,1) DataFrame failure.
    # The default conversion raises ValueError for columns stored as 2-D arrays
    # in R (e.g. pData2 files from minfi/Bioconductor pipelines).
    try:
        converted = rdata.conversion.convert(parsed)
    except ValueError as e:
        if "1-dimensional" in str(e):
            # Fall back to a raw dict-level repair: re-parse and squeeze arrays.
            converted = _rdata_convert_squeeze(path, object_name)
        else:
            raise

    if object_name not in converted:
        available = list(converted.keys())
        raise KeyError(
            f"Object '{object_name}' not found in {path.name}. "
            f"Available: {available}"
        )

    obj = converted[object_name]
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    raise TypeError(
        f"Object '{object_name}' loaded as {type(obj).__name__}, expected DataFrame."
    )


def _rdata_convert_squeeze(path: Path, object_name: str) -> dict[str, pd.DataFrame]:
    """Fallback for rdata files with ndarray-(N,1) columns.

    Parses the R object manually, squeezes all (N,1) numpy arrays to 1-D,
    then builds a DataFrame column-by-column.
    """
    import numpy as np  # type: ignore[import-untyped]
    import rdata  # type: ignore[import-untyped]

    parsed = rdata.parser.parse_file(str(path))

    # Walk the parsed structure to find the target object.
    # parsed is an RData container; its value is a tagged list of objects.
    # We use convert() with a custom class that handles ndarray columns.
    class SqueezeConverter(rdata.conversion.SimpleConverter):  # type: ignore[misc]
        def _array_constructor(  # type: ignore[override]
            self, value: object, attrs: object
        ) -> object:
            result = super()._array_constructor(value, attrs)  # type: ignore[misc]
            if isinstance(result, np.ndarray) and result.ndim == 2 and result.shape[1] == 1:
                return result.squeeze(axis=1)
            return result

    try:
        converted = SqueezeConverter().convert(parsed)
        return {k: v for k, v in converted.items() if isinstance(v, pd.DataFrame)}
    except Exception:
        # Last-resort: use rdata internal R-object access and build manually
        return _rdata_manual_parse(path, object_name)


def _rdata_manual_parse(path: Path, object_name: str) -> dict[str, pd.DataFrame]:
    """Last-resort RData parser: builds DataFrame column-by-column, squeezing arrays."""
    import numpy as np  # type: ignore[import-untyped]
    import rdata  # type: ignore[import-untyped]

    parsed = rdata.parser.parse_file(str(path))
    # Traverse the RData structure manually
    r_obj = parsed.object
    # r_obj is typically a pairlist; find the named element
    result: dict[str, pd.DataFrame] = {}

    def _squeeze(arr: object) -> object:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == 1:
            return arr.squeeze(axis=1)
        return arr

    # For a data.frame-like object, attributes include 'names' and 'row.names'
    if hasattr(r_obj, "value") and hasattr(r_obj, "attributes"):
        attrs = r_obj.attributes
        if attrs and "names" in attrs:
            col_names = list(attrs["names"].value)
            columns_data = {}
            for i, col in enumerate(col_names):
                raw = r_obj.value[i] if hasattr(r_obj, "value") else None
                if raw is not None and hasattr(raw, "value"):
                    val = np.array(raw.value)
                    columns_data[col] = _squeeze(val)
            if columns_data:
                result[object_name] = pd.DataFrame(columns_data)

    return result


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


def load_emory_bvals(data_dir: Optional[Path] = None) -> pd.DataFrame:
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


def load_emory_pdata2(data_dir: Optional[Path] = None) -> pd.DataFrame:
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


def load_best_bvals(data_dir: Optional[Path] = None) -> pd.DataFrame:
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


def load_best_pdata2(data_dir: Optional[Path] = None) -> pd.DataFrame:
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
