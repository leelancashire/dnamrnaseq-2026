"""Snakemake script: load a DNAm cohort (bVals + pData2) and write parquet.

Called by preprocessing.smk rules load_emory and load_best.
Uses snakemake.input / snakemake.output / snakemake.log objects.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Snakemake injects `snakemake` into the global namespace when run as a script
# via `script:` directive. We access it directly.
# ---------------------------------------------------------------------------


def setup_logging(log_path: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def main() -> None:
    # snakemake object available in namespace when called via `script:` directive
    sm = snakemake  # type: ignore[name-defined]  # noqa: F821

    log_path = sm.log[0] if sm.log else None
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    bvals_path = Path(sm.input.bvals)
    pdata_path = Path(sm.input.pdata)
    output_path = Path(sm.output.data)

    logger.info("Loading %s", bvals_path.name)

    import pandas as pd
    import pyreadr

    # Infer object names from file name
    bvals_obj = bvals_path.stem.replace("-", ".").replace("_architecture", ".architecture")
    pdata_obj = pdata_path.stem

    logger.info("Loading bVals: %s -> %s", bvals_path, bvals_obj)
    bvals_result = pyreadr.read_r(str(bvals_path))
    # pyreadr returns OrderedDict; take first (and only) object
    bvals_key = list(bvals_result.keys())[0]
    bvals: pd.DataFrame = bvals_result[bvals_key]
    logger.info("bVals shape: %s", bvals.shape)

    logger.info("Loading pData2: %s -> %s", pdata_path, pdata_obj)
    pdata_result = pyreadr.read_r(str(pdata_path))
    pdata_key = list(pdata_result.keys())[0]
    pdata: pd.DataFrame = pdata_result[pdata_key]
    logger.info("pData2 shape: %s", pdata.shape)

    # Set SampleName as index if available
    if "SampleName" in pdata.columns:
        pdata = pdata.set_index("SampleName")

    # Sample alignment check
    bvals_samples = set(bvals.columns)
    pdata_samples = set(pdata.index)
    overlap = bvals_samples & pdata_samples
    logger.info("Sample alignment: %d/%d bVals samples found in pData2", len(overlap), len(bvals_samples))

    # Write to parquet (bVals transposed: samples x CpGs for easier downstream use)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bvals.T.to_parquet(str(output_path))
    logger.info("Written: %s", output_path)


main()
