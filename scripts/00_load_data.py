#!/usr/bin/env python
"""Day-0 verification: load all data files and report shapes.

This script reproduces Kai's Day-0 verification (2026-05-17 S3 data
verification note). Run after `cp config.yaml.example config.yaml`
and editing data paths if needed.

Expected output:
    emory.bVals.architecture: (292674, 388)  [CpG sites x samples]
    emory_pData2: (388, 366)  [samples x covariates]
    best.bVals.architecture: (292973, 141)  [CpG sites x samples]
    best_pData2: (141, 678)  [samples x covariates]

Usage:
    python scripts/00_load_data.py
    python scripts/00_load_data.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add the repo root to sys.path so this script works without `pip install -e .`
# (i.e. after a bare clone before installation).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Day-0 data verification for dnamrnaseq-2026.")
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable DEBUG logging."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data directory from config.yaml.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Import after sys.path manipulation
    from dnamrnaseq2026.data.loaders import (
        check_sample_alignment,
        load_best_bvals,
        load_best_pdata2,
        load_emory_bvals,
        load_emory_pdata2,
    )

    data_dir = args.data_dir

    # -----------------------------------------------------------------------
    # Load all four matrices
    # -----------------------------------------------------------------------
    try:
        emory_bvals = load_emory_bvals(data_dir=data_dir)
        emory_pdata = load_emory_pdata2(data_dir=data_dir)
        best_bvals = load_best_bvals(data_dir=data_dir)
        best_pdata = load_best_pdata2(data_dir=data_dir)
    except FileNotFoundError as e:
        logger.error("\nData load failed: %s", e)
        logger.error(
            "\nCheck config.yaml: data.emory_dnam_dir should point to the OneDrive mount."
        )
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Report shapes
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  dnamrnaseq-2026  Day-0 Data Verification")
    print("=" * 60)
    print()
    print("Emory cohort:")
    print(f"  emory.bVals.architecture : {emory_bvals.shape[0]:>7,} CpGs x {emory_bvals.shape[1]:>3} samples")
    print(f"  emory_pData2             : {emory_pdata.shape[0]:>7} samples x {emory_pdata.shape[1]:>3} covariates")
    print()
    print("BEST cohort:")
    print(f"  best.bVals.architecture  : {best_bvals.shape[0]:>7,} CpGs x {best_bvals.shape[1]:>3} samples")
    print(f"  best_pData2              : {best_pdata.shape[0]:>7} samples x {best_pdata.shape[1]:>3} covariates")

    # -----------------------------------------------------------------------
    # Reference values from Kai's Day-0 verification (2026-05-17)
    # -----------------------------------------------------------------------
    EXPECTED: dict[str, tuple[int, int]] = {
        "emory_bvals":  (292674, 388),
        "emory_pdata":  (388, 366),
        "best_bvals":   (292973, 141),
        "best_pdata":   (141, 678),
    }

    actual = {
        "emory_bvals":  tuple(emory_bvals.shape),
        "emory_pdata":  tuple(emory_pdata.shape),
        "best_bvals":   tuple(best_bvals.shape),
        "best_pdata":   tuple(best_pdata.shape),
    }

    print()
    print("Shape check vs. Day-0 reference:")
    all_match = True
    for key, expected_shape in EXPECTED.items():
        got = actual[key]
        status = "OK" if got == expected_shape else "MISMATCH"
        if status == "MISMATCH":
            all_match = False
        print(f"  {key:<15}: expected {expected_shape}  got {got}  [{status}]")

    if not all_match:
        print()
        print("[WARNING] Some shapes differ from the Day-0 reference.")
        print("This may be expected if the data files have been updated.")
        print("Verify manually before proceeding to Phase 0 gates.")

    # -----------------------------------------------------------------------
    # Sample-ID alignment
    # -----------------------------------------------------------------------
    print()
    print("Sample-ID alignment check:")
    try:
        check_sample_alignment(emory_bvals, emory_pdata, cohort="Emory")
        check_sample_alignment(best_bvals, best_pdata, cohort="BEST")
    except ValueError as e:
        logger.error("\nAlignment failure: %s", e)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Response value counts
    # -----------------------------------------------------------------------
    print()
    if "Response" in emory_pdata.columns:
        print("Response value counts (Emory):")
        for val, count in emory_pdata["Response"].value_counts().items():
            print(f"  {val}: {count}")
    else:
        print("[INFO] No 'Response' column found in emory_pData2.")

    if "Response" in best_pdata.columns:
        print()
        print("Response value counts (BEST):")
        for val, count in best_pdata["Response"].value_counts().items():
            print(f"  {val}: {count}")
    else:
        print("[INFO] No 'Response' column found in best_pData2.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    if all_match:
        print("  All checks passed. Shapes match Day-0 reference.")
    else:
        print("  Checks passed with shape warnings (see above).")
    print("=" * 60)
    print()

    sys.exit(0)


if __name__ == "__main__":
    main()
