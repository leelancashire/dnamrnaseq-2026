#!/usr/bin/env python
"""Download external cohort data for the dnamrnaseq-2026 analysis.

Currently downloads:
  - GSE98793 (NCBI GEO): whole-blood Affymetrix microarray, MDD vs control,
    192 samples. Used in Gate 0-X cross-disorder centroid analysis.

Outputs are cached in the directory specified by config.yaml external_data_dir
(default: data/external/ inside the repo root).

Usage
-----
    # From repo root, with conda env active:
    python scripts/download_external.py

    # Override cache directory:
    python scripts/download_external.py --cache-dir ~/data/external

    # Force re-download even if cache exists:
    python scripts/download_external.py --force

    # Dry-run: report status without downloading:
    python scripts/download_external.py --status

Reproducibility
---------------
This script is the single entry point for all external data downloads.
Running it from a clean clone (conda env active, config.yaml present)
should reproduce the complete external dataset set.

Re-running is a no-op if the cached files exist and MD5s match.

Manual fallback
---------------
If NCBI GEO FTP is unavailable, download the SOFT file manually from:
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793
Place 'GSE98793_family.soft.gz' in the cache directory and re-run;
the script will detect the file and skip the download.

License
-------
GSE98793 is distributed by NCBI under the NCBI public-data policy.
No data use agreement is required. Free for unrestricted research use.
See: https://www.ncbi.nlm.nih.gov/geo/info/faq.html
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download external cohort data for dnamrnaseq-2026.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for downloaded files. "
            "Defaults to config.yaml external_data_dir, "
            "or data/external/ in the repo root."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if cached file exists and MD5 matches.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        default=False,
        help="Report cache status without downloading.",
    )
    return p.parse_args()


def resolve_cache_dir(cli_cache_dir: Path | None) -> Path:
    """Resolve cache directory from CLI arg, config.yaml, or default."""
    if cli_cache_dir is not None:
        return Path(os.path.expanduser(str(cli_cache_dir))).resolve()

    try:
        from dnamrnaseq2026.data.config import load_config

        cfg = load_config()
        ext_dir = cfg.get("data", {}).get("external_data_dir")
        if ext_dir:
            return Path(os.path.expanduser(str(ext_dir))).resolve()
    except Exception:
        pass

    return (_REPO_ROOT / "data" / "external").resolve()


def report_status(cache_dir: Path) -> None:
    """Report cache status for all expected external files."""
    from dnamrnaseq2026.external_projection.datasets import (
        GSE98793_SOFT_FILENAME,
        GSE98793_SOFT_MD5,
        _md5_of_file,
    )

    soft_path = cache_dir / GSE98793_SOFT_FILENAME
    meta_path = cache_dir / (GSE98793_SOFT_FILENAME + ".meta.json")

    print(f"Cache directory: {cache_dir}")
    print()

    if soft_path.exists():
        actual_md5 = _md5_of_file(soft_path)
        md5_ok = actual_md5 == GSE98793_SOFT_MD5
        status = "HIT (MD5 OK)" if md5_ok else f"STALE (MD5 mismatch: got {actual_md5})"
        size_mb = soft_path.stat().st_size / 1e6
        print(f"  GSE98793 SOFT: {status} ({size_mb:.1f} MB)")
        if meta_path.exists():
            import json

            with meta_path.open() as fh:
                meta = json.load(fh)
            print(f"    Downloaded: {meta.get('download_timestamp_utc', 'unknown')}")
            print(f"    GEOparse: {meta.get('geoparse_version', 'unknown')}")
    else:
        print(f"  GSE98793 SOFT: MISSING ({soft_path})")
        print("    Run: python scripts/download_external.py")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cache_dir = resolve_cache_dir(args.cache_dir)
    print(f"External data cache: {cache_dir}")
    print()

    if args.status:
        report_status(cache_dir)
        return

    # Import here so errors surface clearly
    from dnamrnaseq2026.external_projection.datasets import (
        GSE98793_SOFT_FILENAME,
        GSE98793_SOFT_MD5,
        _md5_of_file,
        download_gse98793,
    )

    # GSE98793
    soft_path = cache_dir / GSE98793_SOFT_FILENAME

    if soft_path.exists() and not args.force:
        existing_md5 = _md5_of_file(soft_path)
        if existing_md5 == GSE98793_SOFT_MD5:
            size_mb = soft_path.stat().st_size / 1e6
            print(f"  GSE98793: cache HIT ({size_mb:.1f} MB, MD5 OK) -- {soft_path}")
            print("  No download needed. Use --force to re-download.")
            print()
            print("All external datasets up to date.")
            return
        else:
            print(
                f"  GSE98793: MD5 mismatch (expected {GSE98793_SOFT_MD5}, "
                f"got {existing_md5}). Re-downloading."
            )

    print("  GSE98793: downloading from NCBI GEO FTP...")
    try:
        gse = download_gse98793(cache_dir=cache_dir, force=args.force)
        final_path = cache_dir / GSE98793_SOFT_FILENAME
        final_md5 = _md5_of_file(final_path)
        size_mb = final_path.stat().st_size / 1e6
        print(f"  GSE98793: downloaded OK ({size_mb:.1f} MB, MD5={final_md5})")
        print(f"  File: {final_path}")
        print(f"  Samples: {len(gse.gsms)}")
        print(f"  Platforms: {list(gse.gpls.keys())}")
    except RuntimeError as exc:
        print(f"  GSE98793: FAILED -- {exc}", file=sys.stderr)
        sys.exit(1)

    print()
    print("All external datasets downloaded.")
    print()
    print("Next step: run Gate 0-X")
    print("  python scripts/01_phase0_gate_X.py")


if __name__ == "__main__":
    main()
