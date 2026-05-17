"""GEO dataset downloader for external cohort projection.

Provides reproducible, cached download of GEO datasets used in the
cross-disorder centroid analysis (Gate 0-X and Phase 3).

Currently supports:
  - GSE98793: Affymetrix whole-blood microarray, MDD vs healthy control,
    192 samples on GPL570 (HG-U133 Plus 2.0). Used as the TRD-inflammatory
    external reference in Gate 0-X.

Download and licensing
----------------------
All datasets obtained from NCBI GEO, which distributes data under the
NCBI public-data policy. No data use agreement required. Data is free for
unrestricted research use. See: https://www.ncbi.nlm.nih.gov/geo/info/faq.html

Manual fallback
---------------
If GEO FTP is unavailable, download the SOFT file directly:
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793
  Click "Download family" -> "SOFT" -> "GSE98793_family.soft.gz"
Then set config.yaml external_data_dir to the parent directory.

Usage
-----
    from dnamrnaseq2026.external_projection.datasets import download_gse98793
    gse_obj = download_gse98793(cache_dir=Path("data/external"))

The returned GEOparse.GSE object is ready to pass into
gse98793_loader.build_gse98793_expression_matrix().
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GSE98793_ACCESSION = "GSE98793"
GSE98793_SOFT_FILENAME = "GSE98793_family.soft.gz"

# MD5 checksum of the SOFT file as of 2026-05-17.
# Recomputed on each download; if the remote file changes this triggers
# a re-download and a warning (GEO data is immutable for published series,
# so checksum drift should not occur outside GEO internal re-exports).
GSE98793_SOFT_MD5 = "e64ca998a9ede4d9804e1d7a050c9f58"

# Canonical GEO FTP path (used by GEOparse internally)
GSE98793_MANUAL_DOWNLOAD_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793"
GSE98793_FTP_URL = (
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE98nnn/GSE98793/soft/GSE98793_family.soft.gz"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex MD5 digest of a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    """Expand and resolve cache directory path.

    If None, falls back to the config.yaml external_data_dir value, then
    to <repo_root>/data/external/ as a last resort.
    """
    if cache_dir is not None:
        return Path(os.path.expanduser(str(cache_dir))).resolve()

    # Try to get from config
    try:
        from dnamrnaseq2026.data.config import load_config

        cfg = load_config()
        ext_dir = cfg.get("data", {}).get("external_data_dir")
        if ext_dir:
            return Path(os.path.expanduser(str(ext_dir))).resolve()
    except Exception:
        pass

    # Fallback: repo root / data / external
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data" / "external"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_gse98793(
    cache_dir: Path | str | None = None,
    force: bool = False,
) -> Any:
    """Download GSE98793 SOFT file from GEO and return a loaded GSE object.

    The file is cached in cache_dir; subsequent calls with the same cache_dir
    are no-ops if the file exists and the MD5 matches the pinned checksum.

    Parameters
    ----------
    cache_dir:
        Directory to store the downloaded SOFT file.
        Resolved from config.yaml external_data_dir if None.
        Expanduser-aware so '~/data/external' works.
    force:
        Re-download even if the cached file exists and MD5 matches.

    Returns
    -------
    GEOparse.GEOTypes.GSE
        Loaded GSE object with .gsms (samples) and .gpls (platforms) populated.

    Raises
    ------
    RuntimeError
        If the download fails and no valid cached file exists.
    FileNotFoundError
        Should not normally raise; included for the no-network + no-cache path.

    Notes
    -----
    Manual fallback if GEO FTP is unavailable:
      Download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793
      Place the .soft.gz file in cache_dir.
    """
    try:
        import GEOparse
    except ImportError as exc:
        raise ImportError(
            "GEOparse is required for downloading GEO datasets. "
            "Install it: pip install GEOparse>=2.0"
        ) from exc

    resolved_cache = _resolve_cache_dir(cache_dir)
    resolved_cache.mkdir(parents=True, exist_ok=True)
    soft_path = resolved_cache / GSE98793_SOFT_FILENAME
    meta_path = resolved_cache / (GSE98793_SOFT_FILENAME + ".meta.json")

    # Cache hit check
    if soft_path.exists() and not force:
        existing_md5 = _md5_of_file(soft_path)
        if existing_md5 == GSE98793_SOFT_MD5:
            logger.info("Cache hit: %s (MD5 match). Skipping download.", soft_path)
        else:
            logger.warning(
                "Cache file exists but MD5 mismatch (expected %s, got %s). " "Re-downloading.",
                GSE98793_SOFT_MD5,
                existing_md5,
            )
            soft_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    # Download if needed
    if not soft_path.exists():
        logger.info(
            "Downloading %s from NCBI GEO FTP to %s...",
            GSE98793_ACCESSION,
            resolved_cache,
        )
        logger.info(
            "Manual fallback URL if FTP fails: %s",
            GSE98793_MANUAL_DOWNLOAD_URL,
        )
        try:
            GEOparse.get_GEO(
                geo=GSE98793_ACCESSION,
                destdir=str(resolved_cache),
                silent=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {GSE98793_ACCESSION} from NCBI GEO FTP.\n"
                f"Manual download: {GSE98793_MANUAL_DOWNLOAD_URL}\n"
                f"Place '{GSE98793_SOFT_FILENAME}' in {resolved_cache}.\n"
                f"Original error: {exc}"
            ) from exc

        if not soft_path.exists():
            raise RuntimeError(
                f"Download appeared to succeed but {soft_path} not found. "
                f"Check permissions on {resolved_cache}."
            )

        # Verify MD5 of freshly downloaded file
        dl_md5 = _md5_of_file(soft_path)
        if dl_md5 != GSE98793_SOFT_MD5:
            logger.warning(
                "Downloaded file MD5 (%s) differs from pinned checksum (%s). "
                "GEO may have updated the file. Proceeding, but check for "
                "format changes.",
                dl_md5,
                GSE98793_SOFT_MD5,
            )

        # Write .meta.json provenance
        meta: dict[str, Any] = {
            "geo_accession": GSE98793_ACCESSION,
            "soft_filename": GSE98793_SOFT_FILENAME,
            "md5": dl_md5,
            "download_timestamp_utc": datetime.now(UTC).isoformat(),
            "geoparse_version": GEOparse.__version__,
            "ftp_url": GSE98793_FTP_URL,
            "manual_download_url": GSE98793_MANUAL_DOWNLOAD_URL,
            "license": "NCBI public data — unrestricted research use",
        }
        with meta_path.open("w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Provenance written to %s", meta_path)

    # Load and return GSE object
    logger.info("Parsing %s...", soft_path)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Columns.*have mixed types")
        gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)

    logger.info(
        "GSE98793 loaded: %d samples on %s.",
        len(gse.gsms),
        list(gse.gpls.keys()),
    )
    return gse
