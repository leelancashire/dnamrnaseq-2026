"""Phase 3: project external cohorts into the atlas latent space.

Entry point for the Phase 3 external-cohort projection step (design doc
Section 5.3 Step 3.3). Loads the Phase 2 Wave 1 prepared external-cohort
files (GTEx + GSE98793) and the winning Phase 2 atlas checkpoint, projects
both reference cohorts into the latent space, builds the two-anchor recovery
axis, and saves the ProjectionResult consumed by the Phase 3.3 proximity test.

Usage
-----
    python scripts/30_phase3_project_external_cohorts.py \\
        --atlas-checkpoint results/phase2/winning_arm/checkpoint.pt \\
        --atlas-arm arm_c \\
        --gtex-path data/external/gtex_whole_blood_emory_aligned.parquet \\
        --gse-centroid-path data/external/gse98793_group_centroids.parquet \\
        --ptsd-terminus-path results/phase3/ptsd_terminus_latent.parquet \\
        --output-dir results/phase3/external_projection

The script is scaffold-only at this stage. The real atlas checkpoint
and external-cohort files do not exist yet; the script will raise
FileNotFoundError with clear guidance when run against real data.

A ``--dry-run`` flag exercises the full code path on synthetic fixtures
so CI can verify the wiring without any real data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3.project_external")

# ---------------------------------------------------------------------------
# Synthetic atlas encoder (dry-run only)
# ---------------------------------------------------------------------------


class _DryRunEncoder:
    """Fixed random linear projection; used only for --dry-run validation."""

    def __init__(self, d_rna_in: int, d_latent: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self._W = rng.standard_normal((d_rna_in, d_latent)).astype(np.float64)
        self._W /= np.linalg.norm(self._W, axis=0, keepdims=True)

    def encode(self, rna_matrix: np.ndarray) -> np.ndarray:
        return (rna_matrix.astype(np.float64) @ self._W).astype(np.float64)


# ---------------------------------------------------------------------------
# Atlas encoder loader (real-data path)
# ---------------------------------------------------------------------------


def load_atlas_encoder(
    checkpoint_path: Path,
    arm: str,
) -> object:
    """Load the winning Phase 2 arm encoder from checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path to the Phase 2 checkpoint (e.g. ``.pt`` file from
        ``scripts/20_phase2_train_embedding.py``).
    arm:
        One of ``'arm_a'``, ``'arm_b'``, ``'arm_c'``. Selects the encoder
        class to load the checkpoint into.

    Returns
    -------
    An object satisfying the AtlasEncoder protocol (has ``.encode(rna_matrix)``).

    Notes
    -----
    This function is intentionally not fully implemented at scaffold time.
    It will raise NotImplementedError until the Phase 2 checkpoint API is
    finalised. Implement by copying the relevant encoder class from
    ``src/dnamrnaseq2026/embedding/`` and loading the ``.pt`` state dict.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Atlas checkpoint not found: {checkpoint_path}\n"
            "Phase 2 training must complete and a winning arm must be chosen "
            "before running Phase 3 on real data."
        )
    raise NotImplementedError(
        f"Real atlas checkpoint loading not yet implemented (arm={arm!r}). "
        "This scaffold runs against synthetic fixtures with --dry-run. "
        "Implement this function once the Phase 2 leaderboard picks a winner."
    )


# ---------------------------------------------------------------------------
# PTSD terminus loader
# ---------------------------------------------------------------------------


def load_ptsd_terminus(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Phase 2 Step 3.0 PTSD POST-IOP latent coordinates.

    Expected parquet schema:
        subject_id (str), response (str or int), cohort (str),
        latent_z0 ... latent_z{d-1} (float64).

    Returns
    -------
    (terminus_latent, subject_ids, response, cohort) as numpy arrays.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"PTSD terminus file not found: {path}\n"
            "Phase 2 Step 3.0 must run first to produce per-subject POST-IOP "
            "latent coordinates."
        )
    df = pd.read_parquet(path)
    latent_cols = [c for c in df.columns if c.startswith("latent_z")]
    if not latent_cols:
        raise ValueError(
            f"No latent_z* columns found in {path}. Check that this is the Phase 2 Step 3.0 output."
        )
    terminus_latent = df[latent_cols].values.astype(np.float64)
    subject_ids = df["subject_id"].values.astype(object)
    response = df["response"].values.astype(object)
    cohort = df["cohort"].values.astype(object)
    logger.info(
        "Loaded PTSD terminus: %d subjects, d_latent=%d",
        terminus_latent.shape[0],
        terminus_latent.shape[1],
    )
    return terminus_latent, subject_ids, response, cohort


# ---------------------------------------------------------------------------
# Dry-run synthetic fixture generator
# ---------------------------------------------------------------------------


def _make_dry_run_data(
    n_gtex: int = 50,
    n_gse_trd: int = 30,
    n_ptsd: int = 40,
    n_rna: int = 50,
    d_latent: int = 16,
    *,
    seed: int = 42,
) -> tuple[object, object, object]:
    """Generate synthetic data for --dry-run validation."""
    from dnamrnaseq2026.trajectory.external_projection import ExternalCohortData, PtsdAtlasData

    rng = np.random.default_rng(seed)
    n_responders = n_ptsd // 2

    # External cohorts with planted separation
    gtex_rna = (rng.standard_normal((n_gtex, n_rna)) + 2.0).astype(np.float64)
    gse_trd_rna = (rng.standard_normal((n_gse_trd, n_rna)) - 2.0).astype(np.float64)

    ext = ExternalCohortData(
        gtex_rna=gtex_rna,
        gtex_sample_ids=np.array([f"GTEX_{i:04d}" for i in range(n_gtex)], dtype=object),
        gse_trd_rna=gse_trd_rna,
        gse_trd_sample_ids=np.array([f"GSE_{i:04d}" for i in range(n_gse_trd)], dtype=object),
        gse_trd_centroid_rna=gse_trd_rna.mean(axis=0),
        feature_names=np.array([f"GENE_{j}" for j in range(n_rna)], dtype=object),
    )

    # Encoder
    enc = _DryRunEncoder(d_rna_in=n_rna, d_latent=d_latent)

    # PTSD termini (pre-projected since dry-run skips Step 3.0)
    terminus_latent = rng.standard_normal((n_ptsd, d_latent)).astype(np.float64)
    terminus_latent[:n_responders, 0] += 3.0
    terminus_latent[n_responders:, 0] -= 3.0
    response = np.array(["R"] * n_responders + ["NR"] * (n_ptsd - n_responders), dtype=object)
    ptsd = PtsdAtlasData(
        terminus_latent=terminus_latent,
        subject_ids=np.array([f"DRY{i:04d}" for i in range(n_ptsd)], dtype=object),
        response=response,
        cohort=np.array(["Emory"] * n_ptsd, dtype=object),
    )
    return enc, ext, ptsd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3: project external cohorts into atlas latent space."
    )
    p.add_argument(
        "--atlas-checkpoint",
        type=Path,
        default=Path("results/phase2/winning_arm/checkpoint.pt"),
        help="Path to the Phase 2 winning arm checkpoint (.pt file).",
    )
    p.add_argument(
        "--atlas-arm",
        choices=["arm_a", "arm_b", "arm_c"],
        default="arm_c",
        help="Which Phase 2 arm to load (chosen by leaderboard).",
    )
    p.add_argument(
        "--gtex-path",
        type=Path,
        default=Path("data/external/gtex_whole_blood_emory_aligned.parquet"),
        help="Path to GTEx v10 whole blood Emory-aligned parquet (Phase 2 Wave 1 output).",
    )
    p.add_argument(
        "--gse-centroid-path",
        type=Path,
        default=Path("data/external/gse98793_group_centroids.parquet"),
        help="Path to GSE98793 group-centroid parquet (Phase 2 Wave 1 output).",
    )
    p.add_argument(
        "--gse-sample-path",
        type=Path,
        default=None,
        help="Optional: individual GSE98793 TRD sample expression parquet.",
    )
    p.add_argument(
        "--ptsd-terminus-path",
        type=Path,
        default=Path("results/phase3/ptsd_terminus_latent.parquet"),
        help="Phase 2 Step 3.0 PTSD POST-IOP latent coordinates parquet.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase3/external_projection"),
        help="Output directory for ProjectionResult files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on synthetic fixtures only; no real data required. Used in CI.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from dnamrnaseq2026.trajectory.external_projection import (
        PtsdAtlasData,
        load_external_cohorts_from_parquet,
        project_external_cohorts,
        save_projection_result,
    )

    if args.dry_run:
        logger.info("--dry-run: using synthetic fixtures (no real data)")
        enc, ext, ptsd = _make_dry_run_data()
        encoder_name = "dry_run_synthetic_encoder"
        gtex_source = "synthetic_gtex"
        gse_source = "synthetic_gse98793_trd"
    else:
        logger.info(
            "Loading real atlas encoder from %s (arm=%s)", args.atlas_checkpoint, args.atlas_arm
        )
        enc = load_atlas_encoder(args.atlas_checkpoint, args.atlas_arm)  # type: ignore[assignment]
        encoder_name = args.atlas_arm

        logger.info("Loading external cohort files")
        ext = load_external_cohorts_from_parquet(
            gtex_path=args.gtex_path,
            gse_centroid_path=args.gse_centroid_path,
            gse_sample_path=args.gse_sample_path,
        )
        gtex_source = str(args.gtex_path)
        gse_source = str(args.gse_centroid_path)

        logger.info("Loading PTSD terminus latent coords from %s", args.ptsd_terminus_path)
        terminus_latent, subject_ids, response, cohort = load_ptsd_terminus(args.ptsd_terminus_path)
        ptsd = PtsdAtlasData(
            terminus_latent=terminus_latent,
            subject_ids=subject_ids,
            response=response,
            cohort=cohort,
        )

    logger.info("Running Phase 3 external-cohort projection")
    result = project_external_cohorts(
        enc,
        ext,
        ptsd,
        encoder_name=encoder_name,
        gtex_source=gtex_source,
        gse_source=gse_source,
    )

    logger.info(
        "Projection complete: axis_norm=%.6f, recovery_score R-mean=%.4f NR-mean=%.4f",
        float(np.linalg.norm(result.recovery_axis)),
        float(result.terminus_recovery_score[result.response == "R"].mean())
        if np.any(result.response == "R")
        else float("nan"),
        float(result.terminus_recovery_score[result.response == "NR"].mean())
        if np.any(result.response == "NR")
        else float("nan"),
    )

    logger.info("Saving ProjectionResult to %s", args.output_dir)
    save_projection_result(result, args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main(sys.argv[1:])
