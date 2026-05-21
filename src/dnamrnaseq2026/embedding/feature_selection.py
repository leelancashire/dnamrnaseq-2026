"""Two-tier feature subsetting + Phase 1 artefact interface (design Section 2.0 / 3v).

The Phase 2 design doc (Kai override, 2026-05-19) replaces the pure
variance-filter default with a two-tier scheme:

- **Tier 1 (preferred, biology-led).** CellDMC-prioritised DNAm CpGs with a
  significant Response x cell-fraction interaction (FDR < 0.10) from Phase 1
  step 1.2; RNA-side = cis-genes within +/-100 kb of those CpGs.
- **Tier 2 (fallback).** Biology-informed variance filter: drop cross-reactive
  and sex-chromosome probes, apply a beta-range floor, then variance-rank.

The Phase 1 artefacts that feed Tier 1 do not exist until the Phase 1 re-run
completes on real EpiDISH cell fractions. Every reader here degrades gracefully:
if the artefact is missing or empty it returns ``None`` / an empty frame and the
caller falls back to Tier 2. This is what makes the scaffold runnable on
synthetic fixtures before Phase 1 lands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Tier 2 fallback sizing (design Section 2.0). Tier 1 sizes depend on the
# CellDMC re-run output and are not knowable in advance.
TIER2_DNAM_TOP = 5000
TIER2_RNA_TOP = 2000
BETA_RANGE_FLOOR = 0.05
CELLDMC_FDR_THRESHOLD = 0.10

# Default Phase 1 artefact directory (design Section 3v artefact interface).
PHASE1_ARTEFACT_DIR = Path("analysis/latest")


@dataclass
class FeatureTier:
    """Resolved feature lists plus the tier that produced them."""

    tier: int  # 1 = CellDMC-prioritised, 2 = variance-filter fallback
    dnam_cpgs: list[str]
    rna_genes: list[str]
    rationale: str


# CellDMC interaction artefact filenames, tried in order. The design doc
# Section 3v names a parquet; the real v5 Phase 1 run writes the per-contrast
# TSVs (celldmc_delta_emory.tsv etc.). Both carry the same schema.
_CELLDMC_FILENAMES = (
    "celldmc_interaction_results.parquet",
    "celldmc_delta_emory.tsv",
)


def read_celldmc_interactions(
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
) -> pd.DataFrame | None:
    """Read the Phase 1 CellDMC interaction table, or None if absent/empty.

    Expected schema (design Section 3v):
    ``cpg, cell_type, coef, se, t_stat, p_val, fdr, sig``.
    Accepts both the design-doc parquet name and the real v5 TSV
    (``celldmc_delta_emory.tsv``). A zero-byte stub is treated as absent.
    """
    for name in _CELLDMC_FILENAMES:
        path = artefact_dir / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
        if df.empty:
            continue
        logger.info("CellDMC interaction artefact loaded from %s (%s)", path, df.shape)
        return df
    logger.info("CellDMC interaction artefact missing/stub in %s", artefact_dir)
    return None


def read_pathway_activity(
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
) -> pd.DataFrame | None:
    """Read PROGENy pathway activity scores (n_samples, ~14), or None if absent."""
    path = artefact_dir / "progeny_pathway_activity.parquet"
    if not path.exists() or path.stat().st_size == 0:
        logger.info("PROGENy pathway artefact missing/stub at %s", path)
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def read_tf_activity(
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
) -> pd.DataFrame | None:
    """Read decoupleR/CollecTRI TF activity scores (n_samples, n_tfs)."""
    path = artefact_dir / "decoupler_tf_activity.parquet"
    if not path.exists() or path.stat().st_size == 0:
        logger.info("decoupleR TF activity artefact missing/stub at %s", path)
        return None
    df = pd.read_parquet(path)
    return df if not df.empty else None


def select_tier1_cpgs(
    celldmc: pd.DataFrame,
    fdr_threshold: float = CELLDMC_FDR_THRESHOLD,
) -> list[str]:
    """Return CpGs with a significant CellDMC interaction at FDR < threshold.

    A CpG is kept if it is significant for *any* cell type.
    """
    sig = celldmc[celldmc["fdr"] < fdr_threshold]
    cpgs = sorted(sig["cpg"].astype(str).unique().tolist())
    logger.info("Tier 1: %d CellDMC-prioritised CpGs at FDR < %.2f", len(cpgs), fdr_threshold)
    return cpgs


def map_cpgs_to_cis_genes(
    cpgs: list[str],
    cpg_gene_map: pd.DataFrame,
    *,
    window_kb: int = 100,
) -> list[str]:
    """Map CpGs to cis-genes within +/-window_kb (design Section 2.0).

    Parameters
    ----------
    cpgs:
        Tier 1 CpG list.
    cpg_gene_map:
        Annotation frame with columns ``cpg``, ``gene``, ``distance_kb``.
    window_kb:
        cis window half-width in kb.
    """
    in_window = cpg_gene_map[
        cpg_gene_map["cpg"].isin(cpgs) & (cpg_gene_map["distance_kb"].abs() <= window_kb)
    ]
    genes = sorted(in_window["gene"].astype(str).unique().tolist())
    logger.info("Tier 1: %d cis-genes within +/-%d kb", len(genes), window_kb)
    return genes


def variance_filter_dnam(
    bvals: pd.DataFrame,
    *,
    top_n: int = TIER2_DNAM_TOP,
    cross_reactive: set[str] | None = None,
    sex_chr_cpgs: set[str] | None = None,
    beta_range_floor: float = BETA_RANGE_FLOOR,
) -> list[str]:
    """Tier 2 biology-informed variance filter for DNAm (design Section 2.0).

    Filters applied BEFORE variance ranking: drop cross-reactive probes
    (Pidsley 2016 / Zhou 2017 blacklists), drop sex-chromosome probes, drop
    near-constant probes (max - min <= beta_range_floor). Then variance-rank.

    Parameters
    ----------
    bvals:
        (n_cpgs, n_samples) beta-value matrix indexed by CpG id.
    top_n:
        Number of CpGs to retain after variance ranking.
    cross_reactive, sex_chr_cpgs:
        Blacklist sets; empty by default (real blacklists supplied at runtime).
    beta_range_floor:
        Minimum (max - min) range across samples to keep a probe.
    """
    cross_reactive = cross_reactive or set()
    sex_chr_cpgs = sex_chr_cpgs or set()
    blacklist = cross_reactive | sex_chr_cpgs

    keep = bvals.loc[~bvals.index.isin(blacklist)]
    beta_range = keep.max(axis=1) - keep.min(axis=1)
    keep = keep.loc[beta_range > beta_range_floor]

    variances = keep.var(axis=1).sort_values(ascending=False)
    selected = [str(c) for c in variances.head(top_n).index]
    logger.info(
        "Tier 2 DNAm: %d CpGs after blacklist/range filter, top %d by variance",
        len(keep),
        len(selected),
    )
    return selected


def variance_filter_rna(
    expr: pd.DataFrame,
    *,
    top_n: int = TIER2_RNA_TOP,
) -> list[str]:
    """Tier 2 HVG selection for RNA (top_n highest-variance genes).

    ``expr`` is (n_genes, n_samples), indexed by gene id. Caller must pass the
    training-fold subset only; HVG selection on the full data leaks.
    """
    variances = expr.var(axis=1).sort_values(ascending=False)
    selected = [str(g) for g in variances.head(top_n).index]
    logger.info("Tier 2 RNA: top %d HVGs of %d genes", len(selected), expr.shape[0])
    return selected


def resolve_feature_tier(
    bvals: pd.DataFrame,
    expr: pd.DataFrame,
    *,
    artefact_dir: Path = PHASE1_ARTEFACT_DIR,
    cpg_gene_map: pd.DataFrame | None = None,
    cross_reactive: set[str] | None = None,
    sex_chr_cpgs: set[str] | None = None,
) -> FeatureTier:
    """Resolve the active feature tier (design Section 2.0 decision logic).

    Tier 1 is used iff the CellDMC artefact exists, is non-empty, and yields at
    least one significant CpG. Otherwise Tier 2 variance filter is used. This is
    the single decision point that lets the same Phase 2 code run on synthetic
    fixtures (always Tier 2) and on real post-Phase-1 data (Tier 1 when signal
    exists).

    ``bvals`` / ``expr`` must already be the training-fold subset.
    """
    celldmc = read_celldmc_interactions(artefact_dir)
    if celldmc is not None:
        tier1_cpgs = select_tier1_cpgs(celldmc)
        if tier1_cpgs and cpg_gene_map is not None:
            tier1_cpgs = [c for c in tier1_cpgs if c in set(bvals.index.astype(str))]
            tier1_genes = map_cpgs_to_cis_genes(tier1_cpgs, cpg_gene_map)
            tier1_genes = [g for g in tier1_genes if g in set(expr.index.astype(str))]
            if tier1_cpgs and tier1_genes:
                return FeatureTier(
                    tier=1,
                    dnam_cpgs=tier1_cpgs,
                    rna_genes=tier1_genes,
                    rationale=(
                        f"Tier 1: {len(tier1_cpgs)} CellDMC-prioritised CpGs "
                        f"(FDR < {CELLDMC_FDR_THRESHOLD}) + {len(tier1_genes)} cis-genes"
                    ),
                )

    dnam = variance_filter_dnam(bvals, cross_reactive=cross_reactive, sex_chr_cpgs=sex_chr_cpgs)
    rna = variance_filter_rna(expr)
    return FeatureTier(
        tier=2,
        dnam_cpgs=dnam,
        rna_genes=rna,
        rationale=(
            f"Tier 2 fallback: variance filter ({len(dnam)} CpGs, {len(rna)} genes); "
            "CellDMC artefact absent/empty (pending Phase 1 re-run)"
        ),
    )


def assemble_rna_activity_features(
    pathway: pd.DataFrame | None,
    tf_activity: pd.DataFrame | None,
    *,
    top_tf_by_variance: int = 150,
) -> pd.DataFrame | None:
    """Concatenate PROGENy + top-variance TF activity into the Arm A RNA input.

    Returns the (n_samples, ~120-220) activity-score matrix described in design
    Section 2.1, or None if neither artefact is available (synthetic-only mode,
    where the harness substitutes variance-filtered raw expression instead).
    """
    if pathway is None and tf_activity is None:
        return None
    parts: list[pd.DataFrame] = []
    if pathway is not None:
        parts.append(pathway)
    if tf_activity is not None:
        tf_var = tf_activity.var(axis=0).sort_values(ascending=False)
        top_tfs = tf_var.head(top_tf_by_variance).index
        parts.append(tf_activity[top_tfs])
    combined = pd.concat(parts, axis=1)
    logger.info("RNA activity features assembled: %s", combined.shape)
    return combined
