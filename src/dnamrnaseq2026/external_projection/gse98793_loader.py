"""GSE98793 expression matrix loader with probe-to-gene rollup.

GSE98793 is a whole-blood Affymetrix microarray study (GPL570,
HG-U133 Plus 2.0) containing 192 samples:
  - 128 MDD (major depressive disorder) cases
  - 64 healthy controls

Probe-to-gene mapping is performed using the committed reference annotation
at resources/hgu133plus2_probe_to_gene.csv (derived from GPL570 platform
table, 45,782 probes, 22,880 unique gene symbols).

Rollup strategy: max-mean (per-gene, take the probe with the highest mean
expression across all samples). This is the bioinformatics standard for
Affymetrix HG-U133 and reduces 54,675 probes to ~22,880 gene-level rows
suitable for cross-platform harmonisation with Emory RNA-seq.

Reference
---------
Reference annotation file:
  src/dnamrnaseq2026/external_projection/resources/hgu133plus2_probe_to_gene.csv
  Source: GPL570 platform table downloaded via GEOparse 2.0.4, 2026-05-17.
  Columns: probe_id, gene_symbol

ANALYSIS_PLAN.md: Step 0-X, Method steps 1-3.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Path to committed probe-to-gene reference CSV
_RESOURCES_DIR = Path(__file__).parent / "resources"
_DEFAULT_PROBE_GENE_CSV = _RESOURCES_DIR / "hgu133plus2_probe_to_gene.csv"

# Expected GPL
EXPECTED_GPL = "GPL570"

# Subject group values in GSE98793 sample characteristics
CNTL_GROUP_MARKER = "CNTL"
CASE_GROUP_MARKER = "CASE"


# ---------------------------------------------------------------------------
# Phenotype extraction
# ---------------------------------------------------------------------------


def extract_gse98793_phenotypes(gse: Any) -> pd.DataFrame:
    """Extract sample-level phenotype table from a loaded GSE98793 object.

    Parameters
    ----------
    gse:
        GEOparse.GSE object from download_gse98793().

    Returns
    -------
    pd.DataFrame
        Index = GSM accession IDs. Columns: subject_group (CASE/CNTL),
        is_mdd (bool), is_control (bool), gender, age, anxiety, batch.
    """
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        row: dict[str, Any] = {"gsm_id": gsm_id}
        for c in chars:
            c = c.strip()
            if ":" in c:
                key, val = c.split(":", 1)
                row[key.strip().lower().replace(" ", "_")] = val.strip()
        rows.append(row)

    pheno = pd.DataFrame(rows).set_index("gsm_id")

    # Parse subject group
    if "subject_group" in pheno.columns:
        pheno["is_mdd"] = pheno["subject_group"].str.contains(CASE_GROUP_MARKER, na=False)
        pheno["is_control"] = pheno["subject_group"].str.contains(CNTL_GROUP_MARKER, na=False)
    else:
        logger.warning(
            "subject_group column not found in GSE98793 phenotypes. "
            "Defaulting all to is_mdd=True, is_control=False."
        )
        pheno["is_mdd"] = True
        pheno["is_control"] = False

    logger.info(
        "GSE98793 phenotypes: %d total, %d MDD, %d control.",
        len(pheno),
        pheno["is_mdd"].sum(),
        pheno["is_control"].sum(),
    )
    return pheno


# ---------------------------------------------------------------------------
# Probe-to-gene mapping
# ---------------------------------------------------------------------------


def load_probe_to_gene_map(
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """Load the HG-U133 Plus 2.0 probe-to-gene reference annotation.

    Parameters
    ----------
    csv_path:
        Path to probe-to-gene CSV. Defaults to the committed reference at
        resources/hgu133plus2_probe_to_gene.csv (GPL570 platform table,
        45,782 probes to 22,880 gene symbols, derived 2026-05-17).

    Returns
    -------
    pd.DataFrame
        Columns: probe_id (str), gene_symbol (str). Indexed by probe_id.
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_PROBE_GENE_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Probe-to-gene reference not found at {path}. "
            "It should be committed to the repository at "
            "src/dnamrnaseq2026/external_projection/resources/hgu133plus2_probe_to_gene.csv"
        )
    df = pd.read_csv(str(path), dtype={"probe_id": str, "gene_symbol": str})
    df = df.dropna(subset=["probe_id", "gene_symbol"])
    df = df[df["gene_symbol"].str.strip() != ""]
    df = df.set_index("probe_id")
    logger.info(
        "Loaded probe-to-gene map: %d probes, %d unique genes.",
        len(df),
        df["gene_symbol"].nunique(),
    )
    return df


# ---------------------------------------------------------------------------
# Expression matrix construction
# ---------------------------------------------------------------------------


def build_gse98793_expression_matrix(
    gse: Any,
    probe_gene_map: pd.DataFrame | None = None,
    rollup: str = "max_mean",
) -> pd.DataFrame:
    """Build a gene x sample expression matrix from a loaded GSE object.

    Pivots per-sample probe values into a genes x samples DataFrame using
    probe-to-gene rollup. Values are taken as-is (already RMA-normalised by
    the submitter; see GSE98793 series metadata).

    Parameters
    ----------
    gse:
        GEOparse.GSE object from download_gse98793().
    probe_gene_map:
        probe_id-indexed DataFrame with gene_symbol column. If None, loads
        the committed reference annotation.
    rollup:
        Probe-to-gene rollup strategy. Only 'max_mean' is implemented:
        for each gene, retain the probe with the highest mean expression
        across all samples (standard for Affymetrix HG-U133 Plus 2.0).

    Returns
    -------
    pd.DataFrame
        Gene symbols as index, GSM accession IDs as columns.
        Shape: (~22,880 genes, 192 samples) for the full GSE98793.

    Raises
    ------
    ValueError
        If rollup is not 'max_mean'.
    """
    if rollup != "max_mean":
        raise ValueError(f"Unsupported rollup strategy: {rollup!r}. Use 'max_mean'.")

    if probe_gene_map is None:
        probe_gene_map = load_probe_to_gene_map()

    # Build probe x sample matrix
    logger.info("Building probe x sample matrix from %d GSMs...", len(gse.gsms))
    sample_dfs = []
    for gsm_id, gsm in gse.gsms.items():
        s = gsm.table.set_index("ID_REF")["VALUE"].rename(gsm_id)
        s = pd.to_numeric(s, errors="coerce")
        sample_dfs.append(s)

    probe_matrix = pd.DataFrame(sample_dfs).T  # probes x samples
    logger.info("Probe x sample matrix shape: %s", probe_matrix.shape)

    # Restrict to probes with gene annotations
    annotated_probes = probe_matrix.index.intersection(probe_gene_map.index)
    n_total = len(probe_matrix)
    n_annotated = len(annotated_probes)
    logger.info(
        "Annotated probes: %d / %d (%.1f%%).",
        n_annotated,
        n_total,
        100 * n_annotated / max(n_total, 1),
    )
    probe_matrix = probe_matrix.loc[annotated_probes].copy()
    probe_matrix["gene_symbol"] = probe_gene_map.loc[annotated_probes, "gene_symbol"].values

    # Max-mean rollup: per gene, pick probe with highest row mean
    probe_means = probe_matrix.drop(columns=["gene_symbol"]).mean(axis=1)
    probe_matrix["probe_mean"] = probe_means

    # For each gene, keep the probe with the highest mean expression
    best_probe_idx = probe_matrix.groupby("gene_symbol")["probe_mean"].idxmax().dropna()
    gene_matrix = probe_matrix.loc[best_probe_idx].drop(columns=["gene_symbol", "probe_mean"])
    # Set gene symbol as index
    gene_matrix.index = probe_gene_map.loc[best_probe_idx.values, "gene_symbol"].values

    logger.info(
        "Gene x sample matrix shape after rollup: %s (strategy: %s).",
        gene_matrix.shape,
        rollup,
    )
    return gene_matrix


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------


def load_gse98793_gene_matrix(
    gse: Any | None = None,
    cache_dir: Path | str | None = None,
    probe_gene_csv: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSE98793 gene expression matrix and phenotype table.

    Downloads and caches the GSE98793 SOFT file if not already present, then
    returns the gene x sample matrix and sample phenotype table.

    Parameters
    ----------
    gse:
        Pre-loaded GEOparse.GSE object. If None, calls download_gse98793().
    cache_dir:
        Cache directory for download_gse98793(). Ignored if gse is provided.
    probe_gene_csv:
        Override path for the probe-to-gene reference CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (gene_matrix, phenotypes)
        gene_matrix: genes x samples (float expression values).
        phenotypes: GSM-indexed DataFrame with is_mdd, is_control columns.
    """
    if gse is None:
        from dnamrnaseq2026.external_projection.datasets import download_gse98793

        gse = download_gse98793(cache_dir=cache_dir)

    probe_gene_map = load_probe_to_gene_map(probe_gene_csv)
    gene_matrix = build_gse98793_expression_matrix(gse, probe_gene_map=probe_gene_map)
    phenotypes = extract_gse98793_phenotypes(gse)

    # Align phenotype index to matrix columns
    phenotypes = phenotypes.reindex(gene_matrix.columns)

    return gene_matrix, phenotypes


# ---------------------------------------------------------------------------
# TRD-inflammatory subset definition
# ---------------------------------------------------------------------------


def reindex_emory_by_gene_symbol(emory_rnaseq: pd.DataFrame) -> pd.DataFrame:
    """Reindex an Emory RNA-seq matrix from Ensembl IDs to gene symbols.

    Emory RNA-seq uses GENCODE-style IDs of the form 'ENSGXXXXXX.X_SYMBOL'
    (e.g. 'ENSG00000227232.5_WASH7P'). To harmonise with GSE98793's gene-symbol
    index, this function extracts the symbol component and collapses any
    duplicate symbols by keeping the row with the highest mean expression
    (same max-mean strategy used for Affymetrix probe rollup).

    Parameters
    ----------
    emory_rnaseq:
        Emory RNA-seq log-CPM matrix, indexed by Ensembl gene IDs with
        symbol suffix (genes x samples).

    Returns
    -------
    pd.DataFrame
        Same matrix re-indexed by gene symbol. Rows without extractable symbols
        are dropped. Duplicate symbols are collapsed by max-mean.
    """

    def _extract_symbol(ensembl_gene_id: str) -> str | None:
        parts = str(ensembl_gene_id).split("_", 1)
        if len(parts) > 1 and not parts[1].startswith("ENSG"):
            sym = parts[1].strip()
            return sym if sym else None
        return None

    symbols = emory_rnaseq.index.map(_extract_symbol)
    has_symbol = symbols.notna()
    logger.info(
        "Emory RNA-seq: %d / %d genes have extractable gene symbols.",
        int(has_symbol.sum()),
        len(emory_rnaseq),
    )

    mat = emory_rnaseq.loc[has_symbol].copy()
    mat.index = pd.Index(symbols[has_symbol].tolist(), name=None)

    # Collapse duplicates: keep row with highest row mean
    # Use integer positional index for groupby to avoid string-index NA issues
    mat = mat.reset_index(names=["gene_symbol"])
    row_means = mat.drop(columns=["gene_symbol"]).mean(axis=1)
    mat["_row_mean"] = row_means
    best_pos = mat.groupby("gene_symbol")["_row_mean"].idxmax()
    mat = mat.loc[best_pos.values].set_index("gene_symbol").drop(columns=["_row_mean"])

    n_dupes = int(has_symbol.sum()) - len(mat)
    logger.info(
        "After duplicate-symbol collapse: %d unique gene symbols (%d duplicates dropped).",
        len(mat),
        n_dupes,
    )
    return mat


def define_trd_inflammatory_mask(
    phenotypes: pd.DataFrame,
    gene_matrix: pd.DataFrame,
    n_top_quartile: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Define TRD-inflammatory and control masks for Gate 0-X.

    Per ANALYSIS_PLAN.md Step 0-X: TRD-inflammatory = GSE98793 MDD samples
    (all CASE samples). For Phase 3.3, refine with high-inflammation criterion
    (top-quartile of an inflammation-gene-set GSVA score). For Phase 0 gate,
    all MDD cases are used as the TRD proxy.

    Parameters
    ----------
    phenotypes:
        Output of extract_gse98793_phenotypes(). Index = GSM IDs.
    gene_matrix:
        Gene x sample matrix. Columns = GSM IDs.
    n_top_quartile:
        Unused in Phase 0; included for API forward-compatibility.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        (trd_mask, control_mask): boolean Series indexed by sample columns
        of gene_matrix.
    """
    sample_ids = gene_matrix.columns
    pheno_aligned = phenotypes.reindex(sample_ids)

    trd_mask = pheno_aligned["is_mdd"].fillna(False)
    control_mask = pheno_aligned["is_control"].fillna(False)

    logger.info(
        "TRD mask: %d MDD samples; control mask: %d control samples.",
        int(trd_mask.sum()),
        int(control_mask.sum()),
    )
    return trd_mask, control_mask
