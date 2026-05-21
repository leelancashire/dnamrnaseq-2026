"""Cross-disorder centroid projection for Gate 0-X.

Tests whether Emory non-responder baseline RNA-seq centroid sits closer
to the GSE98793 TRD-inflammatory centroid than the Emory responder centroid.
This is the quantitative anchor for the Phase 3.3 headline figure.

Acceptance thresholds (ANALYSIS_PLAN.md Step 0-X):
  - PASS: d(Emory NR, GSE TRD-inflammatory) < d(Emory R, GSE TRD-inflammatory)
          at permutation p < 0.05.
  - MARGINAL: correct direction but p in [0.05, 0.15].
  - FAIL: wrong direction or non-significant.

Analysis plan reference: ANALYSIS_PLAN.md Step 0-X.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer

logger = logging.getLogger(__name__)

# Acceptance thresholds
PERMUTATION_PASS_THRESHOLD = 0.05
PERMUTATION_MARGINAL_THRESHOLD = 0.15
BOOTSTRAP_N = 2000
TOP_GENES_N = 2000


def load_gse98793(data_path: Path) -> pd.DataFrame:
    """Load GSE98793 expression matrix from local TSV/CSV file.

    Expected format: genes x samples TSV/CSV with gene IDs as index.
    The file should be the soft-matrix or series matrix normalised expression.

    Parameters
    ----------
    data_path:
        Path to the local GSE98793 expression file (TSV or CSV).

    Returns
    -------
    pd.DataFrame
        Genes x samples expression matrix.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"GSE98793 expression file not found at {data_path}. "
            "Set config.yaml data.external.gse98793 to the local file path. "
            "Download from NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793"
        )

    sep = "\t" if data_path.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(str(data_path), sep=sep, index_col=0)
    logger.info("Loaded GSE98793: %s", df.shape)
    return df


def harmonise_expression_matrices(
    emory_rnaseq: pd.DataFrame,
    gse_expr: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantile-normalise and intersect Emory RNA-seq with GSE98793 expression.

    Per ANALYSIS_PLAN.md Step 0-X Method step 2: quantile-normalise across
    cohorts (crude, for Phase 0 gate), restrict to gene intersection.

    Parameters
    ----------
    emory_rnaseq:
        Emory RNA-seq log-CPM, shape (n_genes, n_samples).
    gse_expr:
        GSE98793 expression matrix, shape (n_genes, n_samples).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Both matrices restricted to shared gene IDs, quantile-normalised.
    """
    shared_genes = emory_rnaseq.index.intersection(gse_expr.index)
    n_shared = len(shared_genes)
    logger.info(
        "Gene intersection: %d (Emory=%d, GSE=%d)",
        n_shared,
        len(emory_rnaseq.index),
        len(gse_expr.index),
    )
    if n_shared < 5000:
        logger.warning(
            "Fewer than 5000 shared genes (%d). " "Check gene ID format (symbol vs Ensembl).",
            n_shared,
        )

    emory_sub = emory_rnaseq.loc[shared_genes]
    gse_sub = gse_expr.loc[shared_genes]

    # Quantile-normalise: transpose to samples x genes, fit on combined, split back
    # ANALYSIS_PLAN.md: "crude harmonisation, document as load-bearing caveat"
    combined = pd.concat([emory_sub, gse_sub], axis=1)  # genes x all_samples
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    combined_norm = pd.DataFrame(
        qt.fit_transform(combined.T).T,
        index=combined.index,
        columns=combined.columns,
    )
    emory_norm = combined_norm[emory_sub.columns]
    gse_norm = combined_norm[gse_sub.columns]
    logger.info("Quantile-normalised combined matrix: %s -> split back.", combined.shape)
    return emory_norm, gse_norm


def compute_centroids(
    emory_rnaseq: pd.DataFrame,
    gse_expr: pd.DataFrame,
    emory_response: pd.Series,
    gse_trd_mask: pd.Series,
    gse_control_mask: pd.Series,
    top_n_genes: int = TOP_GENES_N,
) -> dict[str, Any]:
    """Compute centroids in top-variance gene space.

    Per ANALYSIS_PLAN.md Step 0-X Method steps 3-5.

    Parameters
    ----------
    emory_rnaseq:
        Emory baseline (PRE) RNA-seq, shape (n_genes, n_emory_subjects).
        Already harmonised and quantile-normalised.
    gse_expr:
        GSE98793 expression, shape (n_genes, n_gse_samples).
        Already harmonised and quantile-normalised.
    emory_response:
        Series indexed by Emory sample column names, values 'R' or 'NR'.
    gse_trd_mask:
        Boolean Series indexed by GSE sample column names; True = TRD-inflammatory.
    gse_control_mask:
        Boolean Series indexed by GSE sample column names; True = control.
    top_n_genes:
        Number of genes to retain by combined-cohort variance.

    Returns
    -------
    dict
        Keys: 'emory_r_centroid', 'emory_nr_centroid',
        'gse_trd_centroid', 'gse_control_centroid',
        'top_genes', 'n_emory_r', 'n_emory_nr', 'n_gse_trd', 'n_gse_control'.
    """
    # Variance filter on combined cohort
    combined = pd.concat([emory_rnaseq, gse_expr], axis=1)
    gene_var = combined.var(axis=1)
    top_genes = gene_var.nlargest(min(top_n_genes, len(gene_var))).index
    logger.info("Top %d genes by combined variance retained.", len(top_genes))

    emory_filt = emory_rnaseq.loc[top_genes]
    gse_filt = gse_expr.loc[top_genes]

    # Compute centroids (mean across samples per group)
    r_samples = emory_response[emory_response == "R"].index
    nr_samples = emory_response[emory_response == "NR"].index
    r_samples_valid = [s for s in r_samples if s in emory_filt.columns]
    nr_samples_valid = [s for s in nr_samples if s in emory_filt.columns]
    trd_samples = gse_trd_mask[gse_trd_mask].index.tolist()
    trd_samples_valid = [s for s in trd_samples if s in gse_filt.columns]
    ctrl_samples = gse_control_mask[gse_control_mask].index.tolist()
    ctrl_samples_valid = [s for s in ctrl_samples if s in gse_filt.columns]

    logger.info(
        "Group sizes: Emory R=%d, NR=%d, GSE TRD=%d, control=%d",
        len(r_samples_valid),
        len(nr_samples_valid),
        len(trd_samples_valid),
        len(ctrl_samples_valid),
    )

    emory_r_centroid = emory_filt[r_samples_valid].mean(axis=1)
    emory_nr_centroid = emory_filt[nr_samples_valid].mean(axis=1)
    gse_trd_centroid = gse_filt[trd_samples_valid].mean(axis=1) if trd_samples_valid else None
    gse_ctrl_centroid = gse_filt[ctrl_samples_valid].mean(axis=1) if ctrl_samples_valid else None

    return {
        "emory_r_centroid": emory_r_centroid,
        "emory_nr_centroid": emory_nr_centroid,
        "gse_trd_centroid": gse_trd_centroid,
        "gse_control_centroid": gse_ctrl_centroid,
        "top_genes": top_genes,
        "n_emory_r": len(r_samples_valid),
        "n_emory_nr": len(nr_samples_valid),
        "n_gse_trd": len(trd_samples_valid),
        "n_gse_control": len(ctrl_samples_valid),
        "emory_filt": emory_filt,
        "gse_filt": gse_filt,
        "emory_response": emory_response,
        "gse_trd_mask": gse_trd_mask,
    }


def run_permutation_test(
    emory_filt: pd.DataFrame,
    emory_response: pd.Series,
    gse_trd_centroid: pd.Series,
    n_permutations: int = BOOTSTRAP_N,
    seed: int = 42,
) -> dict[str, Any]:
    """Permutation test: is d(NR, TRD) < d(R, TRD) beyond chance?

    Permutes Response labels among Emory subjects B times and recomputes
    the centroid distance difference each time.

    Parameters
    ----------
    emory_filt:
        Emory expression in top-gene space (n_genes x n_samples).
    emory_response:
        Response labels ('R'/'NR') indexed by sample column name.
    gse_trd_centroid:
        GSE TRD-inflammatory centroid, shape (n_genes,).
    n_permutations:
        Number of permutations.
    seed:
        Random seed.

    Returns
    -------
    dict
        Keys: 'observed_d_nr', 'observed_d_r', 'observed_delta',
        'perm_deltas', 'p_value', 'direction_correct'.
    """
    r_samps = [s for s in emory_response[emory_response == "R"].index if s in emory_filt.columns]
    nr_samps = [s for s in emory_response[emory_response == "NR"].index if s in emory_filt.columns]
    all_samps = r_samps + nr_samps
    n_nr = len(nr_samps)

    trd_vec: np.ndarray[Any, Any] = np.asarray(gse_trd_centroid.to_numpy(), dtype=float)

    def euclidean_centroid_distance(samp_list: list[str]) -> float:
        cent: np.ndarray[Any, Any] = np.asarray(
            emory_filt[samp_list].mean(axis=1).to_numpy(), dtype=float
        )
        return float(np.linalg.norm(cent - trd_vec))

    obs_d_nr = euclidean_centroid_distance(nr_samps)
    obs_d_r = euclidean_centroid_distance(r_samps)
    obs_delta = obs_d_nr - obs_d_r  # negative = NR closer to TRD (expected direction)

    rng = np.random.default_rng(seed)
    perm_deltas = []
    all_samps_arr = np.array(all_samps)
    for _ in range(n_permutations):
        perm = rng.permutation(len(all_samps_arr))
        perm_nr = list(all_samps_arr[perm[:n_nr]])
        perm_r = list(all_samps_arr[perm[n_nr:]])
        perm_d_nr = euclidean_centroid_distance(perm_nr)
        perm_d_r = euclidean_centroid_distance(perm_r)
        perm_deltas.append(perm_d_nr - perm_d_r)

    perm_deltas_arr = np.array(perm_deltas)
    # One-tailed p: fraction of permutations where perm_delta <= obs_delta
    # (i.e. NR was as close or closer to TRD than observed)
    p_value = float(np.mean(perm_deltas_arr <= obs_delta))

    logger.info(
        "Permutation test: d(NR,TRD)=%.4f, d(R,TRD)=%.4f, delta=%.4f, p=%.4f",
        obs_d_nr,
        obs_d_r,
        obs_delta,
        p_value,
    )

    return {
        "observed_d_nr": obs_d_nr,
        "observed_d_r": obs_d_r,
        "observed_delta": obs_delta,
        "perm_deltas": perm_deltas_arr.tolist(),
        "p_value": p_value,
        "direction_correct": obs_delta < 0,  # NR closer to TRD
    }


def project_to_pca_2d(
    emory_filt: pd.DataFrame,
    gse_filt: pd.DataFrame,
    emory_response: pd.Series,
    gse_trd_mask: pd.Series,
    centroids: dict[str, Any],
) -> dict[str, Any]:
    """Project all samples and centroids into 2D PCA for visualisation.

    Fits PCA on combined Emory + GSE matrix, projects samples and centroids.

    Returns
    -------
    dict
        Keys: 'pca', 'emory_pcs', 'gse_pcs', 'centroid_pcs'.
    """
    combined = pd.concat([emory_filt, gse_filt], axis=1)
    pca = PCA(n_components=2, random_state=42)
    combined_pcs = pca.fit_transform(combined.T.values)  # (n_samples, 2)

    n_emory = emory_filt.shape[1]
    emory_pcs = pd.DataFrame(
        combined_pcs[:n_emory], index=emory_filt.columns, columns=["PC1", "PC2"]
    )
    emory_pcs["Response"] = emory_response.reindex(emory_pcs.index)
    emory_pcs["cohort"] = "Emory"

    gse_pcs = pd.DataFrame(combined_pcs[n_emory:], index=gse_filt.columns, columns=["PC1", "PC2"])
    gse_pcs["is_trd"] = gse_trd_mask.reindex(gse_pcs.index).fillna(False)
    gse_pcs["cohort"] = "GSE98793"

    # Project centroids
    centroid_names = [
        "emory_r_centroid",
        "emory_nr_centroid",
        "gse_trd_centroid",
        "gse_control_centroid",
    ]
    centroid_points = {}
    for name in centroid_names:
        c = centroids.get(name)
        if c is not None:
            pc_coords = pca.transform(c.values.reshape(1, -1))[0]
            centroid_points[name] = pc_coords.tolist()

    return {
        "pca": pca,
        "emory_pcs": emory_pcs,
        "gse_pcs": gse_pcs,
        "centroid_pcs": centroid_points,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def determine_gate_0x_verdict(
    perm_results: dict[str, Any],
) -> str:
    """Return PASS, MARGINAL, or FAIL verdict for Gate 0-X.

    Parameters
    ----------
    perm_results:
        Output of run_permutation_test.

    Returns
    -------
    str
        'PASS', 'MARGINAL', or 'FAIL'.
    """
    p = perm_results["p_value"]
    direction_ok = perm_results["direction_correct"]

    if not direction_ok:
        return "FAIL"
    if p < PERMUTATION_PASS_THRESHOLD:
        return "PASS"
    if p < PERMUTATION_MARGINAL_THRESHOLD:
        return "MARGINAL"
    return "FAIL"
