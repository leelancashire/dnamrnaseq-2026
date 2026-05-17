#!/usr/bin/env python
"""Gate 0-X entry-point: cross-disorder centroid projection (Emory vs GSE98793).

Reads config.yaml, downloads GSE98793 via the cached downloader if needed,
loads Emory baseline RNA-seq, harmonises both datasets, computes centroids,
runs permutation test.

GSE98793 is an Affymetrix GPL570 whole-blood microarray study with 192
samples (128 MDD cases, 64 healthy controls). Probe-level data is rolled
up to gene-level using max-mean rollup against the committed reference
annotation (src/dnamrnaseq2026/external_projection/resources/
hgu133plus2_probe_to_gene.csv). For Phase 0, all MDD (CASE) samples are
used as the TRD-inflammatory proxy; Phase 3.3 will refine this with an
inflammation-gene-set GSVA score.

Outputs (to analysis/2026-05-17-phase-0/0-X/):
  gate_0X_centroids.json              -- distances, p-values, verdict
  gate_0X_genes_used.csv              -- 2000-gene set with per-group means
  gate_0X_centroid_projection.png     -- 2D scatter with centroid markers
  gate_0X_centroid_projection.svg

License: GSE98793 data from NCBI GEO (NCBI public data, no DUA required).
  https://www.ncbi.nlm.nih.gov/geo/info/faq.html

Manual fallback if GEO FTP is down:
  Download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793
  Place 'GSE98793_family.soft.gz' in data/external/ and re-run.

Usage:
    python scripts/01_phase0_gate_X.py
    python scripts/01_phase0_gate_X.py --n-perm 2000
    python scripts/01_phase0_gate_X.py --cache-dir ~/data/external
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

OUT_DIR = _REPO_ROOT / "analysis/2026-05-17-phase-0/0-X"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0-X: cross-disorder centroid projection.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override cache directory for GSE98793 SOFT file (else config.yaml / data/external/).",
    )
    p.add_argument("--n-perm", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Gate 0-X starting.")

    import pandas as pd

    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import load_emory_rnaseq, load_emory_subject_data
    from dnamrnaseq2026.external_projection.cross_disorder_centroid import (
        compute_centroids,
        determine_gate_0x_verdict,
        harmonise_expression_matrices,
        project_to_pca_2d,
        run_permutation_test,
    )
    from dnamrnaseq2026.external_projection.datasets import download_gse98793
    from dnamrnaseq2026.external_projection.gse98793_loader import (
        build_gse98793_expression_matrix,
        define_trd_inflammatory_mask,
        extract_gse98793_phenotypes,
        reindex_emory_by_gene_symbol,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]

    # Resolve cache directory
    cache_dir = args.cache_dir
    if cache_dir is None:
        ext_data_dir = cfg.get("data", {}).get("external_data_dir")
        if ext_data_dir:
            cache_dir = Path(ext_data_dir).expanduser()
        else:
            cache_dir = _REPO_ROOT / "data" / "external"

    # Download GSE98793 (cache hit if already present)
    logger.info("Ensuring GSE98793 is available in %s...", cache_dir)
    try:
        gse = download_gse98793(cache_dir=cache_dir)
    except RuntimeError as exc:
        logger.error("Failed to obtain GSE98793: %s", exc)
        out: dict[str, Any] = {
            "gate": "0-X",
            "verdict": "BLOCKED",
            "reason": (
                f"GSE98793 download failed: {exc}. "
                "Manual fallback: download from "
                "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793 "
                f"and place GSE98793_family.soft.gz in {cache_dir}."
            ),
        }
        with (OUT_DIR / "gate_0X_centroids.json").open("w") as fh:
            json.dump(out, fh, indent=2)
        print("\nGate 0-X: BLOCKED (GSE98793 not available)")
        sys.exit(0)

    # Build gene x sample matrix from GSE98793 (probe-to-gene rollup)
    logger.info("Building gene x sample matrix from GSE98793 (max-mean rollup)...")
    gse_gene_matrix = build_gse98793_expression_matrix(gse, rollup="max_mean")
    gse_phenotypes = extract_gse98793_phenotypes(gse)

    # Define TRD-inflammatory and control masks
    # Phase 0: all MDD (CASE) samples as TRD proxy; all CNTL as controls.
    trd_mask, ctrl_mask = define_trd_inflammatory_mask(gse_phenotypes, gse_gene_matrix)

    # Load Emory RNA-seq and build PRE baseline matrix
    logger.info("Loading Emory RNA-seq and subject data...")
    emory_rnaseq = load_emory_rnaseq()
    subject_data = load_emory_subject_data()

    pre_rows = subject_data[subject_data["Visit"] == "PRE-IOP"].copy()

    # Build PRE sample lookup: try SampleName_RNASeq first, fall back to
    # '{Subcode}-{Visit}' format (same fix applied in delta_construction.py)
    sample_name_values = set(pre_rows["SampleName_RNASeq"].values)
    rnaseq_cols = set(emory_rnaseq.columns)
    use_sample_col = bool(sample_name_values & rnaseq_cols)

    if use_sample_col:
        col_to_subcode = pre_rows.set_index("SampleName_RNASeq")["Subcode"].to_dict()
        col_to_response = pre_rows.set_index("SampleName_RNASeq")["Response"].to_dict()
        pre_cols = [c for c in pre_rows["SampleName_RNASeq"].values if c in rnaseq_cols]
    else:
        logger.warning(
            "SampleName_RNASeq values (%s ...) not found in RNA-seq columns (%s ...). "
            "Falling back to '{Subcode}-PRE-IOP' column format.",
            list(sample_name_values)[:2],
            list(emory_rnaseq.columns[:2]),
        )
        # Build {Subcode}-PRE-IOP column names
        subcode_to_response = pre_rows.set_index("Subcode")["Response"].to_dict()
        pre_cols = [
            f"{row['Subcode']}-PRE-IOP"
            for _, row in pre_rows.iterrows()
            if f"{row['Subcode']}-PRE-IOP" in rnaseq_cols
        ]
        col_to_response = {c: subcode_to_response[c.split("-PRE-IOP")[0]] for c in pre_cols}

    emory_pre = emory_rnaseq[pre_cols]
    response_map = pd.Series({c: col_to_response.get(c, "unknown") for c in pre_cols})
    response_map = response_map[response_map.isin(["R", "NR"])]
    emory_pre = emory_pre[[c for c in emory_pre.columns if c in response_map.index]]
    logger.info(
        "Emory PRE baseline: %d genes x %d samples (R=%d, NR=%d)",
        emory_pre.shape[0],
        emory_pre.shape[1],
        int((response_map == "R").sum()),
        int((response_map == "NR").sum()),
    )

    # Reindex Emory from Ensembl IDs (ENSGXXXXXX.X_SYMBOL) to gene symbols
    # so the intersection with GSE98793 gene symbols is non-empty.
    logger.info("Reindexing Emory RNA-seq from Ensembl IDs to gene symbols...")
    emory_pre_sym = reindex_emory_by_gene_symbol(emory_pre)
    logger.info(
        "Emory gene-symbol matrix: %d genes x %d samples.",
        emory_pre_sym.shape[0],
        emory_pre_sym.shape[1],
    )

    # Harmonise: gene-symbol intersection + quantile normalisation
    emory_norm, gse_norm = harmonise_expression_matrices(emory_pre_sym, gse_gene_matrix)

    # Align masks to the harmonised (potentially column-reordered) gse_norm
    trd_mask = trd_mask.reindex(gse_norm.columns).fillna(False)
    ctrl_mask = ctrl_mask.reindex(gse_norm.columns).fillna(False)

    logger.info(
        "GSE98793 samples: %d total, TRD mask=%d, control mask=%d.",
        gse_norm.shape[1],
        int(trd_mask.sum()),
        int(ctrl_mask.sum()),
    )

    # Compute centroids
    centroids = compute_centroids(
        emory_norm,
        gse_norm,
        emory_response=response_map,
        gse_trd_mask=trd_mask,
        gse_control_mask=ctrl_mask,
        top_n_genes=2000,
    )

    if centroids["gse_trd_centroid"] is None:
        logger.error("GSE TRD centroid is None (no TRD samples found). Gate 0-X FAIL.")
        out = {
            "gate": "0-X",
            "verdict": "FAIL",
            "reason": "No TRD-inflammatory samples identified in GSE98793.",
        }
        with (OUT_DIR / "gate_0X_centroids.json").open("w") as fh:
            json.dump(out, fh, indent=2)
        sys.exit(0)

    # Permutation test
    emory_filt = centroids["emory_filt"]
    gse_filt = centroids["gse_filt"]
    perm = run_permutation_test(
        emory_filt=emory_filt,
        emory_response=response_map,
        gse_trd_centroid=centroids["gse_trd_centroid"],
        n_permutations=args.n_perm,
        seed=seed,
    )
    verdict = determine_gate_0x_verdict(perm)
    logger.info("Gate 0-X verdict: %s", verdict)

    # Write results JSON
    gse_gene_n, gse_sample_n = gse_gene_matrix.shape
    gene_intersection_n = len(
        emory_pre_sym.index.intersection(gse_gene_matrix.index)
    )
    out = {
        "gate": "0-X",
        "verdict": verdict,
        "n_emory_r": centroids["n_emory_r"],
        "n_emory_nr": centroids["n_emory_nr"],
        "n_gse_trd": centroids["n_gse_trd"],
        "n_gse_control": centroids["n_gse_control"],
        "n_genes": int(len(centroids["top_genes"])),
        "gene_intersection_pre_filter": gene_intersection_n,
        "d_emory_nr_to_gse_trd": float(perm["observed_d_nr"]),
        "d_emory_r_to_gse_trd": float(perm["observed_d_r"]),
        "observed_delta": float(perm["observed_delta"]),
        "direction_correct": bool(perm["direction_correct"]),
        "permutation_p": float(perm["p_value"]),
        "n_permutations": args.n_perm,
        "gse98793_info": {
            "n_probes_total": 54675,
            "n_probes_annotated": 45782,
            "n_genes_after_rollup": gse_gene_n,
            "n_samples": gse_sample_n,
            "platform": "GPL570 (Affymetrix HG-U133 Plus 2.0)",
            "trd_subset": "all MDD CASE samples (Phase 0 proxy; Phase 3 refines with GSVA)",
            "probe_to_gene_rollup": "max_mean",
        },
        "harmonisation_note": (
            "Quantile normalisation across cohorts (crude Phase-0 harmonisation). "
            "Phase 3 will use ComBat or COCONUT for cross-platform correction. "
            "Gene IDs are gene symbols; intersection with Emory RNA-seq gene symbols."
        ),
        "seed": seed,
    }
    with (OUT_DIR / "gate_0X_centroids.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    # Genes used CSV
    top_genes = centroids["top_genes"]
    emory_resp = centroids["emory_response"]
    r_samples = [s for s in emory_resp[emory_resp == "R"].index if s in emory_filt.columns]
    nr_samples = [s for s in emory_resp[emory_resp == "NR"].index if s in emory_filt.columns]
    genes_df = pd.DataFrame(
        {
            "gene": top_genes,
            "emory_r_mean": emory_filt.loc[top_genes, r_samples].mean(axis=1).values,
            "emory_nr_mean": emory_filt.loc[top_genes, nr_samples].mean(axis=1).values,
        }
    )
    genes_df.to_csv(str(OUT_DIR / "gate_0X_genes_used.csv"), index=False)

    # PCA projection visualisation
    pca_result = project_to_pca_2d(
        emory_filt=emory_filt,
        gse_filt=gse_filt,
        emory_response=response_map,
        gse_trd_mask=trd_mask,
        centroids=centroids,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    emory_pcs = pca_result["emory_pcs"]
    gse_pcs = pca_result["gse_pcs"]
    centroid_pcs = pca_result["centroid_pcs"]
    ev = pca_result["explained_variance_ratio"]

    for resp, color, marker in [("R", "#2196F3", "o"), ("NR", "#F44336", "^")]:
        mask_resp = emory_pcs["Response"] == resp
        ax.scatter(
            emory_pcs.loc[mask_resp, "PC1"],
            emory_pcs.loc[mask_resp, "PC2"],
            c=color,
            marker=marker,
            s=50,
            alpha=0.6,
            label=f"Emory {resp}",
        )

    trd_gse = gse_pcs[gse_pcs["is_trd"]]
    ctrl_gse = gse_pcs[~gse_pcs["is_trd"]]
    ax.scatter(
        trd_gse["PC1"], trd_gse["PC2"], c="#FF9800", marker="s", s=30, alpha=0.5, label="GSE MDD"
    )
    ax.scatter(
        ctrl_gse["PC1"],
        ctrl_gse["PC2"],
        c="#4CAF50",
        marker="D",
        s=30,
        alpha=0.5,
        label="GSE ctrl",
    )

    centroid_colors = {
        "emory_r_centroid": "#2196F3",
        "emory_nr_centroid": "#F44336",
        "gse_trd_centroid": "#FF9800",
        "gse_control_centroid": "#4CAF50",
    }
    for cname, coords in centroid_pcs.items():
        ax.scatter(
            coords[0],
            coords[1],
            c=centroid_colors.get(cname, "black"),
            marker="*",
            s=400,
            zorder=6,
            edgecolors="black",
            linewidth=1.0,
            label=f"{cname.replace('_centroid', '')}",
        )

    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)" if len(ev) > 1 else "PC2")
    ax.set_title(
        f"Gate 0-X: Emory vs GSE98793 centroid projection\n"
        f"d(NR,MDD)={perm['observed_d_nr']:.4f}, d(R,MDD)={perm['observed_d_r']:.4f}, "
        f"p={perm['p_value']:.3f}\nVerdict: {verdict}"
    )
    ax.legend(loc="best", fontsize=7)
    fig.savefig(str(OUT_DIR / "gate_0X_centroid_projection.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(OUT_DIR / "gate_0X_centroid_projection.svg"), bbox_inches="tight")
    plt.close(fig)

    print()
    print("=" * 60)
    print("Gate 0-X: Cross-disorder centroid projection")
    print("=" * 60)
    print(f"Emory R: {out['n_emory_r']}, NR: {out['n_emory_nr']}")
    print(f"GSE MDD (TRD proxy): {out['n_gse_trd']}, control: {out['n_gse_control']}")
    print(f"Genes in common space: {out['n_genes']}")
    print(f"d(NR, MDD): {out['d_emory_nr_to_gse_trd']:.4f}")
    print(f"d(R, MDD):  {out['d_emory_r_to_gse_trd']:.4f}")
    print(f"Direction correct (NR closer to MDD): {out['direction_correct']}")
    print(f"Permutation p: {out['permutation_p']:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
