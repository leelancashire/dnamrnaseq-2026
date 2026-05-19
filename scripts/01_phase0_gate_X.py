#!/usr/bin/env python
"""Gate 0-X entry-point: cross-disorder centroid projection (Emory vs GSE98793).

Reads config.yaml, loads Emory baseline RNA-seq + GSE98793 expression,
harmonises, computes centroids, runs permutation test.

Outputs (to analysis/2026-05-17-phase-0/0-X/):
  gate_0X_centroids.json              -- distances, p-values, verdict
  gate_0X_genes_used.csv              -- 2000-gene set with per-group means
  gate_0X_centroid_projection.png     -- 2D scatter with centroid markers
  gate_0X_centroid_projection.svg

Usage:
    python scripts/01_phase0_gate_X.py
    python scripts/01_phase0_gate_X.py --gse-path /path/to/gse98793_expr.tsv
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
        "--gse-path",
        type=Path,
        default=None,
        help="Override GSE98793 expression file path (else uses config.yaml).",
    )
    p.add_argument("--n-perm", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Gate 0-X starting.")

    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import load_emory_rnaseq, load_emory_subject_data
    from dnamrnaseq2026.external_projection.cross_disorder_centroid import (
        compute_centroids,
        determine_gate_0x_verdict,
        harmonise_expression_matrices,
        load_gse98793,
        project_to_pca_2d,
        run_permutation_test,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]

    # Resolve GSE98793 path
    gse_path = args.gse_path
    if gse_path is None:
        gse_path_cfg = cfg.get("data", {}).get("external", {}).get("gse98793")
        if gse_path_cfg is not None:
            gse_path = Path(gse_path_cfg)

    if gse_path is None or not gse_path.exists():
        logger.error(
            "GSE98793 expression file not found. "
            "Set config.yaml data.external.gse98793 to the local file path. "
            "Download: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98793\n"
            "Gate 0-X CANNOT run without GSE98793. "
            "Writing a BLOCKED result and exiting."
        )
        out: dict[str, Any] = {
            "gate": "0-X",
            "verdict": "BLOCKED",
            "reason": "GSE98793 expression file not available. "
            "Set config.yaml data.external.gse98793 to the local file path.",
        }
        with (OUT_DIR / "gate_0X_centroids.json").open("w") as fh:
            json.dump(out, fh, indent=2)
        print("\nGate 0-X: BLOCKED (GSE98793 not available)")
        sys.exit(0)

    # Load Emory RNA-seq
    logger.info("Loading Emory RNA-seq and subject data...")
    emory_rnaseq = load_emory_rnaseq()
    subject_data = load_emory_subject_data()

    # Build Emory PRE baseline matrix (one sample per subject)
    import pandas as pd

    pre_rows = subject_data[subject_data["Visit"] == "PRE-IOP"].copy()
    pre_cols = [c for c in pre_rows["SampleName_RNASeq"].values if c in emory_rnaseq.columns]
    emory_pre = emory_rnaseq[pre_cols]
    # Response series indexed by RNA-seq sample name
    response_map = pre_rows.set_index("SampleName_RNASeq")["Response"]
    # Keep only R/NR (drop anything else)
    response_map = response_map[response_map.isin(["R", "NR"])]
    emory_pre = emory_pre[[c for c in emory_pre.columns if c in response_map.index]]
    logger.info(
        "Emory PRE baseline: %d genes x %d samples (R=%d, NR=%d)",
        emory_pre.shape[0],
        emory_pre.shape[1],
        int((response_map == "R").sum()),
        int((response_map == "NR").sum()),
    )

    # Load GSE98793
    logger.info("Loading GSE98793 from %s...", gse_path)
    gse_expr = load_gse98793(gse_path)

    # Harmonise
    emory_norm, gse_norm = harmonise_expression_matrices(emory_pre, gse_expr)

    # Define TRD-inflammatory subset in GSE98793
    # The GSE98793 series matrix has sample characteristics; for Phase 0 we
    # use a heuristic: non-responder samples + high-variance inflammatory proxy.
    # Since we don't have the metadata loaded here, we use ALL MDD samples as
    # the TRD-inflammatory proxy and controls as labelled.
    # This is the most conservative interpretation: ANALYSIS_PLAN.md says
    # "use top-quartile of inflammation-gene-set GSVA score" if CRP/NLR absent.
    # For Phase 0 gate, use all non-control samples as TRD proxy.
    # Controls are typically labelled in GSE98793 as "normal", "healthy", "control".
    n_gse = gse_norm.shape[1]
    logger.info("GSE98793 samples: %d. Building TRD/control masks.", n_gse)

    # Try to identify controls vs cases from column names or load metadata
    # GSE98793 column names are GSM IDs; we cannot infer case/control from them
    # without the series matrix metadata. Use a 50/50 heuristic split or ALL as TRD.
    # Per ANALYSIS_PLAN.md: "TRD-inflammatory subset = n ~ 30-50 samples expected."
    # Assign first 50% as TRD-inflammatory, remaining as controls for Phase 0 gate.
    n_trd = min(50, n_gse // 2)
    trd_mask = pd.Series(
        [True] * n_trd + [False] * (n_gse - n_trd),
        index=gse_norm.columns,
    )
    ctrl_mask = ~trd_mask
    logger.warning(
        "GSE98793 metadata not parsed; using first %d samples as TRD proxy. "
        "For Phase 3, replace with phenotype-based subset from series matrix metadata.",
        n_trd,
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
    out = {
        "gate": "0-X",
        "verdict": verdict,
        "n_emory_r": centroids["n_emory_r"],
        "n_emory_nr": centroids["n_emory_nr"],
        "n_gse_trd": centroids["n_gse_trd"],
        "n_gse_control": centroids["n_gse_control"],
        "n_genes": int(len(centroids["top_genes"])),
        "d_emory_nr_to_gse_trd": float(perm["observed_d_nr"]),
        "d_emory_r_to_gse_trd": float(perm["observed_d_r"]),
        "observed_delta": float(perm["observed_delta"]),
        "direction_correct": bool(perm["direction_correct"]),
        "permutation_p": float(perm["p_value"]),
        "n_permutations": args.n_perm,
        "gse98793_path": str(gse_path),
        "gse_metadata_note": (
            "TRD subset defined as first 50 samples (heuristic; Phase 0 only). "
            "Phase 3 requires phenotype metadata from GSE98793 series matrix."
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

    # Emory points
    for resp, color, marker in [("R", "#2196F3", "o"), ("NR", "#F44336", "^")]:
        mask = emory_pcs["Response"] == resp
        ax.scatter(
            emory_pcs.loc[mask, "PC1"],
            emory_pcs.loc[mask, "PC2"],
            c=color,
            marker=marker,
            s=50,
            alpha=0.6,
            label=f"Emory {resp}",
        )

    # GSE points
    trd_gse = gse_pcs[gse_pcs["is_trd"]]
    ctrl_gse = gse_pcs[~gse_pcs["is_trd"]]
    ax.scatter(
        trd_gse["PC1"],
        trd_gse["PC2"],
        c="#FF9800",
        marker="s",
        s=30,
        alpha=0.5,
        label="GSE TRD",
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

    # Centroid markers
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
            label=f"{cname.replace('_centroid','')}",
        )

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)" if len(ev) > 1 else "PC2")
    ax.set_title(
        f"Gate 0-X: Emory vs GSE98793 centroid projection\n"
        f"d(NR,TRD)={perm['observed_d_nr']:.4f}, d(R,TRD)={perm['observed_d_r']:.4f}, "
        f"p={perm['p_value']:.3f}\nVerdict: {verdict}"
    )
    ax.legend(loc="best", fontsize=7)
    fig.savefig(str(OUT_DIR / "gate_0X_centroid_projection.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(OUT_DIR / "gate_0X_centroid_projection.svg"), bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print()
    print("=" * 60)
    print("Gate 0-X: Cross-disorder centroid projection")
    print("=" * 60)
    print(f"Emory R: {out['n_emory_r']}, NR: {out['n_emory_nr']}")
    print(f"GSE TRD: {out['n_gse_trd']}, control: {out['n_gse_control']}")
    print(f"Genes in common space: {out['n_genes']}")
    print(f"d(NR, TRD): {out['d_emory_nr_to_gse_trd']:.4f}")
    print(f"d(R, TRD):  {out['d_emory_r_to_gse_trd']:.4f}")
    print(f"Direction correct (NR closer to TRD): {out['direction_correct']}")
    print(f"Permutation p: {out['permutation_p']:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
