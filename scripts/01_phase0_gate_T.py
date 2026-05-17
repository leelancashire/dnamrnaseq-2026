#!/usr/bin/env python
"""Gate 0-T entry-point: PCA of paired delta-vectors with permutation test.

Reads config.yaml, loads Emory DNAm bVals + RNA-seq, builds joint delta-feature
matrix, runs PCA, computes PERMANOVA and Cohen's d, writes results.

Outputs (to analysis/2026-05-17-phase-0/0-T/):
  gate_0T_results.json      -- PERMANOVA p, Cohen's d, Hotelling T^2, verdict
  gate_0T_loadings.csv      -- top 50 features by PC1/PC2 loading
  gate_0T_pca_arrows.png    -- arrow plot coloured by Response
  gate_0T_pca_arrows.svg    -- same, vector format

Usage:
    python scripts/01_phase0_gate_T.py
    python scripts/01_phase0_gate_T.py --n-cpgs 5000 --n-genes 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

OUT_DIR = _REPO_ROOT / "analysis/2026-05-17-phase-0/0-T"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0-T: PCA of Emory paired delta-vectors.")
    p.add_argument("--n-cpgs", type=int, default=5000, help="Top N CpGs by delta variance.")
    p.add_argument("--n-genes", type=int, default=2000, help="Top N genes by delta variance.")
    p.add_argument("--n-perm", type=int, default=2000, help="PERMANOVA permutations.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import (
        load_emory_bvals,
        load_emory_rnaseq,
        load_emory_subject_data,
    )
    from dnamrnaseq2026.preprocessing.delta_construction import (
        build_dnam_delta_matrix,
        build_joint_delta_matrix,
        build_rnaseq_delta_matrix,
        identify_paired_subjects,
    )
    from dnamrnaseq2026.preprocessing.gate_t_pca import (
        compute_cohens_d_per_pc,
        determine_gate_0t_verdict,
        run_hotelling_t2,
        run_pca,
        run_permanova,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]
    logger.info(
        "Gate 0-T starting. Seed=%d, top_cpgs=%d, top_genes=%d",
        seed, args.n_cpgs, args.n_genes,
    )

    # Load data
    logger.info("Loading Emory data...")
    bvals = load_emory_bvals()
    rnaseq = load_emory_rnaseq()
    subject_data = load_emory_subject_data()

    # Identify paired subjects
    paired_info = identify_paired_subjects(subject_data)
    response = paired_info.set_index("Subcode")["Response"]

    # Build delta matrices
    dnam_delta = build_dnam_delta_matrix(
        bvals, subject_data, top_n_cpgs=args.n_cpgs
    )
    rna_delta = build_rnaseq_delta_matrix(
        rnaseq, subject_data, top_n_genes=args.n_genes
    )
    joint = build_joint_delta_matrix(dnam_delta, rna_delta, scale=True)
    logger.info("Joint delta matrix: %s", joint.shape)

    # PCA
    pc_scores, pca = run_pca(joint, n_components=5)

    # Align response to pc_scores index
    response_aligned = response.reindex(pc_scores.index)

    # PERMANOVA
    permanova = run_permanova(pc_scores, response_aligned, n_permutations=args.n_perm, seed=seed)

    # Cohen's d
    cohens_d = compute_cohens_d_per_pc(pc_scores, response_aligned)

    # Hotelling T^2
    hotelling = run_hotelling_t2(pc_scores, response_aligned)

    # Verdict
    verdict = determine_gate_0t_verdict(permanova, cohens_d)
    logger.info("Gate 0-T verdict: %s", verdict)

    # Write results JSON
    results = {
        "gate": "0-T",
        "verdict": verdict,
        "n_paired_subjects": int(len(pc_scores)),
        "n_responders": int((response_aligned == "R").sum()),
        "n_non_responders": int((response_aligned == "NR").sum()),
        "n_features_total": int(joint.shape[1]),
        "n_cpgs_used": int(dnam_delta.shape[1]),
        "n_genes_used": int(rna_delta.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "permanova_f": float(permanova["f_statistic"]),
        "permanova_p": float(permanova["p_value"]),
        "permanova_n_perm": int(permanova["n_permutations"]),
        "cohens_d_per_pc": cohens_d,
        "max_cohens_d": float(max(cohens_d.values())) if cohens_d else 0.0,
        "hotelling_p": hotelling["hotelling_p"],
        "per_pc_t_p": hotelling["per_pc_t_p"],
        "seed": seed,
    }
    results_path = OUT_DIR / "gate_0T_results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results written to %s", results_path)

    # Loadings CSV: top 50 features by absolute loading on PC1 + PC2
    loadings = {}
    for i, pc_name in enumerate(pc_scores.columns):
        top50 = (
            abs(pca.components_[i])
            .argsort()[::-1][:50]
        )
        loadings[pc_name] = [(joint.columns[j], float(pca.components_[i][j])) for j in top50]

    import pandas as pd
    rows = []
    for pc_name, feats in loadings.items():
        for rank, (feat, loading) in enumerate(feats):
            rows.append({"PC": pc_name, "rank": rank + 1, "feature": feat, "loading": loading})
    loadings_df = pd.DataFrame(rows)
    loadings_path = OUT_DIR / "gate_0T_loadings.csv"
    loadings_df.to_csv(str(loadings_path), index=False)
    logger.info("Loadings written to %s", loadings_path)

    # Arrow plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"R": "#2196F3", "NR": "#F44336"}
    markers = {"R": "o", "NR": "^"}

    for subcode in pc_scores.index:
        resp = response_aligned.get(subcode, "Unknown")
        color = colors.get(resp, "#999999")
        marker = markers.get(resp, "s")
        ax.annotate(
            "",
            xy=(float(pc_scores.loc[subcode, "PC1"]), float(pc_scores.loc[subcode, "PC2"])),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.6, "lw": 1.0},
        )
        ax.scatter(
            float(pc_scores.loc[subcode, "PC1"]),
            float(pc_scores.loc[subcode, "PC2"]),
            c=color, marker=marker, s=40, zorder=3, alpha=0.8,
        )

    # Centroid markers
    for resp_label, color in colors.items():
        mask = response_aligned == resp_label
        if mask.sum() > 0:
            cx = float(pc_scores.loc[mask, "PC1"].mean())
            cy = float(pc_scores.loc[mask, "PC2"].mean())
            ax.scatter(cx, cy, c=color, marker="*", s=300, zorder=5,
                       edgecolors="black", linewidth=1.0, label=f"{resp_label} centroid")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
               markersize=8, label="R"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#F44336",
               markersize=8, label="NR"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)" if len(ev) > 1 else "PC2")
    ax.set_title(
        f"Gate 0-T: Emory Δ-vector PCA\n"
        f"PERMANOVA p={permanova['p_value']:.3f}, max Cohen's d={max(cohens_d.values()):.3f}\n"
        f"Verdict: {verdict}"
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")

    fig_path_png = OUT_DIR / "gate_0T_pca_arrows.png"
    fig_path_svg = OUT_DIR / "gate_0T_pca_arrows.svg"
    fig.savefig(str(fig_path_png), dpi=150, bbox_inches="tight")
    fig.savefig(str(fig_path_svg), bbox_inches="tight")
    plt.close(fig)
    logger.info("Figures saved to %s", OUT_DIR)

    # Print summary
    print()
    print("=" * 60)
    print("Gate 0-T: Trajectory-structure visibility test")
    print("=" * 60)
    print(f"Paired subjects: {results['n_paired_subjects']} "
          f"(R={results['n_responders']}, NR={results['n_non_responders']})")
    print(f"Features: {results['n_features_total']} "
          f"({results['n_cpgs_used']} CpGs + {results['n_genes_used']} genes)")
    print(f"PERMANOVA: F={results['permanova_f']:.4f}, p={results['permanova_p']:.4f}")
    print(f"Max Cohen's d (across PCs): {results['max_cohens_d']:.4f}")
    if results["hotelling_p"] is not None:
        print(f"Hotelling T^2 p: {results['hotelling_p']:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
