#!/usr/bin/env python
"""Gate 0-T re-run entry-point: PCA of cell-type-corrected paired-Δ vectors.

Re-runs Gate 0-T (PERMANOVA + Cohen's d on paired-Δ PCA) using cell-type-
corrected paired-delta matrices. Input artefacts:

  - analysis/latest/cell_props_emory.csv (Phase 1 step 1.1 output)
  - analysis/latest/pdata_emory_with_epidish.csv (Phase 1 step 1.1 output)
  - Emory bVals + RNA-seq via dnamrnaseq2026.data.loaders

The Δ-cell-fractions are computed inside this script (POST - PRE per paired
subject) and used to residualise the per-feature paired-Δ matrices.

Outputs (to analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/):
  gate_0T_rerun_results.json        -- PERMANOVA p, Cohen's d, Hotelling T^2, verdict
  gate_0T_rerun_loadings.csv        -- top 50 features by PC1/PC2 loading
  gate_0T_rerun_pca_arrows.png      -- arrow plot coloured by Response
  gate_0T_rerun_pca_arrows.svg      -- same, vector format
  results.md                        -- gate-style markdown summary

Usage:
    python scripts/01_phase0_gate_T_rerun_cellDMC.py
    python scripts/01_phase0_gate_T_rerun_cellDMC.py --n-cpgs 5000 --n-genes 2000
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

OUT_DIR = _REPO_ROOT / "analysis/2026-05-17-phase-0/gate_t_rerun_celldmc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_PROPS_PATH = _REPO_ROOT / "analysis/latest/cell_props_emory.csv"
PDATA_AUG_PATH = _REPO_ROOT / "analysis/latest/pdata_emory_with_epidish.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate 0-T re-run on cell-type-corrected paired-Δ matrices."
    )
    p.add_argument(
        "--n-cpgs", type=int, default=5000, help="Top N CpGs by post-correction variance."
    )
    p.add_argument(
        "--n-genes", type=int, default=2000, help="Top N genes by post-correction variance."
    )
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

    import numpy as np
    import pandas as pd
    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import (
        load_emory_bvals,
        load_emory_rnaseq,
        load_emory_subject_data,
    )
    from dnamrnaseq2026.preprocessing.cell_type_correction import beta_to_m
    from dnamrnaseq2026.preprocessing.delta_construction import (
        filter_paired_ids,
        identify_paired_subjects,
    )
    from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
        build_corrected_paired_delta,
        build_joint_corrected_delta,
        run_gate_t_rerun,
        select_top_variance_features,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]
    logger.info(
        "Gate 0-T re-run starting. Seed=%d, top_cpgs=%d, top_genes=%d",
        seed,
        args.n_cpgs,
        args.n_genes,
    )

    if not CELL_PROPS_PATH.exists() or not PDATA_AUG_PATH.exists():
        logger.error(
            "Phase 1 step 1.1 outputs missing. Required:\n"
            "  - %s\n  - %s\nRun scripts/11_phase1_epidish.py first.",
            CELL_PROPS_PATH,
            PDATA_AUG_PATH,
        )
        sys.exit(2)

    # Load data
    logger.info("Loading Emory data and Phase 1 cell-type artefacts.")
    bvals = load_emory_bvals()
    rnaseq = load_emory_rnaseq()
    subject_data = load_emory_subject_data()
    cell_props_raw = pd.read_csv(CELL_PROPS_PATH, index_col=0)
    pdata_aug = pd.read_csv(PDATA_AUG_PATH, index_col=0)

    # Cell-props remap: cell_props_raw index is SentrixIDs (from run_epidish.R output).
    # RNA-seq and subject pairing use AMC-IDs (Subcode).  Build SentrixID->AMC-ID
    # map from pdata "Subcode" column (canonical source after load_cohort.R rewrite).
    # Fallback: if pdata index already aligns with cell_props, no remap needed.
    if (
        "Subcode" in pdata_aug.columns
        and len(pdata_aug.index.intersection(cell_props_raw.index)) > 0
    ):
        # pdata index and cell_props index are both SentrixIDs.
        # Map: SentrixID -> AMC-ID via Subcode column.
        sentrix_to_amc = pdata_aug["Subcode"].dropna()
        cell_props = cell_props_raw.reindex(sentrix_to_amc.index).set_axis(sentrix_to_amc.values)
        cell_props = cell_props[~cell_props.index.duplicated(keep="first")]
        logger.info(
            "Remapped cell_props from SentrixID -> AMC-ID (via Subcode): %d rows aligned.",
            int(cell_props.notna().any(axis=1).sum()),
        )
    elif (
        "SampleName_DNAm" in pdata_aug.columns
        and len(pdata_aug.index.intersection(cell_props_raw.index)) == 0
    ):
        dnam_map = pdata_aug["SampleName_DNAm"].dropna()
        cell_props = cell_props_raw.reindex(dnam_map.values).set_axis(dnam_map.index)
        logger.info(
            "Remapped cell_props from SentrixID -> AMC-ID (via SampleName_DNAm): %d rows aligned.",
            int(cell_props.notna().any(axis=1).sum()),
        )
    else:
        cell_props = cell_props_raw

    # Paired subject IDs
    paired_info = identify_paired_subjects(subject_data)
    response = paired_info.set_index("Subcode")["Response"]
    logger.info("Paired subjects: %d.", len(paired_info))

    # DNAm pairing (SentrixID indices on bvals.columns).
    # The pData2 from load_cohort.R has column "SampleName" (not "SampleName_DNAm")
    # for the Sentrix barcode IDs that key into bvals.columns.
    paired_subjects, pre_dnam, post_dnam = filter_paired_ids(
        pdata_aug, dnam_sample_col="SampleName"
    )

    # RNA pairing.
    # Step 1: get paired Subcodes via pdata_aug "Subcode" column (bare AMC-IDs).
    #         These key into cell_props (now AMC-ID indexed after our remap).
    # Step 2: build {Subcode}-{Visit} IDs to key into rnaseq.columns
    #         (format: "AMC-280058-PRE-IOP", "AMC-280058-POST-IOP").
    paired_rna, pre_rna_subcode, post_rna_subcode = filter_paired_ids(
        pdata_aug, dnam_sample_col="Subcode"
    )
    pre_rna = [f"{sc}-PRE-IOP" for sc in pre_rna_subcode]
    post_rna = [f"{sc}-POST-IOP" for sc in post_rna_subcode]
    rna_col_set = set(rnaseq.columns)

    # Δ-cell-fractions (POST - PRE) keyed by paired subject ID, computed from
    # AMC-ID indexed cell_props. Also require both RNA sample IDs to exist in
    # rnaseq.columns (not all pData2-paired subjects have RNA-seq coverage).
    valid_pairs = [
        (sc, p, q, pr, po)
        for sc, p, q, pr, po in zip(
            paired_rna, pre_rna_subcode, post_rna_subcode, pre_rna, post_rna, strict=False
        )
        if p in cell_props.index
        and q in cell_props.index
        and pr in rna_col_set
        and po in rna_col_set
    ]
    if not valid_pairs:
        logger.error("No paired subjects with cell_props entries; cannot residualise.")
        sys.exit(3)
    # Unpack: sc=Subcode, pre/post_cp=AMC-ID for cell_props, pre/post_rna={AMC}-{Visit} for rnaseq
    sc_rna_list = [v[0] for v in valid_pairs]
    pre_cp_list = [v[1] for v in valid_pairs]
    post_cp_list = [v[2] for v in valid_pairs]
    pre_rna_ids = [v[3] for v in valid_pairs]
    post_rna_ids = [v[4] for v in valid_pairs]
    sc_rna = sc_rna_list
    delta_cell_props = cell_props.loc[post_cp_list].values - cell_props.loc[pre_cp_list].values
    delta_cell_props_df = pd.DataFrame(delta_cell_props, index=sc_rna, columns=cell_props.columns)
    logger.info(
        "Δ-cell-fractions computed for %d paired subjects (RNA-aligned).",
        len(sc_rna),
    )

    # The DNAm pairing produced paired_subjects in SubcodeXXX order; restrict to
    # the intersection with sc_rna (RNA-pairable subjects with cell-props rows).
    common_subjects = [s for s in paired_subjects if s in set(sc_rna)]
    sc_idx = {s: i for i, s in enumerate(paired_subjects)}
    sc_rna_idx = {s: i for i, s in enumerate(sc_rna)}
    dnam_pre_aligned = [pre_dnam[sc_idx[s]] for s in common_subjects]
    dnam_post_aligned = [post_dnam[sc_idx[s]] for s in common_subjects]
    # Use {Subcode}-{Visit} IDs to key into rnaseq.columns
    rna_pre_aligned = [pre_rna_ids[sc_rna_idx[s]] for s in common_subjects]
    rna_post_aligned = [post_rna_ids[sc_rna_idx[s]] for s in common_subjects]
    delta_cell_aligned = delta_cell_props_df.loc[common_subjects]
    logger.info(
        "Subjects with both DNAm + RNA + Δ-cell-fractions: %d.",
        len(common_subjects),
    )

    # Compute M-values
    cpg_ids = list(bvals.index)
    m_matrix = beta_to_m(bvals.values.astype(np.float64))
    logger.info("M-values computed: %s.", m_matrix.shape)

    # Residualised paired-Δ DNAm
    corrected_dnam_delta = build_corrected_paired_delta(
        feature_matrix=m_matrix,
        feature_ids=cpg_ids,
        sample_ids_pre=dnam_pre_aligned,
        sample_ids_post=dnam_post_aligned,
        all_sample_ids=list(bvals.columns),
        delta_cell_props=delta_cell_aligned,
        paired_subject_ids=common_subjects,
    )
    logger.info("Corrected DNAm Δ matrix: %s.", corrected_dnam_delta.shape)

    # Residualised paired-Δ RNA-seq
    rna_matrix = rnaseq.values.astype(np.float64)
    rna_sample_ids = list(rnaseq.columns)
    corrected_rna_delta = build_corrected_paired_delta(
        feature_matrix=rna_matrix,
        feature_ids=list(rnaseq.index),
        sample_ids_pre=rna_pre_aligned,
        sample_ids_post=rna_post_aligned,
        all_sample_ids=rna_sample_ids,
        delta_cell_props=delta_cell_aligned,
        paired_subject_ids=common_subjects,
    )
    logger.info("Corrected RNA-seq Δ matrix: %s.", corrected_rna_delta.shape)

    # Variance filter: take top-N features by post-correction variance
    corrected_dnam_top = select_top_variance_features(corrected_dnam_delta, args.n_cpgs)
    corrected_rna_top = select_top_variance_features(corrected_rna_delta, args.n_genes)
    logger.info(
        "Variance-filtered: DNAm=%s, RNA=%s.",
        corrected_dnam_top.shape,
        corrected_rna_top.shape,
    )

    # Joint scaled matrix
    joint = build_joint_corrected_delta(corrected_dnam_top, corrected_rna_top, scale=True)
    logger.info("Joint corrected Δ matrix: %s.", joint.shape)

    # Run gate machinery
    result = run_gate_t_rerun(
        joint_corrected_delta=joint,
        response=response,
        n_permutations=args.n_perm,
        seed=seed,
    )

    pc_scores = result["pc_scores"]
    pca = result["pca"]
    permanova = result["permanova"]
    cohens_d = result["cohens_d_per_pc"]
    hotelling = result["hotelling"]
    verdict = result["verdict"]

    # Results JSON
    results = {
        "gate": "0-T-rerun-cellDMC",
        "verdict": verdict,
        "n_paired_subjects": int(result["n_subjects"]),
        "n_responders": int(result["n_r"]),
        "n_non_responders": int(result["n_nr"]),
        "n_features_total": int(result["n_features"]),
        "n_cpgs_used": int(corrected_dnam_top.shape[1]),
        "n_genes_used": int(corrected_rna_top.shape[1]),
        "explained_variance_ratio": result["explained_variance_ratio"],
        "permanova_f": float(permanova["f_statistic"]),
        "permanova_p": float(permanova["p_value"]),
        "permanova_n_perm": int(permanova["n_permutations"]),
        "cohens_d_per_pc": cohens_d,
        "max_cohens_d": float(max(cohens_d.values())) if cohens_d else 0.0,
        "hotelling_p": hotelling["hotelling_p"],
        "per_pc_t_p": hotelling["per_pc_t_p"],
        "seed": seed,
        "input_cell_props_csv": str(CELL_PROPS_PATH.relative_to(_REPO_ROOT)),
        "input_pdata_csv": str(PDATA_AUG_PATH.relative_to(_REPO_ROOT)),
        "rerun_reference_raw_gate": "analysis/2026-05-17-phase-0/0-T/gate_0T_results.json",
    }
    results_path = OUT_DIR / "gate_0T_rerun_results.json"
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results written to %s.", results_path)

    # Loadings CSV
    loadings_rows: list[dict[str, object]] = []
    for i, pc_name in enumerate(pc_scores.columns):
        top50 = (-np.abs(pca.components_[i])).argsort()[:50]
        for rank, j in enumerate(top50):
            loadings_rows.append(
                {
                    "PC": pc_name,
                    "rank": rank + 1,
                    "feature": joint.columns[j],
                    "loading": float(pca.components_[i][j]),
                }
            )
    loadings_df = pd.DataFrame(loadings_rows)
    loadings_path = OUT_DIR / "gate_0T_rerun_loadings.csv"
    loadings_df.to_csv(str(loadings_path), index=False)
    logger.info("Loadings written to %s.", loadings_path)

    # Arrow plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    response_aligned = response.reindex(pc_scores.index)
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"R": "#2196F3", "NR": "#F44336"}
    markers = {"R": "o", "NR": "^"}
    for subcode in pc_scores.index:
        resp = response_aligned.get(subcode, "Unknown")
        color = colors.get(resp, "#999999")
        marker = markers.get(resp, "s")
        ax.annotate(
            "",
            xy=(
                float(pc_scores.loc[subcode, "PC1"]),
                float(pc_scores.loc[subcode, "PC2"]),
            ),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": color, "alpha": 0.6, "lw": 1.0},
        )
        ax.scatter(
            float(pc_scores.loc[subcode, "PC1"]),
            float(pc_scores.loc[subcode, "PC2"]),
            c=color,
            marker=marker,
            s=40,
            zorder=3,
            alpha=0.8,
        )
    for resp_label, color in colors.items():
        mask = response_aligned == resp_label
        if int(mask.sum()) > 0:
            cx = float(pc_scores.loc[mask, "PC1"].mean())
            cy = float(pc_scores.loc[mask, "PC2"].mean())
            ax.scatter(
                cx,
                cy,
                c=color,
                marker="*",
                s=300,
                zorder=5,
                edgecolors="black",
                linewidth=1.0,
                label=f"{resp_label} centroid",
            )
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=8, label="R"),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor="#F44336", markersize=8, label="NR"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% var)" if len(ev) > 1 else "PC2")
    ax.set_title(
        "Gate 0-T re-run (CellDMC-corrected Δ)\n"
        f"PERMANOVA p={permanova['p_value']:.3f}, max Cohen's d="
        f"{max(cohens_d.values()):.3f}\n"
        f"Verdict: {verdict}"
    )
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    fig_png = OUT_DIR / "gate_0T_rerun_pca_arrows.png"
    fig_svg = OUT_DIR / "gate_0T_rerun_pca_arrows.svg"
    fig.savefig(str(fig_png), dpi=150, bbox_inches="tight")
    fig.savefig(str(fig_svg), bbox_inches="tight")
    plt.close(fig)
    logger.info("Figures saved to %s.", OUT_DIR)

    # results.md: fill the template now that real numbers are in hand.
    md_lines = [
        "# Gate 0-T re-run: cell-type-corrected Δ matrices",
        "",
        "**Date:** filled at run time",
        f"**Verdict:** {verdict}",
        "**Reviewer:** Kai (primary, CellDMC plumbing); "
        "Tobias (secondary, permutation invariance).",
        "**Comparison reference:** "
        "[Gate 0-T raw verdict](../0-T/results.md) (MARGINAL, p=0.111, max d=0.267).",
        "",
        "## Inputs",
        "",
        "- ``analysis/latest/cell_props_emory.csv`` (Phase 1 step 1.1 output)",
        "- ``analysis/latest/pdata_emory_with_epidish.csv`` (Phase 1 step 1.1 output)",
        "- Emory bVals + RNA-seq via ``dnamrnaseq2026.data.loaders``",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Threshold |",
        "|---|---|---|",
        f"| PERMANOVA p | {permanova['p_value']:.4f} | < 0.05 PASS |",
        f"| PERMANOVA F | {permanova['f_statistic']:.4f} | |",
        f"| Max Cohen's d (across PCs) | {max(cohens_d.values()):.4f} | >= 0.30 PASS |",
        f"| Hotelling T^2 p | "
        f"{hotelling['hotelling_p'] if hotelling['hotelling_p'] is not None else 'N/A'} | |",
        f"| n paired subjects (with R/NR + cell-props) | {result['n_subjects']} "
        f"(R={result['n_r']}, NR={result['n_nr']}) | |",
        f"| n features (CpGs + genes) | {result['n_features']} | |",
        "",
        "## Per-PC breakdown",
        "",
        "| PC | Explained variance | Cohen's d | t-test p |",
        "|---|---|---|---|",
    ]
    for i, pc in enumerate(pc_scores.columns):
        ev_pc = (
            f"{result['explained_variance_ratio'][i] * 100:.2f}%"
            if i < len(result["explained_variance_ratio"])
            else "N/A"
        )
        md_lines.append(
            f"| {pc} | {ev_pc} | {cohens_d.get(pc, float('nan')):.3f} | "
            f"{hotelling['per_pc_t_p'].get(pc, float('nan')):.4f} |"
        )

    md_lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Comparison to raw-Δ Gate 0-T",
        "",
        "| Run | PERMANOVA p | Max Cohen's d | Verdict |",
        "|---|---|---|---|",
        "| Raw-Δ (2026-05-17) | 0.111 | 0.267 | MARGINAL |",
        f"| CellDMC-corrected Δ (this run) | {permanova['p_value']:.3f} | "
        f"{max(cohens_d.values()):.3f} | {verdict} |",
        "",
        "## Concepts discussed",
        "",
        "- [[celldmc]]",
        "- [[paired-design]]",
        "- [[trajectory-atlas]]",
        "- [[gate-zero-t]]",
    ]
    md_path = OUT_DIR / "results.md"
    md_path.write_text("\n".join(md_lines))
    logger.info("results.md updated at %s.", md_path)

    # Console summary
    print()
    print("=" * 60)
    print("Gate 0-T re-run: CellDMC-corrected paired-Δ PCA")
    print("=" * 60)
    print(f"Paired subjects: {result['n_subjects']} (R={result['n_r']}, NR={result['n_nr']})")
    print(
        f"Features: {result['n_features']} ({corrected_dnam_top.shape[1]} CpGs + "
        f"{corrected_rna_top.shape[1]} genes)"
    )
    print(f"PERMANOVA: F={permanova['f_statistic']:.4f}, p={permanova['p_value']:.4f}")
    print(f"Max Cohen's d (across PCs): {max(cohens_d.values()):.4f}")
    if hotelling["hotelling_p"] is not None:
        print(f"Hotelling T^2 p: {hotelling['hotelling_p']:.4f}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
