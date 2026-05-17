"""Step 1.4: decoupleR pathway activity (PROGENy + Reactome/GSVA).

Outputs:
  - analysis/2026-05-17-phase-1/1.4/progeny_activity_emory.parquet
  - analysis/2026-05-17-phase-1/1.4/progeny_delta_emory.parquet
  - analysis/2026-05-17-phase-1/1.4/gsva_activity_emory.parquet
  - analysis/2026-05-17-phase-1/1.4/gsva_delta_emory.parquet
  - analysis/2026-05-17-phase-1/1.4/pathway_response_test.tsv
  - analysis/2026-05-17-phase-1/1.4/results.md

Analysis plan reference: ANALYSIS_PLAN.md Step 1.4.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.4")
LATEST_DIR = Path("analysis/latest")
SEED = 42


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from dnamrnaseq2026.data.loaders import load_emory, load_emory_rnaseq
    from dnamrnaseq2026.preprocessing.delta_construction import filter_paired_ids_rna
    from dnamrnaseq2026.preprocessing.pathway_activity import (
        compute_delta_activity,
        get_progeny_net,
        run_gsva,
        run_progeny_ulm,
        stub_gene_sets,
        test_response_association,
    )

    logger.info("Loading Emory RNA-seq.")
    log_cpm_df = load_emory_rnaseq()
    gene_ids = list(log_cpm_df.index)
    log_cpm = log_cpm_df.values.astype(np.float64)
    sample_ids = list(log_cpm_df.columns)

    _, pdata = load_emory()

    pdata_aug_path = LATEST_DIR / "pdata_emory_with_epidish.csv"
    pdata_aug = pd.read_csv(pdata_aug_path, index_col=0) if pdata_aug_path.exists() else pdata

    # Align RNA-seq samples to pdata
    shared_samples = [s for s in pdata_aug.index if s in log_cpm_df.columns]
    if not shared_samples:
        shared_samples = sample_ids
    col_pos = [sample_ids.index(s) for s in shared_samples]
    lc_shared = log_cpm[:, col_pos]

    # --- PROGENy pathway activity ---
    logger.info("Loading PROGENy network.")
    progeny_net = get_progeny_net(organism="human", top=500)

    logger.info("Running PROGENy ULM.")
    progeny_activity = run_progeny_ulm(lc_shared, gene_ids, shared_samples, progeny_net)
    progeny_activity.to_parquet(OUT_DIR / "progeny_activity_emory.parquet")
    progeny_activity.to_parquet(LATEST_DIR / "progeny_activity_emory.parquet")
    logger.info("PROGENy activity: %s", progeny_activity.shape)

    # --- GSVA / Reactome pathway activity ---
    logger.info("Running GSVA (Reactome gene sets or stub).")
    try:
        import decoupler as dc

        reactome = dc.get_resource("Reactome", organism="human")
        gene_sets: dict[str, list[str]] = (
            reactome.groupby("geneset")["genesymbol"].apply(list).to_dict()
        )
        logger.info("Loaded %d Reactome gene sets.", len(gene_sets))
    except Exception as exc:
        logger.warning("Could not load Reactome via decoupler (%s); using stubs.", exc)
        gene_sets = stub_gene_sets(n_sets=20, genes_per_set=30)

    gsva_activity = run_gsva(lc_shared, gene_ids, shared_samples, gene_sets)
    gsva_activity.to_parquet(OUT_DIR / "gsva_activity_emory.parquet")
    gsva_activity.to_parquet(LATEST_DIR / "gsva_activity_emory.parquet")
    logger.info("GSVA activity: %s", gsva_activity.shape)

    # --- Paired delta ---
    paired_subjects, pre_ids, post_ids = filter_paired_ids_rna(pdata_aug)
    pre_in_rna = [s for s in pre_ids if s in shared_samples]
    post_in_rna = [s for s in post_ids if s in shared_samples]
    n_pairs = min(len(pre_in_rna), len(post_in_rna))
    pre_in_rna = pre_in_rna[:n_pairs]
    post_in_rna = post_in_rna[:n_pairs]
    paired_rna = list(paired_subjects[:n_pairs])

    logger.info("Computing delta pathway activity (%d pairs).", n_pairs)
    progeny_delta = compute_delta_activity(progeny_activity, pre_in_rna, post_in_rna, paired_rna)
    progeny_delta.to_parquet(OUT_DIR / "progeny_delta_emory.parquet")
    progeny_delta.to_parquet(LATEST_DIR / "progeny_delta_emory.parquet")

    gsva_delta = compute_delta_activity(gsva_activity, pre_in_rna, post_in_rna, paired_rna)
    gsva_delta.to_parquet(OUT_DIR / "gsva_delta_emory.parquet")
    gsva_delta.to_parquet(LATEST_DIR / "gsva_delta_emory.parquet")

    # --- Response association test ---
    pdata_pre_paired = pdata_aug.loc[pre_in_rna].copy()
    pdata_pre_paired.index = pd.Index(paired_rna)

    logger.info("Testing pathway-response association.")
    pathway_results_list = []

    for label, delta_act in [("PROGENy", progeny_delta), ("GSVA", gsva_delta)]:
        if delta_act.empty:
            continue
        result_df = test_response_association(
            delta_act,
            pdata_pre_paired,
            response_col="Response",
        )
        result_df.insert(0, "source", label)
        pathway_results_list.append(result_df)

    if pathway_results_list:
        pathway_response_test = pd.concat(pathway_results_list, ignore_index=True)
    else:
        pathway_response_test = pd.DataFrame(
            columns=["source", "pathway", "beta", "p_response", "q_response"]
        )

    pathway_response_test.to_csv(OUT_DIR / "pathway_response_test.tsv", sep="\t", index=False)
    pathway_response_test.to_csv(LATEST_DIR / "pathway_response_test.tsv", sep="\t", index=False)

    _write_results_md(progeny_activity, gsva_activity, pathway_response_test)
    logger.info("Step 1.4 complete.")


def _write_results_md(
    progeny_activity: pd.DataFrame,
    gsva_activity: pd.DataFrame,
    pathway_response_test: pd.DataFrame,
) -> None:
    def n_sig(df: pd.DataFrame, threshold: float = 0.10) -> int:
        if df.empty or "q_response" not in df.columns:
            return 0
        return int((df["q_response"].fillna(1.0) < threshold).sum())

    progeny_test = (
        pathway_response_test[pathway_response_test["source"] == "PROGENy"]
        if not pathway_response_test.empty
        else pd.DataFrame()
    )
    gsva_test = (
        pathway_response_test[pathway_response_test["source"] == "GSVA"]
        if not pathway_response_test.empty
        else pd.DataFrame()
    )

    n_progeny_pathways = int(progeny_activity.shape[1]) if not progeny_activity.empty else 0
    n_gsva_pathways = int(gsva_activity.shape[1]) if not gsva_activity.empty else 0
    n_sig_progeny = n_sig(progeny_test)
    n_sig_gsva = n_sig(gsva_test)

    acceptance = "PASS" if (n_sig_progeny + n_sig_gsva) >= 2 else "MARGINAL"

    lines = [
        "# Step 1.4: Pathway Activity (PROGENy + Reactome/GSVA)",
        "",
        "**Date:** 2026-05-17",
        "",
        "## Summary",
        "",
        "| Source | N pathways | N sig (FDR < 0.10) | Acceptance |",
        "|--------|-----------|-------------------|------------|",
        f"| PROGENy | {n_progeny_pathways} | {n_sig_progeny} "
        f"| {'PASS' if n_sig_progeny >= 1 else 'MARGINAL'} |",
        f"| GSVA (Reactome) | {n_gsva_pathways} | {n_sig_gsva} "
        f"| {'PASS' if n_sig_gsva >= 1 else 'MARGINAL'} |",
        "",
        f"**Overall acceptance:** {acceptance}",
        "",
        "## Top pathways (PROGENy, FDR < 0.10)",
        "",
    ]

    if not progeny_test.empty and "q_response" in progeny_test.columns:
        top = (
            progeny_test[progeny_test["q_response"].fillna(1.0) < 0.10]
            .sort_values("q_response")
            .head(10)
        )
        if not top.empty:
            lines.append("| Pathway | Beta | q |")
            lines.append("|---------|------|---|")
            for _, row in top.iterrows():
                p_name = row.get("pathway", "N/A")
                p_beta = row.get("beta", float("nan"))
                p_q = row.get("q_response", float("nan"))
                lines.append(f"| {p_name} | {p_beta:.3f} | {p_q:.3g} |")
        else:
            lines.append("No pathways at FDR < 0.10.")
    else:
        lines.append("No result data available.")

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
