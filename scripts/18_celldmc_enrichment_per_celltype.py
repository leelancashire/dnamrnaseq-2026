"""Step 1.8: Per-cell-type biological coherence enrichment of the 555 CellDMC delta hits.

Deliverable: leaderboard metric (v) -- biological coherence of the CellDMC delta layer.

Runs GO/pathway enrichment per cell type for the 555 Emory CellDMC delta FDR<0.05 CpGs.
CpG -> gene mapping via EPIC v2 annotation (annEPIC_filt3.RData, UCSC_RefGene_Name).
Enrichment via gseapy Enrichr API (GO Biological Process, KEGG, Reactome).

Outputs:
  - analysis/2026-05-17-phase-1/1.8/enrichment_per_celltype.tsv
  - analysis/2026-05-17-phase-1/1.8/cpg_gene_map.csv
  - analysis/2026-05-17-phase-1/1.8/results.md
  - analysis/latest/celldmc_enrichment_per_celltype.tsv (symlink copy)

Analysis plan reference: ANALYSIS_PLAN.md leaderboard metric (v).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.8")
LATEST_DIR = Path("analysis/latest")

# EPIC annotation path (OneDrive mirror)
ANN_EPIC_PATH = "/mnt/d/lee/onedrive/work/nicol healthtech/cvb/Emory-DNAm/annEPIC_filt3.RData"

# Enrichr gene-set libraries to query
ENRICHR_LIBRARIES = [
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

# Hypothesis-relevant keywords for coherence verdict
HYPOTHESIS_KEYWORDS = {
    "monocyte": ["monocyte", "mononuclear", "macrophage", "myeloid"],
    "inflammatory": [
        "inflammatory",
        "inflammation",
        "cytokine",
        "interleukin",
        "nfkb",
        "nf-kb",
        "tnf",
        "interferon",
        "immune",
        "innate immunity",
    ],
    "wnt": ["wnt", "wingless", "frizzled", "beta-catenin"],
    "cd8t": ["t cell", "cytotoxic", "adaptive immunity", "lymphocyte", "cd8"],
    "stress_trauma": ["stress", "glucocorticoid", "cortisol", "hpa axis", "ptsd"],
    "epigenetic": ["methylation", "chromatin", "histone", "epigenetic"],
}


def load_cpg_gene_map(ann_path: str) -> pd.DataFrame:
    """Load CpG -> gene mapping from EPIC annotation RData via Rscript subprocess."""
    import subprocess

    r_script = f"""
options(warn=-1)
suppressMessages(library(S4Vectors))
load("{ann_path}")
df <- data.frame(
  cpg  = as.character(annEPIC_filt3$CpG_name),
  ucsc_gene = as.character(annEPIC_filt3$UCSC_RefGene_Name),
  gencode_gene = as.character(annEPIC_filt3$GencodeV41_Name),
  stringsAsFactors = FALSE
)
write.csv(df, "/tmp/cpg_gene_map_tmp.csv", row.names=FALSE)
cat("nrow:", nrow(df), "\\n")
"""
    r_bin = None
    for candidate in [
        "/home/llanc/dnamrnaseq-2026/.snakemake/conda/afb2b31adda110801b44795aad175ffa_/bin/Rscript",
        "Rscript",
    ]:
        import shutil

        if shutil.which(candidate) or Path(candidate).exists():
            r_bin = candidate
            break

    if r_bin is None:
        raise RuntimeError("Rscript not found; cannot load EPIC annotation.")

    result = subprocess.run(
        [r_bin, "--vanilla", "-e", r_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("R stderr: %s", result.stderr[:500])
        raise RuntimeError("R script failed while loading EPIC annotation.")

    logger.info("R output: %s", result.stdout.strip())
    cpg_map = pd.read_csv("/tmp/cpg_gene_map_tmp.csv")
    logger.info("Loaded EPIC annotation: %d CpGs.", len(cpg_map))
    return cpg_map


def cpg_list_to_genes(cpg_ids: list[str], cpg_map: pd.DataFrame) -> list[str]:
    """Map a list of CpG IDs to unique gene symbols.

    UCSC_RefGene_Name may contain semicolon-separated gene names (e.g. 'GENE1;GENE1;GENE2').
    Falls back to GencodeV41_Name if UCSC is empty.
    """
    subset = cpg_map[cpg_map["cpg"].isin(cpg_ids)].copy()
    genes: set[str] = set()
    for _, row in subset.iterrows():
        raw = str(row["ucsc_gene"]).strip()
        if raw in ("", "NA", "nan", "None"):
            raw = str(row["gencode_gene"]).strip()
        if raw in ("", "NA", "nan", "None"):
            continue
        # split on semicolons and deduplicate
        for g in raw.split(";"):
            g = g.strip()
            if g:
                genes.add(g)
    return sorted(genes)


def run_enrichr(gene_list: list[str], library: str, max_retries: int = 3) -> pd.DataFrame:
    """Query Enrichr API and return results DataFrame.

    Returns empty DataFrame on failure.
    """
    import requests

    if not gene_list:
        return pd.DataFrame()

    base_url = "https://maayanlab.cloud/Enrichr"
    # Step 1: add gene list
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url}/addList",
                files={"list": (None, "\n".join(gene_list)), "description": (None, library)},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("Enrichr addList HTTP %s (attempt %d)", resp.status_code, attempt)
                time.sleep(2**attempt)
                continue
            user_list_id = resp.json().get("userListId")
            break
        except Exception as exc:
            logger.warning("Enrichr addList error (attempt %d): %s", attempt, exc)
            time.sleep(2**attempt)
    else:
        return pd.DataFrame()

    # Step 2: get enrichment results
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                f"{base_url}/enrich",
                params={"userListId": user_list_id, "backgroundType": library},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning("Enrichr enrich HTTP %s (attempt %d)", resp.status_code, attempt)
                time.sleep(2**attempt)
                continue
            data = resp.json().get(library, [])
            break
        except Exception as exc:
            logger.warning("Enrichr enrich error (attempt %d): %s", attempt, exc)
            time.sleep(2**attempt)
    else:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    # Enrichr returns: rank, term, p, z, combined_score, genes, adj_p, old_p, old_adj_p
    rows = []
    for entry in data:
        if len(entry) >= 7:
            rows.append(
                {
                    "rank": entry[0],
                    "term": entry[1],
                    "p_value": entry[2],
                    "z_score": entry[3],
                    "combined_score": entry[4],
                    "genes": ";".join(entry[5]) if isinstance(entry[5], list) else str(entry[5]),
                    "adj_p": entry[6],
                }
            )
    return pd.DataFrame(rows)


def check_hypothesis_coherence(enrichment_df: pd.DataFrame) -> dict[str, object]:
    """Score enrichment results for hypothesis-relevant biology.

    Returns a dict with keyword hits and a coherent/not verdict.
    """
    if enrichment_df.empty or "term" not in enrichment_df.columns:
        return {"coherent": False, "keyword_hits": {}, "top_sig_terms": []}

    sig = enrichment_df[enrichment_df["adj_p"].fillna(1.0) < 0.05]
    all_terms_lower = sig["term"].str.lower().tolist() if not sig.empty else []

    keyword_hits: dict[str, list[str]] = {}
    for category, kws in HYPOTHESIS_KEYWORDS.items():
        matched = [t for t in all_terms_lower if any(kw in t for kw in kws)]
        if matched:
            keyword_hits[category] = matched[:3]

    coherent = len(keyword_hits) >= 2 or any(
        k in keyword_hits for k in ["monocyte", "inflammatory", "wnt"]
    )

    top_sig = sig.sort_values("adj_p").head(10)["term"].tolist() if not sig.empty else []

    return {
        "coherent": coherent,
        "keyword_hits": keyword_hits,
        "top_sig_terms": top_sig,
        "n_sig_terms": len(sig),
        "n_total_terms": len(enrichment_df),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load CellDMC delta hits
    celldmc_path = LATEST_DIR / "celldmc_delta_emory.tsv"
    celldmc_delta = pd.read_csv(celldmc_path, sep="\t")
    sig = celldmc_delta[celldmc_delta["sig"]].copy()
    logger.info("CellDMC delta sig hits: %d across cell types.", len(sig))

    cell_type_counts = sig["cell_type"].value_counts()
    logger.info("Per cell type: %s", cell_type_counts.to_dict())

    # Load EPIC CpG -> gene annotation
    logger.info("Loading EPIC annotation from RData.")
    cpg_map = load_cpg_gene_map(ANN_EPIC_PATH)
    cpg_map.to_csv(OUT_DIR / "cpg_gene_map.csv", index=False)
    logger.info("Saved cpg_gene_map.csv: %d rows.", len(cpg_map))

    # Per-cell-type enrichment
    all_enrichment_rows: list[dict] = []
    coherence_summary: dict[str, dict] = {}

    cell_types_to_test = sig["cell_type"].unique().tolist()

    for ct in cell_types_to_test:
        ct_cpgs = sig[sig["cell_type"] == ct]["cpg"].tolist()
        ct_genes = cpg_list_to_genes(ct_cpgs, cpg_map)
        logger.info("Cell type %s: %d CpGs -> %d unique genes.", ct, len(ct_cpgs), len(ct_genes))

        if len(ct_genes) < 5:
            logger.warning("Too few genes (%d) for %s; skipping enrichment.", len(ct_genes), ct)
            coherence_summary[ct] = {
                "n_cpgs": len(ct_cpgs),
                "n_genes": len(ct_genes),
                "coherent": False,
                "reason": "too_few_genes",
                "keyword_hits": {},
                "top_sig_terms": [],
                "n_sig_terms": 0,
            }
            continue

        ct_enrich_rows: list[dict] = []
        for lib in ENRICHR_LIBRARIES:
            logger.info("  Running Enrichr %s for %s...", lib, ct)
            enr = run_enrichr(ct_genes, lib)
            if not enr.empty:
                enr["cell_type"] = ct
                enr["library"] = lib
                enr["n_input_genes"] = len(ct_genes)
                ct_enrich_rows.append(enr)
                logger.info(
                    "  %s/%s: %d terms, %d sig (adj_p<0.05).",
                    ct,
                    lib,
                    len(enr),
                    int((enr["adj_p"].fillna(1.0) < 0.05).sum()),
                )
            else:
                logger.warning("  No results from Enrichr %s for %s.", lib, ct)
            time.sleep(0.5)  # rate-limit courtesy

        if ct_enrich_rows:
            ct_df = pd.concat(ct_enrich_rows, ignore_index=True)
            all_enrichment_rows.append(ct_df)
            coherence = check_hypothesis_coherence(ct_df)
            coherence_summary[ct] = {
                "n_cpgs": len(ct_cpgs),
                "n_genes": len(ct_genes),
                **coherence,
            }
        else:
            coherence_summary[ct] = {
                "n_cpgs": len(ct_cpgs),
                "n_genes": len(ct_genes),
                "coherent": False,
                "reason": "no_enrichr_results",
                "keyword_hits": {},
                "top_sig_terms": [],
                "n_sig_terms": 0,
            }

    # Save combined enrichment table
    if all_enrichment_rows:
        combined = pd.concat(all_enrichment_rows, ignore_index=True)
        combined.to_csv(OUT_DIR / "enrichment_per_celltype.tsv", sep="\t", index=False)
        combined.to_csv(LATEST_DIR / "celldmc_enrichment_per_celltype.tsv", sep="\t", index=False)
        logger.info("Saved enrichment table: %d rows.", len(combined))
    else:
        combined = pd.DataFrame()
        logger.warning("No enrichment results to save.")

    _write_results_md(coherence_summary, combined)
    logger.info("Step 1.8 complete.")


def _write_results_md(
    coherence_summary: dict[str, dict],
    combined: pd.DataFrame,
) -> None:
    lines = [
        "# Step 1.8: Per-Cell-Type Biological Coherence Enrichment of CellDMC Delta Hits",
        "",
        "**Date:** 2026-05-22",
        "**Analyst:** Lee Lancashire",
        "",
        "## Overview",
        "",
        "Enrichment of the 555 Emory CellDMC delta FDR<0.05 CpGs.",
        "CpG->gene mapping: EPIC v2 annotation (UCSC_RefGene_Name / GencodeV41_Name).",
        "Enrichment: Enrichr API (GO_BP_2023, KEGG_2021, Reactome_2022, MSigDB_Hallmark_2020).",
        "",
        "## Per-cell-type summary",
        "",
        "| Cell type | N CpGs | N genes | N sig terms | Coherent | Top keyword hits |",
        "|-----------|--------|---------|-------------|----------|-----------------|",
    ]

    for ct, info in sorted(coherence_summary.items()):
        n_cpgs = info.get("n_cpgs", 0)
        n_genes = info.get("n_genes", 0)
        n_sig = info.get("n_sig_terms", 0)
        coherent = "YES" if info.get("coherent") else "NO"
        kw_hits = info.get("keyword_hits", {})
        kw_str = ", ".join(list(kw_hits.keys())[:3]) if kw_hits else "none"
        lines.append(f"| {ct} | {n_cpgs} | {n_genes} | {n_sig} | {coherent} | {kw_str} |")

    lines += [
        "",
        "## Top enriched terms per cell type (adj_p < 0.05)",
        "",
    ]

    for ct, info in sorted(coherence_summary.items()):
        top_terms = info.get("top_sig_terms", [])
        lines.append(f"### {ct}")
        if top_terms:
            for t in top_terms[:5]:
                lines.append(f"- {t}")
        else:
            lines.append("- No significant terms (adj_p < 0.05)")
        lines.append("")

    lines += [
        "## Biological coherence verdict",
        "",
    ]

    n_coherent = sum(1 for v in coherence_summary.values() if v.get("coherent"))
    n_total = len(coherence_summary)

    if n_coherent >= 2:
        verdict = (
            "COHERENT: >= 2 cell types show enrichment for hypothesis-relevant biology "
            "(monocyte/inflammatory/WNT/stress programmes). CellDMC layer is biologically "
            "plausible."
        )
    elif n_coherent == 1:
        verdict = (
            "MARGINAL: 1 cell type shows hypothesis-relevant enrichment. "
            "CellDMC layer has limited biological coherence."
        )
    else:
        verdict = (
            "NOT COHERENT: No cell types show enrichment for hypothesis-relevant biology. "
            "Pattern is consistent with FDR noise."
        )

    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append(f"Cell types coherent: {n_coherent} / {n_total}")

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
