"""Step 1.9: BEST CellDMC delta run + replication of 555 Emory hits.

Deliverable: BEST replication of the CellDMC delta layer (decisive test).

Runs CellDMC delta on the BEST cohort (n=50 paired), then tests whether
the 555 Emory FDR<0.05 hits replicate: sign concordance, effect-size
correlation (Spearman), pi1 / replication rate on the 555-hit set.

Outputs:
  - analysis/2026-05-17-phase-1/1.9/celldmc_delta_best.tsv
  - analysis/2026-05-17-phase-1/1.9/replication_555hits.tsv
  - analysis/2026-05-17-phase-1/1.9/concordance_stats.json
  - analysis/2026-05-17-phase-1/1.9/results.md
  - analysis/latest/celldmc_delta_best.tsv
  - analysis/latest/replication_555hits.tsv

Analysis plan reference: TASK-5090-2026-05-22-005.md deliverable 2.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

OUT_DIR = Path("analysis/2026-05-17-phase-1/1.9")
LATEST_DIR = Path("analysis/latest")
SEED = 42

# BEST response encoding: group col (R_BL/NR_BL)
RESPONSE_COL = "group"


def _beta_to_m(
    beta: np.ndarray,
    clip_lo: float = 1e-6,
    clip_hi: float = 8.0,
) -> np.ndarray:
    b = np.clip(beta, clip_lo, 1.0 - clip_lo)
    return np.clip(np.log2(b / (1.0 - b)), -clip_hi, clip_hi)


def build_best_paired(
    pdata: pd.DataFrame,
    subcode_col: str = "Subcode",
    visit_col: str = "Visit",
    pre_label: str = "BL",
    post_label: str = "12W",
    response_col: str = "group",
) -> tuple[list[str], list[str], list[str], pd.DataFrame]:
    """Build BEST paired subject IDs and paired pdata.

    Returns:
        subjects: list of Subcode values with paired BL+12W visits AND R/NR response.
        pre_ids: SampleName for BL visit, aligned to subjects.
        post_ids: SampleName for 12W visit, aligned to subjects.
        pdata_paired: pdata rows for BL visit (one row per subject).
    """
    pre_mask = pdata[visit_col].astype(str).str.upper() == pre_label.upper()
    post_mask = pdata[visit_col].astype(str).str.upper() == post_label.upper()

    pre_map: dict[str, str] = dict(
        zip(pdata.loc[pre_mask, subcode_col], pdata.index[pre_mask], strict=False)
    )
    post_map: dict[str, str] = dict(
        zip(pdata.loc[post_mask, subcode_col], pdata.index[post_mask], strict=False)
    )

    paired_subcodes = sorted(set(pre_map) & set(post_map))
    logger.info("BEST subjects with BL+12W: %d", len(paired_subcodes))

    # Filter to subjects with valid R/NR status (group col: R_BL or NR_BL)
    valid_resp = pdata.loc[
        pdata.index.isin([pre_map[s] for s in paired_subcodes]), response_col
    ].astype(str).str.upper()
    valid_subs = [
        s for s in paired_subcodes
        if str(pdata.loc[pre_map[s], response_col]).upper() in ("R_BL", "NR_BL")
    ]
    logger.info("Subjects with R/NR response: %d", len(valid_subs))

    pre_ids = [pre_map[s] for s in valid_subs]
    post_ids = [post_map[s] for s in valid_subs]

    pdata_paired = pdata.loc[pre_ids].copy()
    pdata_paired.index = pd.Index(valid_subs)

    return valid_subs, pre_ids, post_ids, pdata_paired


def encode_response(pdata_paired: pd.DataFrame, response_col: str = "group") -> np.ndarray:
    """Encode R_BL=1, NR_BL=0 as float array aligned to pdata_paired.index."""
    enc = (
        pdata_paired[response_col]
        .astype(str)
        .str.upper()
        .map({"R_BL": 1.0, "NR_BL": 0.0})
        .values.astype(float)
    )
    return enc


def _ols_celldmc_cpg(
    m_vals: np.ndarray,
    cell_fracs: np.ndarray,
    response: np.ndarray,
    covariates: np.ndarray,
    cell_type_names: list[str],
) -> list[dict[str, Any]]:
    """CellDMC OLS for a single CpG across all cell types."""
    from statsmodels.regression.linear_model import OLS

    n = len(m_vals)
    intercept = np.ones((n, 1))
    results = []
    for ci, ct in enumerate(cell_type_names):
        frac_c = cell_fracs[:, ci : ci + 1]
        interaction = response.reshape(-1, 1) * frac_c
        design = np.hstack([intercept, response.reshape(-1, 1), frac_c, interaction, covariates])
        valid = ~(np.isnan(m_vals) | np.isnan(design).any(axis=1))
        if valid.sum() < design.shape[1] + 2:
            results.append({
                "cell_type": ct, "coef": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_val": np.nan,
            })
            continue
        try:
            fit = OLS(m_vals[valid], design[valid]).fit()
            coef = fit.params[3]
            se = fit.bse[3]
            t = fit.tvalues[3]
            p = fit.pvalues[3]
            results.append({"cell_type": ct, "coef": float(coef), "se": float(se),
                            "t_stat": float(t), "p_val": float(p)})
        except Exception:
            results.append({
                "cell_type": ct, "coef": np.nan, "se": np.nan,
                "t_stat": np.nan, "p_val": np.nan,
            })
    return results


def run_celldmc_best(
    delta_m: np.ndarray,
    cpg_ids: list[str],
    cell_fracs: np.ndarray,
    pdata_paired: pd.DataFrame,
    response: np.ndarray,
    cell_type_names: list[str],
    n_jobs: int = -1,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """Run CellDMC interaction model on BEST delta-M matrix."""
    from joblib import Parallel, delayed

    covariate_cols = [
        c for c in ["Age", "sex", "smokingScore", "PC1", "PC2", "PC3"] if c in pdata_paired.columns
    ]
    if covariate_cols:
        from dnamrnaseq2026.preprocessing.cell_type_correction import _encode_covariates
        cov_matrix = _encode_covariates(pdata_paired[covariate_cols])
    else:
        cov_matrix = np.empty((len(pdata_paired), 0))
        logger.warning("No covariate columns available for BEST CellDMC.")

    n_cpg = delta_m.shape[0]
    chunks = [range(i, min(i + chunk_size, n_cpg)) for i in range(0, n_cpg, chunk_size)]

    def process_chunk(idxs: range) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for ci in idxs:
            res = _ols_celldmc_cpg(
                delta_m[ci], cell_fracs, response, cov_matrix, cell_type_names
            )
            for r in res:
                r["cpg"] = cpg_ids[ci]
                rows.append(r)
        return rows

    logger.info(
        "Running CellDMC on BEST: %d CpGs x %d samples x %d cell types.",
        n_cpg, delta_m.shape[1], len(cell_type_names)
    )
    results_nested = Parallel(n_jobs=n_jobs)(delayed(process_chunk)(ch) for ch in chunks)
    all_rows: list[dict[str, Any]] = []
    for chunk_rows in results_nested:
        all_rows.extend(chunk_rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # FDR per cell type
    df["fdr"] = np.nan
    for ct in df["cell_type"].unique():
        mask = df["cell_type"] == ct
        p_vals = df.loc[mask, "p_val"].fillna(1.0).values
        df.loc[mask, "fdr"] = multipletests(p_vals, method="fdr_bh")[1]

    df["sig"] = df["fdr"] < 0.05
    col_order = ["cpg", "cell_type", "coef", "se", "t_stat", "p_val", "fdr", "sig"]
    return df[[c for c in col_order if c in df.columns]]


def pi1_estimate(p_values: np.ndarray, lambda_val: float = 0.5) -> float:
    """Storey's pi1 estimate (proportion of true non-nulls).

    pi0 = #{p > lambda} / (n * (1 - lambda))
    pi1 = 1 - pi0
    """
    p = p_values[~np.isnan(p_values)]
    if len(p) == 0:
        return np.nan
    pi0 = float(np.sum(p > lambda_val)) / (len(p) * (1.0 - lambda_val))
    pi0 = min(pi0, 1.0)
    return round(1.0 - pi0, 4)


def run_replication_555(
    celldmc_emory: pd.DataFrame,
    celldmc_best: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute replication statistics for the 555 Emory hits in BEST.

    For each (cpg, cell_type) Emory hit, look up the matching BEST coefficient.
    Compute:
      - sign concordance: fraction of matched hits with same sign
      - Spearman rho: correlation of Emory coef vs BEST coef
      - pi1: Storey's replication rate on BEST p-values for the 555 hit set
      - n_sig_best_FDR05: hits also FDR<0.05 in BEST
    """
    emory_sig = celldmc_emory[celldmc_emory["sig"] == True].copy()
    emory_sig = emory_sig.rename(columns={"coef": "coef_emory", "fdr": "fdr_emory"})

    # Match on cpg + cell_type
    best_lookup = celldmc_best[["cpg", "cell_type", "coef", "p_val", "fdr"]].copy()
    best_lookup = best_lookup.rename(
        columns={"coef": "coef_best", "p_val": "p_val_best", "fdr": "fdr_best"}
    )

    merged = emory_sig.merge(best_lookup, on=["cpg", "cell_type"], how="left")
    logger.info("Emory hits merged with BEST: %d / %d matched.", merged["coef_best"].notna().sum(), len(merged))

    matched = merged[merged["coef_best"].notna()].copy()
    n_emory_hits = len(emory_sig)
    n_matched = len(matched)

    if n_matched == 0:
        stats_out: dict[str, Any] = {
            "n_emory_hits": n_emory_hits,
            "n_matched_in_best": 0,
            "sign_concordance": np.nan,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pi1_best": np.nan,
            "n_sig_best_fdr05": 0,
            "verdict": "NO_MATCH",
        }
        return merged, stats_out

    matched["same_sign"] = np.sign(matched["coef_emory"]) == np.sign(matched["coef_best"])
    sign_conc = float(matched["same_sign"].mean())

    spearman_rho, spearman_p = stats.spearmanr(
        matched["coef_emory"].values, matched["coef_best"].values
    )

    pi1 = pi1_estimate(matched["p_val_best"].dropna().values)

    n_sig_best = int((matched["fdr_best"].fillna(1.0) < 0.05).sum())
    n_sig_uncorr = int((matched["p_val_best"].fillna(1.0) < 0.05).sum())

    # Verdict: pass = sign concordance >= 0.55 AND pi1 >= 0.30 AND rho >= 0.15
    if sign_conc >= 0.55 and pi1 >= 0.30 and spearman_rho >= 0.15:
        verdict = "REPLICATES"
    elif sign_conc >= 0.50 and pi1 >= 0.20:
        verdict = "MARGINAL"
    else:
        verdict = "DOES_NOT_REPLICATE"

    stats_out = {
        "n_emory_hits": n_emory_hits,
        "n_matched_in_best": n_matched,
        "sign_concordance": round(sign_conc, 4),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": round(float(spearman_p), 6),
        "pi1_best": pi1,
        "n_sig_best_fdr05": n_sig_best,
        "n_sig_best_uncorrected_p05": n_sig_uncorr,
        "verdict": verdict,
    }

    return merged, stats_out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load BEST data
    logger.info("Loading BEST data.")
    best_df = pd.read_parquet(LATEST_DIR / "data_best.parquet")
    pdata_best = pd.read_csv(LATEST_DIR / "pdata_best_with_epidish.csv", index_col=0)

    # Index pdata by SampleName
    if "SampleName" in pdata_best.columns:
        pdata_best = pdata_best.set_index("SampleName", drop=False)

    logger.info("BEST data: %d CpGs x %d samples.", len(best_df) - 1, len(best_df.columns) - 1)

    # Build BEST paired IDs
    subjects, pre_ids, post_ids, pdata_paired = build_best_paired(
        pdata_best,
        subcode_col="Subcode",
        visit_col="Visit",
        pre_label="BL",
        post_label="12W",
        response_col="group",
    )
    logger.info("BEST paired subjects with R/NR: %d", len(subjects))

    if len(subjects) < 10:
        logger.error("Too few paired subjects to run CellDMC; aborting.")
        return

    # Build beta matrix for BEST (CpGs x paired samples)
    cpg_col = best_df["cpg"].tolist()
    beta_matrix_full = best_df.drop(columns=["cpg"]).values.astype(np.float32)
    sample_names_full = best_df.drop(columns=["cpg"]).columns.tolist()

    sample_idx = {s: i for i, s in enumerate(sample_names_full)}
    pre_cols = np.array([sample_idx[s] for s in pre_ids if s in sample_idx])
    post_cols = np.array([sample_idx[s] for s in post_ids if s in sample_idx])

    n_pairs_ok = min(len(pre_cols), len(post_cols))
    pre_cols = pre_cols[:n_pairs_ok]
    post_cols = post_cols[:n_pairs_ok]
    subjects_ok = subjects[:n_pairs_ok]
    pdata_paired = pdata_paired.iloc[:n_pairs_ok]

    logger.info("Computing BEST delta-M: %d CpGs x %d pairs.", len(cpg_col), n_pairs_ok)
    bvals_pre = beta_matrix_full[:, pre_cols].astype(np.float64)
    bvals_post = beta_matrix_full[:, post_cols].astype(np.float64)
    delta_m = _beta_to_m(bvals_post) - _beta_to_m(bvals_pre)

    # Cell fractions (EpiDISH fresh columns in pdata)
    ct_cols = ["EpiDISH_fresh_Bcell", "EpiDISH_fresh_CD4T", "EpiDISH_fresh_CD8T",
               "EpiDISH_fresh_Mono", "EpiDISH_fresh_Neu", "EpiDISH_fresh_NK"]
    ct_names_map = {
        "EpiDISH_fresh_Bcell": "B",
        "EpiDISH_fresh_CD4T": "CD4T",
        "EpiDISH_fresh_CD8T": "CD8T",
        "EpiDISH_fresh_Mono": "Mono",
        "EpiDISH_fresh_Neu": "Neutro",
        "EpiDISH_fresh_NK": "NK",
    }
    available_ct_cols = [c for c in ct_cols if c in pdata_paired.columns]
    cell_fracs_df = pdata_paired[available_ct_cols].fillna(0.0)
    cell_type_names = [ct_names_map[c] for c in available_ct_cols]

    # Match delta cell fracs to pre_ids (already aligned by subjects_ok)
    cell_fracs_arr = cell_fracs_df.values.astype(np.float64)

    response_arr = encode_response(pdata_paired, response_col="group")
    logger.info(
        "Response: %d R, %d NR.", int(response_arr.sum()), int((response_arr == 0).sum())
    )

    # Run CellDMC on BEST delta-M
    celldmc_best = run_celldmc_best(
        delta_m,
        cpg_col,
        cell_fracs_arr,
        pdata_paired,
        response_arr,
        cell_type_names,
        n_jobs=-1,
    )

    celldmc_best.to_csv(OUT_DIR / "celldmc_delta_best.tsv", sep="\t", index=False)
    celldmc_best.to_csv(LATEST_DIR / "celldmc_delta_best.tsv", sep="\t", index=False)
    n_sig_best = int(celldmc_best["sig"].sum()) if not celldmc_best.empty else 0
    logger.info(
        "BEST CellDMC delta done: %d sig hits (FDR<0.05).", n_sig_best
    )

    # Load Emory 555 hits
    celldmc_emory = pd.read_csv(LATEST_DIR / "celldmc_delta_emory.tsv", sep="\t")

    # Run replication concordance on the 555-hit set
    merged_rep, concordance_stats = run_replication_555(celldmc_emory, celldmc_best)

    merged_rep.to_csv(OUT_DIR / "replication_555hits.tsv", sep="\t", index=False)
    merged_rep.to_csv(LATEST_DIR / "replication_555hits.tsv", sep="\t", index=False)

    concordance_path = OUT_DIR / "concordance_stats.json"
    concordance_path.write_text(json.dumps(concordance_stats, indent=2, default=str))
    logger.info("Concordance stats: %s", concordance_stats)

    _write_results_md(celldmc_best, concordance_stats, n_sig_best)
    logger.info("Step 1.9 complete.")


def _write_results_md(
    celldmc_best: pd.DataFrame,
    concordance_stats: dict[str, Any],
    n_sig_best: int,
) -> None:
    verdict = concordance_stats.get("verdict", "UNKNOWN")
    n_emory = concordance_stats.get("n_emory_hits", 0)
    n_matched = concordance_stats.get("n_matched_in_best", 0)
    sign_conc = concordance_stats.get("sign_concordance", np.nan)
    rho = concordance_stats.get("spearman_rho", np.nan)
    rho_p = concordance_stats.get("spearman_p", np.nan)
    pi1 = concordance_stats.get("pi1_best", np.nan)
    n_sig_fdr05 = concordance_stats.get("n_sig_best_fdr05", 0)
    n_sig_uncorr = concordance_stats.get("n_sig_best_uncorrected_p05", 0)

    # Per-cell-type sig counts in BEST
    ct_counts_best = {}
    if not celldmc_best.empty:
        for ct in celldmc_best["cell_type"].unique():
            ct_mask = celldmc_best["cell_type"] == ct
            ct_counts_best[ct] = int(celldmc_best.loc[ct_mask, "sig"].sum())

    lines = [
        "# Step 1.9: BEST CellDMC Delta + Replication of 555 Emory Hits",
        "",
        "**Date:** 2026-05-22",
        "**Analyst:** Lee Lancashire",
        "",
        "## BEST CellDMC delta (full genome)",
        "",
        f"- Sig hits FDR<0.05 in BEST: {n_sig_best}",
        "",
        "Per cell type (BEST FDR<0.05):",
        "",
        "| Cell type | N sig (FDR<0.05) |",
        "|-----------|-----------------|",
    ]
    for ct, n in sorted(ct_counts_best.items()):
        lines.append(f"| {ct} | {n} |")

    lines += [
        "",
        "## Replication of 555 Emory hits in BEST",
        "",
        f"- Emory hits (FDR<0.05): {n_emory}",
        f"- Matched in BEST (same cpg + cell_type): {n_matched}",
        f"- Sign concordance: {sign_conc:.3f}" if not isinstance(sign_conc, float) or not np.isnan(sign_conc) else "- Sign concordance: N/A",
        f"- Spearman rho (Emory coef vs BEST coef): {rho:.3f} (p={rho_p:.4f})" if not isinstance(rho, float) or not np.isnan(rho) else "- Spearman rho: N/A",
        f"- pi1 (Storey, lambda=0.5): {pi1:.3f}" if not isinstance(pi1, float) or not np.isnan(pi1) else "- pi1: N/A",
        f"- BEST FDR<0.05 on the 555-hit set: {n_sig_fdr05}",
        f"- BEST p<0.05 (uncorrected) on the 555-hit set: {n_sig_uncorr}",
        "",
        "## Replication criteria",
        "",
        "- REPLICATES: sign concordance >= 0.55 AND pi1 >= 0.30 AND Spearman rho >= 0.15",
        "- MARGINAL: sign concordance >= 0.50 AND pi1 >= 0.20",
        "- DOES_NOT_REPLICATE: below thresholds",
        "",
        f"**Verdict: {verdict}**",
    ]

    out_path = OUT_DIR / "results.md"
    out_path.write_text("\n".join(lines))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
