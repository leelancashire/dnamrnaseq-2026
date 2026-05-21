#!/usr/bin/env Rscript
# run_celldmc.R
# Phase 1 Step 1.2: CellDMC — cell-type-specific differentially methylated CpGs.
#
# CellDMC() is an exported function from the EpiDISH package. It fits a
# cell-type interaction model:
#   M_cpg ~ phenotype + cell_type_1 + ... + cell_type_K
#             + phenotype:cell_type_1 + ... + phenotype:cell_type_K + covariates
#
# The interaction terms (phenotype:cell_type_j) identify CpGs whose methylation
# changes with phenotype are specific to cell-type j. This is the correct model
# for Step 1.2; the Phase 1 PR #4 failure was caused by degenerate cell
# fractions (pData2 fallback columns, near-constant) which collapsed these
# interaction terms to null.
#
# Usage (Snakemake invokes this via the r-bioc conda env):
#   Rscript workflow/scripts/run_celldmc.R \
#     --bvals  analysis/latest/data_emory.parquet \
#     --fracs  analysis/latest/cell_props_emory.csv \
#     --pdata  analysis/latest/pdata_emory.csv \
#     --pheno  Response \
#     --visit  PRE_IOP \
#     --covars Age,Sex \
#     --output analysis/latest/celldmc_pre_emory.tsv \
#     --fdr    0.05 \
#     --ncore  4
#
# Inputs:
#   --bvals  : parquet, CpG x sample (beta values)
#   --fracs  : CSV, sample x cell-type fractions (output of run_epidish.R)
#   --pdata  : CSV, sample x covariates (Response, Visit, Age, Sex, ...)
#   --pheno  : column name in pdata for the phenotype (default: Response)
#   --visit  : visit filter value; only samples matching this Visit are used.
#              Pass "ALL" to include all visits (for delta analysis).
#   --covars : comma-separated covariate column names to include in the model
#   --output : output TSV (per-CpG, per-cell-type DMP results)
#   --fdr    : FDR threshold for reporting (default: 0.05; all CpGs are
#              written to output, this controls a summary column)
#   --ncore  : number of cores for parallel CellDMC (default: 1)
#
# Outputs:
#   TSV with columns: cpg, cell_type, coef, se, t_stat, p_val, fdr, sig
#   One row per (CpG, cell-type) interaction term.
#
# Column naming note (schema divergence vs Python path):
#   This R script uses t_stat for the t-statistic column. The Python CellDMC
#   implementation in src/dnamrnaseq2026/preprocessing/cell_type_correction.py
#   uses a different schema: beta_response, beta_interaction, p_response,
#   p_interaction, q_response, q_interaction (no t column). These are parallel
#   code paths producing different output schemas from the same conceptual model.
#   Any Phase 2 integration code that reads both must handle the rename:
#   Python "p_interaction" ~ R "p_val"; Python "beta_interaction" ~ R "coef".
#   See PR #5 review comment for tracking context.
#
# Design note on the phenotype encoding:
#   CellDMC expects a numeric phenotype vector. Response (R/NR) is encoded as
#   R=1, NR=0. BEST cohort uses BEST Response categories: 1,2,3 encoded as
#   treatment_response (1/2/3 -> 1.0/0.5/0.0 normalised to [0,1]). The caller
#   is responsible for passing the correct --pheno column.

suppressPackageStartupMessages({
  library(EpiDISH)
  library(optparse)
  library(arrow)
  library(dplyr)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

option_list <- list(
  make_option(c("--bvals"),  type = "character", help = "Beta-value parquet"),
  make_option(c("--fracs"),  type = "character", help = "Cell-fraction CSV"),
  make_option(c("--pdata"),  type = "character", help = "Sample metadata CSV"),
  make_option(c("--pheno"),  type = "character", default = "Response",
              help = "Phenotype column in pdata [default: Response]"),
  make_option(c("--visit"),  type = "character", default = "ALL",
              help = "Visit filter (PRE_IOP, POST_IOP, or ALL) [default: ALL]"),
  make_option(c("--covars"), type = "character", default = "",
              help = "Comma-separated covariate columns [default: none]"),
  make_option(c("--output"), type = "character", help = "Output TSV"),
  make_option(c("--fdr"),    type = "double",    default = 0.05,
              help = "FDR threshold for 'sig' column [default: 0.05]"),
  make_option(c("--ncore"),  type = "integer",   default = 1L,
              help = "Parallel cores [default: 1]")
)

opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$bvals),  file.exists(opt$bvals))
stopifnot(!is.null(opt$fracs),  file.exists(opt$fracs))
stopifnot(!is.null(opt$pdata),  file.exists(opt$pdata))
stopifnot(!is.null(opt$output))

cat(sprintf("[run_celldmc] bvals  : %s\n", opt$bvals))
cat(sprintf("[run_celldmc] fracs  : %s\n", opt$fracs))
cat(sprintf("[run_celldmc] pdata  : %s\n", opt$pdata))
cat(sprintf("[run_celldmc] pheno  : %s\n", opt$pheno))
cat(sprintf("[run_celldmc] visit  : %s\n", opt$visit))
cat(sprintf("[run_celldmc] covars : %s\n", opt$covars))
cat(sprintf("[run_celldmc] output : %s\n", opt$output))
cat(sprintf("[run_celldmc] fdr    : %.4f\n", opt$fdr))
cat(sprintf("[run_celldmc] ncore  : %d\n", opt$ncore))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

cat("[run_celldmc] Loading beta matrix...\n")
beta_df <- arrow::read_parquet(opt$bvals)

if (colnames(beta_df)[1] %in% c("__index_level_0__", "index", "cpg", "CpG")) {
  cpg_ids  <- beta_df[[1]]
  beta_df  <- beta_df[, -1, drop = FALSE]
} else {
  cpg_ids  <- rownames(beta_df)
}
beta_mat <- as.matrix(beta_df)
rownames(beta_mat) <- cpg_ids

cat("[run_celldmc] Loading cell fractions...\n")
frac_df <- read.csv(opt$fracs, row.names = 1, check.names = FALSE)
frac_mat <- as.matrix(frac_df)

cat("[run_celldmc] Loading sample metadata...\n")
pdata <- read.csv(opt$pdata, row.names = 1, check.names = FALSE)

cat(sprintf("[run_celldmc] Beta: %d CpGs x %d samples\n",
            nrow(beta_mat), ncol(beta_mat)))
cat(sprintf("[run_celldmc] Fracs: %d samples x %d cell types\n",
            nrow(frac_mat), ncol(frac_mat)))
cat(sprintf("[run_celldmc] pData: %d samples x %d columns\n",
            nrow(pdata), ncol(pdata)))

# ---------------------------------------------------------------------------
# Visit filter
# ---------------------------------------------------------------------------

if (opt$visit != "ALL") {
  if (!"Visit" %in% colnames(pdata)) {
    stop("[run_celldmc] FATAL: --visit filter specified but 'Visit' column not in pdata.")
  }
  keep_samples <- rownames(pdata)[pdata$Visit == opt$visit]
  if (length(keep_samples) == 0) {
    stop(sprintf("[run_celldmc] FATAL: No samples match Visit == '%s'.", opt$visit))
  }
  pdata    <- pdata[keep_samples, , drop = FALSE]
  frac_mat <- frac_mat[keep_samples, , drop = FALSE]
  beta_mat <- beta_mat[, keep_samples, drop = FALSE]
  cat(sprintf("[run_celldmc] After visit filter (%s): %d samples retained.\n",
              opt$visit, length(keep_samples)))
}

# ---------------------------------------------------------------------------
# Align sample IDs across all three matrices
# ---------------------------------------------------------------------------

common_samples <- Reduce(intersect, list(
  colnames(beta_mat),
  rownames(frac_mat),
  rownames(pdata)
))

if (length(common_samples) < 5) {
  stop(sprintf(
    "[run_celldmc] FATAL: Only %d common samples across beta/fracs/pdata. Check ID alignment.",
    length(common_samples)
  ))
}

cat(sprintf("[run_celldmc] Common samples: %d\n", length(common_samples)))
beta_mat <- beta_mat[, common_samples, drop = FALSE]
frac_mat <- frac_mat[common_samples, , drop = FALSE]
pdata    <- pdata[common_samples, , drop = FALSE]

# ---------------------------------------------------------------------------
# Phenotype vector
# ---------------------------------------------------------------------------

if (!opt$pheno %in% colnames(pdata)) {
  stop(sprintf("[run_celldmc] FATAL: phenotype column '%s' not in pdata.", opt$pheno))
}

pheno_raw <- pdata[[opt$pheno]]

# Encode: if character (R/NR), convert. If numeric, use as-is.
if (is.character(pheno_raw) || is.factor(pheno_raw)) {
  pheno_raw <- as.character(pheno_raw)
  unique_vals <- unique(pheno_raw[!is.na(pheno_raw)])
  if (setequal(unique_vals, c("R", "NR")) || setequal(unique_vals, c("NR", "R"))) {
    pheno_vec <- as.numeric(pheno_raw == "R")  # R=1, NR=0
    cat("[run_celldmc] Encoded Response: R=1, NR=0\n")
  } else {
    stop(sprintf(
      "[run_celldmc] FATAL: Cannot auto-encode phenotype values: %s. ",
      paste(unique_vals, collapse = ", ")
    ))
  }
} else {
  pheno_vec <- as.numeric(pheno_raw)
}

names(pheno_vec) <- common_samples

# Drop NA phenotype samples. CellDMC rejects any NA in pheno.v.
na_mask <- is.na(pheno_vec)
if (any(na_mask)) {
  n_dropped <- sum(na_mask)
  cat(sprintf("[run_celldmc] Dropping %d samples with NA phenotype.\n", n_dropped))
  keep_pheno <- !na_mask
  pheno_vec  <- pheno_vec[keep_pheno]
  frac_mat   <- frac_mat[names(pheno_vec), , drop = FALSE]
  beta_mat   <- beta_mat[, names(pheno_vec), drop = FALSE]
  pdata      <- pdata[names(pheno_vec), , drop = FALSE]
  cat(sprintf("[run_celldmc] After NA drop: %d samples retained.\n", length(pheno_vec)))
}

if (var(pheno_vec, na.rm = TRUE) < 1e-8) {
  stop("[run_celldmc] FATAL: Phenotype vector is constant. CellDMC cannot run.")
}

cat(sprintf("[run_celldmc] Phenotype: N=%d, mean=%.3f, var=%.3f\n",
            length(pheno_vec), mean(pheno_vec), var(pheno_vec)))

# ---------------------------------------------------------------------------
# Covariate matrix
# ---------------------------------------------------------------------------

covar_names <- trimws(strsplit(opt$covars, ",")[[1]])
covar_names <- covar_names[nchar(covar_names) > 0]

if (length(covar_names) > 0) {
  missing_covars <- setdiff(covar_names, colnames(pdata))
  if (length(missing_covars) > 0) {
    stop(sprintf("[run_celldmc] FATAL: Covariate columns not in pdata: %s",
                 paste(missing_covars, collapse = ", ")))
  }
  covar_mat <- as.matrix(pdata[, covar_names, drop = FALSE])
  # Encode any character covariates as numeric
  for (col in covar_names) {
    if (!is.numeric(covar_mat[, col])) {
      covar_mat[, col] <- as.numeric(as.factor(covar_mat[, col]))
    }
  }
  covar_mat <- matrix(as.numeric(covar_mat), nrow = nrow(covar_mat),
                      dimnames = list(rownames(covar_mat), covar_names))
  cat(sprintf("[run_celldmc] Covariates: %s\n", paste(covar_names, collapse = ", ")))
} else {
  covar_mat <- NULL
  cat("[run_celldmc] No covariates.\n")
}

# ---------------------------------------------------------------------------
# Variance filter: top 50,000 most variable CpGs for speed
# (CellDMC on 850K+ CpGs is slow; full run is available by setting --ncore)
# ---------------------------------------------------------------------------

N_CPG_MAX <- 50000L
if (nrow(beta_mat) > N_CPG_MAX) {
  cat(sprintf("[run_celldmc] Variance-filtering to top %d CpGs...\n", N_CPG_MAX))
  row_vars <- apply(beta_mat, 1, var, na.rm = TRUE)
  keep_cpgs <- order(row_vars, decreasing = TRUE)[seq_len(N_CPG_MAX)]
  beta_mat <- beta_mat[keep_cpgs, , drop = FALSE]
  cat(sprintf("[run_celldmc] Retained %d CpGs.\n", nrow(beta_mat)))
} else {
  cat(sprintf("[run_celldmc] Using all %d CpGs (below threshold).\n", nrow(beta_mat)))
}

# ---------------------------------------------------------------------------
# Run CellDMC
# ---------------------------------------------------------------------------

cat(sprintf("[run_celldmc] Running CellDMC (%d CpGs x %d samples x %d cell types, ncore=%d)...\n",
            nrow(beta_mat), ncol(beta_mat), ncol(frac_mat), opt$ncore))

celldmc_out <- CellDMC(
  beta.m     = beta_mat,
  pheno.v    = pheno_vec,
  frac.m     = frac_mat,
  cov.mod    = covar_mat,
  adjPMethod = "BH",
  adjPThresh = opt$fdr,
  mc.cores   = opt$ncore
)

# CellDMC returns:
#   $coe  : list of data.frames, one per cell type. Each has columns:
#            Estimate, SE, t, p-value, Adjusted.P.Value
cat("[run_celldmc] CellDMC complete. Collating results...\n")

cell_types <- names(celldmc_out$coe)
cat(sprintf("[run_celldmc] Cell types in output: %s\n",
            paste(cell_types, collapse = ", ")))

# Collate into a long-format TSV: cpg x cell_type rows
result_list <- lapply(cell_types, function(ct) {
  df <- celldmc_out$coe[[ct]]
  data.frame(
    cpg       = rownames(df),
    cell_type = ct,
    coef      = df[["Estimate"]],
    se        = df[["SE"]],
    t_stat    = df[["t"]],
    p_val     = df[["p-value"]],
    fdr       = df[["Adjusted.P.Value"]],
    sig       = df[["Adjusted.P.Value"]] < opt$fdr,
    stringsAsFactors = FALSE
  )
})

results_df <- do.call(rbind, result_list)

n_sig <- sum(results_df$sig, na.rm = TRUE)
cat(sprintf("[run_celldmc] Significant (FDR<%.2f): %d / %d CpG-cell_type pairs\n",
            opt$fdr, n_sig, nrow(results_df)))

# ---------------------------------------------------------------------------
# Write output TSV
# ---------------------------------------------------------------------------

out_dir <- dirname(opt$output)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

write.table(results_df, file = opt$output, sep = "\t", row.names = FALSE,
            quote = FALSE)
cat(sprintf("[run_celldmc] Wrote %d rows to: %s\n", nrow(results_df), opt$output))
cat("[run_celldmc] DONE\n")
