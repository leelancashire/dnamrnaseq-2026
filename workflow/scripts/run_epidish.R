#!/usr/bin/env Rscript
# run_epidish.R
# Phase 1 Step 1.1: EpiDISH cell-type fraction estimation.
#
# Reads a beta-value parquet (CpG x sample) written by the Python load_cohort
# rule, runs EpiDISH::epidish() with the chosen reference panel, and writes a
# cell-fraction CSV (sample x cell-type).
#
# Usage (Snakemake invokes this via the r-bioc conda env):
#   Rscript workflow/scripts/run_epidish.R \
#     --input  analysis/latest/data_emory.parquet \
#     --output analysis/latest/cell_props_emory.csv \
#     --ref    centDHSbloodDMC.m \
#     --method RPC
#
# Inputs:
#   --input   : parquet file, CpG rows x sample columns (beta values in [0,1])
#   --output  : output CSV, samples x cell-types (rows = samples)
#   --ref     : EpiDISH reference panel name. Must be a dataset that ships with
#               EpiDISH 2.16.0. Default: centDHSbloodDMC.m (see notes below).
#               The reference panel object is loaded via data() from the package.
#               No external download required.
#   --method  : EpiDISH method. One of: RPC, CBS, CP. Default: RPC.
#
# Outputs:
#   cell-fraction CSV: columns depend on the chosen reference. With the default
#   centDHSbloodDMC.m: B, NK, CD4T, CD8T, Mono, Neutro, Eosino (7 types).
#   Row names = sample IDs.
#
# Notes on EpiDISH 2.16.0 reference panels (as confirmed by data(package="EpiDISH")):
#
#   BLOOD REFERENCES (whole blood, use for PBMC/blood methylation arrays):
#   centDHSbloodDMC.m : 333 CpGs, 7 types: B, NK, CD4T, CD8T, Mono, Neutro, Eosino.
#                       Default for this project (see reference-choice rationale below).
#   centBloodSub.m    : 188 CpGs, 7 types (same cell type set, sparser probe set).
#   cent12CT.m        : 600 CpGs, 12 types: CD4Tnv, Baso, CD4Tmem, Bmem, Bnv, Treg,
#                       CD8Tmem, CD8Tnv, Eos, NK, Neu, Mono. Higher resolution but
#                       lower power per type in CellDMC interaction models.
#   cent12CT450k.m    : 450k array version of cent12CT.m.
#
#   TISSUE REFERENCES (non-blood -- not appropriate for this project):
#   centEpiFibFatIC.m : breast tissue (Epi/Fib/Fat/IC)
#   centEpiFibIC.m    : epithelial tissue (Epi/Fib/IC)
#
#   NOTE: centEpicV1 and centEpicV2 DO NOT EXIST in EpiDISH 2.16.0. Earlier
#   versions of this script incorrectly referenced centEpicV2 as the default.
#   That object is not shipped with this package version and will produce a
#   hard error: "object 'centEpicV2' not found".
#
# Reference panel choice rationale (2026-05-21):
#   centDHSbloodDMC.m (7 types, default) was chosen over cent12CT.m (12 types)
#   for this project. The study hypothesis targets a monocyte-specific mechanism.
#   Monocytes are represented in both references. The 12-type reference fragments
#   T-cell and B-cell compartments into subsets that are not the biological target.
#   At n~168-208 per visit stratum after NA-phenotype filtering, fitting 12
#   CellDMC interaction terms per CpG costs meaningful power per type vs 7.
#   centDHSbloodDMC.m gives adequate monocyte resolution without power dilution.
#
# CellDMC note: CellDMC() is exported from the EpiDISH package. This script
# handles only the cell-fraction estimation step. run_celldmc.R handles the
# CellDMC interaction-term modelling using these fractions as input.

suppressPackageStartupMessages({
  library(EpiDISH)
  library(optparse)
  library(arrow)    # parquet I/O
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

option_list <- list(
  make_option(c("--input"),  type = "character", help = "Input parquet (CpG x sample)"),
  make_option(c("--output"), type = "character", help = "Output cell-fraction CSV"),
  make_option(c("--ref"),    type = "character", default = "centDHSbloodDMC.m",
              help = "EpiDISH reference panel [default: centDHSbloodDMC.m]"),
  make_option(c("--method"), type = "character", default = "RPC",
              help = "EpiDISH method: RPC, CBS, or CP [default: RPC]")
)

opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$input),  file.exists(opt$input))
stopifnot(!is.null(opt$output))

cat(sprintf("[run_epidish] input  : %s\n", opt$input))
cat(sprintf("[run_epidish] output : %s\n", opt$output))
cat(sprintf("[run_epidish] ref    : %s\n", opt$ref))
cat(sprintf("[run_epidish] method : %s\n", opt$method))

# ---------------------------------------------------------------------------
# Load beta matrix
# ---------------------------------------------------------------------------

cat("[run_epidish] Loading beta matrix from parquet...\n")
beta_df <- arrow::read_parquet(opt$input)

# Expected layout: CpG as row index (first column named __index_level_0__ or
# similar), samples as remaining columns. Normalise to matrix.
if (colnames(beta_df)[1] %in% c("__index_level_0__", "index", "cpg", "CpG")) {
  rownames_vec <- beta_df[[1]]
  beta_df <- beta_df[, -1, drop = FALSE]
} else {
  # Assume row names are stored as actual row names (arrow preserves them
  # when written with row.names=TRUE). If not, use integer indices.
  rownames_vec <- rownames(beta_df)
}

beta_mat <- as.matrix(beta_df)
rownames(beta_mat) <- rownames_vec

cat(sprintf("[run_epidish] Beta matrix: %d CpGs x %d samples\n",
            nrow(beta_mat), ncol(beta_mat)))

# Sanity check: beta values must be in [0, 1]
brange <- range(beta_mat, na.rm = TRUE)
if (brange[1] < -0.01 || brange[2] > 1.01) {
  stop(sprintf(
    "[run_epidish] FATAL: beta values outside [0,1] range: [%.4f, %.4f]. ",
    brange[1], brange[2]
  ))
}

# ---------------------------------------------------------------------------
# Load EpiDISH reference panel
# ---------------------------------------------------------------------------

cat(sprintf("[run_epidish] Loading reference panel: %s\n", opt$ref))

# Reference panels ship with EpiDISH. Access via data() call.
# Valid blood references in EpiDISH 2.16.0 (confirmed via data(package="EpiDISH")).
# centEpicV1 and centEpicV2 do NOT exist in this version.
valid_refs <- c("centDHSbloodDMC.m", "centBloodSub.m", "cent12CT.m",
                "cent12CT450k.m", "centEpiFibFatIC.m", "centEpiFibIC.m")
if (!opt$ref %in% valid_refs) {
  stop(sprintf("[run_epidish] Unknown ref '%s'. Choose from: %s",
               opt$ref, paste(valid_refs, collapse = ", ")))
}

data(list = opt$ref, package = "EpiDISH", envir = environment())
ref_mat <- get(opt$ref)

cat(sprintf("[run_epidish] Reference: %d CpGs x %d cell types\n",
            nrow(ref_mat), ncol(ref_mat)))
cat(sprintf("[run_epidish] Cell types: %s\n",
            paste(colnames(ref_mat), collapse = ", ")))

# ---------------------------------------------------------------------------
# Run EpiDISH
# ---------------------------------------------------------------------------

cat(sprintf("[run_epidish] Running epidish (method=%s)...\n", opt$method))

epi_out <- epidish(
  beta.m  = beta_mat,
  ref.m   = ref_mat,
  method  = opt$method
)

# epidish returns list with $estF = estimated cell fractions (sample x cell-type)
frac_mat <- epi_out$estF

cat(sprintf("[run_epidish] Cell fractions: %d samples x %d cell types\n",
            nrow(frac_mat), ncol(frac_mat)))

# Row-sum check: fractions should sum to ~1 per sample
row_sums <- rowSums(frac_mat)
cat(sprintf("[run_epidish] Fraction row-sum range: [%.4f, %.4f]\n",
            min(row_sums), max(row_sums)))

if (any(abs(row_sums - 1.0) > 0.05)) {
  warning(sprintf(
    "[run_epidish] %d samples have fraction row-sum >5%% from 1.0. Check input.",
    sum(abs(row_sums - 1.0) > 0.05)
  ))
}

# Variance check: all-constant columns indicate degenerate deconvolution
col_vars <- apply(frac_mat, 2, var)
const_cols <- names(col_vars)[col_vars < 1e-6]
if (length(const_cols) > 0) {
  warning(sprintf(
    "[run_epidish] Near-constant cell-type columns detected: %s. ",
    paste(const_cols, collapse = ", ")
  ))
}

# ---------------------------------------------------------------------------
# Write output CSV
# ---------------------------------------------------------------------------

out_dir <- dirname(opt$output)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

write.csv(frac_mat, file = opt$output, row.names = TRUE)
cat(sprintf("[run_epidish] Wrote cell fractions to: %s\n", opt$output))
cat("[run_epidish] DONE\n")
