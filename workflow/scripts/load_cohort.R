#!/usr/bin/env Rscript
# load_cohort.R
# Phase 1 data loading: read RData files (bVals matrix + pData2 frame) and
# write output parquet + pdata CSV.
#
# Replaces scripts/snakemake/load_cohort.py (pyreadr path) because pyreadr
# cannot handle R matrix objects (PyreadrError: matrix, array or table object
# with more than one vector). R load() reads all RData types natively.
#
# Usage (Snakemake invokes via the r-bioconductor.yaml conda env):
#   Rscript workflow/scripts/load_cohort.R \
#     --bvals   "/path/to/cohort.bVals.architecture.RData" \
#     --pdata   "/path/to/cohort_pData2.RData" \
#     --out_data    "analysis/latest/data_emory.parquet" \
#     --out_pdata   "analysis/latest/pdata_emory.csv"   # optional
#
# RData structure (confirmed by probe 2026-05-21):
#   emory.bVals.architecture: matrix double, 292674 CpGs x 388 samples
#   emory_pData2: data.frame, 388 x 366; rownames = array IDs (Sentrix),
#     SampleName == rownames, Response in {R, NR, NA}, Visit in {PRE-IOP, POST-IOP}
#   best.bVals.architecture: matrix double, 292973 CpGs x 141 samples
#   best_pData2: data.frame, 141 x 678; same index scheme
#
# Parquet output orientation: CpG rows x sample columns (first column = "cpg").
#   run_epidish.R and run_celldmc.R both expect this orientation.
#   They detect the "cpg" first column, extract it as rownames, and treat the
#   remaining columns as samples.  This is the INVERSE of what the old Python
#   load_cohort.py wrote (bvals.T = samples x CpGs); the Python transpose was
#   incorrect for the R-direct pipeline.
#
# pData CSV: all columns from pData2, with rownames written as first column
#   "SampleName".  run_celldmc.R reads with row.names=1 (first col as index).
#
# Arguments:
#   --bvals     : path to bVals .RData file
#   --pdata     : path to pData2 .RData file
#   --out_data  : output parquet path (CpG x sample)
#   --out_pdata : output pdata CSV path (optional; skip for BEST if not needed)

suppressPackageStartupMessages({
  library(arrow)
  library(optparse)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

option_list <- list(
  make_option(c("--bvals"),     type = "character", help = "bVals .RData file"),
  make_option(c("--pdata"),     type = "character", help = "pData2 .RData file"),
  make_option(c("--out_data"),  type = "character", help = "Output parquet (CpG x sample)"),
  make_option(c("--out_pdata"), type = "character", default = NULL,
              help = "Output pdata CSV [optional]")
)

opt <- parse_args(OptionParser(option_list = option_list))

stopifnot(!is.null(opt$bvals),    file.exists(opt$bvals))
stopifnot(!is.null(opt$pdata),    file.exists(opt$pdata))
stopifnot(!is.null(opt$out_data))

cat(sprintf("[load_cohort] bvals     : %s\n", opt$bvals))
cat(sprintf("[load_cohort] pdata     : %s\n", opt$pdata))
cat(sprintf("[load_cohort] out_data  : %s\n", opt$out_data))
if (!is.null(opt$out_pdata)) {
  cat(sprintf("[load_cohort] out_pdata : %s\n", opt$out_pdata))
}

# ---------------------------------------------------------------------------
# Load bVals matrix
# ---------------------------------------------------------------------------

cat("[load_cohort] Loading bVals RData...\n")
bvals_env <- new.env()
load(opt$bvals, envir = bvals_env)
bvals_objs <- ls(bvals_env)
cat(sprintf("[load_cohort] bVals objects in RData: %s\n",
            paste(bvals_objs, collapse = ", ")))

# Take the first (and typically only) object
bvals_raw <- get(bvals_objs[1], envir = bvals_env)
rm(bvals_env)

# Ensure it is a matrix with CpG rownames and sample colnames
if (!is.matrix(bvals_raw)) {
  bvals_raw <- as.matrix(bvals_raw)
}

cpg_ids     <- rownames(bvals_raw)
sample_ids  <- colnames(bvals_raw)

cat(sprintf("[load_cohort] bVals matrix: %d CpGs x %d samples\n",
            length(cpg_ids), length(sample_ids)))

if (is.null(cpg_ids) || is.null(sample_ids)) {
  stop("[load_cohort] FATAL: bVals matrix has no row or column names.")
}

# Sanity check: beta values in [0, 1]
brange <- range(bvals_raw, na.rm = TRUE)
cat(sprintf("[load_cohort] Beta value range: [%.6f, %.6f]\n", brange[1], brange[2]))
if (brange[1] < -0.01 || brange[2] > 1.01) {
  stop(sprintf(
    "[load_cohort] FATAL: beta values outside [0,1]: [%.4f, %.4f]. Not a methylation beta matrix.",
    brange[1], brange[2]
  ))
}

# ---------------------------------------------------------------------------
# Load pData2 frame
# ---------------------------------------------------------------------------

cat("[load_cohort] Loading pData2 RData...\n")
pdata_env <- new.env()
load(opt$pdata, envir = pdata_env)
pdata_objs <- ls(pdata_env)
cat(sprintf("[load_cohort] pData objects in RData: %s\n",
            paste(pdata_objs, collapse = ", ")))

pdata_raw <- get(pdata_objs[1], envir = pdata_env)
rm(pdata_env)

if (!is.data.frame(pdata_raw)) {
  pdata_raw <- as.data.frame(pdata_raw)
}

cat(sprintf("[load_cohort] pData2 shape: %d samples x %d columns\n",
            nrow(pdata_raw), ncol(pdata_raw)))

# Use SampleName as index if present (SampleName == rownames in both cohorts,
# confirmed by probe). If SampleName is a column, set it as rownames.
if ("SampleName" %in% colnames(pdata_raw)) {
  if (!identical(rownames(pdata_raw), as.character(pdata_raw$SampleName))) {
    # SampleName differs from current rownames -- use SampleName as index
    rownames(pdata_raw) <- as.character(pdata_raw$SampleName)
    cat("[load_cohort] Set rownames from SampleName column.\n")
  } else {
    cat("[load_cohort] rownames already match SampleName column.\n")
  }
}

pdata_sample_ids <- rownames(pdata_raw)

# ---------------------------------------------------------------------------
# Sample alignment check
# ---------------------------------------------------------------------------

overlap    <- intersect(sample_ids, pdata_sample_ids)
bvals_only <- setdiff(sample_ids, pdata_sample_ids)
pdata_only <- setdiff(pdata_sample_ids, sample_ids)

cat(sprintf("[load_cohort] Sample alignment: %d bVals / %d pData2 / %d overlap\n",
            length(sample_ids), length(pdata_sample_ids), length(overlap)))

if (length(bvals_only) > 0) {
  cat(sprintf("[load_cohort] WARNING: %d samples in bVals not in pData2: %s...\n",
              length(bvals_only), paste(head(bvals_only, 5), collapse = ", ")))
}
if (length(pdata_only) > 0) {
  cat(sprintf("[load_cohort] WARNING: %d samples in pData2 not in bVals: %s...\n",
              length(pdata_only), paste(head(pdata_only, 5), collapse = ", ")))
}
if (length(overlap) == 0) {
  stop("[load_cohort] FATAL: Zero sample overlap between bVals and pData2.")
}

# Log key clinical columns for downstream awareness
key_cols <- c("Visit", "Response", "response_category", "Age", "Sex", "sex",
              "Gender", "Subcode")
present_key_cols <- intersect(key_cols, colnames(pdata_raw))
cat(sprintf("[load_cohort] Key clinical columns present: %s\n",
            paste(present_key_cols, collapse = ", ")))
for (col in present_key_cols) {
  vals <- table(pdata_raw[[col]], useNA = "always")
  cat(sprintf("[load_cohort]   %s: %s\n", col,
              paste(paste0(names(vals), "=", as.vector(vals)), collapse=", ")))
}

# ---------------------------------------------------------------------------
# Write bVals parquet (CpG x sample orientation)
#
# Layout written: data.frame with first column "cpg" (CpG probe IDs) and
# remaining columns named by sample IDs. run_epidish.R and run_celldmc.R both
# detect "cpg" as the first column and extract it as rownames.
#
# We write ALL samples (not only aligned overlap) to preserve the full matrix
# for steps that may do their own alignment. The pData CSV provides the
# per-sample metadata; downstream rules join on sample ID.
# ---------------------------------------------------------------------------

cat("[load_cohort] Writing bVals parquet (CpG x sample)...\n")

out_dir <- dirname(opt$out_data)
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  cat(sprintf("[load_cohort] Created output directory: %s\n", out_dir))
}

# Convert matrix to data.frame with cpg as first column
bvals_df <- as.data.frame(bvals_raw)
bvals_df <- cbind(cpg = rownames(bvals_df), bvals_df)
# Ensure numeric columns (not factors)
for (cn in colnames(bvals_df)[-1]) {
  bvals_df[[cn]] <- as.numeric(bvals_df[[cn]])
}

arrow::write_parquet(bvals_df, opt$out_data)
cat(sprintf("[load_cohort] Written: %s (%d CpGs x %d samples)\n",
            opt$out_data, nrow(bvals_df), ncol(bvals_df) - 1L))

# ---------------------------------------------------------------------------
# Write pData CSV (if --out_pdata provided)
# ---------------------------------------------------------------------------

if (!is.null(opt$out_pdata)) {
  cat("[load_cohort] Writing pData CSV...\n")
  pdata_dir <- dirname(opt$out_pdata)
  if (!dir.exists(pdata_dir)) {
    dir.create(pdata_dir, recursive = TRUE)
  }
  # Write with rownames (first column becomes the index column; run_celldmc.R
  # reads with row.names=1 so this aligns correctly)
  write.csv(pdata_raw, file = opt$out_pdata, row.names = TRUE)
  cat(sprintf("[load_cohort] Written pData: %s (%d samples x %d columns)\n",
              opt$out_pdata, nrow(pdata_raw), ncol(pdata_raw)))
}

cat("[load_cohort] DONE\n")
