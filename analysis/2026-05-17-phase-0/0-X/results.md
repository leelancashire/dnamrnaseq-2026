# Gate 0-X results: cross-disorder centroid projection

**Date:** 2026-05-17 (downloader + real run)
**Seed:** 42
**Verdict: MARGINAL**

## Quantitative findings

| Metric | Value |
|--------|-------|
| d(Emory NR, GSE MDD) | 70.5815 |
| d(Emory R, GSE MDD) | 71.1164 |
| Observed delta (NR - R) | -0.5349 |
| Direction correct (NR closer to MDD) | True |
| Permutation p (one-tailed, B=2000, seed=42) | 0.1115 |
| Genes in common space | 2000 |
| Gene intersection pre-filter | 13,192 |

## Method

- GSE98793: 192 whole-blood Affymetrix GPL570 samples (128 MDD CASE, 64 CNTL).
  Downloaded via GEOparse 2.0.4 from NCBI GEO FTP.
- Probe-to-gene rollup: max-mean strategy, committed reference annotation
  (src/dnamrnaseq2026/external_projection/resources/hgu133plus2_probe_to_gene.csv).
  54,675 probes to 22,880 gene symbols.
- Emory RNA-seq: 176 PRE-IOP samples (112 R, 64 NR). GENCODE Ensembl IDs
  reindexed to gene symbols (19,349 unique after duplicate collapse).
- Harmonisation: gene-symbol intersection (13,192 genes), quantile normalisation
  across combined 368-sample matrix.
- Variance filter to top 2,000 genes by combined-cohort variance.
- Centroids: Emory R mean, Emory NR mean, GSE MDD mean (all 128 CASE samples).
- Permutation test: B=2000 permutations of Emory R/NR labels.

## Verdict justification

MARGINAL: direction correct (NR centroid 0.53 units closer to GSE MDD
than R centroid) but permutation p=0.111 exceeds the PASS threshold of
0.05 and falls within the MARGINAL band [0.05, 0.15]. Per ANALYSIS_PLAN.md
Step 0-X acceptance criteria, MARGINAL allows the Phase 3.3 cross-disorder
figure to be built with the result documented honestly.

## Known limitations

1. TRD subset: all 128 MDD CASE samples used as TRD proxy. GSE98793 metadata
   does not include antidepressant response labels. Phase 3.3 should refine
   with a high-inflammation criterion (top-quartile GSVA score on an
   inflammation gene set).
2. Cross-platform harmonisation: quantile normalisation (crude, documented
   as load-bearing caveat per ANALYSIS_PLAN.md). Phase 3 will apply ComBat
   or COCONUT for a more principled approach.
3. Sample size asymmetry: Emory NR n=64 vs R n=112. The permutation test
   accounts for this, but the smaller NR group increases variance in the
   centroid estimate.

## Files

- `gate_0X_centroids.json` -- full results
- `gate_0X_genes_used.csv` -- 2000-gene set with Emory R/NR mean expression
- `gate_0X_centroid_projection.png` -- 2D PCA scatter with centroid markers
- `gate_0X_centroid_projection.svg`
