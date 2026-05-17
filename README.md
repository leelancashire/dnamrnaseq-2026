# dnamrnaseq-2026

PTSD treatment-response trajectory atlas. Companion code for the manuscript:
*"Treatment response trajectories in PTSD trace toward healthy-state biology and
away from the treatment-resistant-depression inflammatory state."*

Joint DNAm + RNA-seq multi-omics analysis. Emory + BEST cohorts.

**Status:** Scaffold (v0.1.0). Data loaders and preprocessing Snakemake rules are functional.
Embedding, trajectory, and figure rules are stubs pending Phase 0 gate results.

---

## Quickstart (10 minutes)

### Prerequisites

- conda >= 23 (or mamba)
- Access to the Emory/BEST OneDrive data mount (internal collaborators only)
- GitHub access to `leelancashire/dnamrnaseq-2026` (private repo)

### Steps

```bash
# 1. Clone
git clone git@github.com:leelancashire/dnamrnaseq-2026.git
cd dnamrnaseq-2026

# 2. Create environment
conda env create -f environment.yml
conda activate dnamrnaseq2026

# 3. Install the package in editable mode
pip install -e .

# 4. Configure data paths
cp config.yaml.example config.yaml
# Edit config.yaml if your OneDrive mount differs from the default WSL2 path.
# Default: /mnt/d/lee/onedrive/work/nicol healthtech/cvb/emory-dnam

# 5. Verify data access (Day-0 verification)
python scripts/00_load_data.py
```

**Expected output from `00_load_data.py`:**

```
Loading Emory DNAm bVals (architecture subset)...
  emory.bVals.architecture: (292674, 388)  [CpG sites x samples]
Loading Emory pData2...
  emory_pData2: (388, 366)  [samples x covariates]
Loading BEST DNAm bVals (architecture subset)...
  best.bVals.architecture: (292973, 141)  [CpG sites x samples]
Loading BEST pData2...
  best_pData2: (141, 678)  [samples x covariates]

Sample-ID alignment check:
  Emory bVals cols in pData2 index: 388/388
  BEST bVals cols in pData2 index: 141/141

Response value counts (Emory):
  [... from pData2 Response column ...]
Response value counts (BEST):
  [... from pData2 Response column ...]

All checks passed. Exit 0.
```

If shapes differ from above, check that the OneDrive data path in `config.yaml` is correct.

---

## Repo Layout

```
dnamrnaseq-2026/
├── README.md                   # this file
├── LICENSE                     # MIT
├── pyproject.toml              # package config (PEP 621)
├── environment.yml             # conda env spec
├── config.yaml.example         # template; copy to config.yaml (gitignored)
├── .pre-commit-config.yaml     # ruff + mypy + nbstripout + safety checks
├── .github/workflows/
│   ├── ci.yml                  # lint + test on every PR
│   └── smoke-pipeline.yml      # Snakemake end-to-end on synthetic data
├── Snakefile                   # top-level Snakemake DAG entry point
├── workflow/
│   ├── rules/                  # Snakemake rule modules (.smk)
│   ├── envs/                   # per-rule conda env specs
│   └── schemas/                # config YAML schemas
├── src/dnamrnaseq2026/         # importable library (src-layout)
│   ├── data/                   # loaders, harmonisation, sample-ID linkage
│   ├── preprocessing/          # CellDMC, EpiDISH, covariate residualisation
│   ├── embedding/              # FM arm, MOFA+ arm, contrastive arm
│   ├── trajectory/             # archetype clustering, recovery-axis annotation
│   ├── conformal/              # Mondrian weighted conformal prediction sets
│   ├── mediation/              # HIMA / BAMA / E-value wrappers
│   ├── external_projection/    # GSE98793, GTEx, AURORA harmonisation + projection
│   └── viz/                    # headline trajectory-atlas figure generator
├── scripts/                    # runnable entry-points (numbered by phase)
│   ├── 00_load_data.py         # Day-0 verification
│   ├── 01_phase0_gates.py      # Gates 0-T, 0-Δ, 0-K, 0-H
│   └── ...                     # see scripts/ for full list
├── tests/                      # pytest; smoke tests on synthetic fixtures
│   └── fixtures/               # 10 subjects x 100 CpGs x 50 genes synthetic data
├── notebooks/                  # EDA only — NOT canonical analysis
├── analysis/                   # dated analysis runs (config + log + outputs)
└── manuscript/
    ├── figures/                # code-generated figures bound to the paper
    └── supplementary/          # SI tables from analysis runs
```

**Key principle:** `src/` is importable library code. `scripts/` are runnable entry-points.
`analysis/` holds dated run directories. `notebooks/` are for exploration only.

---

## Running the Snakemake Pipeline

```bash
# Dry run (check DAG without executing)
snakemake --use-conda --cores 4 -n

# Run preprocessing phase
snakemake --use-conda --cores 4 preprocess_emory preprocess_best

# Run full pipeline (when all rules are implemented)
snakemake --use-conda --cores 4 all
```

---

## Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Hooks: ruff format + lint, mypy strict on `src/`, nbstripout, large-file guard (2 MB limit).

---

## Analysis Plan

**Execution-ready step-by-step plan: [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md).** One section per step (~30 steps across Phase 0 gates through Phase 4 deliverables), each with concrete inputs, methods, outputs, acceptance criteria, and Snakefile rule names. This is what a team member opens when they pick up a step.

Full v2.2 strategic plan (trajectory atlas methodology, Phase 0 gates, embedding arms,
conformal prediction, mediation analysis):

`04-projects/dnamrnaseq/2026-05-17-integrated-analysis-plan-v2.md` in the companion knowledge
vault (at `/home/llanc/claude-code/` on Lee's machine). Section 13 is the repo scaffold spec
this repo was built from. v2.2 is the *why*; `docs/ANALYSIS_PLAN.md` is the *how*.

**Key architectural choices (from Section 13):**
- `src/` layout forces editable installs; avoids the "passes locally, fails in CI" import trap.
- Snakemake DAG with `workflow/rules/` handles partial reruns and DAG-aware caching.
- `analysis/YYYY-MM-DD-slug/` dated run directories with committed config snapshots ensure
  every result is traceable to an exact code version + config + data snapshot.
- Manuscript `figures/` and `supplementary/` live in the repo (code-generated); manuscript prose
  stays in the vault (knowledge layer).

## CI Status

Two GitHub Actions workflows:
- `ci.yml` runs on every push/PR: ruff lint + format, mypy strict on `src/`, pytest on
  synthetic fixtures. Target: <5 min.
- `smoke-pipeline.yml` runs on push to main + weekly: Snakemake DAG parse + preprocessing
  stub rules on synthetic data. Target: <10 min.

---

## Data Access

Data is not in this repo. The canonical source is Lee's local OneDrive mount. See `config.yaml`
(copy from `config.yaml.example`) for the path configuration.

Files used:
- `emory.bVals.architecture.RData` — Emory DNAm beta values (architecture CpG subset)
- `best.bVals.architecture.RData` — BEST DNAm beta values (architecture CpG subset)
- `emory_pData2.RData` — Emory sample metadata (covariates, Response, visit labels)
- `best_pData2.RData` — BEST sample metadata

External cohorts (GSE98793, GTEx, AURORA) are downloaded separately when needed.

---

## Troubleshooting

**conda env create fails with conflicts (rpy2 + torch + transformers):**
The `cpuonly` constraint in `environment.yml` prevents PyTorch from pulling in CUDA libraries.
If you have a GPU, remove the `cpuonly` line and install the GPU variant of PyTorch after
conda env creation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Bioconductor packages fail to install (EpiDISH, CellDMC):**
Try installing manually within the activated conda env:
```r
BiocManager::install("EpiDISH")
```
EpiDISH ships CellDMC; installing EpiDISH is sufficient.

**pyreadr fails on pData2 files (ndarray-of-shape-(N,1) issue):**
Use the `rdata` fallback loader. The loaders in `src/dnamrnaseq2026/data/loaders.py`
try `pyreadr` first and fall back to `rdata` automatically.

**00_load_data.py: FileNotFoundError:**
Check `config.yaml` paths. The default assumes WSL2 with OneDrive mounted at `/mnt/d/`.
Adjust `data.emory_dnam_dir` if your mount path differs.

---

## License

MIT. See LICENSE.
