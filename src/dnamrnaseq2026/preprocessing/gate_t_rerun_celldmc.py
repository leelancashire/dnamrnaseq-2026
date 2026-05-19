"""Gate 0-T re-run on cell-type-corrected paired-delta matrices.

Re-runs the Gate 0-T machinery (PERMANOVA on PC scores + per-PC t-tests +
Cohen's d) using cell-type-corrected paired-delta matrices instead of raw
delta-M-value / delta-logCPM matrices.

Why this exists
---------------

[FACT, A] Gate 0-T (raw paired-delta PCA on Emory n=164) returned MARGINAL on
2026-05-17: PERMANOVA p=0.111 (2000 perms, seed=42), max Cohen's d=0.267 (PC2),
PC1 captures 15.7% of variance. See
``analysis/2026-05-17-phase-0/0-T/gate_0T_results.json`` and the project note
``04-projects/dnamrnaseq/2026-05-17-phase-0-results.md``.

[INFERENCE, B] The leading PC of the raw paired-Δ space is plausibly dominated
by within-subject changes in blood cell composition (Neu, CD4T, Mono, etc.)
between PRE and POST visits. Treatment-response signal (R vs NR) lives on a
direction that is at least partly orthogonal to cell-composition drift. If so,
residualising the per-CpG and per-gene Δ-values on Δ-cell-fractions (CellDMC
adjustment style) should remove the cell-composition variance from the PC basis
and unmask the treatment-response signal. This is the rescue hypothesis stated
in the Phase 0 results note and ANALYSIS_PLAN.md Step 0-T MARGINAL recovery
path.

[FACT, B] Phase 1 step 1.2 already runs a tightly-scoped instance of this
analysis as ``rescue_check_1_2_5``: it residualises the joint Δ-matrices on
Δ-cell-fractions, runs PCA, and produces PERMANOVA + Cohen's d. The
``2026-05-17-phase-1/1.2/rescue_1_2_5.json`` artefact reports the rescue
verdict. As of the Phase 1 PR #4 merge (2026-05-19), that verdict is MARGINAL
(p=0.107, max d=0.272).

[INFERENCE, B] The current ``rescue_check_1_2_5`` is a self-contained
in-script check inside Step 1.2; it does not produce a gate-style artefact set
(loadings CSV, PCA arrow figure, gate-verdict-formatted JSON, results.md). It
also does not provide a Hotelling's T-squared block to mirror the original
gate. This module wraps the same statistical machinery into a standalone
gate-0-T re-run pipeline that produces a drop-in replacement artefact set in
``analysis/2026-05-17-phase-0/gate_t_rerun_celldmc/`` and applies the canonical
Gate 0-T thresholds (PASS p<0.05 AND max d>=0.30; MARGINAL p in [0.05, 0.15];
FAIL otherwise). This is the formal Gate 0-T re-run, not the in-line Phase 1
sanity check.

Methodological choices
----------------------

1. **PC basis is re-derived from the corrected Δ-matrix, not projected onto the
   original (raw-Δ) basis.** [INFERENCE, B] Justification: the original PC basis
   was contaminated by cell-type-proportion variance, which is exactly the
   signal being removed. Projecting onto the contaminated basis would defeat the
   purpose of correction. The new PC basis represents directions of variance in
   the corrected space.

2. **PERMANOVA: same B=2000 permutations, seed=42 as the original Gate 0-T.**
   [FACT, A] Identical permutation budget to enable like-for-like comparison of
   p-values between raw-Δ and corrected-Δ runs.

3. **Permutation invariance.** [INFERENCE, B] Residualisation on cell fractions
   is a deterministic linear-OLS pre-processing step applied identically to all
   subjects' Δ-vectors. Under the null (no R vs NR signal in residualised
   space), label permutation remains valid: the residualised Δ-matrix is
   exchangeable across subjects once labels are scrambled, because the
   residualisation operator is label-free (depends on cell fractions, not on
   R/NR). Effective N for the permutation test is unchanged at N=164 (or the
   paired-subject count actually achievable).

4. **Sample size and PC retention.** [FACT, A] The PCA is fit with
   ``n_components = min(5, n_subjects - 1, n_features)``. For Emory with 164
   paired subjects and ~7000 joint features, this is 5 components. Cohen's d is
   computed per PC and the max-over-PCs is taken as the test statistic for the
   verdict thresholds, matching the original Gate 0-T convention.

5. **Multiple testing on per-PC t-tests.** [INFERENCE, B] The original Gate 0-T
   reports raw per-PC t-test p-values without correction; the PERMANOVA is the
   primary multivariate test and per-PC t-tests are descriptive. This module
   follows the same convention. If a future deliverable claims any individual
   PC as significant, Bonferroni or Holm correction across the 5 PCs would be
   the minimum acceptable adjustment.

6. **Subject-ID-based alignment (Kai pre-condition).** [FACT, A] Kai's PR #4
   post-merge review (2026-05-19) flagged a positional-slicing risk in
   ``scripts/12_phase1_celldmc.py`` where ``pdata_paired`` is constructed by
   slicing ``pre_ids_rna`` by length. This module avoids that pattern: every
   paired-Δ row is keyed on the subject Subcode via explicit dict lookups
   (``sc_idx``, ``sc_rna_idx`` in the entry-point script) and the
   ``common_subjects`` intersection enforces subject-ID-based join semantics.
   The corrected Δ-matrices, Δ-cell-fractions, and response Series are all
   row-indexed on the same paired subject IDs in this module's public API
   (``build_corrected_paired_delta`` takes ``paired_subject_ids`` as the row
   index and ``delta_cell_props.loc[paired_subject_ids]`` enforces ID-keyed
   alignment, not positional).

Inputs and dependencies
-----------------------

Hard dependencies:

- ``src/dnamrnaseq2026/preprocessing/cell_type_correction.py``:
  ``residualise_on_cell_props``, ``beta_to_m``. Available on ``main`` as of
  Phase 1 PR #4 merge (2026-05-19).
- ``src/dnamrnaseq2026/preprocessing/delta_construction.py``:
  ``filter_paired_ids``, ``filter_paired_ids_rna``, ``build_paired_delta``.
  Available on ``main`` after the same merge.

Input artefacts consumed at runtime (NOT yet present on main; produced by
Phase 1 step 1.1 on a real-data run; staged for execution after the next
Phase 1 real-data run):

- ``analysis/latest/cell_props_emory.csv``: EpiDISH 7-cell fractions for the
  Emory PRE+POST samples, indexed by sample ID and aligned to the columns of
  the bVals + RNA-seq matrices.
- ``analysis/latest/pdata_emory_with_epidish.csv``: pData2 augmented with
  EpiDISH fractions, used for the sex/age/PC1-6 nuisance covariates.

The raw bVals and RNA-seq matrices are loaded via the standard
``dnamrnaseq2026.data.loaders`` interface (same as the original Gate 0-T
script). The corrected Δ-matrices are computed in-memory by residualising
per-CpG Δ-M-values and per-gene Δ-log-CPM on Δ-cell-fractions.

Acceptance thresholds (mirror Gate 0-T canonical thresholds)
------------------------------------------------------------

[FACT, A] ``PERMANOVA_PASS_THRESHOLD = 0.05``,
``PERMANOVA_MARGINAL_THRESHOLD = 0.15``, ``COHENS_D_THRESHOLD = 0.30``,
``N_PCS = 5``, ``BOOTSTRAP_N = 2000``. Identical to the original
``gate_t_pca.py`` constants.

Verdict definitions:

- ``PASS``: PERMANOVA p < 0.05 AND max per-PC Cohen's d >= 0.30
- ``MARGINAL``: 0.05 <= PERMANOVA p < 0.15, OR p < 0.05 with max d < 0.30
- ``FAIL``: PERMANOVA p >= 0.15

Reviewer
--------

Primary: Kai (ran original Gate 0-C EpiDISH gate, owns CellDMC plumbing).
Secondary (optional): Tobias (PERMANOVA permutation-invariance check if
flagged).

References
----------

- ``analysis/2026-05-17-phase-0/0-T/results.md`` (original Gate 0-T verdict)
- ``analysis/2026-05-17-phase-1/1.2/results.md`` (in-line rescue_check_1_2_5)
- ``04-projects/dnamrnaseq/2026-05-17-phase-0-results.md`` (recovery plan)
- ``docs/ANALYSIS_PLAN.md`` Step 0-T MARGINAL recovery section
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Acceptance thresholds: identical to the original Gate 0-T (ANALYSIS_PLAN.md
# Step 0-T) to allow direct comparison of raw-Δ vs corrected-Δ verdicts.
PERMANOVA_PASS_THRESHOLD = 0.05
PERMANOVA_MARGINAL_THRESHOLD = 0.15
COHENS_D_THRESHOLD = 0.30
N_PCS = 5
BOOTSTRAP_N = 2000


def build_corrected_paired_delta(
    feature_matrix: np.ndarray[Any, Any],
    feature_ids: list[str],
    sample_ids_pre: list[str],
    sample_ids_post: list[str],
    all_sample_ids: list[str],
    delta_cell_props: pd.DataFrame,
    paired_subject_ids: list[str],
) -> pd.DataFrame:
    """Build a paired-Δ matrix with per-feature residualisation on Δ-cell-fractions.

    Pipeline:
      1. Subset the per-sample feature matrix to PRE / POST positions.
      2. Compute Δ = POST - PRE per subject.
      3. Residualise each row (feature) on Δ-cell-fractions via OLS.

    The result is a (n_paired_subjects x n_features) DataFrame with the
    cell-type-proportion variance regressed out of each feature.

    Parameters
    ----------
    feature_matrix:
        2-D numpy array of shape (n_features, n_samples). For DNAm this is the
        M-value matrix; for RNA-seq this is the log-CPM matrix.
    feature_ids:
        Row labels for ``feature_matrix`` (CpG IDs or gene IDs).
    sample_ids_pre, sample_ids_post:
        Lists of PRE-visit and POST-visit sample identifiers, aligned by
        subject (i.e. ``sample_ids_pre[i]`` and ``sample_ids_post[i]`` belong to
        the same subject).
    all_sample_ids:
        Column labels for ``feature_matrix`` (must contain every ID listed in
        ``sample_ids_pre`` and ``sample_ids_post``).
    delta_cell_props:
        DataFrame of Δ-cell-fractions (POST - PRE), indexed by paired-subject
        ID, with one column per cell type.
    paired_subject_ids:
        Subject identifiers in the same order as ``sample_ids_pre`` /
        ``sample_ids_post``, used as the row index of the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Shape (n_paired_subjects, n_features). Cell-type-corrected paired-Δ
        values; column index is ``feature_ids``, row index is
        ``paired_subject_ids``.
    """
    from .cell_type_correction import residualise_on_cell_props

    if len(sample_ids_pre) != len(sample_ids_post):
        raise ValueError(
            "sample_ids_pre and sample_ids_post must be the same length "
            f"(got {len(sample_ids_pre)} vs {len(sample_ids_post)})."
        )
    if len(sample_ids_pre) != len(paired_subject_ids):
        raise ValueError(
            "paired_subject_ids must align with sample_ids_pre / "
            f"sample_ids_post (got {len(paired_subject_ids)} vs "
            f"{len(sample_ids_pre)})."
        )

    sample_index = {sid: i for i, sid in enumerate(all_sample_ids)}
    pre_pos = [sample_index[s] for s in sample_ids_pre]
    post_pos = [sample_index[s] for s in sample_ids_post]

    delta = feature_matrix[:, post_pos] - feature_matrix[:, pre_pos]

    aligned_props = delta_cell_props.loc[paired_subject_ids]
    corrected = residualise_on_cell_props(delta, aligned_props, paired_subject_ids)

    # residualise_on_cell_props returns (n_features, n_subjects); transpose to
    # (n_subjects, n_features) so subjects index rows for downstream PCA.
    corrected_df = pd.DataFrame(
        corrected.T, index=paired_subject_ids, columns=feature_ids
    )
    return corrected_df


def select_top_variance_features(
    corrected_delta: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    """Keep the ``top_n`` features with highest variance across subjects.

    Mirrors the variance filter applied in the original Gate 0-T
    (``build_dnam_delta_matrix`` / ``build_rnaseq_delta_matrix``).

    Parameters
    ----------
    corrected_delta:
        DataFrame of shape (n_subjects, n_features).
    top_n:
        Number of features to keep.

    Returns
    -------
    pd.DataFrame
        Reduced to (n_subjects, min(top_n, n_features)) columns.
    """
    n_features = corrected_delta.shape[1]
    if n_features <= top_n:
        return corrected_delta
    variances = corrected_delta.var(axis=0).fillna(0.0)
    top_features = variances.nlargest(top_n).index
    return corrected_delta[top_features]


def build_joint_corrected_delta(
    corrected_dnam_delta: pd.DataFrame,
    corrected_rna_delta: pd.DataFrame,
    scale: bool = True,
) -> pd.DataFrame:
    """Concatenate corrected DNAm and RNA-seq Δ-matrices, optionally column-scaling.

    Mirrors ``build_joint_delta_matrix`` from ``delta_construction.py`` but on
    cell-type-corrected inputs. The two input DataFrames must share their row
    index (paired-subject IDs).

    Parameters
    ----------
    corrected_dnam_delta:
        Shape (n_paired, n_cpgs).
    corrected_rna_delta:
        Shape (n_paired, n_genes).
    scale:
        If True (default), z-score each column to mean 0 and SD 1. Matches the
        original Gate 0-T preprocessing.

    Returns
    -------
    pd.DataFrame
        Shape (n_paired, n_cpgs + n_genes).
    """
    shared_index = corrected_dnam_delta.index.intersection(corrected_rna_delta.index)
    dnam = corrected_dnam_delta.loc[shared_index].copy()
    rna = corrected_rna_delta.loc[shared_index].copy()
    joint = pd.concat([dnam, rna], axis=1)

    # NaN imputation: column mean (matches build_joint_delta_matrix behaviour
    # used in the original gate, and is also what rescue_check_1_2_5 does).
    col_means = joint.mean(axis=0)
    joint = joint.fillna(col_means).fillna(0.0)

    if scale:
        col_std = joint.std(axis=0, ddof=0)
        col_std = col_std.where(col_std > 0, 1.0)
        joint = (joint - joint.mean(axis=0)) / col_std

    return joint


def run_gate_t_rerun(
    joint_corrected_delta: pd.DataFrame,
    response: pd.Series,
    n_permutations: int = BOOTSTRAP_N,
    seed: int = 42,
    n_components: int = N_PCS,
) -> dict[str, Any]:
    """Run the full Gate 0-T pipeline on a corrected joint Δ-matrix.

    Steps: PCA -> per-PC Cohen's d -> PERMANOVA -> per-PC t-tests + Hotelling's
    T-squared -> verdict.

    Re-uses the original Gate 0-T helpers in ``gate_t_pca.py`` for PCA,
    Cohen's d, PERMANOVA, and Hotelling's T-squared, so the statistical
    machinery is identical to the raw-Δ run; only the input matrix differs.

    Parameters
    ----------
    joint_corrected_delta:
        DataFrame (n_subjects, n_features), already cell-type-corrected and
        column-scaled. Row index = subject IDs.
    response:
        Response labels (``R`` / ``NR``) indexed by subject ID.
    n_permutations:
        Number of permutations for the PERMANOVA test (default 2000, matching
        the original Gate 0-T).
    seed:
        Random seed for the permutation test (default 42).
    n_components:
        Number of principal components to retain (default 5).

    Returns
    -------
    dict
        Keys:
          - ``verdict``: ``PASS`` / ``MARGINAL`` / ``FAIL``
          - ``pc_scores``: DataFrame of PC scores
          - ``pca``: fitted ``sklearn.decomposition.PCA`` object
          - ``permanova``: dict with ``f_statistic``, ``p_value``,
            ``n_permutations``
          - ``cohens_d_per_pc``: dict[str, float]
          - ``hotelling``: dict with ``per_pc_t_p`` and ``hotelling_p``
          - ``n_subjects``, ``n_r``, ``n_nr``, ``n_features``,
            ``explained_variance_ratio``
    """
    from .gate_t_pca import (
        compute_cohens_d_per_pc,
        determine_gate_0t_verdict,
        run_hotelling_t2,
        run_pca,
        run_permanova,
    )

    aligned_response = response.reindex(joint_corrected_delta.index)
    valid_mask = aligned_response.isin(["R", "NR"])
    if int(valid_mask.sum()) < 10:
        raise ValueError(
            "Fewer than 10 paired subjects with R/NR labels after alignment; "
            "PERMANOVA is not interpretable. Check the response Series."
        )
    joint_valid = joint_corrected_delta.loc[valid_mask[valid_mask].index]
    response_valid = aligned_response.loc[joint_valid.index]

    pc_scores, pca = run_pca(joint_valid, n_components=n_components)
    cohens_d = compute_cohens_d_per_pc(pc_scores, response_valid)
    permanova = run_permanova(
        pc_scores, response_valid, n_permutations=n_permutations, seed=seed
    )
    hotelling = run_hotelling_t2(pc_scores, response_valid)
    verdict = determine_gate_0t_verdict(permanova, cohens_d)

    n_r = int((response_valid == "R").sum())
    n_nr = int((response_valid == "NR").sum())

    logger.info(
        "Gate 0-T re-run verdict=%s, PERMANOVA p=%.4f, max d=%.3f, n=%d (R=%d, NR=%d)",
        verdict,
        permanova["p_value"],
        max(cohens_d.values()) if cohens_d else 0.0,
        len(joint_valid),
        n_r,
        n_nr,
    )

    return {
        "verdict": verdict,
        "pc_scores": pc_scores,
        "pca": pca,
        "permanova": permanova,
        "cohens_d_per_pc": cohens_d,
        "hotelling": hotelling,
        "n_subjects": int(len(joint_valid)),
        "n_r": n_r,
        "n_nr": n_nr,
        "n_features": int(joint_valid.shape[1]),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
