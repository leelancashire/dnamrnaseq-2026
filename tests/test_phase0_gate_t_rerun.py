"""Smoke tests for Gate 0-T re-run on cell-type-corrected Δ matrices.

Exercises ``src/dnamrnaseq2026/preprocessing/gate_t_rerun_celldmc.py``
end-to-end on a tiny synthetic 10-subject x 200-CpG (+ 100-gene) paired Δ
matrix. CI-safe; no OneDrive or R-Bioconductor dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Synthetic dimensions
N_SUBJECTS = 10
N_CPGS = 200
N_GENES = 100
N_CELL_TYPES = 6


@pytest.fixture
def synthetic_paired_delta_inputs():
    """Build PRE/POST sample matrices + Δ-cell-fractions for a 10-subject cohort.

    DNAm M-values (200 CpGs x 20 samples) and RNA log-CPM (100 genes x 20
    samples) are constructed with a deliberately injected R vs NR effect on a
    handful of CpGs (independent of cell composition) so the pipeline is
    actually exercising signal-detection logic.
    """
    rng = np.random.default_rng(7)

    # Build cell-type fractions per sample (PRE + POST stacked).
    n_samples = N_SUBJECTS * 2
    cell_types = ["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]
    cell_props_raw = rng.dirichlet(alpha=np.ones(N_CELL_TYPES) * 2.0, size=n_samples)
    sample_ids = [f"S{i:02d}_PRE" for i in range(N_SUBJECTS)] + [
        f"S{i:02d}_POST" for i in range(N_SUBJECTS)
    ]
    cell_props = pd.DataFrame(cell_props_raw, index=sample_ids, columns=cell_types)

    # DNAm M-values: random + small cell-composition coupling on CpGs 50-100,
    # plus a small R-vs-NR signal injected on the first 20 CpGs at POST visit.
    m_matrix = rng.normal(0.0, 1.0, size=(N_CPGS, n_samples))
    cell_loading = rng.normal(0.0, 0.5, size=(50, N_CELL_TYPES))
    m_matrix[50:100, :] += cell_loading @ cell_props.values.T

    # Response labels (5 R, 5 NR).
    response_array = np.array(["R"] * 5 + ["NR"] * 5)

    # Inject R-vs-NR signal on first 20 CpGs at POST visit only (Δ-signal).
    post_cols = np.arange(N_SUBJECTS, 2 * N_SUBJECTS)
    r_mask = response_array == "R"
    signal = np.zeros((20, N_SUBJECTS))
    signal[:, r_mask] = rng.normal(0.8, 0.2, size=(20, int(r_mask.sum())))
    signal[:, ~r_mask] = rng.normal(-0.8, 0.2, size=(20, int((~r_mask).sum())))
    m_matrix[:20, post_cols] += signal

    # RNA log-CPM: random + tiny cell-composition coupling. No deliberate signal.
    rna_matrix = rng.normal(4.0, 1.5, size=(N_GENES, n_samples))

    cpg_ids = [f"cg{i:08d}" for i in range(N_CPGS)]
    gene_ids = [f"GENE{i:05d}" for i in range(N_GENES)]
    paired_subject_ids = [f"S{i:02d}" for i in range(N_SUBJECTS)]
    pre_ids = [f"S{i:02d}_PRE" for i in range(N_SUBJECTS)]
    post_ids = [f"S{i:02d}_POST" for i in range(N_SUBJECTS)]

    delta_cell_props = cell_props.loc[post_ids].values - cell_props.loc[pre_ids].values
    delta_cell_props_df = pd.DataFrame(
        delta_cell_props, index=paired_subject_ids, columns=cell_types
    )

    response = pd.Series(response_array, index=paired_subject_ids, name="Response")

    return {
        "m_matrix": m_matrix,
        "rna_matrix": rna_matrix,
        "cpg_ids": cpg_ids,
        "gene_ids": gene_ids,
        "sample_ids": sample_ids,
        "pre_ids": pre_ids,
        "post_ids": post_ids,
        "paired_subject_ids": paired_subject_ids,
        "delta_cell_props_df": delta_cell_props_df,
        "response": response,
    }


class TestCorrectedPairedDelta:
    """Tests for ``build_corrected_paired_delta``."""

    def test_dnam_corrected_shape(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
        )

        inputs = synthetic_paired_delta_inputs
        corrected = build_corrected_paired_delta(
            feature_matrix=inputs["m_matrix"],
            feature_ids=inputs["cpg_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        assert corrected.shape == (N_SUBJECTS, N_CPGS)
        assert list(corrected.index) == inputs["paired_subject_ids"]
        assert list(corrected.columns) == inputs["cpg_ids"]

    def test_rna_corrected_shape(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
        )

        inputs = synthetic_paired_delta_inputs
        corrected = build_corrected_paired_delta(
            feature_matrix=inputs["rna_matrix"],
            feature_ids=inputs["gene_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        assert corrected.shape == (N_SUBJECTS, N_GENES)
        assert not corrected.isna().all().any()

    def test_length_mismatch_raises(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
        )

        inputs = synthetic_paired_delta_inputs
        with pytest.raises(ValueError):
            build_corrected_paired_delta(
                feature_matrix=inputs["m_matrix"],
                feature_ids=inputs["cpg_ids"],
                sample_ids_pre=inputs["pre_ids"],
                sample_ids_post=inputs["post_ids"][:-1],
                all_sample_ids=inputs["sample_ids"],
                delta_cell_props=inputs["delta_cell_props_df"],
                paired_subject_ids=inputs["paired_subject_ids"],
            )


class TestTopVarianceFilter:
    """Tests for ``select_top_variance_features``."""

    def test_keeps_requested_count(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
            select_top_variance_features,
        )

        inputs = synthetic_paired_delta_inputs
        corrected = build_corrected_paired_delta(
            feature_matrix=inputs["m_matrix"],
            feature_ids=inputs["cpg_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        reduced = select_top_variance_features(corrected, top_n=50)
        assert reduced.shape == (N_SUBJECTS, 50)

    def test_passthrough_when_top_n_exceeds_features(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
            select_top_variance_features,
        )

        inputs = synthetic_paired_delta_inputs
        corrected = build_corrected_paired_delta(
            feature_matrix=inputs["m_matrix"],
            feature_ids=inputs["cpg_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        reduced = select_top_variance_features(corrected, top_n=N_CPGS + 100)
        assert reduced.shape == corrected.shape


class TestJointCorrectedDelta:
    """Tests for ``build_joint_corrected_delta``."""

    def test_concatenation_and_scaling(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
            build_joint_corrected_delta,
            select_top_variance_features,
        )

        inputs = synthetic_paired_delta_inputs
        dnam_corr = select_top_variance_features(
            build_corrected_paired_delta(
                feature_matrix=inputs["m_matrix"],
                feature_ids=inputs["cpg_ids"],
                sample_ids_pre=inputs["pre_ids"],
                sample_ids_post=inputs["post_ids"],
                all_sample_ids=inputs["sample_ids"],
                delta_cell_props=inputs["delta_cell_props_df"],
                paired_subject_ids=inputs["paired_subject_ids"],
            ),
            top_n=30,
        )
        rna_corr = select_top_variance_features(
            build_corrected_paired_delta(
                feature_matrix=inputs["rna_matrix"],
                feature_ids=inputs["gene_ids"],
                sample_ids_pre=inputs["pre_ids"],
                sample_ids_post=inputs["post_ids"],
                all_sample_ids=inputs["sample_ids"],
                delta_cell_props=inputs["delta_cell_props_df"],
                paired_subject_ids=inputs["paired_subject_ids"],
            ),
            top_n=20,
        )
        joint = build_joint_corrected_delta(dnam_corr, rna_corr, scale=True)
        assert joint.shape == (N_SUBJECTS, 50)
        # After scaling each column should have ~0 mean.
        assert np.abs(joint.mean(axis=0)).max() < 1e-9


class TestRunGateTRerun:
    """End-to-end pipeline smoke test."""

    def test_full_pipeline_returns_verdict(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
            build_joint_corrected_delta,
            run_gate_t_rerun,
            select_top_variance_features,
        )

        inputs = synthetic_paired_delta_inputs

        dnam_corr = build_corrected_paired_delta(
            feature_matrix=inputs["m_matrix"],
            feature_ids=inputs["cpg_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        rna_corr = build_corrected_paired_delta(
            feature_matrix=inputs["rna_matrix"],
            feature_ids=inputs["gene_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        dnam_top = select_top_variance_features(dnam_corr, top_n=40)
        rna_top = select_top_variance_features(rna_corr, top_n=20)
        joint = build_joint_corrected_delta(dnam_top, rna_top, scale=True)

        result = run_gate_t_rerun(
            joint_corrected_delta=joint,
            response=inputs["response"],
            n_permutations=200,
            seed=42,
        )
        assert result["verdict"] in {"PASS", "MARGINAL", "FAIL"}
        assert 0.0 <= result["permanova"]["p_value"] <= 1.0
        assert result["n_subjects"] == N_SUBJECTS
        assert result["n_r"] + result["n_nr"] == N_SUBJECTS
        assert "PC1" in result["cohens_d_per_pc"]
        # Effective N preserved under residualisation: every subject with an
        # R/NR label survives into the PCA stage.
        assert result["pc_scores"].shape[0] == N_SUBJECTS

    def test_no_response_labels_raises(self, synthetic_paired_delta_inputs):
        from dnamrnaseq2026.preprocessing.gate_t_rerun_celldmc import (
            build_corrected_paired_delta,
            build_joint_corrected_delta,
            run_gate_t_rerun,
        )

        inputs = synthetic_paired_delta_inputs
        dnam_corr = build_corrected_paired_delta(
            feature_matrix=inputs["m_matrix"],
            feature_ids=inputs["cpg_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        rna_corr = build_corrected_paired_delta(
            feature_matrix=inputs["rna_matrix"],
            feature_ids=inputs["gene_ids"],
            sample_ids_pre=inputs["pre_ids"],
            sample_ids_post=inputs["post_ids"],
            all_sample_ids=inputs["sample_ids"],
            delta_cell_props=inputs["delta_cell_props_df"],
            paired_subject_ids=inputs["paired_subject_ids"],
        )
        joint = build_joint_corrected_delta(dnam_corr, rna_corr, scale=True)

        # All labels NaN: pipeline should raise informative error.
        empty_response = pd.Series(
            [None] * N_SUBJECTS, index=inputs["paired_subject_ids"], dtype=object
        )
        with pytest.raises(ValueError, match="Fewer than 10 paired subjects"):
            run_gate_t_rerun(
                joint_corrected_delta=joint,
                response=empty_response,
                n_permutations=50,
                seed=42,
            )
