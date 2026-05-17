"""Synthetic-fixture smoke tests for Phase 0 gate modules.

These tests do NOT require OneDrive access and must pass in CI.
They exercise the core logic of each gate using small synthetic DataFrames.

Gates tested:
  - 0-T: delta construction, PCA, statistical tests
  - 0-C: cell-type deconvolution validation
  - 0-S: source-domain classifier harmonisation
  - 0-X: cross-disorder centroid projection (harmonisation + centroid logic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data dimensions (tiny: CI-safe, no OneDrive access needed)
# ---------------------------------------------------------------------------
N_CPGS = 200
N_GENES = 100
N_SUBJECTS = 20  # 10 paired, 10 with pre only (mix)
N_PAIRED = 10
N_GSE_SAMPLES = 40
N_GSE_GENES = 80


@pytest.fixture
def synthetic_bvals():
    """Beta values (N_CPGS x N_PAIRED*2 samples), paired PRE/POST."""
    rng = np.random.default_rng(10)
    data = rng.uniform(0.05, 0.95, size=(N_CPGS, N_PAIRED * 2))
    cpg_ids = [f"cg{i:08d}" for i in range(N_CPGS)]
    # Paired: e.g. subj01_PRE, subj01_POST
    sample_ids = []
    for i in range(N_PAIRED):
        sample_ids.append(f"subj{i:02d}_PRE")
        sample_ids.append(f"subj{i:02d}_POST")
    return pd.DataFrame(data, index=cpg_ids, columns=sample_ids)


@pytest.fixture
def synthetic_rnaseq():
    """Log-CPM matrix (N_GENES x N_PAIRED*2 samples)."""
    rng = np.random.default_rng(11)
    data = rng.normal(4.0, 1.5, size=(N_GENES, N_PAIRED * 2))
    gene_ids = [f"GENE{i:05d}" for i in range(N_GENES)]
    sample_ids = []
    for i in range(N_PAIRED):
        sample_ids.append(f"subj{i:02d}_PRE")
        sample_ids.append(f"subj{i:02d}_POST")
    return pd.DataFrame(data, index=gene_ids, columns=sample_ids)


@pytest.fixture
def synthetic_subject_data():
    """Subject metadata with paired PRE/POST samples and R/NR labels."""
    rows = []
    for i in range(N_PAIRED):
        response = "R" if i < N_PAIRED // 2 else "NR"
        rows.append({
            "Subcode": f"subj{i:02d}",
            "Visit": "PRE-IOP",
            "Response": response,
            "SampleName_DNAm": f"subj{i:02d}_PRE",
            "SampleName_RNASeq": f"subj{i:02d}_PRE",
        })
        rows.append({
            "Subcode": f"subj{i:02d}",
            "Visit": "POST-IOP",
            "Response": response,
            "SampleName_DNAm": f"subj{i:02d}_POST",
            "SampleName_RNASeq": f"subj{i:02d}_POST",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_pdata(synthetic_bvals):
    """pData2 indexed by SampleName with cell-type fractions and N2LR."""
    rng = np.random.default_rng(13)
    sample_ids = list(synthetic_bvals.columns)
    n = len(sample_ids)
    bcell = rng.uniform(0.03, 0.10, n)
    cd4t = rng.uniform(0.10, 0.25, n)
    cd8t = rng.uniform(0.05, 0.15, n)
    mono = rng.uniform(0.05, 0.20, n)
    neu = rng.uniform(0.40, 0.70, n)
    nk = rng.uniform(0.02, 0.08, n)
    lymph = bcell + cd4t + cd8t + nk
    n2lr = neu / np.maximum(lymph, 1e-6)
    df = pd.DataFrame({
        "Bcell": bcell,
        "CD4T": cd4t,
        "CD8T": cd8t,
        "Mono": mono,
        "Neu": neu,
        "NK": nk,
        "N2LR": n2lr,
        "Response": ["R"] * (n // 2) + ["NR"] * (n - n // 2),
        "Visit": ["PRE-IOP", "POST-IOP"] * (n // 2),
    }, index=sample_ids)
    return df


# ---------------------------------------------------------------------------
# Gate 0-T: delta construction and PCA
# ---------------------------------------------------------------------------


class TestDeltaConstruction:
    """Tests for delta_construction.py."""

    def test_beta_to_mvalue_range(self):
        """M-values are clipped to [-3, 3]."""
        from dnamrnaseq2026.preprocessing.delta_construction import beta_to_mvalue

        beta = np.array([0.001, 0.5, 0.999, 0.0, 1.0])
        m = beta_to_mvalue(beta)
        assert m.min() >= -3.0
        assert m.max() <= 3.0

    def test_identify_paired_subjects(self, synthetic_subject_data):
        """Paired subject identification returns correct count."""
        from dnamrnaseq2026.preprocessing.delta_construction import identify_paired_subjects

        paired = identify_paired_subjects(synthetic_subject_data)
        assert len(paired) == N_PAIRED
        assert set(paired["Response"].unique()).issubset({"R", "NR"})

    def test_build_dnam_delta_matrix(self, synthetic_bvals, synthetic_subject_data):
        """Delta-M matrix shape is (n_paired, n_cpgs)."""
        from dnamrnaseq2026.preprocessing.delta_construction import build_dnam_delta_matrix

        delta = build_dnam_delta_matrix(
            synthetic_bvals,
            synthetic_subject_data,
            top_n_cpgs=50,
        )
        assert delta.shape[0] == N_PAIRED
        assert delta.shape[1] <= 50
        assert not delta.isnull().any().any()

    def test_build_rnaseq_delta_matrix(self, synthetic_rnaseq, synthetic_subject_data):
        """Delta-logCPM matrix shape is (n_paired, n_genes)."""
        from dnamrnaseq2026.preprocessing.delta_construction import build_rnaseq_delta_matrix

        delta = build_rnaseq_delta_matrix(
            synthetic_rnaseq,
            synthetic_subject_data,
            top_n_genes=30,
        )
        assert delta.shape[0] == N_PAIRED
        assert delta.shape[1] <= 30

    def test_build_joint_delta_matrix(
        self, synthetic_bvals, synthetic_rnaseq, synthetic_subject_data
    ):
        """Joint delta matrix is scaled and has correct shape."""
        from dnamrnaseq2026.preprocessing.delta_construction import (
            build_dnam_delta_matrix,
            build_joint_delta_matrix,
            build_rnaseq_delta_matrix,
        )

        dnam_delta = build_dnam_delta_matrix(
            synthetic_bvals, synthetic_subject_data, top_n_cpgs=20
        )
        rna_delta = build_rnaseq_delta_matrix(
            synthetic_rnaseq, synthetic_subject_data, top_n_genes=10
        )
        joint = build_joint_delta_matrix(dnam_delta, rna_delta, scale=True)
        assert joint.shape[1] == 30
        assert joint.shape[0] == N_PAIRED
        # Scaled: column means should be near 0
        assert abs(joint.mean(axis=0).mean()) < 1e-10


class TestGateTPCA:
    """Tests for gate_t_pca.py."""

    def test_run_pca_shape(self, synthetic_bvals, synthetic_rnaseq, synthetic_subject_data):
        """PCA returns correct number of components."""
        from dnamrnaseq2026.preprocessing.delta_construction import (
            build_dnam_delta_matrix,
            build_joint_delta_matrix,
            build_rnaseq_delta_matrix,
        )
        from dnamrnaseq2026.preprocessing.gate_t_pca import run_pca

        dnam_delta = build_dnam_delta_matrix(
            synthetic_bvals, synthetic_subject_data, top_n_cpgs=20
        )
        rna_delta = build_rnaseq_delta_matrix(
            synthetic_rnaseq, synthetic_subject_data, top_n_genes=10
        )
        joint = build_joint_delta_matrix(dnam_delta, rna_delta, scale=True)
        scores, pca = run_pca(joint, n_components=5)
        assert scores.shape[0] == N_PAIRED
        assert scores.shape[1] <= 5
        assert pca.n_components_ >= 1

    def test_cohens_d_is_non_negative(
        self, synthetic_bvals, synthetic_rnaseq, synthetic_subject_data
    ):
        """Cohen's d values are non-negative floats."""
        from dnamrnaseq2026.preprocessing.delta_construction import (
            build_dnam_delta_matrix,
            build_joint_delta_matrix,
            build_rnaseq_delta_matrix,
            identify_paired_subjects,
        )
        from dnamrnaseq2026.preprocessing.gate_t_pca import (
            compute_cohens_d_per_pc,
            run_pca,
        )

        dnam_delta = build_dnam_delta_matrix(
            synthetic_bvals, synthetic_subject_data, top_n_cpgs=20
        )
        rna_delta = build_rnaseq_delta_matrix(
            synthetic_rnaseq, synthetic_subject_data, top_n_genes=10
        )
        joint = build_joint_delta_matrix(dnam_delta, rna_delta, scale=True)
        scores, _ = run_pca(joint, n_components=3)

        paired = identify_paired_subjects(synthetic_subject_data)
        response = paired.set_index("Subcode")["Response"]
        d = compute_cohens_d_per_pc(scores, response)
        for v in d.values():
            assert v >= 0.0

    def test_permanova_returns_p(
        self, synthetic_bvals, synthetic_rnaseq, synthetic_subject_data
    ):
        """PERMANOVA returns a p-value in [0, 1]."""
        from dnamrnaseq2026.preprocessing.delta_construction import (
            build_dnam_delta_matrix,
            build_joint_delta_matrix,
            build_rnaseq_delta_matrix,
            identify_paired_subjects,
        )
        from dnamrnaseq2026.preprocessing.gate_t_pca import run_pca, run_permanova

        dnam_delta = build_dnam_delta_matrix(
            synthetic_bvals, synthetic_subject_data, top_n_cpgs=20
        )
        rna_delta = build_rnaseq_delta_matrix(
            synthetic_rnaseq, synthetic_subject_data, top_n_genes=10
        )
        joint = build_joint_delta_matrix(dnam_delta, rna_delta, scale=True)
        scores, _ = run_pca(joint, n_components=3)

        paired = identify_paired_subjects(synthetic_subject_data)
        response = paired.set_index("Subcode")["Response"]
        result = run_permanova(scores, response, n_permutations=50, seed=42)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Gate 0-C: cell-type deconvolution validation
# ---------------------------------------------------------------------------


class TestGateC:
    """Tests for cell_type_deconv.py."""

    def test_compute_n2lr_uses_existing(self, synthetic_pdata):
        """compute_n2lr returns existing N2LR column if present."""
        from dnamrnaseq2026.preprocessing.cell_type_deconv import compute_n2lr

        n2lr = compute_n2lr(synthetic_pdata)
        assert len(n2lr) == len(synthetic_pdata)
        assert (n2lr > 0).all()

    def test_validate_delta_cell_fractions(self, synthetic_pdata, synthetic_subject_data):
        """Validation runs and returns expected keys."""
        from dnamrnaseq2026.preprocessing.cell_type_deconv import validate_delta_cell_fractions

        results = validate_delta_cell_fractions(
            pdata=synthetic_pdata,
            subject_data=synthetic_subject_data,
        )
        assert "validation_2_pass" in results
        assert "validation_3_pass" in results
        assert "mono_n2lr_r" in results
        assert isinstance(results["mono_n2lr_r"], float)
        assert 0 < results["n_paired"] <= N_PAIRED

    def test_verdict_pass(self, synthetic_pdata, synthetic_subject_data):
        """determine_gate_0c_verdict returns a string."""
        from dnamrnaseq2026.preprocessing.cell_type_deconv import (
            determine_gate_0c_verdict,
            validate_delta_cell_fractions,
        )

        results = validate_delta_cell_fractions(
            pdata=synthetic_pdata,
            subject_data=synthetic_subject_data,
        )
        verdict = determine_gate_0c_verdict(results)
        assert verdict in {"PASS", "MARGINAL", "FAIL"}


# ---------------------------------------------------------------------------
# Gate 0-S: source-domain classifier
# ---------------------------------------------------------------------------


class TestGateS:
    """Tests for covariate_shift.py."""

    @pytest.fixture
    def emory_best_deltas(self, synthetic_bvals, synthetic_rnaseq, synthetic_subject_data):
        """Build small Emory and BEST delta matrices with shared features."""
        from dnamrnaseq2026.preprocessing.delta_construction import (
            build_dnam_delta_matrix,
            build_joint_delta_matrix,
            build_rnaseq_delta_matrix,
        )

        rng = np.random.default_rng(20)
        # BEST: slightly shifted bVals and rnaseq
        best_bvals = pd.DataFrame(
            rng.uniform(0.1, 0.9, size=synthetic_bvals.shape),
            index=synthetic_bvals.index,
            columns=synthetic_bvals.columns,
        )
        best_rnaseq = pd.DataFrame(
            synthetic_rnaseq.values + rng.normal(0.5, 0.3, size=synthetic_rnaseq.shape),
            index=synthetic_rnaseq.index,
            columns=synthetic_rnaseq.columns,
        )
        emory_dnam = build_dnam_delta_matrix(
            synthetic_bvals, synthetic_subject_data, top_n_cpgs=20
        )
        emory_rna = build_rnaseq_delta_matrix(
            synthetic_rnaseq, synthetic_subject_data, top_n_genes=10
        )
        emory_joint = build_joint_delta_matrix(emory_dnam, emory_rna, scale=True)

        best_dnam = build_dnam_delta_matrix(best_bvals, synthetic_subject_data, top_n_cpgs=20)
        best_rna = build_rnaseq_delta_matrix(best_rnaseq, synthetic_subject_data, top_n_genes=10)
        best_joint = build_joint_delta_matrix(best_dnam, best_rna, scale=True)
        return emory_joint, best_joint

    def test_harmonise_feature_sets(self, emory_best_deltas):
        """Harmonised matrices have same columns."""
        from dnamrnaseq2026.preprocessing.covariate_shift import harmonise_feature_sets

        emory, best = emory_best_deltas
        emory_h, best_h = harmonise_feature_sets(emory, best, min_features=1)
        assert set(emory_h.columns) == set(best_h.columns)

    def test_classifier_returns_auc(self, emory_best_deltas):
        """Source-domain classifier returns AUC in [0, 1]."""
        from dnamrnaseq2026.preprocessing.covariate_shift import (
            harmonise_feature_sets,
            train_source_domain_classifier,
        )

        emory, best = emory_best_deltas
        emory_h, best_h = harmonise_feature_sets(emory, best, min_features=1)
        results = train_source_domain_classifier(
            emory_h, best_h, seed=42, n_jobs=1
        )
        assert 0.0 <= results["lr_mean_auc"] <= 1.0
        assert 0.0 <= results["rf_mean_auc"] <= 1.0

    def test_verdict_is_valid(self):
        """determine_gate_0s_verdict returns valid string."""
        from dnamrnaseq2026.preprocessing.covariate_shift import determine_gate_0s_verdict

        assert determine_gate_0s_verdict(0.60) == "PASS"
        assert determine_gate_0s_verdict(0.80) == "MARGINAL"
        assert determine_gate_0s_verdict(0.90) == "FAIL"


# ---------------------------------------------------------------------------
# Gate 0-X: cross-disorder centroid projection
# ---------------------------------------------------------------------------


class TestGateX:
    """Tests for cross_disorder_centroid.py."""

    @pytest.fixture
    def synthetic_gse_expr(self, synthetic_rnaseq):
        """Synthetic GSE expression matrix with overlapping gene IDs."""
        rng = np.random.default_rng(30)
        # Use same gene IDs as rnaseq for overlap
        data = rng.normal(5.0, 2.0, size=(N_GENES, N_GSE_SAMPLES))
        sample_ids = [f"GSE_samp_{i:03d}" for i in range(N_GSE_SAMPLES)]
        return pd.DataFrame(data, index=synthetic_rnaseq.index, columns=sample_ids)

    def test_harmonise_expression_matrices(self, synthetic_rnaseq, synthetic_gse_expr):
        """Harmonised matrices have same gene index (intersection)."""
        from dnamrnaseq2026.external_projection.cross_disorder_centroid import (
            harmonise_expression_matrices,
        )

        emory_norm, gse_norm = harmonise_expression_matrices(synthetic_rnaseq, synthetic_gse_expr)
        assert set(emory_norm.index) == set(gse_norm.index)
        assert emory_norm.shape[1] == synthetic_rnaseq.shape[1]
        assert gse_norm.shape[1] == synthetic_gse_expr.shape[1]

    def test_compute_centroids(self, synthetic_rnaseq, synthetic_gse_expr, synthetic_subject_data):
        """Centroids are computed for each group."""
        from dnamrnaseq2026.external_projection.cross_disorder_centroid import (
            compute_centroids,
            harmonise_expression_matrices,
        )

        # Build Emory baseline matrix (PRE samples only)
        pre_cols = [c for c in synthetic_rnaseq.columns if c.endswith("_PRE")]
        emory_pre = synthetic_rnaseq[pre_cols]

        # Build response Series indexed by PRE sample column
        resp_map = synthetic_subject_data[
            synthetic_subject_data["Visit"] == "PRE-IOP"
        ].set_index("SampleName_RNASeq")["Response"]

        emory_norm, gse_norm = harmonise_expression_matrices(emory_pre, synthetic_gse_expr)

        # Build TRD + control masks (first half TRD, second half controls)
        n_gse = gse_norm.shape[1]
        trd_mask = pd.Series(
            [True] * (n_gse // 2) + [False] * (n_gse - n_gse // 2),
            index=gse_norm.columns,
        )
        ctrl_mask = ~trd_mask

        centroids = compute_centroids(
            emory_norm,
            gse_norm,
            emory_response=resp_map,
            gse_trd_mask=trd_mask,
            gse_control_mask=ctrl_mask,
            top_n_genes=20,
        )
        assert "emory_r_centroid" in centroids
        assert "emory_nr_centroid" in centroids
        assert "gse_trd_centroid" in centroids
        assert centroids["n_emory_r"] > 0
        assert centroids["n_emory_nr"] > 0

    def test_verdict_function(self):
        """determine_gate_0x_verdict returns valid string."""
        from dnamrnaseq2026.external_projection.cross_disorder_centroid import (
            determine_gate_0x_verdict,
        )

        assert determine_gate_0x_verdict({"p_value": 0.02, "direction_correct": True}) == "PASS"
        assert determine_gate_0x_verdict({"p_value": 0.10, "direction_correct": True}) == "MARGINAL"
        assert determine_gate_0x_verdict({"p_value": 0.02, "direction_correct": False}) == "FAIL"
        assert determine_gate_0x_verdict({"p_value": 0.20, "direction_correct": True}) == "FAIL"
