"""Synthetic-fixture smoke tests for Phase 1 modules.

All tests use synthetic data only; no OneDrive access, no rpy2 required.
Tests validate that each module runs end-to-end with the right output shape
and column names, not that the biology is correct (that requires real data).

CI must stay green without any R/Bioconductor dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def n_samples() -> int:
    return 40  # 20 PRE + 20 POST = 10 paired subjects


@pytest.fixture()
def n_cpgs() -> int:
    return 500


@pytest.fixture()
def n_genes() -> int:
    return 200


@pytest.fixture()
def cell_type_cols() -> list[str]:
    return ["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]


@pytest.fixture()
def synthetic_pdata(
    rng: np.random.Generator, n_samples: int, cell_type_cols: list[str]
) -> pd.DataFrame:
    """Synthetic pData2 with 20 PRE and 20 POST samples."""
    n_subj = n_samples // 2
    subcodes = [f"SUBJ{i:03d}" for i in range(n_subj)] * 2
    visits = ["PRE"] * n_subj + ["POST"] * n_subj
    sample_names = [
        f"{s}-PRE" if v == "PRE" else f"{s}-POST" for s, v in zip(subcodes, visits, strict=False)
    ]
    responses = (["R"] * (n_subj // 2) + ["NR"] * (n_subj - n_subj // 2)) * 2

    # Cell-type fractions (rows sum to ~1)
    fracs = rng.dirichlet(np.ones(len(cell_type_cols)), size=n_samples)
    pdata = pd.DataFrame(
        {
            "Subcode": subcodes,
            "Visit": visits,
            "Response": responses,
            "Age": rng.integers(25, 55, size=n_samples).astype(float),
            "sex": rng.choice(["M", "F"], size=n_samples),
            "smokingScore": rng.uniform(0, 1, size=n_samples),
            **{ct: fracs[:, i] for i, ct in enumerate(cell_type_cols)},
        },
        index=sample_names,
    )
    return pdata


@pytest.fixture()
def synthetic_bvals(rng: np.random.Generator, n_cpgs: int, n_samples: int) -> np.ndarray:
    """Synthetic beta values in (0,1), shape (n_cpgs, n_samples)."""
    return rng.beta(2, 5, size=(n_cpgs, n_samples))


@pytest.fixture()
def synthetic_log_cpm(rng: np.random.Generator, n_genes: int, n_samples: int) -> np.ndarray:
    """Synthetic log-CPM matrix, shape (n_genes, n_samples)."""
    return rng.normal(5.0, 2.0, size=(n_genes, n_samples))


@pytest.fixture()
def cpg_ids(n_cpgs: int) -> list[str]:
    return [f"cg{i:08d}" for i in range(n_cpgs)]


@pytest.fixture()
def gene_ids(n_genes: int) -> list[str]:
    return [f"GENE_{i}" for i in range(n_genes)]


@pytest.fixture()
def cell_fracs(synthetic_pdata: pd.DataFrame, cell_type_cols: list[str]) -> pd.DataFrame:
    return synthetic_pdata[cell_type_cols]


# ---------------------------------------------------------------------------
# Step 1.1 / 1.2: cell_type_correction
# ---------------------------------------------------------------------------


class TestCellTypeCorrection:
    def test_beta_to_m_shape(self, synthetic_bvals: np.ndarray) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import beta_to_m

        m = beta_to_m(synthetic_bvals)
        assert m.shape == synthetic_bvals.shape

    def test_beta_to_m_clipped(self, synthetic_bvals: np.ndarray) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import MVAL_CLIP, beta_to_m

        m = beta_to_m(synthetic_bvals)
        assert np.all(np.isfinite(m))
        assert np.all(np.abs(m) <= MVAL_CLIP + 1e-6)

    def test_run_epidish_from_pdata(self, synthetic_pdata: pd.DataFrame) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import run_epidish_from_pdata

        fracs = run_epidish_from_pdata(synthetic_pdata)
        assert fracs.shape[1] >= 2
        assert set(fracs.index) == set(synthetic_pdata.index)

    def test_run_celldmc_small(
        self,
        rng: np.random.Generator,
        synthetic_bvals: np.ndarray,
        cpg_ids: list[str],
        cell_fracs: pd.DataFrame,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        """CellDMC on a 50-CpG slice returns the correct output shape."""
        from dnamrnaseq2026.preprocessing.cell_type_correction import beta_to_m, run_celldmc

        m_vals = beta_to_m(synthetic_bvals[:50, :])
        result = run_celldmc(
            m_matrix=m_vals,
            cpg_ids=cpg_ids[:50],
            cell_fracs=cell_fracs,
            pdata=synthetic_pdata,
            n_jobs=1,
            chunk_size=10,
        )
        assert isinstance(result, pd.DataFrame)
        assert "cpg" in result.columns
        assert "cell_type" in result.columns
        assert "q_interaction" in result.columns
        # Should have rows for multiple cell types
        assert len(result) > 0

    def test_residualise_on_cell_props(
        self,
        synthetic_log_cpm: np.ndarray,
        cell_fracs: pd.DataFrame,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import residualise_on_cell_props

        sample_ids = list(synthetic_pdata.index)
        residuals = residualise_on_cell_props(synthetic_log_cpm[:10, :], cell_fracs, sample_ids)
        assert residuals.shape == synthetic_log_cpm[:10, :].shape

    def test_annotate_cross_contrast(
        self,
        rng: np.random.Generator,
        cpg_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import annotate_cross_contrast

        # Build minimal mock CellDMC output
        cell_types = ["Bcell", "CD4T"]
        rows = []
        for cpg in cpg_ids[:20]:
            for ct in cell_types:
                rows.append(
                    {
                        "cpg": cpg,
                        "cell_type": ct,
                        "q_interaction": rng.uniform(0, 1),
                    }
                )
        df = pd.DataFrame(rows)
        result = annotate_cross_contrast(df, df, df)
        assert "cross_contrast_class" in result.columns

    def test_rescue_check_1_2_5(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
        cpg_ids: list[str],
        gene_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.cell_type_correction import rescue_check_1_2_5

        n_subj = 20
        delta_m = rng.normal(0, 1, size=(len(cpg_ids), n_subj))
        delta_rna = rng.normal(0, 1, size=(len(gene_ids), n_subj))
        pdata_paired = synthetic_pdata.iloc[:n_subj].copy()
        pdata_paired.index = [f"SUBJ{i:03d}" for i in range(n_subj)]

        result = rescue_check_1_2_5(
            delta_m,
            delta_rna,
            cpg_ids,
            gene_ids,
            pdata_paired,
            n_permutations=50,
        )
        assert "verdict" in result
        assert result["verdict"] in {"RESCUE_PASS", "MARGINAL", "FAIL"}
        assert "permanova_p" in result
        assert "rescue_passed" in result


# ---------------------------------------------------------------------------
# Step 1.3: rnaseq_differential
# ---------------------------------------------------------------------------


class TestRnaseqDifferential:
    def test_make_cell_frac_pc1_shape(
        self,
        cell_fracs: pd.DataFrame,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.rnaseq_differential import make_cell_frac_pc1

        sample_ids = list(synthetic_pdata.index[:20])
        pc1 = make_cell_frac_pc1(cell_fracs, sample_ids)
        assert pc1.shape == (20,)

    def test_run_de_ols(
        self,
        synthetic_log_cpm: np.ndarray,
        gene_ids: list[str],
        synthetic_pdata: pd.DataFrame,
        cell_fracs: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.rnaseq_differential import run_de_ols

        result = run_de_ols(
            log_cpm=synthetic_log_cpm[:20, :],
            gene_ids=gene_ids[:20],
            pdata=synthetic_pdata,
            cell_fracs=cell_fracs,
            n_jobs=1,
        )
        assert isinstance(result, pd.DataFrame)
        assert "gene" in result.columns
        assert "q_response" in result.columns
        assert len(result) == 20

    def test_run_de_delta(
        self,
        rng: np.random.Generator,
        synthetic_log_cpm: np.ndarray,
        gene_ids: list[str],
        synthetic_pdata: pd.DataFrame,
        cell_fracs: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.rnaseq_differential import run_de_delta

        n_subj = 20
        delta_lc = rng.normal(0, 1, size=(len(gene_ids[:20]), n_subj))
        pdata_paired = synthetic_pdata.iloc[:n_subj].copy()
        pdata_paired.index = [f"SUBJ{i:03d}" for i in range(n_subj)]
        delta_cf = cell_fracs.iloc[:n_subj].copy()
        delta_cf.index = pdata_paired.index

        result = run_de_delta(
            delta_log_cpm=delta_lc,
            gene_ids=gene_ids[:20],
            pdata_paired=pdata_paired,
            delta_cell_fracs=delta_cf,
            n_jobs=1,
        )
        assert "gene" in result.columns
        assert len(result) == 20


# ---------------------------------------------------------------------------
# Step 1.4: pathway_activity
# ---------------------------------------------------------------------------


class TestPathwayActivity:
    def test_get_progeny_stub(self) -> None:
        from dnamrnaseq2026.preprocessing.pathway_activity import get_progeny_net

        # Without decoupler installed, should return synthetic stub without error
        net = get_progeny_net()
        assert isinstance(net, pd.DataFrame)
        assert "source" in net.columns

    def test_run_progeny_ulm_stub(
        self,
        rng: np.random.Generator,
        synthetic_log_cpm: np.ndarray,
        gene_ids: list[str],
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.pathway_activity import (
            _synthetic_progeny_net,
            run_progeny_ulm,
        )

        net = _synthetic_progeny_net()
        # Only keep genes that appear in the synthetic net
        net_genes = net["target"].unique().tolist()
        sample_ids = list(synthetic_pdata.index)

        # Build a tiny matrix aligned to net_genes
        n_net_genes = len(net_genes)
        tiny_mat = rng.normal(5, 1, size=(n_net_genes, len(sample_ids)))
        result = run_progeny_ulm(tiny_mat, net_genes, sample_ids, net)
        assert isinstance(result, pd.DataFrame)

    def test_compute_delta_activity(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.pathway_activity import compute_delta_activity

        n_subj = 10
        sample_names = [f"SUBJ{i:03d}-PRE" for i in range(n_subj)] + [
            f"SUBJ{i:03d}-POST" for i in range(n_subj)
        ]
        act = pd.DataFrame(
            rng.normal(0, 1, size=(2 * n_subj, 3)),
            index=sample_names,
            columns=["PATH1", "PATH2", "PATH3"],
        )
        pre_ids = [f"SUBJ{i:03d}-PRE" for i in range(n_subj)]
        post_ids = [f"SUBJ{i:03d}-POST" for i in range(n_subj)]
        delta = compute_delta_activity(
            act, pre_ids, post_ids, [f"SUBJ{i:03d}" for i in range(n_subj)]
        )
        assert delta.shape == (n_subj, 3)

    def test_test_response_association(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.pathway_activity import test_response_association

        activity = pd.DataFrame(
            rng.normal(0, 1, size=(len(synthetic_pdata), 5)),
            index=synthetic_pdata.index,
            columns=[f"PATH_{i}" for i in range(5)],
        )
        result = test_response_association(activity, synthetic_pdata)
        assert "q_response" in result.columns
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Step 1.5: tf_activity
# ---------------------------------------------------------------------------


class TestTfActivity:
    def test_get_collectri_stub(self) -> None:
        from dnamrnaseq2026.preprocessing.tf_activity import get_collectri_net

        net = get_collectri_net()
        assert isinstance(net, pd.DataFrame)
        assert "source" in net.columns

    def test_run_tf_ulm_stub(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.tf_activity import (
            _synthetic_collectri_net,
            run_tf_ulm,
        )

        net = _synthetic_collectri_net(n_tfs=3, n_targets=10)
        net_genes = net["target"].unique().tolist()
        sample_ids = list(synthetic_pdata.index)
        mat = rng.normal(5, 1, size=(len(net_genes), len(sample_ids)))
        result = run_tf_ulm(mat, net_genes, sample_ids, net)
        assert isinstance(result, pd.DataFrame)

    def test_test_tf_response_association(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.tf_activity import test_tf_response_association

        n_subj = 20
        pdata_paired = synthetic_pdata.iloc[:n_subj].copy()
        pdata_paired.index = [f"SUBJ{i:03d}" for i in range(n_subj)]

        tfs = ["NFATC1", "TCF7L2", "TF_A", "TF_B"]
        da = pd.DataFrame(
            rng.normal(0, 1, size=(n_subj, len(tfs))),
            index=pdata_paired.index,
            columns=tfs,
        )
        result = test_tf_response_association(da, pdata_paired)
        assert "tf" in result.columns
        assert "priority_family" in result.columns
        assert len(result) == len(tfs)

    def test_build_priority_tf_table(
        self,
        rng: np.random.Generator,
        synthetic_pdata: pd.DataFrame,
    ) -> None:
        from dnamrnaseq2026.preprocessing.tf_activity import (
            build_priority_tf_table,
            test_tf_response_association,
        )

        n_subj = 20
        pdata_paired = synthetic_pdata.iloc[:n_subj].copy()
        pdata_paired.index = [f"SUBJ{i:03d}" for i in range(n_subj)]
        tfs = ["NFATC1", "TCF7L2", "TF_A"]
        da = pd.DataFrame(
            rng.normal(0, 1, size=(n_subj, len(tfs))),
            index=pdata_paired.index,
            columns=tfs,
        )
        test_result = test_tf_response_association(da, pdata_paired)
        priority_table = build_priority_tf_table(test_result)
        assert "significant" in priority_table.columns
        assert all(priority_table["priority_family"].isin(["NFAT", "WNT"]))


# ---------------------------------------------------------------------------
# Step 1.6: regulatory_enrichment
# ---------------------------------------------------------------------------


class TestRegulatoryEnrichment:
    def test_hypergeometric_enrichment(self) -> None:
        from dnamrnaseq2026.preprocessing.regulatory_enrichment import hypergeometric_enrichment

        result = hypergeometric_enrichment(
            n_sig=100,
            n_background=850_000,
            feature_size=5_000,
            n_overlap=5,
        )
        assert "p_hypergeom" in result
        assert "enrichment" in result
        assert result["enrichment"] > 0

    def test_run_regulatory_enrichment_stub(
        self,
        rng: np.random.Generator,
        cpg_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.regulatory_enrichment import (
            run_regulatory_enrichment,
            stub_cpg_positions,
            stub_encode_features,
        )

        # Build minimal CellDMC output with some significant CpGs
        rows = []
        for cpg in cpg_ids[:50]:
            rows.append(
                {
                    "cpg": cpg,
                    "cell_type": "Mono",
                    "q_interaction": 0.01,
                }
            )
        celldmc = pd.DataFrame(rows)

        cpg_pos = stub_cpg_positions(cpg_ids)
        bg_pos = stub_cpg_positions(cpg_ids)
        features = stub_encode_features(n_features=2, n_intervals=20)

        result = run_regulatory_enrichment(
            celldmc_delta=celldmc,
            cpg_positions=cpg_pos,
            background_cpg_positions=bg_pos,
            encode_features=features,
        )
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            assert "q_hypergeom" in result.columns

    def test_cpg_ids_to_bed(self, cpg_ids: list[str]) -> None:
        from dnamrnaseq2026.preprocessing.regulatory_enrichment import cpg_ids_to_bed

        n = len(cpg_ids)
        chr_vals = (["1", "2", "3"] * ((n // 3) + 1))[:n]
        manifest = pd.DataFrame(
            {
                "CHR": chr_vals,
                "MAPINFO": range(n),
            },
            index=cpg_ids,
        )
        bed = cpg_ids_to_bed(cpg_ids[:30], manifest)
        assert "chrom" in bed.columns
        assert "start" in bed.columns
        assert "end" in bed.columns


# ---------------------------------------------------------------------------
# Step 1.7: replication
# ---------------------------------------------------------------------------


class TestReplication:
    def test_build_best_paired_ids_from_index(self) -> None:
        from dnamrnaseq2026.preprocessing.replication import build_best_paired_ids

        sample_names = [f"SUBJ{i:03d}-BL" for i in range(10)] + [
            f"SUBJ{i:03d}-12W" for i in range(10)
        ]
        pdata = pd.DataFrame(
            {"dummy": range(20)},
            index=sample_names,
        )
        subjects, pre_ids, post_ids = build_best_paired_ids(pdata)
        assert len(subjects) == 10
        assert all("-BL" in s for s in pre_ids)
        assert all("-12W" in s for s in post_ids)

    def test_compute_best_delta_m(
        self,
        rng: np.random.Generator,
        cpg_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.replication import compute_best_delta_m

        n_subj = 10
        n_best_samples = 20
        sample_names = [f"SUBJ{i:03d}-BL" for i in range(n_subj)] + [
            f"SUBJ{i:03d}-12W" for i in range(n_subj)
        ]
        pdata = pd.DataFrame({"dummy": range(n_best_samples)}, index=sample_names)
        bvals = rng.beta(2, 5, size=(len(cpg_ids), n_best_samples))
        pre_ids = [f"SUBJ{i:03d}-BL" for i in range(n_subj)]
        post_ids = [f"SUBJ{i:03d}-12W" for i in range(n_subj)]

        delta_m, kept = compute_best_delta_m(
            bvals,
            cpg_ids,
            pre_ids,
            post_ids,
            pdata,
            sig_cpg_ids=cpg_ids[:50],
        )
        assert delta_m.shape[0] == 50
        assert delta_m.shape[1] == n_subj

    def test_run_replication(
        self,
        rng: np.random.Generator,
        cpg_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.replication import run_replication

        n_subj = 15
        n_cpg_test = 30
        delta_m = rng.normal(0, 1, size=(n_cpg_test, n_subj))
        kept_cpgs = cpg_ids[:n_cpg_test]

        emory_betas = pd.DataFrame(
            {
                "cpg": np.tile(kept_cpgs, 3),
                "cell_type": np.repeat(["Bcell", "CD4T", "Mono"], n_cpg_test),
                "beta_interaction": rng.normal(0, 0.5, size=n_cpg_test * 3),
            }
        )

        pdata_best = pd.DataFrame(
            {
                "Response": (["R"] * 8 + ["NR"] * 7)[:n_subj],
                "Age": rng.uniform(25, 55, size=n_subj),
                "Therapy_type": (["CPT"] * 5 + ["PE"] * 5 + ["None"] * 5)[:n_subj],
            },
            index=[f"SUBJ{i:03d}" for i in range(n_subj)],
        )

        overall, within_mod, interaction = run_replication(
            delta_m_best=delta_m,
            kept_cpg_ids=kept_cpgs,
            emory_betas=emory_betas,
            pdata_best_paired=pdata_best,
            n_jobs=1,
        )
        assert "cpg" in overall.columns
        assert "same_direction" in overall.columns
        assert len(overall) == n_cpg_test

    def test_summarise_replication(
        self,
        rng: np.random.Generator,
        cpg_ids: list[str],
    ) -> None:
        from dnamrnaseq2026.preprocessing.replication import summarise_replication

        n = 50
        overall = pd.DataFrame(
            {
                "cpg": cpg_ids[:n],
                "emory_beta": rng.normal(0, 1, size=n),
                "beta_best": rng.normal(0, 1, size=n),
                "p_best": rng.uniform(0, 1, size=n),
                "same_direction": rng.choice([True, False], size=n),
            }
        )
        summary = summarise_replication(overall)
        assert "verdict" in summary
        assert summary["verdict"] in {"PASS", "FAIL"}
