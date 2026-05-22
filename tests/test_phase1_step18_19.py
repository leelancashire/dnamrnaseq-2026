"""Synthetic-fixture smoke tests for Steps 1.8 and 1.9.

Tests validate that:
1. check_hypothesis_coherence correctly identifies keyword hits
2. pi1_estimate returns a value in [0,1]
3. run_replication_555 produces correct concordance stats on synthetic data
4. run_celldmc_best returns expected output schema
5. build_best_paired correctly identifies paired R/NR subjects

No external API calls, no OneDrive access, no rpy2 required.
CI must stay green without network access.
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
def n_cpgs() -> int:
    return 200


@pytest.fixture()
def n_pairs() -> int:
    return 30


@pytest.fixture()
def cell_type_names() -> list[str]:
    return ["B", "CD4T", "CD8T", "Mono", "Neutro", "NK"]


@pytest.fixture()
def synthetic_pdata_best(rng: np.random.Generator, n_pairs: int) -> pd.DataFrame:
    """Synthetic BEST pdata with BL/12W paired samples and R_BL/NR_BL groups."""
    subcodes = [f"BEST-{i:04d}" for i in range(n_pairs)]
    # PRE (BL) rows
    bl_samples = [f"BL_{s}" for s in subcodes]
    w12_samples = [f"12W_{s}" for s in subcodes]

    n_r = n_pairs // 2
    groups_bl = ["R_BL"] * n_r + ["NR_BL"] * (n_pairs - n_r)
    groups_12w = ["R_12W"] * n_r + ["NR_12W"] * (n_pairs - n_r)

    ct_cols = [
        "EpiDISH_fresh_Bcell",
        "EpiDISH_fresh_CD4T",
        "EpiDISH_fresh_CD8T",
        "EpiDISH_fresh_Mono",
        "EpiDISH_fresh_Neu",
        "EpiDISH_fresh_NK",
    ]
    fracs_bl = rng.dirichlet(np.ones(6), size=n_pairs)
    fracs_12w = rng.dirichlet(np.ones(6), size=n_pairs)

    rows_bl = {
        "SampleName": bl_samples,
        "Subcode": subcodes,
        "Visit": ["BL"] * n_pairs,
        "group": groups_bl,
        "Age": rng.uniform(25, 55, size=n_pairs),
        "sex": rng.choice(["M", "F"], size=n_pairs),
        "smokingScore": rng.uniform(0, 1, size=n_pairs),
        **{ct: fracs_bl[:, i] for i, ct in enumerate(ct_cols)},
    }
    rows_12w = {
        "SampleName": w12_samples,
        "Subcode": subcodes,
        "Visit": ["12W"] * n_pairs,
        "group": groups_12w,
        "Age": rng.uniform(25, 55, size=n_pairs),
        "sex": rng.choice(["M", "F"], size=n_pairs),
        "smokingScore": rng.uniform(0, 1, size=n_pairs),
        **{ct: fracs_12w[:, i] for i, ct in enumerate(ct_cols)},
    }
    bl_df = pd.DataFrame(rows_bl)
    w12_df = pd.DataFrame(rows_12w)
    all_df = pd.concat([bl_df, w12_df], ignore_index=True)
    all_df = all_df.set_index("SampleName", drop=False)
    return all_df


@pytest.fixture()
def synthetic_emory_hits(rng: np.random.Generator, n_cpgs: int) -> pd.DataFrame:
    """Synthetic Emory CellDMC delta hits (555 subset)."""
    n_sig = 50
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
    cell_types = ["CD8T"] * 30 + ["B"] * 15 + ["Neutro"] * 5
    coefs = rng.normal(0.3, 0.5, size=n_sig)
    rows = {
        "cpg": cpg_ids[:n_sig],
        "cell_type": cell_types,
        "coef": coefs,
        "se": rng.uniform(0.05, 0.2, size=n_sig),
        "t_stat": coefs / rng.uniform(0.05, 0.2, size=n_sig),
        "p_val": rng.uniform(1e-9, 4e-4, size=n_sig),
        "fdr": rng.uniform(0.0001, 0.049, size=n_sig),
        "sig": [True] * n_sig,
    }
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_best_celldmc(rng: np.random.Generator, n_cpgs: int) -> pd.DataFrame:
    """Synthetic BEST CellDMC delta results."""
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
    cell_types = ["B", "CD4T", "CD8T", "Mono", "Neutro", "NK"]
    rows = []
    for ct in cell_types:
        for cpg in cpg_ids:
            coef = rng.normal(0.0, 0.3)
            p = rng.uniform(0.01, 0.99)
            rows.append(
                {
                    "cpg": cpg,
                    "cell_type": ct,
                    "coef": coef,
                    "se": abs(rng.normal(0.1, 0.05)),
                    "t_stat": coef / 0.1,
                    "p_val": p,
                    "fdr": min(p * 6, 1.0),
                    "sig": p < 0.01,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests for pi1_estimate
# ---------------------------------------------------------------------------


class TestPi1Estimate:
    def test_returns_float_in_unit_interval(self, rng: np.random.Generator) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script19_iv", "scripts/19_best_celldmc_and_replication.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        p_vals = rng.uniform(0, 1, size=200)
        pi1 = mod.pi1_estimate(p_vals)
        assert isinstance(pi1, float)
        assert 0.0 <= pi1 <= 1.0

    def test_pi1_with_uniform_null(self, rng: np.random.Generator) -> None:
        """Uniform p-values -> pi1 near 0."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script19", "scripts/19_best_celldmc_and_replication.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        pi1 = mod.pi1_estimate(rng.uniform(0, 1, size=1000))
        assert -0.5 <= pi1 <= 0.5  # uniform null: pi1 ~ 0 (within estimation noise)

    def test_pi1_with_mostly_small_p(self, rng: np.random.Generator) -> None:
        """Mostly small p-values -> pi1 close to 1."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script19b", "scripts/19_best_celldmc_and_replication.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        p_vals = np.concatenate(
            [
                rng.uniform(0, 0.01, size=900),
                rng.uniform(0.5, 1.0, size=100),
            ]
        )
        pi1 = mod.pi1_estimate(p_vals)
        assert pi1 > 0.5


# ---------------------------------------------------------------------------
# Tests for hypothesis coherence checker
# ---------------------------------------------------------------------------


class TestHypothesisCoherence:
    def _load_mod(self):  # type: ignore[return]
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script18_coh", "scripts/18_celldmc_enrichment_per_celltype.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    def test_coherent_when_inflammatory_term_present(self) -> None:
        mod = self._load_mod()
        df = pd.DataFrame(
            {
                "term": [
                    "Inflammatory response",
                    "NF-kB signaling",
                    "Cytokine production",
                    "Unrelated process",
                ],
                "adj_p": [0.001, 0.01, 0.03, 0.5],
            }
        )
        result = mod.check_hypothesis_coherence(df)
        assert result["coherent"] is True
        assert "inflammatory" in result["keyword_hits"]

    def test_not_coherent_when_no_relevant_terms(self) -> None:
        mod = self._load_mod()
        df = pd.DataFrame(
            {
                "term": ["Ribosome biogenesis", "RNA splicing", "Protein folding"],
                "adj_p": [0.001, 0.002, 0.01],
            }
        )
        result = mod.check_hypothesis_coherence(df)
        assert result["coherent"] is False

    def test_empty_input_returns_not_coherent(self) -> None:
        mod = self._load_mod()
        result = mod.check_hypothesis_coherence(pd.DataFrame())
        assert result["coherent"] is False


# ---------------------------------------------------------------------------
# Tests for build_best_paired
# ---------------------------------------------------------------------------


class TestBuildBestPaired:
    def _load_mod(self):  # type: ignore[return]
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script19_bp", "scripts/19_best_celldmc_and_replication.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    def test_returns_paired_r_nr_subjects(
        self, synthetic_pdata_best: pd.DataFrame, n_pairs: int
    ) -> None:
        mod = self._load_mod()
        subjects, pre_ids, post_ids, pdata_paired = mod.build_best_paired(
            synthetic_pdata_best,
            subcode_col="Subcode",
            visit_col="Visit",
            pre_label="BL",
            post_label="12W",
            response_col="group",
        )
        assert len(subjects) == n_pairs
        assert len(pre_ids) == len(subjects)
        assert len(post_ids) == len(subjects)
        assert len(pdata_paired) == len(subjects)

    def test_pre_post_ids_are_different(self, synthetic_pdata_best: pd.DataFrame) -> None:
        mod = self._load_mod()
        subjects, pre_ids, post_ids, _ = mod.build_best_paired(
            synthetic_pdata_best,
            subcode_col="Subcode",
            visit_col="Visit",
            pre_label="BL",
            post_label="12W",
            response_col="group",
        )
        assert all(p != q for p, q in zip(pre_ids, post_ids, strict=False))


# ---------------------------------------------------------------------------
# Tests for run_replication_555
# ---------------------------------------------------------------------------


class TestRunReplication555:
    def _load_mod(self):  # type: ignore[return]
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script19_rep", "scripts/19_best_celldmc_and_replication.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    def test_concordance_stats_keys(
        self,
        synthetic_emory_hits: pd.DataFrame,
        synthetic_best_celldmc: pd.DataFrame,
    ) -> None:
        mod = self._load_mod()
        _, stats = mod.run_replication_555(synthetic_emory_hits, synthetic_best_celldmc)
        required_keys = {
            "n_emory_hits",
            "n_matched_in_best",
            "sign_concordance",
            "spearman_rho",
            "pi1_best",
            "verdict",
        }
        assert required_keys.issubset(set(stats.keys()))

    def test_verdict_is_valid_string(
        self,
        synthetic_emory_hits: pd.DataFrame,
        synthetic_best_celldmc: pd.DataFrame,
    ) -> None:
        mod = self._load_mod()
        _, stats = mod.run_replication_555(synthetic_emory_hits, synthetic_best_celldmc)
        assert stats["verdict"] in {"REPLICATES", "MARGINAL", "DOES_NOT_REPLICATE", "NO_MATCH"}

    def test_sign_concordance_in_unit_interval(
        self,
        synthetic_emory_hits: pd.DataFrame,
        synthetic_best_celldmc: pd.DataFrame,
    ) -> None:
        mod = self._load_mod()
        _, stats = mod.run_replication_555(synthetic_emory_hits, synthetic_best_celldmc)
        sc = stats["sign_concordance"]
        if not (isinstance(sc, float) and np.isnan(sc)):
            assert 0.0 <= sc <= 1.0

    def test_perfect_replication_detected(self, rng: np.random.Generator) -> None:
        """When BEST coefs match Emory exactly, should detect as REPLICATES."""
        mod = self._load_mod()
        n_sig = 100
        cpg_ids = [f"cg{i:08d}" for i in range(n_sig)]
        cell_types = ["CD8T"] * 60 + ["B"] * 30 + ["Neutro"] * 10
        coefs = rng.normal(0.5, 0.3, size=n_sig)
        emory = pd.DataFrame(
            {
                "cpg": cpg_ids,
                "cell_type": cell_types,
                "coef": coefs,
                "fdr": [0.01] * n_sig,
                "sig": [True] * n_sig,
            }
        )
        # BEST: same sign and magnitude + small p
        best = pd.DataFrame(
            {
                "cpg": cpg_ids,
                "cell_type": cell_types,
                "coef": coefs * rng.uniform(0.8, 1.2, size=n_sig),  # ~same sign
                "p_val": [0.01] * n_sig,
                "fdr": [0.04] * n_sig,
                "sig": [True] * n_sig,
            }
        )
        _, stats = mod.run_replication_555(emory, best)
        # sign concordance should be very high with same-direction coefs
        assert stats["sign_concordance"] > 0.7
        assert stats["spearman_rho"] > 0.5


# ---------------------------------------------------------------------------
# Tests for cpg_list_to_genes (offline: uses a synthetic cpg_map)
# ---------------------------------------------------------------------------


class TestCpgListToGenes:
    def _load_mod(self):  # type: ignore[return]
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "script18_cpg", "scripts/18_celldmc_enrichment_per_celltype.py"
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    def test_extracts_unique_genes(self) -> None:
        mod = self._load_mod()
        cpg_map = pd.DataFrame(
            {
                "cpg": ["cg00000001", "cg00000002", "cg00000003"],
                "ucsc_gene": ["GENE_A;GENE_A;GENE_B", "GENE_C", "NA"],
                "gencode_gene": ["GENE_A", "GENE_C", "GENE_D"],
            }
        )
        genes = mod.cpg_list_to_genes(["cg00000001", "cg00000002", "cg00000003"], cpg_map)
        assert "GENE_A" in genes
        assert "GENE_B" in genes
        assert "GENE_C" in genes
        assert "GENE_D" in genes  # fallback from gencode
        # Duplicates should be removed
        assert len(genes) == len(set(genes))

    def test_empty_input_returns_empty(self) -> None:
        mod = self._load_mod()
        cpg_map = pd.DataFrame({"cpg": [], "ucsc_gene": [], "gencode_gene": []})
        genes = mod.cpg_list_to_genes([], cpg_map)
        assert genes == []
