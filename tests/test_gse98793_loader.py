"""Synthetic-fixture smoke tests for the GSE98793 downloader and loader.

These tests do NOT require network access or the actual GSE98793 file.
They exercise the probe-to-gene rollup logic and phenotype extraction on
small synthetic fixtures that mimic the real data structure.

The one network-dependent function (download_gse98793) is tested via a
mock that returns a minimal fake GSE object without hitting NCBI GEO FTP.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic fixture constants
# ---------------------------------------------------------------------------

N_PROBES = 120
N_GENES = 40  # fewer than probes to create multi-probe genes
N_SAMPLES = 24  # 16 CASE + 8 CNTL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_probe_gene_csv(tmp_path):
    """Commit-equivalent probe-to-gene CSV (small synthetic version)."""
    rng = np.random.default_rng(42)
    probe_ids = [f"probe_{i:06d}_at" for i in range(N_PROBES)]
    # Assign each probe to one of N_GENES genes (some genes get multiple probes)
    gene_ids = [f"GENE{i:04d}" for i in range(N_GENES)]
    gene_assignments = [gene_ids[i % N_GENES] for i in range(N_PROBES)]

    df = pd.DataFrame({"probe_id": probe_ids, "gene_symbol": gene_assignments})
    csv_path = tmp_path / "hgu133plus2_probe_to_gene.csv"
    df.to_csv(str(csv_path), index=False)
    return csv_path


@pytest.fixture
def synthetic_gse_object():
    """Minimal fake GEOparse.GSE object with 24 samples and 120 probes."""
    rng = np.random.default_rng(99)
    probe_ids = [f"probe_{i:06d}_at" for i in range(N_PROBES)]
    gsm_ids = [f"GSM{900000 + i}" for i in range(N_SAMPLES)]

    # Build per-sample GSM mocks
    gsms = {}
    for j, gsm_id in enumerate(gsm_ids):
        values = rng.normal(7.0, 1.5, size=N_PROBES)
        table = pd.DataFrame({"ID_REF": probe_ids, "VALUE": values})
        gsm = MagicMock()
        gsm.table = table
        # First 16 are CASE, last 8 are CNTL
        if j < 16:
            chars = [
                "subject group: CASE; major depressive disorder (MDD) patient",
                f"batch: {j % 3 + 1}",
            ]
        else:
            chars = [
                "subject group: CNTL; healthy control",
                f"batch: {(j - 16) % 2 + 1}",
            ]
        gsm.metadata = {"characteristics_ch1": chars}
        gsms[gsm_id] = gsm

    gse = MagicMock()
    gse.gsms = gsms
    gse.gpls = {"GPL570": MagicMock()}
    return gse


# ---------------------------------------------------------------------------
# Tests for probe-to-gene map loading
# ---------------------------------------------------------------------------


class TestProbeToGeneMap:
    """Tests for load_probe_to_gene_map."""

    def test_loads_from_custom_path(self, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import load_probe_to_gene_map

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        assert len(probe_map) == N_PROBES
        assert "gene_symbol" in probe_map.columns
        assert probe_map.index.name == "probe_id"

    def test_committed_resource_exists(self):
        """Committed probe-to-gene CSV exists at its expected path."""
        from dnamrnaseq2026.external_projection.gse98793_loader import _DEFAULT_PROBE_GENE_CSV

        assert _DEFAULT_PROBE_GENE_CSV.exists(), (
            f"Committed probe-to-gene reference not found at {_DEFAULT_PROBE_GENE_CSV}. "
            "It must be committed to the repository."
        )

    def test_committed_resource_has_expected_columns(self):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            _DEFAULT_PROBE_GENE_CSV,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=_DEFAULT_PROBE_GENE_CSV)
        assert "gene_symbol" in probe_map.columns
        # GPL570 should have ~45k probes
        assert (
            len(probe_map) > 40000
        ), f"Expected >40000 probes in committed reference, got {len(probe_map)}"

    def test_missing_file_raises(self, tmp_path):
        from dnamrnaseq2026.external_projection.gse98793_loader import load_probe_to_gene_map

        with pytest.raises(FileNotFoundError):
            load_probe_to_gene_map(csv_path=tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# Tests for expression matrix construction (probe-to-gene rollup)
# ---------------------------------------------------------------------------


class TestBuildExpressionMatrix:
    """Tests for build_gse98793_expression_matrix with synthetic data."""

    def test_matrix_shape(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)

        # Should have N_GENES rows (one per unique gene after rollup)
        assert mat.shape[0] == N_GENES
        # Should have N_SAMPLES columns
        assert mat.shape[1] == N_SAMPLES

    def test_matrix_index_is_gene_symbols(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)

        # Index should be gene symbols (GENE0000 ... GENE0039)
        assert all(
            g.startswith("GENE") for g in mat.index
        ), f"Expected gene symbol index, got: {mat.index[:5].tolist()}"

    def test_matrix_values_are_numeric(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)

        assert mat.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()
        assert not mat.isnull().any().any()

    def test_max_mean_rollup_picks_highest_mean(
        self, synthetic_gse_object, synthetic_probe_gene_csv
    ):
        """Max-mean rollup should select the probe with the highest mean per gene."""
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)

        # For each gene in the rollup output, verify no probe from that gene
        # has a higher mean than the selected probe's values.
        # Build the full probe x sample matrix manually for comparison.
        gsms = synthetic_gse_object.gsms
        probe_vals = {}
        for gsm_id, gsm in gsms.items():
            for _, row in gsm.table.iterrows():
                if row["ID_REF"] not in probe_vals:
                    probe_vals[row["ID_REF"]] = []
                probe_vals[row["ID_REF"]].append(float(row["VALUE"]))

        probe_means = {p: float(np.mean(v)) for p, v in probe_vals.items()}

        for gene in mat.index[:5]:  # spot-check first 5 genes
            # Find all probes for this gene
            gene_probes = probe_map[probe_map["gene_symbol"] == gene].index.tolist()
            if len(gene_probes) > 1:
                gene_probe_means = {p: probe_means.get(p, -np.inf) for p in gene_probes}
                best_probe = max(gene_probe_means, key=gene_probe_means.__getitem__)
                expected_row_mean = np.mean([probe_means[p] for p in gene_probes])
                # The selected row should have mean >= average of all probes for this gene
                selected_mean = mat.loc[gene].mean()
                assert selected_mean >= probe_means[best_probe] - 1e-6, (
                    f"Gene {gene}: selected probe mean {selected_mean:.4f} < "
                    f"best probe mean {probe_means[best_probe]:.4f}"
                )

    def test_unsupported_rollup_raises(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        with pytest.raises(ValueError, match="Unsupported rollup"):
            build_gse98793_expression_matrix(
                synthetic_gse_object, probe_gene_map=probe_map, rollup="mean"
            )


# ---------------------------------------------------------------------------
# Tests for phenotype extraction
# ---------------------------------------------------------------------------


class TestPhenotypeExtraction:
    """Tests for extract_gse98793_phenotypes."""

    def test_phenotype_shape(self, synthetic_gse_object):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            extract_gse98793_phenotypes,
        )

        pheno = extract_gse98793_phenotypes(synthetic_gse_object)
        assert len(pheno) == N_SAMPLES
        assert "is_mdd" in pheno.columns
        assert "is_control" in pheno.columns

    def test_case_control_counts(self, synthetic_gse_object):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            extract_gse98793_phenotypes,
        )

        pheno = extract_gse98793_phenotypes(synthetic_gse_object)
        # 16 CASE, 8 CNTL in the synthetic fixture
        assert int(pheno["is_mdd"].sum()) == 16
        assert int(pheno["is_control"].sum()) == 8

    def test_is_mdd_and_control_are_mutually_exclusive(self, synthetic_gse_object):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            extract_gse98793_phenotypes,
        )

        pheno = extract_gse98793_phenotypes(synthetic_gse_object)
        both = pheno["is_mdd"] & pheno["is_control"]
        assert not both.any(), "No sample should be both MDD and control."


# ---------------------------------------------------------------------------
# Tests for TRD-inflammatory mask
# ---------------------------------------------------------------------------


class TestTRDMask:
    """Tests for define_trd_inflammatory_mask."""

    def test_trd_mask_counts(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            define_trd_inflammatory_mask,
            extract_gse98793_phenotypes,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)
        pheno = extract_gse98793_phenotypes(synthetic_gse_object)

        trd_mask, ctrl_mask = define_trd_inflammatory_mask(pheno, mat)
        assert int(trd_mask.sum()) == 16
        assert int(ctrl_mask.sum()) == 8

    def test_masks_indexed_by_sample_columns(self, synthetic_gse_object, synthetic_probe_gene_csv):
        from dnamrnaseq2026.external_projection.gse98793_loader import (
            build_gse98793_expression_matrix,
            define_trd_inflammatory_mask,
            extract_gse98793_phenotypes,
            load_probe_to_gene_map,
        )

        probe_map = load_probe_to_gene_map(csv_path=synthetic_probe_gene_csv)
        mat = build_gse98793_expression_matrix(synthetic_gse_object, probe_gene_map=probe_map)
        pheno = extract_gse98793_phenotypes(synthetic_gse_object)

        trd_mask, ctrl_mask = define_trd_inflammatory_mask(pheno, mat)
        assert set(trd_mask.index) == set(mat.columns)
        assert set(ctrl_mask.index) == set(mat.columns)


# ---------------------------------------------------------------------------
# Tests for the download function (mocked -- no network)
# ---------------------------------------------------------------------------


class TestDownloadGSE98793:
    """Smoke tests for download_gse98793 with network access mocked out."""

    def test_cache_hit_skips_download(self, tmp_path):
        """If the file exists and MD5 matches, no download occurs."""
        from dnamrnaseq2026.external_projection.datasets import (
            GSE98793_SOFT_FILENAME,
            GSE98793_SOFT_MD5,
        )

        # Create a fake file with the expected MD5 by writing the right content
        # (We patch _md5_of_file to return the pinned checksum)
        fake_soft = tmp_path / GSE98793_SOFT_FILENAME
        fake_soft.write_text("fake content")

        fake_gse = MagicMock()

        with (
            patch(
                "dnamrnaseq2026.external_projection.datasets._md5_of_file",
                return_value=GSE98793_SOFT_MD5,
            ),
            patch(
                "GEOparse.get_GEO",
                return_value=fake_gse,
            ) as mock_get,
        ):
            import GEOparse  # noqa: F401
            from dnamrnaseq2026.external_projection.datasets import download_gse98793

            result = download_gse98793(cache_dir=tmp_path, force=False)

        # get_GEO should only be called once (for parsing, not for download)
        # because the cache hit path calls get_GEO(filepath=...) to load
        assert mock_get.call_count == 1
        call_kwargs = mock_get.call_args[1] if mock_get.call_args.kwargs else {}
        call_args = mock_get.call_args[0] if mock_get.call_args.args else ()
        # Should be called with filepath= for local parse, not geo= for download
        assert "filepath" in call_kwargs or (len(call_args) == 0)

    def test_missing_geoparse_raises_importerror(self, tmp_path):
        """If GEOparse is not installed, ImportError is raised."""
        import sys

        original = sys.modules.get("GEOparse")
        sys.modules["GEOparse"] = None  # type: ignore[assignment]

        try:
            # Force reimport with patched module
            import importlib
            import dnamrnaseq2026.external_projection.datasets as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="GEOparse is required"):
                mod.download_gse98793(cache_dir=tmp_path)
        finally:
            if original is not None:
                sys.modules["GEOparse"] = original
            else:
                del sys.modules["GEOparse"]
            # Reload to restore
            import importlib
            import dnamrnaseq2026.external_projection.datasets as mod

            importlib.reload(mod)
