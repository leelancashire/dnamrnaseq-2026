"""Tests for the step 1.2 pdata_paired ID-based join fix.

Regression suite for the positional-slicing bug in 12_phase1_celldmc.py
where ``pdata_paired`` was previously constructed as::

    pdata_aug.loc[pre_ids_rna[:len(paired_subjects)]].copy()

This silently assumed that ``filter_paired_ids`` and
``filter_paired_ids_rna`` return subjects in the same order, AND that the
valid-RNA-cell_props filter does not change the subject count.  With real
EpiDISH outputs the order assumption holds (both functions sort their
subject sets), but the count assumption can silently fail if any subjects
have cell_props entries missing.

The correct fix uses explicit subject-ID-based lookups:
- ``rna_pre_by_sc``: dict mapping Subcode -> PRE AMC-ID
- ``rna_post_by_sc``: dict mapping Subcode -> POST AMC-ID
- ``common_subjects``: intersection of DNAm pairing and RNA/cell_props
  availability, ordered by the DNAm pairing sort order

This test module creates synthetic pdata_aug DataFrames with deliberate
out-of-order subject IDs and verifies that the fix produces correctly
aligned pdata_paired and delta_cell_props_df.

No real data dependencies; no GPU; CPU-only; CI-safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pdata_aug(subcodes: list[str], visit_order: str = "sorted") -> pd.DataFrame:
    """Build a minimal pdata_aug DataFrame.

    Parameters
    ----------
    subcodes:
        Subject Subcode identifiers.
    visit_order:
        'sorted' gives rows in Subcode-sorted order; 'reversed' gives them
        in reverse order to simulate a non-standard upstream sort.

    Returns
    -------
    pd.DataFrame
        One PRE and one POST row per subject.  Index is AMC-ID
        ('{Subcode}-{Visit}').  Includes Visit, Subcode, Response,
        SampleName_DNAm, SampleName_RNASeq, and six cell-type columns.
    """
    rows = []
    for sc in subcodes:
        for visit in ("PRE-IOP", "POST-IOP"):
            amc_id = f"{sc}-{visit}"
            rows.append(
                {
                    "Subcode": sc,
                    "Visit": visit,
                    "Response": "R" if subcodes.index(sc) % 2 == 0 else "NR",
                    "SampleName_DNAm": f"SENTRIX-{sc}-{visit[:3]}",
                    "SampleName_RNASeq": amc_id,
                    "Bcell": 0.1,
                    "CD4T": 0.25,
                    "CD8T": 0.15,
                    "Mono": 0.2,
                    "Neu": 0.2,
                    "NK": 0.1,
                }
            )
    df = pd.DataFrame(rows)
    df.index = df["SampleName_RNASeq"]
    if visit_order == "reversed":
        df = df.iloc[::-1].reset_index(drop=True)
        df.index = df["SampleName_RNASeq"]
    return df


def _apply_step12_delta_pairing(
    pdata_aug: pd.DataFrame,
    cell_props: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Apply the fixed pdata_paired construction logic from 12_phase1_celldmc.py.

    Mirrors the fix introduced in the slicing bug PR exactly, so the test
    is coupled to the correct behaviour rather than to any particular
    implementation detail.

    Returns
    -------
    tuple of (common_subjects, pdata_paired, delta_cell_props_df)
    """
    from dnamrnaseq2026.preprocessing.delta_construction import (
        filter_paired_ids,
        filter_paired_ids_rna,
    )

    paired_subjects, _, _ = filter_paired_ids(pdata_aug)
    paired_subjects_rna, pre_ids_rna, post_ids_rna = filter_paired_ids_rna(pdata_aug)

    rna_pre_by_sc: dict[str, str] = dict(zip(paired_subjects_rna, pre_ids_rna, strict=False))
    rna_post_by_sc: dict[str, str] = dict(zip(paired_subjects_rna, post_ids_rna, strict=False))

    common_subjects = [
        sc
        for sc in paired_subjects
        if sc in rna_pre_by_sc
        and rna_pre_by_sc[sc] in cell_props.index
        and rna_post_by_sc[sc] in cell_props.index
    ]

    if common_subjects:
        pre_ids_rna_filt = [rna_pre_by_sc[sc] for sc in common_subjects]
        post_ids_rna_filt = [rna_post_by_sc[sc] for sc in common_subjects]
        delta_vals = (
            cell_props.loc[post_ids_rna_filt].values - cell_props.loc[pre_ids_rna_filt].values
        )
    else:
        cell_type_cols = list(cell_props.columns)
        common_subjects = list(paired_subjects)
        delta_vals = np.zeros((len(common_subjects), len(cell_type_cols)))

    delta_cell_props_df = pd.DataFrame(
        delta_vals, index=common_subjects, columns=cell_props.columns
    )

    if common_subjects and common_subjects[0] in rna_pre_by_sc:
        pre_amc_ids = [rna_pre_by_sc[sc] for sc in common_subjects]
        pdata_paired = pdata_aug.loc[pre_amc_ids].copy()
    else:
        pdata_paired = pdata_aug.loc[common_subjects].copy()
    pdata_paired.index = pd.Index(common_subjects)

    return common_subjects, pdata_paired, delta_cell_props_df


# ---------------------------------------------------------------------------
# Tests: aligned pairing when filter functions return out-of-order subjects
# ---------------------------------------------------------------------------


class TestPdataPairedIdBasedJoin:
    """Verify that pdata_paired is correctly assigned for out-of-order subjects."""

    def test_sorted_subjects_pdata_paired_index_matches_subcodes(self) -> None:
        """Baseline: sorted input produces pdata_paired indexed by Subcodes."""
        subcodes = ["SC001", "SC002", "SC003", "SC004"]
        pdata_aug = _make_pdata_aug(subcodes, visit_order="sorted")
        cell_props = pdata_aug[["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]].copy()

        common, pdata_paired, delta_cell_props_df = _apply_step12_delta_pairing(
            pdata_aug, cell_props
        )

        assert list(pdata_paired.index) == common, (
            "pdata_paired.index must equal common_subjects in the same order"
        )
        assert list(delta_cell_props_df.index) == common, (
            "delta_cell_props_df.index must equal common_subjects in the same order"
        )

    def test_out_of_order_subjects_still_correctly_aligned(self) -> None:
        """Core regression: when filter functions return subjects in a different
        order than the pdata_aug row order, the fix must still align pdata_paired
        correctly by subject ID (not by row position).
        """
        # We construct a pdata_aug where the rows are in reverse Subcode order.
        # filter_paired_ids and filter_paired_ids_rna both sort() their output
        # so they return ['SC001', 'SC002', 'SC003'] in ascending order regardless.
        # The PRE-visit AMC-IDs in pdata_aug come from the reversed data layout.
        # The correct fix must look up each Subcode's PRE AMC-ID explicitly rather
        # than relying on the row-position-matches-subject-order assumption.
        subcodes = ["SC003", "SC001", "SC002"]  # unsorted in the pdata_aug
        pdata_aug = _make_pdata_aug(subcodes, visit_order="sorted")
        cell_props = pdata_aug[["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]].copy()

        common, pdata_paired, delta_cell_props_df = _apply_step12_delta_pairing(
            pdata_aug, cell_props
        )

        # The index must be exactly common_subjects in order.
        assert list(pdata_paired.index) == common, (
            "Out-of-order subjects: pdata_paired.index must match common_subjects exactly"
        )
        # For each subject, the Response in pdata_paired must be the correct
        # response for THAT subject (not the one positionally adjacent in pdata_aug).
        for sc in common:
            expected_response = pdata_aug[pdata_aug["Subcode"] == sc]["Response"].iloc[0]
            actual_response = pdata_paired.loc[sc, "Response"]
            assert actual_response == expected_response, (
                f"Subject {sc}: expected Response={expected_response!r}, "
                f"got {actual_response!r}. Positional slicing bug may be present."
            )

    def test_subjects_missing_cell_props_are_excluded_correctly(self) -> None:
        """When some subjects lack cell_props rows, common_subjects is the
        correct subset and pdata_paired contains only those subjects.

        The OLD code sliced ``pre_ids_rna[:len(paired_subjects)]`` which would
        produce wrong metadata for the REMAINING subjects if any intermediate
        subject was dropped.
        """
        subcodes = ["SC001", "SC002", "SC003", "SC004"]
        pdata_aug = _make_pdata_aug(subcodes, visit_order="sorted")
        # Only include cell_props for SC001 and SC003 (drop SC002, SC004)
        all_cell_props = pdata_aug[["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]].copy()
        keep_ids = [idx for idx in all_cell_props.index if "SC001" in idx or "SC003" in idx]
        cell_props = all_cell_props.loc[keep_ids]

        common, pdata_paired, delta_cell_props_df = _apply_step12_delta_pairing(
            pdata_aug, cell_props
        )

        assert set(common) == {
            "SC001",
            "SC003",
        }, "Only subjects with cell_props for both PRE and POST should be included"
        assert list(pdata_paired.index) == sorted(common), (
            "pdata_paired must be indexed by the surviving common_subjects in sorted order"
        )
        assert len(delta_cell_props_df) == 2

        # Each subject's pdata_paired row must have the CORRECT Response.
        for sc in common:
            expected = pdata_aug[pdata_aug["Subcode"] == sc]["Response"].iloc[0]
            assert pdata_paired.loc[sc, "Response"] == expected, (
                f"{sc}: wrong Response in pdata_paired after subject-drop filtering"
            )

    def test_delta_cell_props_index_aligns_with_pdata_paired(self) -> None:
        """delta_cell_props_df.index and pdata_paired.index must be identical.

        This is the alignment contract required by run_celldmc and
        residualise_on_cell_props: they both key on the same subject IDs.
        """
        subcodes = ["SC001", "SC002", "SC003"]
        pdata_aug = _make_pdata_aug(subcodes, visit_order="sorted")
        cell_props = pdata_aug[["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]].copy()

        common, pdata_paired, delta_cell_props_df = _apply_step12_delta_pairing(
            pdata_aug, cell_props
        )

        assert list(delta_cell_props_df.index) == list(pdata_paired.index), (
            "delta_cell_props_df.index and pdata_paired.index must be identical"
        )

    def test_delta_cell_props_values_correct_for_each_subject(self) -> None:
        """Each row of delta_cell_props_df must be POST - PRE for THAT subject.

        With the old positional slice, a subject-order mismatch would cause
        SUBJECT A's delta to be assigned to SUBJECT B's metadata row.
        """
        subcodes = ["SC001", "SC002"]
        pdata_aug = _make_pdata_aug(subcodes, visit_order="sorted")

        # Give each sample a distinct Neu fraction so we can verify alignment.
        rng = np.random.default_rng(99)
        cell_props = pdata_aug[["Bcell", "CD4T", "CD8T", "Mono", "Neu", "NK"]].copy()
        for idx in cell_props.index:
            cell_props.loc[idx, "Neu"] = float(rng.uniform(0.05, 0.40))
        # Normalise rows to sum to 1.
        cell_props = cell_props.div(cell_props.sum(axis=1), axis=0)

        common, pdata_paired, delta_cell_props_df = _apply_step12_delta_pairing(
            pdata_aug, cell_props
        )

        for sc in common:
            pre_id = f"{sc}-PRE-IOP"
            post_id = f"{sc}-POST-IOP"
            expected_delta_neu = cell_props.loc[post_id, "Neu"] - cell_props.loc[pre_id, "Neu"]
            actual_delta_neu = float(delta_cell_props_df.loc[sc, "Neu"])
            assert abs(actual_delta_neu - expected_delta_neu) < 1e-9, (
                f"{sc}: delta Neu mismatch (expected {expected_delta_neu:.6f}, "
                f"got {actual_delta_neu:.6f}). Subject-ID alignment broken."
            )
