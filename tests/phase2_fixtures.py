"""Synthetic Phase 2 fixtures: paired (subject, visit) tensors with planted signal.

Used by tests/test_phase2_*.py. The synthetic data is deliberately tiny and
CI-safe (no OneDrive, no GPU). It plants a recoverable trajectory signal so the
metric code can be exercised on something other than pure noise, while staying
small enough that every test runs in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_paired(
    n_subjects: int = 30,
    n_dnam: int = 40,
    n_rna: int = 24,
    n_clinical: int = 5,
    *,
    seed: int = 42,
    plant_signal: bool = True,
    n_classes: int = 2,
) -> dict[str, np.ndarray]:
    """Return a synthetic paired dataset as plain arrays.

    Keys: ``x_pre``, ``x_post`` (n_subjects, n_dnam + n_rna + n_clinical);
    ``rna_pre/post``, ``dnam_pre/post``, ``clin_pre/post`` (split views);
    ``subject_ids``, ``response``, ``responder_mask``, ``delta_pcl``.

    When ``plant_signal`` is True, responders share a common POST-shift
    direction so the recovery axis is recoverable.
    """
    rng = np.random.default_rng(seed)
    d = n_dnam + n_rna + n_clinical

    x_pre = rng.standard_normal((n_subjects, d))
    if n_classes == 2:
        response = rng.choice(["R", "NR"], size=n_subjects)
        responder_mask = response == "R"
    else:
        response = rng.choice([1, 2, 3], size=n_subjects)
        responder_mask = response >= 2

    shift_dir = rng.standard_normal(d)
    shift_dir /= np.linalg.norm(shift_dir)
    delta = rng.standard_normal((n_subjects, d)) * 0.3
    if plant_signal:
        magnitude = np.where(responder_mask, 2.0, 0.2)[:, None]
        delta = delta + magnitude * shift_dir[None, :]
    x_post = x_pre + delta

    delta_pcl = -10.0 * responder_mask.astype(float) + rng.standard_normal(n_subjects) * 2.0

    return {
        "x_pre": x_pre,
        "x_post": x_post,
        "dnam_pre": x_pre[:, :n_dnam],
        "dnam_post": x_post[:, :n_dnam],
        "rna_pre": x_pre[:, n_dnam : n_dnam + n_rna],
        "rna_post": x_post[:, n_dnam : n_dnam + n_rna],
        "clin_pre": x_pre[:, n_dnam + n_rna :],
        "clin_post": x_post[:, n_dnam + n_rna :],
        "subject_ids": np.array([f"SUBJ{i:03d}" for i in range(n_subjects)], dtype=object),
        "response": response,
        "responder_mask": responder_mask,
        "delta_pcl": delta_pcl,
    }


def make_synthetic_sample_frame(
    n_subjects: int = 30,
    n_features: int = 60,
    *,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, pdata) at the sample level for the data-harness tests.

    Each subject contributes one PRE and one POST sample, so the frames have
    2 * n_subjects rows.
    """
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_subjects
    subcodes = [f"SUBJ{i:03d}" for i in range(n_subjects)] * 2
    visits = ["PRE_IOP"] * n_subjects + ["POST_IOP"] * n_subjects
    sample_ids = [f"{s}_{v}" for s, v in zip(subcodes, visits, strict=True)]
    responses = list(rng.choice(["R", "NR"], size=n_subjects)) * 2

    features = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        index=sample_ids,
        columns=[f"feat_{j}" for j in range(n_features)],
    )
    pdata = pd.DataFrame(
        {
            "Subcode": subcodes,
            "Visit": visits,
            "Response": responses,
            "PCL_total": rng.integers(10, 80, size=n_samples).astype(float),
        },
        index=sample_ids,
    )
    return features, pdata
