"""Synthetic Phase 3 fixtures: projected-atlas latent coordinates.

Used by tests/test_phase3_proximity.py. Mimics the structure that the Phase 3
projection pipeline will eventually emit: per-subject trajectory termini
(POST-IOP latent coordinates) for responders and non-responders, plus a
projected GSE98793 TRD-inflammatory reference cloud, all in one shared latent
space.

The fixture is deliberately tiny and CI-safe (no OneDrive, no GPU, no trained
atlas). It can plant a separable signal (non-responder termini shifted toward
the TRD cloud) or generate a pure null (no group difference), so the proximity
test and its pre-registered fail criterion can be exercised on both a clear
PASS case and a clear FAIL case.
"""

from __future__ import annotations

import numpy as np


def make_synthetic_termini(
    n_responders: int = 30,
    n_nonresponders: int = 30,
    n_trd: int = 40,
    d_latent: int = 8,
    *,
    seed: int = 42,
    plant_signal: bool = True,
    signal_strength: float = 2.5,
) -> dict[str, np.ndarray]:
    """Return synthetic projected-atlas coordinates as plain arrays.

    Keys: ``responder_termini`` (n_responders, d_latent),
    ``nonresponder_termini`` (n_nonresponders, d_latent),
    ``trd_reference`` (n_trd, d_latent).

    Geometry
    --------
    The TRD-inflammatory cloud is centred at ``+signal`` along latent axis 0.
    Responder termini are centred at the origin (the "healthy" end). When
    ``plant_signal`` is True, non-responder termini are shifted a fraction of
    the way toward the TRD centroid, so their distance-to-TRD is smaller and
    the proximity test should PASS. When False, non-responder termini share the
    responder distribution exactly: a pure null where the test should FAIL.
    """
    rng = np.random.default_rng(seed)

    # TRD-inflammatory reference cloud: offset along axis 0.
    trd_centre = np.zeros(d_latent)
    trd_centre[0] = signal_strength
    trd_reference = trd_centre[None, :] + rng.standard_normal((n_trd, d_latent)) * 0.6

    # Responder termini: centred at the origin (healthy end of the axis).
    responder_termini = rng.standard_normal((n_responders, d_latent)) * 0.6

    # Non-responder termini: shifted toward TRD when signal is planted.
    nonresponder_termini = rng.standard_normal((n_nonresponders, d_latent)) * 0.6
    if plant_signal:
        shift = np.zeros(d_latent)
        shift[0] = signal_strength * 0.6  # part-way toward the TRD centroid
        nonresponder_termini = nonresponder_termini + shift[None, :]

    return {
        "responder_termini": responder_termini,
        "nonresponder_termini": nonresponder_termini,
        "trd_reference": trd_reference,
    }
