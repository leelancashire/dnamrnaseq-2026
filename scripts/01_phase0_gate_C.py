#!/usr/bin/env python
"""Gate 0-C entry-point: EpiDISH cell-type deconvolution validation.

Reads config.yaml, loads Emory pData2 and subject data, validates
delta-cell-fraction stability and N2LR correlation.

Outputs (to analysis/2026-05-17-phase-0/0-C/):
  gate_0C_results.json            -- all three validations + verdict
  gate_0C_delta_props_hist.png    -- Δ-cell-fraction histograms
  gate_0C_delta_props_hist.svg    -- same, vector

Usage:
    python scripts/01_phase0_gate_C.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

OUT_DIR = _REPO_ROOT / "analysis/2026-05-17-phase-0/0-C"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Gate 0-C starting.")

    from dnamrnaseq2026.data.config import load_config
    from dnamrnaseq2026.data.loaders import load_emory_pdata2, load_emory_subject_data
    from dnamrnaseq2026.preprocessing.cell_type_deconv import (
        determine_gate_0c_verdict,
        validate_delta_cell_fractions,
    )

    cfg = load_config()
    seed = cfg["run"]["seed"]
    logger.info("Seed=%d", seed)

    # Load data
    pdata = load_emory_pdata2()
    subject_data = load_emory_subject_data()

    # Run validation
    results = validate_delta_cell_fractions(pdata=pdata, subject_data=subject_data)
    verdict = determine_gate_0c_verdict(results)
    logger.info("Gate 0-C verdict: %s", verdict)

    # Write results JSON
    out = {
        "gate": "0-C",
        "verdict": verdict,
        "n_paired": int(results["n_paired"]),
        "validation_1_note": (
            "SKIPPED (pData2 columns ARE the reference; rpy2 EpiDISH not run in Phase 0). "
            "Trivially passes."
        ),
        "validation_2_pass": bool(results["validation_2_pass"]),
        "validation_2_mono_sd": float(results["validation_2_mono_sd"]),
        "validation_2_neu_sd": float(results["validation_2_neu_sd"]),
        "validation_2_threshold": 0.02,
        "validation_3_pass": bool(results["validation_3_pass"]),
        "mono_n2lr_pearson_r": float(results["mono_n2lr_r"]),
        "mono_n2lr_pearson_p": float(results["mono_n2lr_p"]),
        "validation_3_threshold": 0.30,
        "available_cell_cols": results["available_cell_cols"],
        "seed": seed,
    }
    results_path = OUT_DIR / "gate_0C_results.json"
    with results_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("Results written to %s", results_path)

    # Delta-cell-fraction histograms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta_props = results["delta_props_df"]
    cell_cols = results["available_cell_cols"]
    n_cols = len(cell_cols)
    fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(12, 6))
    axes = axes.flatten()

    for i, col in enumerate(cell_cols):
        axes[i].hist(
            delta_props[col].dropna(), bins=20, color="#4CAF50", edgecolor="white", alpha=0.8
        )
        axes[i].axvline(0, color="red", lw=1.0, ls="--")
        sd_val = float(delta_props[col].std())
        axes[i].set_title(f"{col} (SD={sd_val:.4f})")
        axes[i].set_xlabel("Δ proportion (POST - PRE)")
        axes[i].set_ylabel("n subjects")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Gate 0-C: Δ-cell-fraction distributions (n={len(delta_props)} paired subjects)\n"
        f"Verdict: {verdict}"
    )
    fig.tight_layout()

    fig.savefig(str(OUT_DIR / "gate_0C_delta_props_hist.png"), dpi=150, bbox_inches="tight")
    fig.savefig(str(OUT_DIR / "gate_0C_delta_props_hist.svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Figures saved to %s", OUT_DIR)

    # Print summary
    print()
    print("=" * 60)
    print("Gate 0-C: EpiDISH cell-type deconvolution validation")
    print("=" * 60)
    print(f"Paired subjects: {out['n_paired']}")
    print("Validation 1 (EpiDISH cross-check): SKIPPED (pData2 = reference)")
    print(f"Validation 2 (Δ-cell SD): "
          f"Mono={out['validation_2_mono_sd']:.4f}, "
          f"Neu={out['validation_2_neu_sd']:.4f} "
          f"(threshold=0.02) -> {'PASS' if out['validation_2_pass'] else 'FAIL'}")
    print(f"Validation 3 (delta_Mono x delta_N2LR Pearson r): "
          f"r={out['mono_n2lr_pearson_r']:.4f} (p={out['mono_n2lr_pearson_p']:.4f}) "
          f"(threshold=0.30) -> {'PASS' if out['validation_3_pass'] else 'FAIL'}")
    print(f"Verdict: {verdict}")
    print("=" * 60)


if __name__ == "__main__":
    main()
