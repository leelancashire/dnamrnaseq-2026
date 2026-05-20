"""Phase 2: train the three embedding arms (A: FM, B: MOFA+, C: contrastive).

Wired into the Snakemake rule ``train_embedding`` (workflow/rules/embedding.smk).

Design reference: 04-projects/dnamrnaseq/2026-05-19-phase-2-design.md (v1.1),
Section 2 (arms) and Section 5 (compute budget).

Critical-path note. This script is GATED on Phase 1 producing real
cell-type-corrected feature inputs (the Tier 1 CellDMC-prioritised feature set,
design Section 2.0). Until Phase 1 step 1.2 lands non-null CellDMC outputs the
arms run on the Tier 2 variance-filter fallback. The scaffold is exercised on
synthetic fixtures via tests/test_phase2_arms.py; this script is the real-data
entrypoint and is NOT invoked during synthetic CI.

Outputs (per the embedding.smk rule):
  - analysis/latest/embedding_fm.pt
  - analysis/latest/embedding_mofa.h5
  - analysis/latest/embedding_contrastive.pt
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LATEST_DIR = Path("analysis/latest")


def main() -> None:
    """Train all three arms on the real Phase 1 feature inputs.

    This entrypoint is intentionally a thin orchestrator. The arm
    implementations live in dnamrnaseq2026.embedding.{arm_a_fm, arm_b_mofa,
    arm_c_contrastive}; the data harness in dnamrnaseq2026.embedding.data_harness.

    Real-data wiring (subject loading, Phase 1 artefact ingestion, the 10-seed
    x 5-fold CV loop) is filled in once Phase 1 completes. The arm modules,
    losses, and training loops are fully implemented and synthetic-tested; what
    this script still needs is the glue to the real Emory/BEST loaders.
    """
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Phase 2 real-data training is gated on Phase 1 cell-type-corrected "
        "outputs. Arm modules are scaffold-complete and synthetic-tested; "
        "real-data CV wiring activates post-Phase-1."
    )
    logger.info(
        "Arm modules: dnamrnaseq2026.embedding.arm_a_fm / arm_b_mofa / "
        "arm_c_contrastive. Run tests/test_phase2_arms.py for synthetic coverage."
    )


if __name__ == "__main__":
    main()
