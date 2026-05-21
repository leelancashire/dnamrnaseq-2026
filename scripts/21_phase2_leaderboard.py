"""Phase 2: build the six-metric embedding-architecture leaderboard.

Wired into the Snakemake rule ``build_trajectory_atlas`` precursor; the
leaderboard picks the winning embedding E* for Phase 3.

Design reference: 04-projects/dnamrnaseq/2026-05-19-phase-2-design.md (v1.1),
Section 3 (six metrics) and Section 3.7 (leaderboard format).

Metric (v) (biological coherence) is a pluggable step: it reads the Phase 1
enrichment artefacts from analysis/latest/ and reports "pending Phase 1 re-run"
when they are stubs. Metrics i-iv and vi score independently.

Output:
  - analysis/latest/phase2_leaderboard.csv
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
    """Aggregate per-arm metric scores into the 6 x 3 leaderboard.

    Thin orchestrator: scoring logic lives in
    dnamrnaseq2026.embedding.leaderboard (score_arm + build_leaderboard).
    Real-data wiring loads the trained per-arm embeddings, computes the six
    metrics per arm, and writes the leaderboard CSV. Activates post-Phase-1
    alongside 20_phase2_train_embedding.py.
    """
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Phase 2 leaderboard is gated on trained embeddings (20_phase2_"
        "train_embedding.py) and Phase 1 enrichment artefacts for metric (v). "
        "Scoring logic is scaffold-complete and synthetic-tested."
    )
    logger.info(
        "Leaderboard logic: dnamrnaseq2026.embedding.leaderboard. "
        "Run tests/test_phase2_leaderboard.py for synthetic coverage."
    )


if __name__ == "__main__":
    main()
