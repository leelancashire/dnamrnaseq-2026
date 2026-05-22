"""Phase 2: assemble the complete 3-arm x 6-metric embedding leaderboard.

Design reference: 04-projects/dnamrnaseq/2026-05-19-phase-2-design.md (v1.1),
Section 3 (six metrics) and Section 3.7 (leaderboard format).

Each arm writes its own single-column leaderboard entry CSV:
  - Arm A: ``arm_a_leaderboard_entry.csv``  (scripts/20, run_arm_ac)
  - Arm B: ``arm_b_leaderboard_entry.csv``  (scripts/23)
  - Arm C: ``arm_c_leaderboard_entry.csv``  (scripts/20, run_arm_ac)

This script joins whichever entries are present on the metric index into the
6-row x N-arm table Lee uses to pick the winning embedding E* (design Section
3.7). No composite score: the table reports per-cell ``value | PASS/FAIL`` and
the decision is team consensus. The leakage-clean per-fold metric (iii) is
applied identically to all three arms (Arm B: per-fold MOFA+ + TF refit;
Arms A/C: per-fold TF refit + encoder retrain), so the columns are comparable.

Output:
  - analysis/latest/phase2_leaderboard.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

LATEST_DIR = Path("analysis/latest")

# Per-arm entry CSVs, in leaderboard column order.
ARM_ENTRIES = (
    ("arm_a", LATEST_DIR / "arm_a_leaderboard_entry.csv"),
    ("arm_b", LATEST_DIR / "arm_b_leaderboard_entry.csv"),
    ("arm_c", LATEST_DIR / "arm_c_leaderboard_entry.csv"),
)

METRIC_ORDER = [
    "i_trajectory_consistency",
    "ii_trait_state_disentanglement",
    "iii_loso_reconstruction",
    "iv_conformal_coverage",
    "v_biological_coherence",
    "vi_archetype_clusterability",
]


def main() -> None:
    """Join the per-arm leaderboard entries into the complete 3-arm table."""
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    columns: dict[str, pd.Series] = {}
    for arm, path in ARM_ENTRIES:
        if not path.exists() or path.stat().st_size == 0:
            logger.warning("%s entry missing at %s; column omitted", arm, path)
            continue
        entry = pd.read_csv(path).set_index("metric")
        # The entry CSV has one data column (the arm's own name); take it.
        columns[str(entry.columns[0])] = entry.iloc[:, 0]
        logger.info("Loaded %s entry from %s", arm, path)

    if not columns:
        logger.error(
            "No per-arm leaderboard entries found. Run scripts/20 (Arms A/C) "
            "and scripts/23 (Arm B) first."
        )
        sys.exit(1)

    leaderboard = pd.DataFrame(columns).reindex(METRIC_ORDER)
    out = LATEST_DIR / "phase2_leaderboard.csv"
    leaderboard.to_csv(out)
    logger.info("Phase 2 leaderboard (%d arms):\n%s", leaderboard.shape[1], leaderboard.to_string())
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
