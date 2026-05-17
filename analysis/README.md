# analysis/

Dated analysis run directories. Each subdirectory is one analysis run.

**Convention:** `YYYY-MM-DD-<slug>/`
- `config.yaml` snapshot used for that run
- log files
- outputs (CSV, figures, etc.)

Outputs are gitignored (see `.gitignore`). Config snapshots are committed
so every run can be reproduced from `config + code version + data snapshot`.

**Example:**
```
analysis/
  2026-05-20-phase0-gates/
    config.yaml        # committed
    run.log            # committed (small)
    outputs/           # gitignored
      gate_0t_results.csv
      gate_0k_results.csv
```
