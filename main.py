"""
uowc/main.py
============
Application entry point.

Separation-of-Concern role
---------------------------
  `main.py` is the wiring layer.  It instantiates the config, calls each
  module in the correct sequence, and measures total wall-clock time.
  It contains no physics, no transport logic, no metrics, and no plot code —
  only the top-level call sequence.

  This design means the entire pipeline can be re-orchestrated (e.g. swap
  out the sweep runner, add checkpointing, or call from a notebook) by
  editing only this file.

Module dependency graph
-----------------------
  main.py
    ├── config          (WaterParams, BeamParams, SIM, ...)
    ├── reporting       (print_run_header, print_summary_tables)
    ├── simulation      (run_sweep)  ──► transport ──► physics
    ├── metrics         (compute_all_metrics)
    └── plotting        (plot_all)
"""

from __future__ import annotations
import os
import sys
import time

# ── Package imports (all from the uowc package) ────────────────────────────
from uowc.config    import SIM, ALL_WATERS, ALL_BEAMS
from uowc.reporting import print_run_header, print_summary_tables
from uowc.simulation import run_sweep, RunKey
from uowc.metrics   import compute_all_metrics
from uowc.plotting  import plot_all


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(out_dir: str = "/mnt/user-data/outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ── 1. Configuration summary ──────────────────────────────────────────
    print_run_header(SIM)

    # ── 2. Run the photon transport sweep ─────────────────────────────────
    #       Returns Dict[RunKey → (weights, times_of_flight)]
    raw_results = run_sweep(SIM, verbose=True)

    # ── 3. Compute channel metrics from raw photon data ───────────────────
    print("  Computing channel metrics ...", end=" ", flush=True)
    metrics = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key          = RunKey(water.name, beam.name, float(Z))
                w_arr, t_arr = raw_results[key]
                metrics[key] = compute_all_metrics(
                    w_arr, t_arr, SIM, water.c, Z
                )
    print("done.")

    # ── 4. Console summary tables ─────────────────────────────────────────
    print_summary_tables(metrics, SIM)

    # ── 5. Save figures ───────────────────────────────────────────────────
    print("  Saving figures ...")
    plot_all(metrics, SIM, save_dir=out_dir)

    # ── 6. Timing ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n  Total wall-clock time : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print("  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Optional: accept an output directory as the first CLI argument
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/outputs"
    main(out_dir=out)
