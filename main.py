"""
uowc/main.py
============
Application entry point.

Wiring layer only — no physics, no transport, no metrics, no plot code.
Calls each module in the correct sequence and measures wall-clock time.

Sweep mode
----------
  Both homogeneous and inhomogeneous sweeps use the adaptive runners so
  that every (medium, beam, range) combination accumulates at least
  cfg.min_captured_photons captured photons before moving on.  This
  guarantees statistically meaningful CIR, delay spread, and bandwidth
  estimates at all depths without wasting photons at short ranges.

  RunResult.n_launched is passed to compute_all_metrics so that received
  power is normalised by the actual total launched — critical because
  n_launched varies per combination in adaptive mode.
"""

from __future__ import annotations
import os
import sys
import time

from config    import SIM, ALL_WATERS, ALL_BEAMS
from medium    import ALL_INHOMOGENEOUS_MEDIA
from reporting import (
    print_run_header, print_summary_tables,
    print_inhomogeneous_header, print_inhomogeneous_summary,
)
from simulation import (
    RunKey,
    run_sweep_adaptive,
    run_sweep_inhomogeneous_adaptive,
)
from metrics   import compute_all_metrics
from plotting  import plot_all, plot_all_inhomogeneous


def main(out_dir: str = "/mnt/user-data/outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════════
    # Part 1 — Homogeneous medium
    # ═══════════════════════════════════════════════════════════════════════
    print_run_header(SIM)

    print("  Running adaptive homogeneous sweep ...")
    raw_homogeneous = run_sweep_adaptive(SIM, verbose=True)

    print("  Computing homogeneous metrics ...", end=" ", flush=True)
    metrics_homogeneous = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key    = RunKey(water.name, beam.name, float(Z))
                result = raw_homogeneous[key]
                metrics_homogeneous[key] = compute_all_metrics(
                    result.weights, result.times, SIM,
                    water.c, Z,
                    n_launched=result.n_launched,   # ← adaptive denominator
                )
    print("done.")

    print_summary_tables(metrics_homogeneous, SIM)
    print("  Saving homogeneous figures ...")
    plot_all(metrics_homogeneous, SIM, save_dir=out_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # Part 2 — Inhomogeneous media (Woodcock delta-tracking)
    # ═══════════════════════════════════════════════════════════════════════
    print_inhomogeneous_header(SIM, ALL_INHOMOGENEOUS_MEDIA)

    print("  Running adaptive inhomogeneous sweep ...")
    raw_inhomogeneous = run_sweep_inhomogeneous_adaptive(
        SIM, media=ALL_INHOMOGENEOUS_MEDIA, verbose=True,
    )

    print("  Computing inhomogeneous metrics ...", end=" ", flush=True)
    metrics_inhomogeneous = {}
    for medium in ALL_INHOMOGENEOUS_MEDIA:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key    = RunKey(medium.name, beam.name, float(Z))
                result = raw_inhomogeneous[key]
                metrics_inhomogeneous[key] = compute_all_metrics(
                    result.weights, result.times, SIM,
                    medium.c_max, Z,
                    n_launched=result.n_launched,   # ← adaptive denominator
                )
    print("done.")

    print_inhomogeneous_summary(metrics_inhomogeneous, SIM, ALL_INHOMOGENEOUS_MEDIA)
    print("  Saving inhomogeneous figures ...")
    plot_all_inhomogeneous(
        metrics_inhomogeneous, SIM, ALL_INHOMOGENEOUS_MEDIA, save_dir=out_dir
    )

    elapsed = time.perf_counter() - t_start
    print(f"\n  Total wall-clock time: {elapsed:.1f} s\n  Done.\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    main(out)