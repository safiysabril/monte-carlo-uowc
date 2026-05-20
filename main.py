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
    ├── medium          (LayeredMedium, GradientMedium, ALL_INHOMOGENEOUS_MEDIA)
    ├── reporting       (print_run_header, print_summary_tables,
    │                    print_inhomogeneous_header, print_inhomogeneous_summary)
    ├── simulation      (run_sweep, run_sweep_inhomogeneous, RunKey)
    ├── metrics         (compute_all_metrics)
    └── plotting        (plot_all, plot_all_inhomogeneous)
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
from simulation import run_sweep, run_sweep_inhomogeneous, RunKey
from metrics   import compute_all_metrics
from plotting  import plot_all, plot_all_inhomogeneous


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(raw_results, waters_or_media, beams, cfg):
    """Compute metrics for every key in raw_results."""
    metrics = {}
    for entity in waters_or_media:
        for beam in beams:
            for Z in cfg.link_ranges_m:
                # entity is either WaterParams (has .c) or MediumProfile (has .c_max)
                c_ref = entity.c if hasattr(entity, 'c') else entity.c_max
                key          = RunKey(entity.name, beam.name, float(Z))
                w_arr, t_arr = raw_results[key]
                metrics[key] = compute_all_metrics(w_arr, t_arr, cfg, c_ref, Z)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(out_dir: str = "/mnt/user-data/outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    t_start = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════════
    # Part 1 — Homogeneous medium  (original pipeline, unchanged)
    # ═══════════════════════════════════════════════════════════════════════

    print_run_header(SIM)

    print("  Running homogeneous sweep ...")
    raw_homogeneous = run_sweep(SIM, verbose=True)

    print("  Computing homogeneous metrics ...", end=" ", flush=True)
    metrics_homogeneous = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key          = RunKey(water.name, beam.name, float(Z))
                w_arr, t_arr = raw_homogeneous[key]
                metrics_homogeneous[key] = compute_all_metrics(
                    w_arr, t_arr, SIM, water.c, Z
                )
    print("done.")

    print_summary_tables(metrics_homogeneous, SIM)

    print("  Saving homogeneous figures ...")
    plot_all(metrics_homogeneous, SIM, save_dir=out_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # Part 2 — Inhomogeneous media  (Woodcock delta-tracking)
    # ═══════════════════════════════════════════════════════════════════════

    print_inhomogeneous_header(SIM, ALL_INHOMOGENEOUS_MEDIA)

    print("  Running inhomogeneous sweep ...")
    raw_inhomogeneous = run_sweep_inhomogeneous(
        SIM,
        media=ALL_INHOMOGENEOUS_MEDIA,
        verbose=True,
    )

    print("  Computing inhomogeneous metrics ...", end=" ", flush=True)
    metrics_inhomogeneous = {}
    for medium in ALL_INHOMOGENEOUS_MEDIA:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key          = RunKey(medium.name, beam.name, float(Z))
                w_arr, t_arr = raw_inhomogeneous[key]
                # Use c_max as the representative attenuation for Beer-Lambert ref.
                # In practice, the effective attenuation is the path-weighted mean.
                metrics_inhomogeneous[key] = compute_all_metrics(
                    w_arr, t_arr, SIM, medium.c_max, Z
                )
    print("done.")

    print_inhomogeneous_summary(metrics_inhomogeneous, SIM, ALL_INHOMOGENEOUS_MEDIA)

    print("  Saving inhomogeneous figures ...")
    plot_all_inhomogeneous(
        metrics_inhomogeneous, SIM, ALL_INHOMOGENEOUS_MEDIA, save_dir=out_dir
    )

    # ═══════════════════════════════════════════════════════════════════════
    # Wall-clock summary
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.perf_counter() - t_start
    print(f"\n  Total wall-clock time: {elapsed:.1f} s")
    print("  Done.\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    main(out)