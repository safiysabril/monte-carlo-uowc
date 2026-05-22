"""
uowc/main.py
============
Application entry point — wiring layer only.

Pipeline
--------
  1. Run adaptive simulations (homogeneous + inhomogeneous)
  2. Compute channel metrics (CIR, delay spread, bandwidth)
  3. Convert RunResults → Pandas DataFrame
  4. Compute capture statistics (with correct n_launched denominators)
  5. Save DataFrame to Parquet (Strategy A — full in-memory then write)
  6. Generate channel figures (plotting module)
  7. Generate diagnostic figures (analysis.plots module)
"""

from __future__ import annotations
import os, sys, time

from uowc.config    import SIM, ALL_WATERS, ALL_BEAMS
from uowc.medium    import ALL_INHOMOGENEOUS_MEDIA
from uowc.reporting import (print_run_header, print_summary_tables,
                             print_inhomogeneous_header,
                             print_inhomogeneous_summary)
from uowc.simulation import (RunKey, run_sweep_adaptive,
                              run_sweep_inhomogeneous_adaptive)
from uowc.metrics   import compute_all_metrics
from uowc.plotting  import plot_all, plot_all_inhomogeneous
from uowc.analysis  import (to_dataframe, to_parquet,
                             capture_statistics_with_launched)
from uowc.analysis.plots import plot_all_diagnostics


def main(out_dir: str = "/uowc/outputs") -> None:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()

    # ── 1. Homogeneous sweep ─────────────────────────────────────────────
    print_run_header(SIM)
    raw_hom = run_sweep_adaptive(SIM, verbose=True)

    metrics_hom = {}
    for water in ALL_WATERS:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key = RunKey(water.name, beam.name, float(Z))
                metrics_hom[key] = compute_all_metrics(
                    raw_hom[key], SIM, water.c, Z)
    print_summary_tables(metrics_hom, SIM)
    plot_all(metrics_hom, SIM, save_dir=out_dir)

    # ── 2. Inhomogeneous sweep ───────────────────────────────────────────
    print_inhomogeneous_header(SIM, ALL_INHOMOGENEOUS_MEDIA)
    raw_inh = run_sweep_inhomogeneous_adaptive(
        SIM, media=ALL_INHOMOGENEOUS_MEDIA, verbose=True)

    metrics_inh = {}
    for medium in ALL_INHOMOGENEOUS_MEDIA:
        for beam in ALL_BEAMS:
            for Z in SIM.link_ranges_m:
                key = RunKey(medium.name, beam.name, float(Z))
                metrics_inh[key] = compute_all_metrics(
                    raw_inh[key], SIM, medium.c_max, Z)
    print_inhomogeneous_summary(metrics_inh, SIM, ALL_INHOMOGENEOUS_MEDIA)
    plot_all_inhomogeneous(metrics_inh, SIM, ALL_INHOMOGENEOUS_MEDIA,
                           save_dir=out_dir)

    # ── 3. Build Pandas DataFrames ───────────────────────────────────────
    print("\n  Building DataFrames ...")
    df_hom = to_dataframe(raw_hom)
    df_inh = to_dataframe(raw_inh)

    to_parquet(df_hom, os.path.join(out_dir, "photons_homogeneous.parquet"))
    to_parquet(df_inh, os.path.join(out_dir, "photons_inhomogeneous.parquet"))

    # ── 4. Capture statistics ────────────────────────────────────────────
    launched_hom = {k: v.n_launched for k, v in raw_hom.items()}
    launched_inh = {k: v.n_launched for k, v in raw_inh.items()}

    stats_hom = capture_statistics_with_launched(df_hom, launched_hom)
    stats_inh = capture_statistics_with_launched(df_inh, launched_inh)

    # ── 5. Diagnostic figures ────────────────────────────────────────────
    print("\n  Generating diagnostic figures ...")
    diag_dir = os.path.join(out_dir, "diagnostics")
    plot_all_diagnostics(df_hom, stats_hom, diag_dir)
    plot_all_diagnostics(df_inh, stats_inh, diag_dir,
                         medium_name=ALL_INHOMOGENEOUS_MEDIA[0].name)

    print(f"\n  Total: {time.perf_counter()-t0:.1f} s\n  Done.\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/uowc/outputs"
    main(out)