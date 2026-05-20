"""
uowc.simulation
===============
Parallel orchestration layer.

Separation-of-Concern role
---------------------------
  This module owns one thing: how to distribute photon batches across CPU
  cores and collect results.  It knows about workers and seeds, but nothing
  about optical physics, channel metrics, or plotting.

  The public surface is a single function `run_sweep()` that iterates over
  all (water, beam, link_range) combinations and returns raw photon data
  (captured weights + times-of-flight) keyed by a named tuple.

Parallelism strategy
--------------------
  ProcessPoolExecutor is used (not ThreadPoolExecutor) because each worker
  is CPU-bound NumPy work, which is subject to the GIL.  Subprocess
  separation also means each worker gets its own memory space and cannot
  accidentally share mutable state.

  SeedSequence.spawn() derives provably independent 128-bit seeds for each
  worker.  Using the legacy np.random.seed() / np.random.rand() API inside
  forked processes would give silent statistical errors (numpy issue #9650).
"""

from __future__ import annotations
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

from uowc.config import (
    WaterParams, BeamParams, SimConfig,
    RECEIVER, ALL_WATERS, ALL_BEAMS,
)
from uowc.transport import propagate_batch


# ─────────────────────────────────────────────────────────────────────────────
# Result key
# ─────────────────────────────────────────────────────────────────────────────

class RunKey(NamedTuple):
    """Unique identifier for one (water, beam, link_range) combination."""
    water_name: str
    beam_name:  str
    link_range: float


# ─────────────────────────────────────────────────────────────────────────────
# Single-combination runner
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    water: WaterParams,
    beam:  BeamParams,
    link_range_m: float,
    cfg:  SimConfig,
    seed: np.random.SeedSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribute `cfg.n_photons` across `cfg.n_workers` processes for a single
    (water, beam, link_range) combination.

    Parameters
    ----------
    water, beam   : optical and geometric configuration
    link_range_m  : receiver depth / link length (m)
    cfg           : simulation parameters
    seed          : SeedSequence for this combination (caller-managed)

    Returns
    -------
    (weights, times_of_flight) — NumPy arrays of captured photon data
    """
    nw      = cfg.n_workers
    base    = cfg.n_photons // nw
    rem     = cfg.n_photons  % nw
    batches = [base + (1 if i < rem else 0) for i in range(nw)]

    worker_seeds = seed.spawn(nw)

    args_list = [
        (
            batches[i],
            worker_seeds[i].generate_state(4, dtype=np.uint64),
            water.c, water.b, water.g, water.omega,
            beam.divergence_rad, beam.waist_m,
            RECEIVER.aperture_radius_m, RECEIVER.fov_rad,
            link_range_m,
            cfg.weight_threshold, cfg.roulette_m,
            cfg.chunk_size,
        )
        for i in range(nw)
    ]

    all_w: List[np.ndarray] = []
    all_t: List[np.ndarray] = []

    with ProcessPoolExecutor(max_workers=nw) as ex:
        for w_arr, t_arr in ex.map(propagate_batch, args_list):
            if w_arr.size > 0:
                all_w.append(w_arr)
                all_t.append(t_arr)

    if all_w:
        return np.concatenate(all_w), np.concatenate(all_t)
    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Full parameter sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    cfg: SimConfig,
    *,
    waters: Tuple[WaterParams, ...] = ALL_WATERS,
    beams:  Tuple[BeamParams,  ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, Tuple[np.ndarray, np.ndarray]]:
    """
    Run the full (water × beam × range) parameter sweep.

    Returns a dict mapping each `RunKey` to `(weights, times_of_flight)`.
    All downstream analysis (metrics, plotting) consumes this dict.

    Parameters
    ----------
    cfg     : simulation configuration
    waters  : water types to sweep (default: all presets)
    beams   : beam types to sweep  (default: all presets)
    verbose : print per-combination timing and key metrics if True
    """
    n_combos    = len(waters) * len(beams) * len(cfg.link_ranges_m)
    root_ss     = np.random.SeedSequence(cfg.master_seed)
    combo_seeds = root_ss.spawn(n_combos)

    results: Dict[RunKey, Tuple[np.ndarray, np.ndarray]] = {}
    seed_idx = 0

    if verbose:
        print(f"\n  Sweep: {n_combos} combinations  |  "
              f"{cfg.n_photons:,} photons each  |  "
              f"{cfg.n_workers} worker(s)\n")
        _print_sweep_header()

    for water in waters:
        for beam in beams:
            for Z in cfg.link_ranges_m:
                key = RunKey(water.name, beam.name, float(Z))
                t0  = time.perf_counter()

                w_arr, t_arr = run_one(
                    water, beam, Z, cfg, combo_seeds[seed_idx]
                )
                seed_idx += 1

                results[key] = (w_arr, t_arr)

                if verbose:
                    elapsed = time.perf_counter() - t0
                    n_cap   = w_arr.size
                    p_norm  = w_arr.sum() / cfg.n_photons if n_cap > 0 else 0.0
                    p_dB    = 10.0 * np.log10(p_norm + 1e-300)
                    _print_sweep_row(
                        water.name, beam.name, Z, elapsed,
                        p_dB, n_cap, seed_idx, n_combos,
                    )

    if verbose:
        print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers — console progress display only
# ─────────────────────────────────────────────────────────────────────────────

def _print_sweep_header() -> None:
    col = "{:<15} {:<24} {:>5}  {:>9}  {:>8}  {:>10}  {:>5}"
    print("  " + col.format("Water", "Beam", "Z(m)",
                             "P_norm(dB)", "N_cap", "Time(s)", "Done"))
    print("  " + "-" * 82)


def _print_sweep_row(
    water_name: str, beam_name: str, Z: float,
    elapsed: float, p_dB: float, n_cap: int,
    done: int, total: int,
) -> None:
    col = "{:<15} {:<24} {:>5.0f}  {:>9.3f}  {:>8,}  {:>10.2f}  {:>5}"
    print("  " + col.format(
        water_name[:15], beam_name[:24], Z,
        p_dB, n_cap, elapsed, f"{done}/{total}",
    ))
