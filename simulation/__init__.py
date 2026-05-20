"""
uowc.simulation
===============
Parallel orchestration layer.

Separation-of-Concern role
---------------------------
  This module owns one thing: how to distribute photon batches across CPU
  cores and collect results.  It knows about workers and seeds, but nothing
  about optical physics, channel metrics, or plotting.

  Two public sweeps are exposed:
    run_sweep              — original homogeneous sweep (unchanged API)
    run_sweep_inhomogeneous — inhomogeneous sweep over a MediumProfile

  Both return the same Dict[RunKey → (weights, times_of_flight)] so all
  downstream modules (metrics, plotting, reporting) work unchanged.

Parallelism strategy
--------------------
  ProcessPoolExecutor (not ThreadPoolExecutor): each worker is CPU-bound
  NumPy work subject to the GIL.  Subprocesses also prevent accidental
  shared-mutable-state bugs.

  SeedSequence.spawn() derives provably independent 128-bit seeds.
  The legacy np.random.seed() API inside forked processes gives silent
  statistical errors (numpy issue #9650) — never use it here.
"""

from __future__ import annotations
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, NamedTuple, Tuple, Optional

import numpy as np

from config import (
    WaterParams, BeamParams, SimConfig,
    RECEIVER, ALL_WATERS, ALL_BEAMS,
)
from transport import propagate_batch, propagate_batch_inhomogeneous


# ─────────────────────────────────────────────────────────────────────────────
# Result key
# ─────────────────────────────────────────────────────────────────────────────

class RunKey(NamedTuple):
    """Unique identifier for one (water-or-medium, beam, link_range) run."""
    water_name: str
    beam_name:  str
    link_range: float


# ─────────────────────────────────────────────────────────────────────────────
# Single-combination runner  (homogeneous)
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    water:        WaterParams,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribute `cfg.n_photons` across `cfg.n_workers` processes for a single
    (water, beam, link_range) combination — homogeneous medium.

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
# Single-combination runner  (inhomogeneous — Woodcock delta-tracking)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_inhomogeneous(
    medium:       object,   # any MediumProfile (HomogeneousMedium, LayeredMedium, ...)
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Distribute `cfg.n_photons` across workers for one (medium, beam, range)
    combination using the Woodcock delta-tracking worker.

    The `medium` object is pickled and sent to each worker process.
    All concrete MediumProfile subclasses are frozen dataclasses whose
    fields are plain Python scalars or tuples → pickle-safe.

    Parameters
    ----------
    medium        : MediumProfile — LayeredMedium, GradientMedium, or
                    HomogeneousMedium (the last routes to the Woodcock worker
                    but the homogeneous fast-path in `propagate_batch` is
                    slightly faster for purely uniform channels).
    beam          : transmitter geometry
    link_range_m  : receiver depth (m)
    cfg           : simulation knobs
    seed          : SeedSequence for this combination

    Returns
    -------
    (weights, times_of_flight) — captured photon arrays
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
            medium,                          # ← MediumProfile object
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
        for w_arr, t_arr in ex.map(propagate_batch_inhomogeneous, args_list):
            if w_arr.size > 0:
                all_w.append(w_arr)
                all_t.append(t_arr)

    if all_w:
        return np.concatenate(all_w), np.concatenate(all_t)
    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Full parameter sweep  (homogeneous — original, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    cfg:     SimConfig,
    *,
    waters:  Tuple[WaterParams, ...] = ALL_WATERS,
    beams:   Tuple[BeamParams,  ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, Tuple[np.ndarray, np.ndarray]]:
    """
    Run the full (water × beam × range) parameter sweep — homogeneous medium.

    Returns a dict mapping each `RunKey` to `(weights, times_of_flight)`.
    All downstream analysis (metrics, plotting) consumes this dict.
    """
    ss      = np.random.SeedSequence(cfg.master_seed)
    combos  = [(w, b, Z) for w in waters for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, Tuple[np.ndarray, np.ndarray]] = {}

    t0 = time.perf_counter()
    for (water, beam, Z), seed in zip(combos, seeds):
        key = RunKey(water.name, beam.name, float(Z))
        if verbose:
            print(f"  [{water.name} | {beam.name} | {Z:4.0f} m] ", end="", flush=True)
        w_arr, t_arr = run_one(water, beam, Z, cfg, seed)
        results[key] = (w_arr, t_arr)
        if verbose:
            n_cap = w_arr.size
            pct   = 100.0 * n_cap / cfg.n_photons
            print(f"captured {n_cap:>7,} / {cfg.n_photons:,}  ({pct:.2f} %)  "
                  f"[{time.perf_counter()-t0:.1f} s]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full parameter sweep  (inhomogeneous — new)
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_inhomogeneous(
    cfg:     SimConfig,
    *,
    media:   Tuple[object, ...],    # tuple of MediumProfile instances
    beams:   Tuple[BeamParams, ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, Tuple[np.ndarray, np.ndarray]]:
    """
    Inhomogeneous parameter sweep over (medium × beam × range).

    Accepts any iterable of MediumProfile instances (LayeredMedium,
    GradientMedium, or HomogeneousMedium).  The RunKey uses `medium.name`
    in place of `water_name` so the results dict has the same type as the
    homogeneous sweep output and can be passed to the same metrics /
    plotting routines.

    Parameters
    ----------
    cfg     : simulation configuration
    media   : tuple of MediumProfile objects to sweep over
    beams   : beam types to sweep (default: ALL_BEAMS)
    verbose : print per-run progress

    Returns
    -------
    Dict[RunKey → (weights, times_of_flight)]
    """
    ss      = np.random.SeedSequence(cfg.master_seed + 1)  # offset to avoid seed collision
    combos  = [(m, b, Z) for m in media for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, Tuple[np.ndarray, np.ndarray]] = {}

    t0 = time.perf_counter()
    for (medium, beam, Z), seed in zip(combos, seeds):
        key = RunKey(medium.name, beam.name, float(Z))
        if verbose:
            print(f"  [{medium.name} | {beam.name} | {Z:4.0f} m] ",
                  end="", flush=True)
        w_arr, t_arr = run_one_inhomogeneous(medium, beam, Z, cfg, seed)
        results[key] = (w_arr, t_arr)
        if verbose:
            n_cap = w_arr.size
            pct   = 100.0 * n_cap / cfg.n_photons
            print(f"captured {n_cap:>7,} / {cfg.n_photons:,}  ({pct:.2f} %)  "
                  f"[{time.perf_counter()-t0:.1f} s]")

    return results