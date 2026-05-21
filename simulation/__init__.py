"""
uowc.simulation
===============
Parallel orchestration layer.

Separation-of-Concern role
---------------------------
  This module owns one thing: how to distribute photon batches across CPU
  cores and collect results.  It knows about workers and seeds, but nothing
  about optical physics, channel metrics, or plotting.

Public API
----------
  RunResult          — named tuple: (weights, times, n_launched)
  RunKey             — named tuple: (water_name, beam_name, link_range)

  Fixed-launch sweeps (one batch of n_photons per run):
    run_sweep(cfg)
    run_sweep_inhomogeneous(cfg, media)

  Adaptive sweeps (repeat batches until min_captured_photons):
    run_sweep_adaptive(cfg)
    run_sweep_inhomogeneous_adaptive(cfg, media)

  Low-level single-run functions (used by sweeps and tests):
    run_one(water, beam, Z, cfg, seed)
    run_one_inhomogeneous(medium, beam, Z, cfg, seed)
    run_one_adaptive(water, beam, Z, cfg, seed)
    run_one_inhomogeneous_adaptive(medium, beam, Z, cfg, seed)

RunResult vs plain tuple
------------------------
  All functions return RunResult(weights, times, n_launched).
  n_launched tracks how many photons were actually launched so that
  received_power_dB can normalise correctly — this matters in adaptive
  mode where n_launched differs per run.

  RunResult unpacks like a 3-tuple:
      w, t, n = result          # explicit
      w, t, _ = result          # ignore n_launched if not needed

Adaptive algorithm
------------------
  1. Launch one batch of cfg.n_photons (distributed across workers).
  2. Accumulate captured weights and times.
  3. Repeat until:
       (a) cumulative captured ≥ cfg.min_captured_photons  (success), or
       (b) cumulative launched ≥ cfg.max_launched_photons  (cap hit).
  4. Return RunResult with the union of all captured photons and the
     total launched count.

  Each round uses fresh seeds derived from the parent SeedSequence by
  calling seed.spawn(n_workers) — SeedSequence tracks an internal spawn
  counter, so successive calls automatically produce independent sub-seeds.

Seed management
---------------
  SeedSequence.spawn() is stateful: each call advances an internal counter
  and returns new child sequences that are provably independent.  This means
  calling run_one_adaptive with the same parent seed always produces the
  same sequence of random numbers regardless of how many rounds are needed —
  reproducibility is guaranteed even when the number of rounds varies across
  runs (e.g. because the capture rate changed).
"""

from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from config import (
    WaterParams, BeamParams, SimConfig,
    RECEIVER, ALL_WATERS, ALL_BEAMS,
)
from medium import MediumProfile
from transport import propagate_batch, propagate_batch_inhomogeneous


# ─────────────────────────────────────────────────────────────────────────────
# Public named tuples
# ─────────────────────────────────────────────────────────────────────────────

class RunKey(NamedTuple):
    """Unique identifier for one (water-or-medium, beam, link_range) run."""
    water_name: str
    beam_name:  str
    link_range: float


class RunResult(NamedTuple):
    """
    Output of every single-run and sweep function.

    Attributes
    ----------
    weights    : captured photon weights, shape (M,)
    times      : captured times-of-flight (s), shape (M,)
    n_launched : total photons launched (including all adaptive rounds)

    n_launched is the correct denominator for received_power_dB.
    In fixed mode it equals cfg.n_photons; in adaptive mode it is
    n_rounds × cfg.n_photons and varies per (medium, beam, range).
    """
    weights:    np.ndarray
    times:      np.ndarray
    n_launched: int


# ─────────────────────────────────────────────────────────────────────────────
# Internal: dispatch one batch of n_photons to worker pool
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_batch_homogeneous(
    water:        WaterParams,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """Launch cfg.n_photons across workers; return (weights, times)."""
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


def _dispatch_batch_inhomogeneous(
    medium:       MediumProfile,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """Launch cfg.n_photons (Woodcock) across workers; return (weights, times)."""
    nw      = cfg.n_workers
    base    = cfg.n_photons // nw
    rem     = cfg.n_photons  % nw
    batches = [base + (1 if i < rem else 0) for i in range(nw)]
    worker_seeds = seed.spawn(nw)

    args_list = [
        (
            batches[i],
            worker_seeds[i].generate_state(4, dtype=np.uint64),
            medium,
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
# Internal: adaptive loop
# ─────────────────────────────────────────────────────────────────────────────

def _adaptive_loop(
    dispatch_fn,          # callable(seed) -> (weights, times)
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
    label:        str,    # for verbose output
    verbose:      bool,
) -> RunResult:
    """
    Repeatedly call dispatch_fn until min_captured_photons is reached
    or max_launched_photons is exhausted.

    dispatch_fn must accept a single SeedSequence argument.
    seed.spawn(n_workers) is called inside each dispatch — because
    SeedSequence.spawn() is stateful, each call to dispatch_fn automatically
    gets fresh, independent sub-seeds.

    Parameters
    ----------
    dispatch_fn : one-argument callable(seed) that runs one batch and
                  returns (weights_array, times_array)
    cfg         : simulation configuration (min_captured, max_launched, n_photons)
    seed        : parent SeedSequence (mutated by each .spawn() call)
    label       : short string for progress output
    verbose     : print per-round progress if True
    """
    all_w: List[np.ndarray] = []
    all_t: List[np.ndarray] = []
    n_captured  = 0
    n_launched  = 0
    round_num   = 0
    cap_hit     = False

    min_cap = cfg.min_captured_photons
    max_lau = cfg.max_launched_photons

    while True:
        round_num += 1
        w, t = dispatch_fn(seed)

        n_launched += cfg.n_photons
        if w.size > 0:
            all_w.append(w)
            all_t.append(t)
            n_captured += w.size

        if verbose:
            rate = n_captured / n_launched * 100 if n_launched > 0 else 0.0
            print(
                f"      round {round_num:>3}: "
                f"+{w.size:>6,} captured  |  "
                f"total {n_captured:>7,} / {min_cap:,}  "
                f"({rate:.3f}%)  "
                f"[{n_launched / 1e6:.1f} M launched]"
            )

        # ── Stop conditions ──────────────────────────────────────────────
        if n_captured >= min_cap:
            break
        if n_launched >= max_lau:
            cap_hit = True
            break

    if cap_hit:
        print(
            f"  ⚠  {label}: cap hit at {n_launched:,} launched — "
            f"only {n_captured:,} captured (target {min_cap:,}). "
            f"Metrics may be noisy."
        )

    if all_w:
        return RunResult(np.concatenate(all_w), np.concatenate(all_t), n_launched)
    return RunResult(np.array([], dtype=np.float64),
                     np.array([], dtype=np.float64),
                     n_launched)


# ─────────────────────────────────────────────────────────────────────────────
# Public single-run functions
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    water:        WaterParams,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> RunResult:
    """Fixed launch: exactly cfg.n_photons — homogeneous medium."""
    w, t = _dispatch_batch_homogeneous(water, beam, link_range_m, cfg, seed)
    return RunResult(w, t, cfg.n_photons)


def run_one_inhomogeneous(
    medium:       MediumProfile,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
) -> RunResult:
    """Fixed launch: exactly cfg.n_photons — Woodcock inhomogeneous."""
    w, t = _dispatch_batch_inhomogeneous(medium, beam, link_range_m, cfg, seed)
    return RunResult(w, t, cfg.n_photons)


def run_one_adaptive(
    water:        WaterParams,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
    verbose:      bool = False,
) -> RunResult:
    """
    Adaptive launch — homogeneous medium.

    Launches batches of cfg.n_photons until at least
    cfg.min_captured_photons are captured, or cfg.max_launched_photons
    total photons have been launched (whichever comes first).

    Returns RunResult with n_launched = total photons launched across
    all rounds, which is the correct denominator for received_power_dB.
    """
    label = f"{water.name} | {beam.name} | {link_range_m:.0f} m"

    def dispatch(s):
        return _dispatch_batch_homogeneous(water, beam, link_range_m, cfg, s)

    return _adaptive_loop(dispatch, cfg, seed, label, verbose)


def run_one_inhomogeneous_adaptive(
    medium:       MediumProfile,
    beam:         BeamParams,
    link_range_m: float,
    cfg:          SimConfig,
    seed:         np.random.SeedSequence,
    verbose:      bool = False,
) -> RunResult:
    """
    Adaptive launch — Woodcock inhomogeneous medium.

    Same contract as run_one_adaptive but uses the Woodcock delta-tracking
    worker, accepting any MediumProfile (HomogeneousMedium, LayeredMedium,
    GradientMedium, or user-defined subclass).
    """
    label = f"{medium.name} | {beam.name} | {link_range_m:.0f} m"

    def dispatch(s):
        return _dispatch_batch_inhomogeneous(medium, beam, link_range_m, cfg, s)

    return _adaptive_loop(dispatch, cfg, seed, label, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Public sweep functions — fixed launch
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep(
    cfg:     SimConfig,
    *,
    waters:  Tuple[WaterParams, ...] = ALL_WATERS,
    beams:   Tuple[BeamParams,  ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, RunResult]:
    """
    Fixed-launch sweep over (water × beam × range) — homogeneous medium.

    Launches exactly cfg.n_photons per combination.  Use run_sweep_adaptive
    for depth-invariant capture counts.
    """
    ss      = np.random.SeedSequence(cfg.master_seed)
    combos  = [(w, b, Z) for w in waters for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, RunResult] = {}

    t0 = time.perf_counter()
    for (water, beam, Z), seed in zip(combos, seeds):
        key = RunKey(water.name, beam.name, float(Z))
        if verbose:
            print(f"  [{water.name} | {beam.name} | {Z:4.0f} m] ", end="", flush=True)
        result = run_one(water, beam, Z, cfg, seed)
        results[key] = result
        if verbose:
            pct = 100.0 * result.weights.size / result.n_launched
            print(f"captured {result.weights.size:>7,} / {result.n_launched:,}  "
                  f"({pct:.2f}%)  [{time.perf_counter()-t0:.1f} s]")

    return results


def run_sweep_inhomogeneous(
    cfg:     SimConfig,
    *,
    media:   Sequence[MediumProfile],
    beams:   Tuple[BeamParams, ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, RunResult]:
    """
    Fixed-launch sweep over (medium × beam × range) — inhomogeneous.

    Launches exactly cfg.n_photons per combination.
    """
    ss      = np.random.SeedSequence(cfg.master_seed + 1)
    combos  = [(m, b, Z) for m in media for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, RunResult] = {}

    t0 = time.perf_counter()
    for (medium, beam, Z), seed in zip(combos, seeds):
        key = RunKey(medium.name, beam.name, float(Z))
        if verbose:
            print(f"  [{medium.name} | {beam.name} | {Z:4.0f} m] ", end="", flush=True)
        result = run_one_inhomogeneous(medium, beam, Z, cfg, seed)
        results[key] = result
        if verbose:
            pct = 100.0 * result.weights.size / result.n_launched
            print(f"captured {result.weights.size:>7,} / {result.n_launched:,}  "
                  f"({pct:.2f}%)  [{time.perf_counter()-t0:.1f} s]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public sweep functions — adaptive launch
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_adaptive(
    cfg:     SimConfig,
    *,
    waters:  Tuple[WaterParams, ...] = ALL_WATERS,
    beams:   Tuple[BeamParams,  ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, RunResult]:
    """
    Adaptive sweep over (water × beam × range) — homogeneous medium.

    Each (water, beam, range) combination launches batches of cfg.n_photons
    until cfg.min_captured_photons captured photons are accumulated, or
    cfg.max_launched_photons is exhausted.

    RunResult.n_launched varies per combination: shallow, clear-water runs
    need one batch; deep, turbid runs may need many.  This variation is the
    correct information for computing received_power_dB — always pass
    result.n_launched as the denominator.

    verbose=True prints a summary line per combination.  For per-round
    progress inside each combination, use run_one_adaptive with verbose=True.
    """
    ss      = np.random.SeedSequence(cfg.master_seed)
    combos  = [(w, b, Z) for w in waters for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, RunResult] = {}

    t0 = time.perf_counter()
    for (water, beam, Z), seed in zip(combos, seeds):
        key   = RunKey(water.name, beam.name, float(Z))
        label = f"{water.name} | {beam.name} | {Z:.0f} m"
        if verbose:
            print(f"  [{label}]")

        result = run_one_adaptive(water, beam, Z, cfg, seed, verbose=verbose)
        results[key] = result

        if verbose:
            n_rounds = result.n_launched // cfg.n_photons
            rate     = (100.0 * result.weights.size / result.n_launched
                        if result.n_launched > 0 else 0.0)
            print(
                f"    → {result.weights.size:,} captured in {n_rounds} round(s)  "
                f"| {result.n_launched/1e6:.1f} M launched  "
                f"| capture rate {rate:.4f}%  "
                f"| {time.perf_counter()-t0:.1f} s total\n"
            )

    return results


def run_sweep_inhomogeneous_adaptive(
    cfg:     SimConfig,
    *,
    media:   Sequence[MediumProfile],
    beams:   Tuple[BeamParams, ...] = ALL_BEAMS,
    verbose: bool = True,
) -> Dict[RunKey, RunResult]:
    """
    Adaptive sweep over (medium × beam × range) — inhomogeneous.

    Same contract as run_sweep_adaptive but uses the Woodcock
    delta-tracking worker for any MediumProfile.
    """
    ss      = np.random.SeedSequence(cfg.master_seed + 1)
    combos  = [(m, b, Z) for m in media for b in beams for Z in cfg.link_ranges_m]
    seeds   = ss.spawn(len(combos))
    results: Dict[RunKey, RunResult] = {}

    t0 = time.perf_counter()
    for (medium, beam, Z), seed in zip(combos, seeds):
        key   = RunKey(medium.name, beam.name, float(Z))
        label = f"{medium.name} | {beam.name} | {Z:.0f} m"
        if verbose:
            print(f"  [{label}]")

        result = run_one_inhomogeneous_adaptive(medium, beam, Z, cfg, seed,
                                                verbose=verbose)
        results[key] = result

        if verbose:
            n_rounds = result.n_launched // cfg.n_photons
            rate     = (100.0 * result.weights.size / result.n_launched
                        if result.n_launched > 0 else 0.0)
            print(
                f"    → {result.weights.size:,} captured in {n_rounds} round(s)  "
                f"| {result.n_launched/1e6:.1f} M launched  "
                f"| capture rate {rate:.4f}%  "
                f"| {time.perf_counter()-t0:.1f} s total\n"
            )

    return results