"""
uowc.transport
==============
Monte Carlo photon propagation engine.

Separation-of-Concern role
---------------------------
  This module owns only the mechanics of moving photons through the medium
  and deciding which ones reach the receiver.  It calls `uowc.physics` for
  all optical calculations and reads geometry from `uowc.config`.
  It has no knowledge of metrics, plotting, or parallelism orchestration.

Algorithm (MCML lineage)
------------------------
  For each photon packet:
    1. Sample free-flight step  s = -ln(ξ) / c         [physics.sample_step_length]
    2. Advance position
    3. Check receiver crossing at z = link_range
    4. Update weight  W *= ω                            [physics.update_weight]
    5. Russian roulette if W < threshold                [physics.russian_roulette]
    6. Sample HG scattering angle and rotate direction  [physics.sample_hg_cos_theta,
                                                          physics.rotate_direction]

References
----------
  Gabriel et al. (2013)  JOCN 5(1):1-12
  Cox (2012)  NC State PhD Dissertation
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Tuple

from uowc.physics import (
    sample_step_length,
    update_weight,
    russian_roulette,
    sample_hg_cos_theta,
    rotate_direction,
    sample_launch_positions,
    sample_launch_directions,
    path_length_to_tof,
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-process worker  (called from the parallel runner via ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch(args: tuple) -> Tuple[ndarray, ndarray]:
    """
    Propagate `n_photons` packets and return (weights, times_of_flight)
    for all packets captured at the receiver plane z = link_range.

    This function is the unit of work dispatched to each worker process.
    It is intentionally free of global state so that it survives the
    `pickle → subprocess → unpickle` round-trip cleanly.

    Parameters  (passed as a single tuple for ProcessPoolExecutor.map)
    ----------
    n_photons      : int   — packets to launch in this batch
    seed_state     : 4-element uint64 array from SeedSequence.generate_state
    c, b, g        : water optical properties
    omega          : single-scattering albedo  b/c
    beam_div       : beam half-angle (rad)
    beam_waist     : beam waist radius (m)
    rx_radius      : receiver aperture radius (m)
    rx_fov         : receiver FOV half-angle (rad)
    link_range     : target depth / link length (m)
    weight_threshold, roulette_m : Monte Carlo controls
    chunk_size     : mini-batch size (vectorisation granularity)

    Returns
    -------
    (captured_weights, captured_tof)  — both shape (M,), M ≤ n_photons
    """
    (n_photons, seed_state,
     c, b, g, omega,
     beam_div, beam_waist,
     rx_radius, rx_fov,
     link_range,
     weight_threshold, roulette_m,
     chunk_size) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))

    cap_w: list[ndarray] = []
    cap_t: list[ndarray] = []

    remaining = n_photons
    while remaining > 0:
        N          = min(chunk_size, remaining)
        remaining -= N

        # ── 1. Launch ─────────────────────────────────────────────────────
        x, y = sample_launch_positions(beam_waist, rng, N)
        z    = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)

        w     = np.ones(N)
        L     = np.zeros(N)      # cumulative optical path length (m)
        alive = np.ones(N, dtype=bool)

        # ── 2. Transport loop ─────────────────────────────────────────────
        for _ in range(8_000):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            n   = idx.size

            # Step 1 — free-flight
            s  = sample_step_length(c, rng.uniform(size=n))
            xn = x[idx] + s * ux[idx]
            yn = y[idx] + s * uy[idx]
            zn = z[idx] + s * uz[idx]

            # Step 2 — receiver crossing detection
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s,
                    link_range, rx_radius, rx_fov,
                    w, alive, cap_w, cap_t,
                )

            # Step 3 — retire and update survivors
            alive_mask_in_step = alive.copy()
            alive_mask_in_step[idx[cross]] = False
            si = np.where(alive_mask_in_step)[0]
            if si.size == 0:
                continue

            # Remap step arrays to survivor subset
            step_mask = np.isin(idx, si)
            s_si  = s[step_mask]
            xn_si = xn[step_mask]
            yn_si = yn[step_mask]
            zn_si = zn[step_mask]

            x[si] = xn_si
            y[si] = yn_si
            z[si] = zn_si
            L[si] += s_si

            # Kill photons that have left the simulation volume
            out = (z[si] < 0) | (z[si] > link_range * 3.0) \
                              | (np.hypot(x[si], y[si]) > 10.0)
            alive[si[out]] = False
            active = si[~out]
            if active.size == 0:
                continue

            # Step 4 — weight update (implicit absorption)
            update_weight(w[active], omega)

            # Step 5 — Russian roulette
            rr_xi = rng.uniform(size=N)
            russian_roulette(w, alive, rr_xi, weight_threshold, roulette_m)
            active = np.where(alive)[0]
            if active.size == 0:
                continue

            # Step 6 — HG scattering + direction rotation
            cos_sc = sample_hg_cos_theta(g, rng.uniform(size=active.size))
            sin_sc = np.sqrt(np.maximum(0.0, 1.0 - cos_sc ** 2))
            phi_sc = rng.uniform(0.0, 2.0 * np.pi, active.size)
            ux[active], uy[active], uz[active] = rotate_direction(
                ux[active], uy[active], uz[active],
                cos_sc, sin_sc, phi_sc,
            )

    if cap_w:
        return np.concatenate(cap_w), np.concatenate(cap_t)
    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: process photons that cross the receiver plane
# ─────────────────────────────────────────────────────────────────────────────

def _handle_receiver_crossing(
    cross: ndarray,       # boolean mask within step-idx
    idx: ndarray,         # global indices of photons in this step
    x: ndarray, y: ndarray, z: ndarray,
    ux: ndarray, uy: ndarray, uz: ndarray,
    L: ndarray,
    s: ndarray,           # step lengths for this iteration
    link_range: float,
    rx_radius: float,
    rx_fov: float,
    w: ndarray,
    alive: ndarray,
    cap_w: list,
    cap_t: list,
) -> None:
    """
    For photons that cross z = link_range in this step, compute the exact
    crossing point, check aperture and FOV acceptance, and append accepted
    photon (weight, time-of-flight) to `cap_w` / `cap_t`.
    Retired (crossed) photons are marked dead in `alive`.
    """
    ci    = idx[cross]
    uz_ci = uz[ci]
    s_ci  = s[cross]

    # Exact fraction of step at the receiver plane
    safe_uz = np.where(np.abs(uz_ci) > 1e-12, uz_ci, 1e-12)
    frac    = np.clip((link_range - z[ci]) / (safe_uz * s_ci), 0.0, 1.0)

    x_rx = x[ci] + frac * s_ci * ux[ci]
    y_rx = y[ci] + frac * s_ci * uy[ci]
    L_rx = L[ci] + frac * s_ci

    r_rx    = np.hypot(x_rx, y_rx)
    cos_inc = np.clip(np.abs(uz_ci), 0.0, 1.0)
    ang_rx  = np.arccos(cos_inc)

    hit = (r_rx <= rx_radius) & (ang_rx <= rx_fov)
    if hit.any():
        cap_w.append(w[ci[hit]].copy())
        cap_t.append(path_length_to_tof(L_rx[hit]))

    alive[ci] = False
