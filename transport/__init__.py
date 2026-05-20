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

Two propagation workers
-----------------------
  propagate_batch             — original homogeneous path (unchanged API)
  propagate_batch_inhomogeneous — Woodcock delta-tracking for any MediumProfile

Both are free-standing picklable functions suitable for ProcessPoolExecutor.

Woodcock delta-tracking algorithm
----------------------------------
  For a medium with spatially varying c(z) ≤ c_max:

    1.  Sample a tentative step    s = -ln(ξ) / c_max
    2.  Advance photon to z_new
    3.  Check receiver crossing (happens regardless of collision type)
    4.  Evaluate local c_local = c(z_new)
    5.  Accept as real collision with probability p = c_local / c_max
        Real collision:
          a. Weight update   W *= ω(z_new) = b(z_new) / c(z_new)
          b. Russian roulette if W < threshold
          c. HG scatter with local g(z_new)
        Null collision:
          — photon position is already updated; no other action

  The thinning theorem of Poisson processes guarantees that the sequence of
  real collisions reproduces the inhomogeneous interaction rate c(z) exactly,
  with no bias relative to the true physics.

References
----------
  Gabriel et al. (2013)  JOCN 5(1):1-12
  Cox (2012)  NC State PhD Dissertation
  Woodcock et al. (1965) — original delta-tracking paper
  Lux & Koblinger (1991) — Monte Carlo Particle Transport, CRC Press
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Tuple

from uowc.physics import (
    sample_step_length,
    sample_step_woodcock,
    accept_real_collision,
    update_weight,
    update_weight_array,
    russian_roulette,
    russian_roulette_targeted,
    sample_hg_cos_theta,
    sample_hg_cos_theta_array,
    rotate_direction,
    sample_launch_positions,
    sample_launch_directions,
    path_length_to_tof,
)

_MAX_ITER = 8_000   # hard cap on transport loop iterations per photon batch


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper: receiver-plane crossing detection
# ─────────────────────────────────────────────────────────────────────────────

def _handle_receiver_crossing(
    cross:      ndarray,   # bool mask within step-idx
    idx:        ndarray,   # global indices of photons in this step
    x:          ndarray,
    y:          ndarray,
    z:          ndarray,
    ux:         ndarray,
    uy:         ndarray,
    uz:         ndarray,
    L:          ndarray,
    s:          ndarray,   # step lengths for this iteration (indexed by step-local i)
    link_range: float,
    rx_radius:  float,
    rx_fov:     float,
    w:          ndarray,
    alive:      ndarray,
    cap_w:      list,
    cap_t:      list,
) -> None:
    """
    For photons that cross z = link_range in this step, compute the exact
    crossing point via linear interpolation, check aperture and FOV
    acceptance, and append (weight, time-of-flight) for captured photons.

    Crossing is detected geometrically — it is independent of whether the
    step was a real or null Woodcock collision.  This ensures that photons
    do not overshoot the receiver plane even during null-collision steps.

    All crossed photons are retired (marked dead) regardless of acceptance.
    """
    ci    = idx[cross]
    uz_ci = uz[ci]
    s_ci  = s[cross]

    # Fraction of the step at which the photon crosses z = link_range
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


# ─────────────────────────────────────────────────────────────────────────────
# Worker 1: Homogeneous medium  (original algorithm — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch(args: tuple) -> Tuple[ndarray, ndarray]:
    """
    Propagate photons through a homogeneous medium.

    This is the original MCML-lineage implementation, preserved without
    modification.  It is dispatched by `simulation.run_one` when the
    medium is homogeneous (scalar c, b, g, omega).

    Parameters  (passed as a single tuple for ProcessPoolExecutor.map)
    ----------
    n_photons      : int   — packets to launch in this batch
    seed_state     : 4-element uint64 array from SeedSequence.generate_state
    c, b, g        : water optical properties (scalars)
    omega          : single-scattering albedo  b/c  (scalar)
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
        L     = np.zeros(N)
        alive = np.ones(N, dtype=bool)

        # ── 2. Transport loop ─────────────────────────────────────────────
        for _ in range(_MAX_ITER):
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

            # Step 3 — retire crossed, update survivors
            alive_mask = alive.copy()
            alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0:
                continue

            step_mask = np.isin(idx, si)
            x[si] = xn[step_mask]
            y[si] = yn[step_mask]
            z[si] = zn[step_mask]
            L[si] += s[step_mask]

            # Kill out-of-bounds
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
# Worker 2: Inhomogeneous medium  (Woodcock delta-tracking)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch_inhomogeneous(args: tuple) -> Tuple[ndarray, ndarray]:
    """
    Propagate photons through an inhomogeneous medium using Woodcock
    delta-tracking.

    This worker is dispatched by `simulation.run_one_inhomogeneous` for any
    medium profile where `medium.is_homogeneous() == False`.  It also works
    correctly for homogeneous media, though `propagate_batch` is faster
    in that case.

    Key differences from `propagate_batch`
    ---------------------------------------
    1. Steps are sampled with the majorant c_max:  s = -ln(ξ) / c_max.
    2. After advancing to the new position, a Woodcock acceptance test
       determines whether the collision is real or null.
    3. Weight update and HG scattering are applied only at real collisions,
       using the LOCAL values  ω(z) and g(z) at the photon's current depth.
    4. Russian roulette is applied only to photons with real collisions
       whose weight just decreased (using `russian_roulette_targeted`).
    5. Receiver crossing is detected geometrically every step, including
       steps that end in null collisions — a photon must not overshoot
       the receiver plane just because its collision was virtual.

    Performance notes
    -----------------
    - Null-collision overhead: roughly (1 - c_mean/c_max) extra steps.
      For clear→coastal→turbid: c_mean ≈ 0.5 m⁻¹, c_max = 2.19 m⁻¹,
      so ~77 % of steps are null.  However, null steps are cheap: one
      position update + one comparison, no RNG for scattering.
    - IOP lookup: `LayeredMedium._lookup` uses np.searchsorted → O(N log K).
      For K ≤ 10 layers and N = 10 000 photons this costs < 0.1 ms per step.
    - For a homogeneous medium wrapped in HomogeneousMedium, this path
      produces ~c_max/c_max = 100 % real collisions and is equivalent to
      `propagate_batch`, just slightly slower due to array IOP lookups.

    Parameters  (passed as a single tuple for ProcessPoolExecutor.map)
    ----------
    n_photons      : int
    seed_state     : 4-element uint64 array
    medium         : MediumProfile — any of HomogeneousMedium, LayeredMedium,
                     GradientMedium, or a user-defined subclass
    beam_div       : beam half-angle (rad)
    beam_waist     : beam waist radius (m)
    rx_radius      : receiver aperture radius (m)
    rx_fov         : receiver FOV half-angle (rad)
    link_range     : target depth / link length (m)
    weight_threshold, roulette_m : Monte Carlo controls
    chunk_size     : mini-batch size

    Returns
    -------
    (captured_weights, captured_tof) — both shape (M,)
    """
    (n_photons, seed_state,
     medium,
     beam_div, beam_waist,
     rx_radius, rx_fov,
     link_range,
     weight_threshold, roulette_m,
     chunk_size) = args

    rng   = np.random.default_rng(np.random.PCG64(seed_state))
    c_max = medium.c_max

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
        L     = np.zeros(N)
        alive = np.ones(N, dtype=bool)

        # ── 2. Transport loop ─────────────────────────────────────────────
        for _ in range(_MAX_ITER):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            n   = idx.size

            # Step 1 — sample tentative step with majorant c_max
            s  = sample_step_woodcock(c_max, rng.uniform(size=n))
            xn = x[idx] + s * ux[idx]
            yn = y[idx] + s * uy[idx]
            zn = z[idx] + s * uz[idx]

            # Step 2 — receiver crossing (geometric; independent of collision type)
            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s,
                    link_range, rx_radius, rx_fov,
                    w, alive, cap_w, cap_t,
                )

            # Step 3 — retire crossed, update survivors
            alive_mask = alive.copy()
            alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0:
                continue

            step_mask = np.isin(idx, si)
            x[si] = xn[step_mask]
            y[si] = yn[step_mask]
            z[si] = zn[step_mask]
            L[si] += s[step_mask]

            # Kill out-of-bounds
            out = (z[si] < 0) | (z[si] > link_range * 3.0) \
                              | (np.hypot(x[si], y[si]) > 10.0)
            alive[si[out]] = False
            active = si[~out]
            if active.size == 0:
                continue

            # ── Woodcock acceptance test ──────────────────────────────────
            # Step 4 — query local IOPs at current photon positions
            c_local = medium.attenuation(z[active])

            # Step 5 — Bernoulli test: real (True) or null (False) collision
            real_mask = accept_real_collision(
                c_local, c_max, rng.uniform(size=active.size)
            )
            real_idx  = active[real_mask]

            # Null-collision photons: position already updated above.
            # No weight change, no scattering — continue to next iteration.
            if real_idx.size == 0:
                continue

            # ── Real-collision operations (subset of active photons) ──────

            # Step 6 — weight update with local single-scattering albedo
            omega_local = medium.albedo(z[real_idx])
            update_weight_array(w[real_idx], omega_local)

            # Step 7 — Russian roulette (only photons whose weight just fell)
            rr_xi = rng.uniform(size=real_idx.size)
            russian_roulette_targeted(
                w, alive, real_idx, rr_xi, weight_threshold, roulette_m
            )

            # Restrict scattering to photons still alive after roulette
            scatter_idx = real_idx[alive[real_idx]]
            if scatter_idx.size == 0:
                continue

            # Step 8 — HG scattering with local asymmetry parameter
            g_local = medium.asymmetry(z[scatter_idx])
            cos_sc  = sample_hg_cos_theta_array(
                g_local, rng.uniform(size=scatter_idx.size)
            )
            sin_sc  = np.sqrt(np.maximum(0.0, 1.0 - cos_sc ** 2))
            phi_sc  = rng.uniform(0.0, 2.0 * np.pi, scatter_idx.size)
            ux[scatter_idx], uy[scatter_idx], uz[scatter_idx] = rotate_direction(
                ux[scatter_idx], uy[scatter_idx], uz[scatter_idx],
                cos_sc, sin_sc, phi_sc,
            )

    if cap_w:
        return np.concatenate(cap_w), np.concatenate(cap_t)
    return np.array([], dtype=np.float64), np.array([], dtype=np.float64)