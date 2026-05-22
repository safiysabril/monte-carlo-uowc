"""
uowc.transport
==============
Monte Carlo photon propagation engine.

Separation-of-Concern role
---------------------------
  This module owns only the mechanics of moving photons through the medium
  and deciding which ones reach the receiver.  It calls `uowc.physics` for
  all optical calculations and reads geometry from `uowc.config`.
  It has no knowledge of metrics, plotting, parallelism, or Pandas.

Two propagation workers
-----------------------
  propagate_batch              — homogeneous medium
  propagate_batch_inhomogeneous — Woodcock delta-tracking for any MediumProfile

Return format
-------------
  Each worker returns a dict of parallel NumPy arrays — one entry per
  captured photon.  This is the "structure-of-arrays" (SoA) layout, which
  is more cache-friendly than a list-of-dicts and maps directly onto a
  Pandas DataFrame via pd.DataFrame(result_dict).

  Keys returned:
    weight        float64   final photon weight
    tof_s         float64   time of flight (s)
    x_m           float32   x position at capture plane
    y_m           float32   y position at capture plane
    path_length_m float32   total optical path length
    n_scatters    int32     number of real HG scattering events
    n_nulls       int32     null (virtual) collisions  [inhom. only; 0 for hom.]

  Empty runs return the same dict with zero-length arrays so callers never
  need to check for None.
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Dict, Tuple

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

_MAX_ITER = 8_000


# ─────────────────────────────────────────────────────────────────────────────
# Internal type alias
# ─────────────────────────────────────────────────────────────────────────────
PhotonRecord = Dict[str, ndarray]

_EMPTY_RECORD: PhotonRecord = {
    "weight":        np.array([], dtype=np.float64),
    "tof_s":         np.array([], dtype=np.float64),
    "x_m":           np.array([], dtype=np.float32),
    "y_m":           np.array([], dtype=np.float32),
    "path_length_m": np.array([], dtype=np.float32),
    "n_scatters":    np.array([], dtype=np.int32),
    "n_nulls":       np.array([], dtype=np.int32),
}


def _empty_record() -> PhotonRecord:
    """Return a fresh empty record dict (avoid shared mutable default)."""
    return {k: v.copy() for k, v in _EMPTY_RECORD.items()}


def _concat_records(records: list[PhotonRecord]) -> PhotonRecord:
    """Concatenate a list of per-batch record dicts into one."""
    if not records:
        return _empty_record()
    return {k: np.concatenate([r[k] for r in records]) for k in _EMPTY_RECORD}


# ─────────────────────────────────────────────────────────────────────────────
# Shared receiver-crossing helper
# ─────────────────────────────────────────────────────────────────────────────

def _handle_receiver_crossing(
    cross:      ndarray,
    idx:        ndarray,
    x:          ndarray,
    y:          ndarray,
    z:          ndarray,
    ux:         ndarray,
    uy:         ndarray,
    uz:         ndarray,
    L:          ndarray,
    s:          ndarray,
    n_sc:       ndarray,
    n_nl:       ndarray,
    link_range: float,
    rx_radius:  float,
    rx_fov:     float,
    w:          ndarray,
    alive:      ndarray,
    records:    list[PhotonRecord],
) -> None:
    """
    Detect receiver-plane crossing, apply aperture+FOV filter,
    and append accepted photon records.

    Now also records x_m, y_m, path_length_m, n_scatters, n_nulls
    so downstream analysis can examine spatial and path statistics.
    """
    ci    = idx[cross]
    uz_ci = uz[ci]
    s_ci  = s[cross]

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
        hi = ci[hit]
        records.append({
            "weight":        w[hi].copy().astype(np.float64),
            "tof_s":         path_length_to_tof(L_rx[hit]).astype(np.float64),
            "x_m":           x_rx[hit].astype(np.float32),
            "y_m":           y_rx[hit].astype(np.float32),
            "path_length_m": L_rx[hit].astype(np.float32),
            "n_scatters":    n_sc[hi].copy().astype(np.int32),
            "n_nulls":       n_nl[hi].copy().astype(np.int32),
        })

    alive[ci] = False


# ─────────────────────────────────────────────────────────────────────────────
# Worker 1: Homogeneous medium
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch(args: tuple) -> PhotonRecord:
    """
    Propagate photons through a homogeneous medium.

    Returns a PhotonRecord dict of parallel arrays for all captured photons.
    """
    (n_photons, seed_state,
     c, b, g, omega,
     beam_div, beam_waist,
     rx_radius, rx_fov,
     link_range,
     weight_threshold, roulette_m,
     chunk_size) = args

    rng = np.random.default_rng(np.random.PCG64(seed_state))
    records: list[PhotonRecord] = []

    remaining = n_photons
    while remaining > 0:
        N          = min(chunk_size, remaining)
        remaining -= N

        x, y  = sample_launch_positions(beam_waist, rng, N)
        z     = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w     = np.ones(N)
        L     = np.zeros(N)
        n_sc  = np.zeros(N, dtype=np.int32)   # scatter counter
        n_nl  = np.zeros(N, dtype=np.int32)   # null counter (always 0 here)
        alive = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            n   = idx.size

            s   = sample_step_length(c, rng.uniform(size=n))
            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]

            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s,
                    n_sc, n_nl, link_range, rx_radius, rx_fov,
                    w, alive, records,
                )

            alive_mask = alive.copy()
            alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0:
                continue

            step_mask = np.isin(idx, si)
            x[si] = xn[step_mask];  y[si] = yn[step_mask];  z[si] = zn[step_mask]
            L[si] += s[step_mask]

            out = (z[si] < 0) | (z[si] > link_range * 3.0) \
                              | (np.hypot(x[si], y[si]) > 10.0)
            alive[si[out]] = False
            active = si[~out]
            if active.size == 0:
                continue

            update_weight(w[active], omega)

            rr_xi = rng.uniform(size=N)
            russian_roulette(w, alive, rr_xi, weight_threshold, roulette_m)
            active = np.where(alive)[0]
            if active.size == 0:
                continue

            # scatter and increment counter
            cos_sc = sample_hg_cos_theta(g, rng.uniform(size=active.size))
            sin_sc = np.sqrt(np.maximum(0.0, 1.0 - cos_sc ** 2))
            phi_sc = rng.uniform(0.0, 2.0 * np.pi, active.size)
            ux[active], uy[active], uz[active] = rotate_direction(
                ux[active], uy[active], uz[active], cos_sc, sin_sc, phi_sc,
            )
            n_sc[active] += 1

    return _concat_records(records)


# ─────────────────────────────────────────────────────────────────────────────
# Worker 2: Inhomogeneous medium  (Woodcock delta-tracking)
# ─────────────────────────────────────────────────────────────────────────────

def propagate_batch_inhomogeneous(args: tuple) -> PhotonRecord:
    """
    Propagate photons through an inhomogeneous medium using Woodcock
    delta-tracking.

    Returns a PhotonRecord dict of parallel arrays for all captured photons.
    n_nulls records null (virtual) collisions for diagnostics.
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
    records: list[PhotonRecord] = []

    remaining = n_photons
    while remaining > 0:
        N          = min(chunk_size, remaining)
        remaining -= N

        x, y  = sample_launch_positions(beam_waist, rng, N)
        z     = np.zeros(N)
        ux, uy, uz = sample_launch_directions(beam_div, rng, N)
        w     = np.ones(N)
        L     = np.zeros(N)
        n_sc  = np.zeros(N, dtype=np.int32)
        n_nl  = np.zeros(N, dtype=np.int32)
        alive = np.ones(N, dtype=bool)

        for _ in range(_MAX_ITER):
            if not alive.any():
                break
            idx = np.where(alive)[0]
            n   = idx.size

            s   = sample_step_woodcock(c_max, rng.uniform(size=n))
            xn  = x[idx] + s * ux[idx]
            yn  = y[idx] + s * uy[idx]
            zn  = z[idx] + s * uz[idx]

            cross = (z[idx] < link_range) & (zn >= link_range)
            if cross.any():
                _handle_receiver_crossing(
                    cross, idx, x, y, z, ux, uy, uz, L, s,
                    n_sc, n_nl, link_range, rx_radius, rx_fov,
                    w, alive, records,
                )

            alive_mask = alive.copy()
            alive_mask[idx[cross]] = False
            si = np.where(alive_mask)[0]
            if si.size == 0:
                continue

            step_mask = np.isin(idx, si)
            x[si] = xn[step_mask];  y[si] = yn[step_mask];  z[si] = zn[step_mask]
            L[si] += s[step_mask]

            out = (z[si] < 0) | (z[si] > link_range * 3.0) \
                              | (np.hypot(x[si], y[si]) > 10.0)
            alive[si[out]] = False
            active = si[~out]
            if active.size == 0:
                continue

            c_local   = medium.attenuation(z[active])
            real_mask = accept_real_collision(c_local, c_max, rng.uniform(size=active.size))
            real_idx  = active[real_mask]
            null_idx  = active[~real_mask]

            # count null collisions for diagnostics
            n_nl[null_idx] += 1

            if real_idx.size == 0:
                continue

            omega_local = medium.albedo(z[real_idx])
            update_weight_array(w[real_idx], omega_local)

            rr_xi = rng.uniform(size=real_idx.size)
            russian_roulette_targeted(
                w, alive, real_idx, rr_xi, weight_threshold, roulette_m
            )

            scatter_idx = real_idx[alive[real_idx]]
            if scatter_idx.size == 0:
                continue

            g_local = medium.asymmetry(z[scatter_idx])
            cos_sc  = sample_hg_cos_theta_array(g_local, rng.uniform(size=scatter_idx.size))
            sin_sc  = np.sqrt(np.maximum(0.0, 1.0 - cos_sc ** 2))
            phi_sc  = rng.uniform(0.0, 2.0 * np.pi, scatter_idx.size)
            ux[scatter_idx], uy[scatter_idx], uz[scatter_idx] = rotate_direction(
                ux[scatter_idx], uy[scatter_idx], uz[scatter_idx],
                cos_sc, sin_sc, phi_sc,
            )
            n_sc[scatter_idx] += 1

    return _concat_records(records)