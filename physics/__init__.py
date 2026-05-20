"""
uowc.physics
============
Pure functions encoding the underwater optical physics.

Separation-of-Concern role
---------------------------
  This module knows about optics and photon-matter interactions only.
  It has no awareness of receivers, parallelism, file I/O, or plotting.
  Every function is stateless and side-effect-free — safe to test in isolation.

References
----------
  Gabriel et al. (2013)  JOCN 5(1):1-12      — HG phase function, IOPs
  Mobley & Preisendorfer (1994)               — seawater radiative transfer
  Haltrin (1999) Applied Optics 38(33):6826   — chlorophyll-based bio-optics
  Wang, Jacques & Zheng (1995) CPC 47:131-146 — MCML algorithm
"""

from __future__ import annotations
import numpy as np
from numpy import ndarray

from uowc.config import C_MEDIUM


# ─────────────────────────────────────────────────────────────────────────────
# Beer-Lambert reference
# ─────────────────────────────────────────────────────────────────────────────

def beer_lambert_power_dB(c: float, distance_m: float) -> float:
    """
    Normalised received power from the Beer-Lambert law (dB).

        P = exp(-c * d)   →   P_dB = -c * d * 10/ln(10)

    No scattering, no geometry — a deterministic lower bound on path loss.
    """
    return 10.0 * np.log10(np.exp(-c * distance_m) + 1e-300)


# ─────────────────────────────────────────────────────────────────────────────
# Free-flight sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_step_length(c: float, xi: ndarray) -> ndarray:
    """
    Sample exponentially distributed free-flight step lengths.

        s = -ln(ξ) / c         [Beer-Lambert inverse-CDF]

    Parameters
    ----------
    c  : beam-attenuation coefficient (m⁻¹)
    xi : uniform random variates in (0, 1], shape (N,)

    Returns
    -------
    s : step lengths (m), shape (N,)
    """
    return -np.log(np.clip(xi, 1e-15, 1.0)) / c


# ─────────────────────────────────────────────────────────────────────────────
# Implicit-absorption weight update
# ─────────────────────────────────────────────────────────────────────────────

def update_weight(w: ndarray, omega: float) -> ndarray:
    """
    Apply the implicit-absorption (MCML) weight reduction.

        W_{n+1} = W_n · ω        ω = b / c  (single-scattering albedo)

    Parameters
    ----------
    w     : current photon weights, shape (N,)
    omega : single-scattering albedo

    Returns
    -------
    Updated weights (in-place modification for efficiency).
    """
    w *= omega
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Russian roulette
# ─────────────────────────────────────────────────────────────────────────────

def russian_roulette(
    w: ndarray,
    alive: ndarray,
    xi: ndarray,
    threshold: float,
    m: int,
) -> ndarray:
    """
    Kill or boost photons whose weight has fallen below `threshold`.

    Photons survive with probability 1/m and have their weight scaled by m,
    preserving the unbiased estimator  E[W_new] = W_old.

    Parameters
    ----------
    w         : photon weights,               shape (N,)
    alive     : boolean liveness mask,        shape (N,)
    xi        : uniform random variates,      shape (N,) — pre-drawn by caller
    threshold : weight kill threshold (e.g. 1e-4)
    m         : roulette multiplier (e.g. 10)

    Returns
    -------
    alive : updated liveness mask (in-place)
    """
    low = alive & (w < threshold)
    if not low.any():
        return alive
    survive = low & (xi < (1.0 / m))
    kill    = low & ~survive
    w[survive] *= m
    alive[kill] = False
    return alive


# ─────────────────────────────────────────────────────────────────────────────
# Henyey-Greenstein phase function
# ─────────────────────────────────────────────────────────────────────────────

def sample_hg_cos_theta(g: float, xi: ndarray) -> ndarray:
    """
    Sample the polar scattering angle cosine from the Henyey-Greenstein CDF.

        cosθ = 1/(2g) · [1 + g² - ((1-g²)/(1-g+2gξ))²]    g ≠ 0
        cosθ = 1 - 2ξ                                        g = 0

    Parameters
    ----------
    g  : asymmetry parameter (0 = isotropic, 1 = fully forward)
    xi : uniform random variates in [0, 1), shape (N,)

    Returns
    -------
    cos_theta : shape (N,), clipped to [-1, 1]
    """
    if abs(g) < 1e-6:
        return np.clip(1.0 - 2.0 * xi, -1.0, 1.0)
    tmp = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
    cos_theta = (1.0 + g * g - tmp * tmp) / (2.0 * g)
    return np.clip(cos_theta, -1.0, 1.0)


def rotate_direction(
    ux: ndarray, uy: ndarray, uz: ndarray,
    cos_sc: ndarray, sin_sc: ndarray,
    phi_sc: ndarray,
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Rotate a batch of unit direction vectors (ux, uy, uz) by a local
    scattering deflection (cos_sc, sin_sc) around azimuth phi_sc.

    Uses the standard MCML frame-rotation formula with a near-z-axis
    singularity fix to avoid division by zero when |uz| → 1.

    Parameters
    ----------
    ux, uy, uz : direction cosines of current propagation direction, shape (N,)
    cos_sc     : cosine of polar scattering angle,  shape (N,)
    sin_sc     : sine  of polar scattering angle,  shape (N,)
    phi_sc     : azimuthal scattering angle (rad), shape (N,)

    Returns
    -------
    (ux_new, uy_new, uz_new) : rotated unit direction cosines
    """
    cp   = np.cos(phi_sc)
    sp   = np.sin(phi_sc)
    denom = np.sqrt(np.maximum(1e-12, 1.0 - uz ** 2))
    near_z = np.abs(uz) > (1.0 - 1e-5)
    sgn    = np.where(uz >= 0.0, 1.0, -1.0)

    ux_new = (sin_sc * (ux * uz * cp - uy * sp) / denom + ux * cos_sc)
    uy_new = (sin_sc * (uy * uz * cp + ux * sp) / denom + uy * cos_sc)
    uz_new = (-sin_sc * cp * denom + uz * cos_sc)

    # Singularity: photon nearly along ±z axis
    ux_new[near_z] = sin_sc[near_z] * cp[near_z]
    uy_new[near_z] = sin_sc[near_z] * sp[near_z] * sgn[near_z]
    uz_new[near_z] = cos_sc[near_z] * sgn[near_z]

    # Renormalise (guards against floating-point drift over many steps)
    norm = np.sqrt(ux_new ** 2 + uy_new ** 2 + uz_new ** 2)
    norm = np.where(norm > 1e-15, norm, 1.0)
    return ux_new / norm, uy_new / norm, uz_new / norm


# ─────────────────────────────────────────────────────────────────────────────
# Photon launch geometry
# ─────────────────────────────────────────────────────────────────────────────

def sample_launch_positions(waist_m: float, rng: np.random.Generator, N: int
                             ) -> tuple[ndarray, ndarray]:
    """
    Sample (x, y) launch positions from a TEM₀₀ Gaussian beam profile.

    The 1/e² intensity radius is `waist_m`.  The radial CDF of a
    2-D Gaussian gives  r = waist · √(-0.5 · ln ξ).

    Returns
    -------
    x, y : Cartesian launch coordinates (m), shape (N,)
    """
    r   = waist_m * np.sqrt(-0.5 * np.log(np.clip(rng.uniform(size=N), 1e-15, 1.0)))
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    return r * np.cos(phi), r * np.sin(phi)


def sample_launch_directions(
    divergence_rad: float, rng: np.random.Generator, N: int
) -> tuple[ndarray, ndarray, ndarray]:
    """
    Sample initial direction cosines uniformly within a cone of half-angle
    `divergence_rad` (solid-angle uniform sampling).

        cos θ₀ = 1 - ξ · (1 - cos θ_max)

    Returns
    -------
    ux, uy, uz : direction cosines, shape (N,), with uz > 0 (downward)
    """
    cos_max = np.cos(divergence_rad)
    cos_th  = 1.0 - rng.uniform(size=N) * (1.0 - cos_max)
    sin_th  = np.sqrt(np.maximum(0.0, 1.0 - cos_th ** 2))
    phi     = rng.uniform(0.0, 2.0 * np.pi, N)
    return sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th


# ─────────────────────────────────────────────────────────────────────────────
# Time-of-flight
# ─────────────────────────────────────────────────────────────────────────────

def path_length_to_tof(path_length_m: ndarray) -> ndarray:
    """
    Convert cumulative optical path length to time-of-flight.

        t = L / v_medium = L · n_water / c₀

    Parameters
    ----------
    path_length_m : cumulative path length (m), shape (N,)

    Returns
    -------
    tof_s : time of flight (s), shape (N,)
    """
    return path_length_m / C_MEDIUM
