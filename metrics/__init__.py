"""
uowc.metrics
============
Channel characterisation metrics derived from captured photon data.

Separation-of-Concern role
---------------------------
  This module transforms raw Monte Carlo output (arrays of captured weights
  and times-of-flight) into physically meaningful channel metrics.
  It knows nothing about how photons were propagated, how results are plotted,
  or how computations were parallelised.

  All public functions are pure (no side-effects, no I/O) and are safe to
  call from tests, notebooks, or post-processing scripts independently.

Metrics computed
----------------
  received_power_dB   — normalised received power (dB)
  compute_cir         — channel impulse response histogram
  rms_delay_spread    — RMS delay spread τ_rms (s)
  frequency_response  — |H(f)| via zero-padded FFT
  bandwidth_3dB       — 3 dB channel bandwidth (Hz)
  compute_all_metrics — convenience wrapper: all metrics in one dict

References
----------
  Zeng et al. (2017)  IEEE Comm. Surveys & Tutorials 19(1):204-238
  Gabriel et al. (2013)  JOCN 5(1):1-12
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Dict, Tuple

from config import SimConfig
from physics import beer_lambert_power_dB


# ─────────────────────────────────────────────────────────────────────────────
# Received power
# ─────────────────────────────────────────────────────────────────────────────

def received_power_dB(weights: ndarray, n_total: int) -> float:
    """
    Normalised received power in dB.

        P_norm = Σ w_i / N_total

    Dividing by N_total (launched, not captured) gives the fraction of input
    power received — a direct estimator of the channel DC gain.

    Parameters
    ----------
    weights : captured photon weights, shape (M,)
    n_total : total photons launched

    Returns
    -------
    P_dB : normalised received power (dB)
    """
    if weights.size == 0:
        return -np.inf
    return float(10.0 * np.log10(np.maximum(weights.sum() / n_total, 1e-300)))


# ─────────────────────────────────────────────────────────────────────────────
# Channel impulse response
# ─────────────────────────────────────────────────────────────────────────────

def compute_cir(
    weights: ndarray,
    times:   ndarray,
    dt:      float,
    n_bins:  int,
) -> Tuple[ndarray, ndarray]:
    """
    Build the normalised channel impulse response (CIR) histogram.

        h[k] = Σ_{t_i ∈ [k·Δt, (k+1)·Δt)} w_i  /  (Σ w_i)

    Parameters
    ----------
    weights : captured photon weights,       shape (M,)
    times   : corresponding times-of-flight, shape (M,) in seconds
    dt      : bin width (s)
    n_bins  : number of time bins

    Returns
    -------
    t_axis : bin-centre time axis (s), shape (n_bins,)
    h_norm : normalised CIR amplitude, shape (n_bins,)
    """
    edges  = np.linspace(0.0, dt * n_bins, n_bins + 1)
    h, _   = np.histogram(times, bins=edges, weights=weights)
    t_axis = 0.5 * (edges[:-1] + edges[1:])
    total  = h.sum() + 1e-30
    return t_axis, h / total


# ─────────────────────────────────────────────────────────────────────────────
# RMS delay spread
# ─────────────────────────────────────────────────────────────────────────────

def rms_delay_spread(weights: ndarray, times: ndarray) -> float:
    """
    RMS delay spread  τ_rms (s) — the second central moment of the CIR.

        τ̄     = Σ w_i · t_i  /  Σ w_i
        τ_rms = √( Σ w_i · (t_i - τ̄)²  /  Σ w_i )

    τ_rms characterises inter-symbol interference: a channel supports a
    data rate R ≈ 1/(2π·τ_rms) without ISI equalisation.

    Parameters
    ----------
    weights : captured photon weights,       shape (M,)
    times   : corresponding times-of-flight, shape (M,) in seconds

    Returns
    -------
    τ_rms in seconds, or NaN if fewer than 2 photons were captured.
    """
    if weights.size < 2 or weights.sum() == 0.0:
        return float("nan")
    w_sum  = weights.sum()
    tau_m  = (weights * times).sum() / w_sum
    return float(np.sqrt(((weights * (times - tau_m) ** 2).sum()) / w_sum))


# ─────────────────────────────────────────────────────────────────────────────
# Frequency response
# ─────────────────────────────────────────────────────────────────────────────

def frequency_response(
    h_norm:     ndarray,
    dt:         float,
    pad_factor: int = 8,
) -> Tuple[ndarray, ndarray]:
    """
    Compute the channel frequency response via zero-padded FFT.

        H(f) = ℱ{ h(t) }     (one-sided, real-input FFT)

    Parameters
    ----------
    h_norm     : normalised CIR, shape (n_bins,)
    dt         : CIR bin width (s)
    pad_factor : zero-padding factor for frequency resolution

    Returns
    -------
    freqs  : positive frequencies (Hz), shape (K,)
    H_norm : |H(f)| normalised to DC  (H[0] = 1), shape (K,)
    """
    N_pad  = len(h_norm) * pad_factor
    H      = np.fft.rfft(h_norm, n=N_pad)
    H_mag  = np.abs(H)
    dc     = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    freqs  = np.fft.rfftfreq(N_pad, d=dt)
    return freqs, H_mag / dc


# ─────────────────────────────────────────────────────────────────────────────
# 3 dB Bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def bandwidth_3dB(freqs: ndarray, H_norm: ndarray) -> float:
    """
    Find the −3 dB channel bandwidth: the first frequency where |H(f)|²
    falls to 0.5 of its DC value (equivalently |H(f)| < 1/√2).

    Parameters
    ----------
    freqs  : frequency axis (Hz), shape (K,)
    H_norm : normalised |H(f)|,   shape (K,)

    Returns
    -------
    Bandwidth in Hz.  Returns freqs[-1] if the response never drops below −3 dB.
    """
    idx = np.where(H_norm < (1.0 / np.sqrt(2.0)))[0]
    return float(freqs[idx[0]]) if idx.size > 0 else float(freqs[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: all metrics for one run
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    weights:    ndarray,
    times:      ndarray,
    cfg:        SimConfig,
    c:          float,
    link_range: float,
) -> Dict:
    """
    Compute every channel metric for a single (water, beam, range) run.

    Parameters
    ----------
    weights, times : raw captured photon data from `simulation.run_one`
    cfg            : simulation config (for n_photons, dt_bin_s, n_time_bins)
    c              : water attenuation coefficient (for Beer-Lambert reference)
    link_range     : link length in metres

    Returns
    -------
    Dict with keys:
      power_dB       — Monte Carlo normalised received power (dB)
      beer_lambert_dB — Beer-Lambert reference power (dB)
      delay_spread_s  — RMS delay spread (s)
      bandwidth_hz    — 3 dB bandwidth (Hz)
      t_axis          — CIR time axis (s), shape (n_bins,)
      cir             — normalised CIR amplitude, shape (n_bins,)
      freqs           — frequency axis (Hz)
      fr              — normalised |H(f)|
    """
    p_dB  = received_power_dB(weights, cfg.n_photons)
    p_bl  = beer_lambert_power_dB(c, link_range)
    ds    = rms_delay_spread(weights, times) if weights.size > 1 else float("nan")
    t_ax, h = compute_cir(weights, times, cfg.dt_bin_s, cfg.n_time_bins)
    freqs, Hn = frequency_response(h, cfg.dt_bin_s)
    bw    = bandwidth_3dB(freqs, Hn)

    return {
        "power_dB":        p_dB,
        "beer_lambert_dB": p_bl,
        "delay_spread_s":  ds,
        "bandwidth_hz":    bw,
        "t_axis":          t_ax,
        "cir":             h,
        "freqs":           freqs,
        "fr":              Hn,
    }
