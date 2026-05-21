"""
uowc.metrics
============
Channel characterisation metrics derived from captured photon data.

Separation-of-Concern role
---------------------------
  This module transforms raw Monte Carlo output (arrays of captured weights
  and times-of-flight) into physically meaningful channel metrics.
  It knows nothing about how photons were propagated, parallelised, or plotted.

compute_all_metrics — n_launched parameter
------------------------------------------
  In adaptive mode n_launched ≠ cfg.n_photons per run (it varies by depth).
  compute_all_metrics therefore accepts n_launched explicitly.
  Passing it is mandatory when using the adaptive sweeps; for fixed-launch
  runs you can pass result.n_launched (== cfg.n_photons) or let the function
  fall back to cfg.n_photons via the optional parameter.

References
----------
  Zeng et al. (2017)  IEEE Comm. Surveys & Tutorials 19(1):204-238
  Gabriel et al. (2013)  JOCN 5(1):1-12
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import numpy as np
from numpy import ndarray

from config import SimConfig
from physics import beer_lambert_power_dB


# ─────────────────────────────────────────────────────────────────────────────
# Received power
# ─────────────────────────────────────────────────────────────────────────────

def received_power_dB(weights: ndarray, n_launched: int) -> float:
    """
    Normalised received power in dB.

        P_norm = Σ w_i / N_launched

    Parameters
    ----------
    weights    : captured photon weights, shape (M,)
    n_launched : total photons launched (the correct denominator).
                 In adaptive mode this equals n_rounds × cfg.n_photons
                 and is obtained from RunResult.n_launched.
    """
    if weights.size == 0 or n_launched == 0:
        return -np.inf
    return float(10.0 * np.log10(np.maximum(weights.sum() / n_launched, 1e-300)))


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
    Normalised channel impulse response (CIR) histogram.

        h[k] = Σ_{t_i in bin k} w_i  /  Σ w_i
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
    RMS delay spread  τ_rms (s) — second central moment of the CIR.

        τ̄     = Σ w_i t_i  /  Σ w_i
        τ_rms = √( Σ w_i (t_i − τ̄)²  /  Σ w_i )
    """
    if weights.size < 2 or weights.sum() == 0.0:
        return float("nan")
    w_sum = weights.sum()
    tau_m = (weights * times).sum() / w_sum
    return float(np.sqrt((weights * (times - tau_m) ** 2).sum() / w_sum))


# ─────────────────────────────────────────────────────────────────────────────
# Frequency response
# ─────────────────────────────────────────────────────────────────────────────

def frequency_response(
    h_norm:     ndarray,
    dt:         float,
    pad_factor: int = 8,
) -> Tuple[ndarray, ndarray]:
    """
    Normalised channel frequency response via zero-padded FFT.

        H(f) = ℱ{ h(t) }   (one-sided real-input FFT, normalised to DC)
    """
    N_pad = len(h_norm) * pad_factor
    H     = np.fft.rfft(h_norm, n=N_pad)
    H_mag = np.abs(H)
    dc    = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    freqs = np.fft.rfftfreq(N_pad, d=dt)
    return freqs, H_mag / dc


# ─────────────────────────────────────────────────────────────────────────────
# 3 dB Bandwidth
# ─────────────────────────────────────────────────────────────────────────────

def bandwidth_3dB(freqs: ndarray, H_norm: ndarray) -> float:
    """First frequency at which |H(f)| drops below 1/√2  (−3 dB)."""
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
    n_launched: Optional[int] = None,
) -> Dict:
    """
    Compute every channel metric for a single (water-or-medium, beam, range) run.

    Parameters
    ----------
    weights, times : raw captured photon data (RunResult.weights / .times)
    cfg            : simulation config (for dt_bin_s, n_time_bins)
    c              : representative attenuation coefficient for Beer-Lambert ref
    link_range     : link length in metres
    n_launched     : total photons launched — pass RunResult.n_launched.
                     Defaults to cfg.n_photons for backward compatibility with
                     fixed-launch code that does not yet use RunResult.

    Returns
    -------
    Dict with keys:
      power_dB        — MC normalised received power (dB)
      beer_lambert_dB — Beer-Lambert reference (dB)
      delay_spread_s  — RMS delay spread τ_rms (s)
      bandwidth_hz    — 3 dB bandwidth (Hz)
      t_axis          — CIR time axis (s), shape (n_bins,)
      cir             — normalised CIR, shape (n_bins,)
      freqs           — frequency axis (Hz)
      fr              — normalised |H(f)|
      n_launched      — total photons launched (useful for reporting tables)
      n_captured      — number of captured photons (useful for quality check)
    """
    # ── denominator for received power ────────────────────────────────────
    n_total = n_launched if n_launched is not None else cfg.n_photons

    p_dB = received_power_dB(weights, n_total)
    p_bl = beer_lambert_power_dB(c, link_range)
    ds   = rms_delay_spread(weights, times) if weights.size > 1 else float("nan")

    t_ax, h    = compute_cir(weights, times, cfg.dt_bin_s, cfg.n_time_bins)
    freqs, Hn  = frequency_response(h, cfg.dt_bin_s)
    bw         = bandwidth_3dB(freqs, Hn)

    return {
        "power_dB":        p_dB,
        "beer_lambert_dB": p_bl,
        "delay_spread_s":  ds,
        "bandwidth_hz":    bw,
        "t_axis":          t_ax,
        "cir":             h,
        "freqs":           freqs,
        "fr":              Hn,
        "n_launched":      n_total,      # store for reporting
        "n_captured":      weights.size, # store for quality diagnostics
    }