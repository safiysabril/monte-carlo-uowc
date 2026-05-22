"""
uowc.metrics
============
Channel characterisation metrics derived from RunResult.

Accepts RunResult directly — no need to manually unpack weights/times.
n_launched is read from RunResult.n_launched for correct normalisation.
"""

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from numpy import ndarray
from uowc.config import SimConfig
from uowc.physics import beer_lambert_power_dB


def received_power_dB(weights: ndarray, n_launched: int) -> float:
    if weights.size == 0 or n_launched == 0:
        return -np.inf
    return float(10.0 * np.log10(np.maximum(weights.sum() / n_launched, 1e-300)))


def compute_cir(weights, times, dt, n_bins):
    edges  = np.linspace(0.0, dt * n_bins, n_bins + 1)
    h, _   = np.histogram(times, bins=edges, weights=weights)
    t_axis = 0.5 * (edges[:-1] + edges[1:])
    return t_axis, h / (h.sum() + 1e-30)


def rms_delay_spread(weights, times) -> float:
    if weights.size < 2 or weights.sum() == 0:
        return float("nan")
    w_sum = weights.sum()
    tau_m = (weights * times).sum() / w_sum
    return float(np.sqrt((weights * (times - tau_m) ** 2).sum() / w_sum))


def frequency_response(h_norm, dt, pad_factor=8):
    N_pad = len(h_norm) * pad_factor
    H     = np.fft.rfft(h_norm, n=N_pad)
    H_mag = np.abs(H)
    dc    = H_mag[0] if H_mag[0] > 1e-30 else 1.0
    return np.fft.rfftfreq(N_pad, d=dt), H_mag / dc


def bandwidth_3dB(freqs, H_norm) -> float:
    idx = np.where(H_norm < (1.0 / np.sqrt(2.0)))[0]
    return float(freqs[idx[0]]) if idx.size > 0 else float(freqs[-1])


def compute_all_metrics(result, cfg: SimConfig, c: float, link_range: float) -> Dict:
    """
    Compute all channel metrics from a RunResult.

    Parameters
    ----------
    result     : RunResult  (has .weights, .times, .n_launched, .record)
    cfg        : SimConfig
    c          : reference attenuation for Beer-Lambert (m⁻¹)
    link_range : link length (m)
    """
    weights    = result.weights
    times      = result.times
    n_launched = result.n_launched

    p_dB  = received_power_dB(weights, n_launched)
    p_bl  = beer_lambert_power_dB(c, link_range)
    ds    = rms_delay_spread(weights, times) if weights.size > 1 else float("nan")
    t_ax, h   = compute_cir(weights, times, cfg.dt_bin_s, cfg.n_time_bins)
    freqs, Hn = frequency_response(h, cfg.dt_bin_s)
    bw        = bandwidth_3dB(freqs, Hn)

    return {
        "power_dB":        p_dB,
        "beer_lambert_dB": p_bl,
        "delay_spread_s":  ds,
        "bandwidth_hz":    bw,
        "t_axis":          t_ax,
        "cir":             h,
        "freqs":           freqs,
        "fr":              Hn,
        "n_launched":      n_launched,
        "n_captured":      weights.size,
    }