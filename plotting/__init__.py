"""
uowc.plotting
=============
Publication-quality figure generation.

Separation-of-Concern role
---------------------------
  This module receives pre-computed metric dicts and turns them into figures.
  It knows about axes, labels, colours, and line styles — nothing else.
  Changing a plot style never touches physics, transport, or metrics code.

  All public functions accept a `save_dir` argument and return nothing;
  figures are written to disk and closed to avoid memory accumulation.

Figures produced
----------------
  fig1 — Normalised received power vs link range
  fig2 — Channel impulse response  (5 m and 25 m)
  fig3 — Frequency response        (5 m and 25 m)
  fig4 — RMS delay spread vs link range
  fig5 — 3 dB channel bandwidth vs link range
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from uowc.config import (
    WaterParams, BeamParams,
    CLEAR_WATER, COASTAL_WATER, COLLIMATED, DIFFUSED,
    ALL_WATERS, ALL_BEAMS, SimConfig,
)
from uowc.simulation import RunKey


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens — change colours / markers in one place
# ─────────────────────────────────────────────────────────────────────────────
_COLOUR: Dict[str, Dict[str, str]] = {
    CLEAR_WATER.name  : {COLLIMATED.name: "#1f77b4", DIFFUSED.name: "#7ab8e8"},
    COASTAL_WATER.name: {COLLIMATED.name: "#d62728", DIFFUSED.name: "#f4a261"},
}
_MARKER: Dict[str, str] = {
    COLLIMATED.name: "o",
    DIFFUSED.name:   "s",
}
_LINE_KW  = dict(linewidth=1.8)
_BEER_KW  = dict(color="black", linestyle="--", linewidth=1.2)
_GRID_KW  = dict(alpha=0.25, linestyle="--")
_TICK_SZ  = 8
_LABEL_SZ = 9
_TITLE_SZ = 9


# ─────────────────────────────────────────────────────────────────────────────
# Private utilities
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=_LABEL_SZ)
    ax.set_ylabel(ylabel, fontsize=_LABEL_SZ)
    ax.set_title(title,   fontsize=_TITLE_SZ, fontweight="bold")
    ax.grid(True, **_GRID_KW)
    ax.tick_params(labelsize=_TICK_SZ)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _colour(water: WaterParams, beam: BeamParams) -> str:
    return _COLOUR[water.name][beam.name]


def _marker(beam: BeamParams) -> str:
    return _MARKER[beam.name]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Received power vs link range
# ─────────────────────────────────────────────────────────────────────────────

def plot_received_power(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    Two-panel figure: one panel per water type.
    Each panel shows MC power for collimated and diffused, plus a
    Beer-Lambert reference line.
    """
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle(
        "Normalised Received Power vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )

    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            pwr = [metrics[RunKey(water.name, beam.name, float(Z))]["power_dB"]
                   for Z in ranges]
            ax.plot(ranges, pwr,
                    marker=_marker(beam), color=_colour(water, beam),
                    label=f"MC — {beam.name}", **_LINE_KW)

        # Beer-Lambert reference
        r_arr = np.array(ranges)
        bl    = 10.0 * np.log10(np.exp(-water.c * r_arr) + 1e-300)
        ax.plot(r_arr, bl, **_BEER_KW, label="Beer-Lambert")
        _style_ax(ax, "Link Range (m)", "Normalised Power (dB)", water.name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig1_received_power.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Channel impulse response  (5 m and 25 m)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cir(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    4 × 2 grid (rows = water types, cols = [Coll@5m, Coll@25m, Diff@5m, Diff@25m]).
    """
    ranges   = list(cfg.link_ranges_m)
    z_show   = [ranges[0], ranges[-1]]     # 5 m and 25 m
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Channel Impulse Response  |  Homogeneous Medium\n"
        "Cols: Collimated @5 m | Collimated @25 m | "
        "Diffused @5 m | Diffused @25 m",
        fontsize=11, fontweight="bold",
    )

    for row, water in enumerate(ALL_WATERS):
        col = 0
        for beam in ALL_BEAMS:
            for Z in z_show:
                key = RunKey(water.name, beam.name, float(Z))
                m   = metrics[key]
                ax  = axes[row, col]
                t_ns = m["t_axis"] * 1e9
                ax.plot(t_ns, m["cir"],
                        color=_colour(water, beam), linewidth=1.2)
                _style_ax(ax, "Time (ns)", "Norm. amplitude",
                          f"{water.name}\n{beam.name} — {Z} m")
                ax.set_xlim([0, t_ns[-1]])
                col += 1

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig2_cir.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Frequency response  (5 m and 25 m)
# ─────────────────────────────────────────────────────────────────────────────

def plot_frequency_response(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """
    4 × 2 grid mirroring the CIR figure; includes −3 dB marker per subplot.
    """
    ranges   = list(cfg.link_ranges_m)
    z_show   = [ranges[0], ranges[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Frequency Response  |  Homogeneous Medium\n"
        "Cols: Collimated @5 m | Collimated @25 m | "
        "Diffused @5 m | Diffused @25 m",
        fontsize=11, fontweight="bold",
    )

    for row, water in enumerate(ALL_WATERS):
        col = 0
        for beam in ALL_BEAMS:
            for Z in z_show:
                key    = RunKey(water.name, beam.name, float(Z))
                m      = metrics[key]
                ax     = axes[row, col]
                H_dB   = 20.0 * np.log10(np.maximum(m["fr"], 1e-12))
                f_MHz  = m["freqs"] / 1e6
                bw_MHz = m["bandwidth_hz"] / 1e6
                mask   = f_MHz > 0
                ax.semilogx(f_MHz[mask], H_dB[mask],
                            color=_colour(water, beam), linewidth=1.2)
                ax.axvline(bw_MHz, color="gray", linestyle="--",
                           linewidth=1.0, label=f"−3 dB: {bw_MHz:.1f} MHz")
                ax.axhline(-3, color="gray",   linestyle=":",  linewidth=0.8)
                ax.set_ylim([-40, 2])
                _style_ax(ax, "Frequency (MHz)", "Power (dB)",
                          f"{water.name}\n{beam.name} — {Z} m")
                ax.legend(fontsize=7)
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{x:.0f}")
                )
                col += 1

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig3_frequency_response.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — RMS delay spread vs link range
# ─────────────────────────────────────────────────────────────────────────────

def plot_delay_spread(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """Semi-log Y: delay spread (s) vs link range (m)."""
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "RMS Delay Spread vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )

    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            ds = [metrics[RunKey(water.name, beam.name, float(Z))]["delay_spread_s"]
                  for Z in ranges]
            ax.semilogy(ranges, ds,
                        marker=_marker(beam), color=_colour(water, beam),
                        label=beam.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "RMS Delay Spread (s)", water.name)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.2e}")
        )

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig4_delay_spread.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — 3 dB Bandwidth vs link range
# ─────────────────────────────────────────────────────────────────────────────

def plot_bandwidth(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """Semi-log Y: channel bandwidth (MHz) vs link range (m)."""
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "3 dB Channel Bandwidth vs Link Range  |  Homogeneous Medium",
        fontsize=12, fontweight="bold",
    )

    for ax, water in zip(axes, ALL_WATERS):
        for beam in ALL_BEAMS:
            bw = [metrics[RunKey(water.name, beam.name, float(Z))]["bandwidth_hz"] / 1e6
                  for Z in ranges]
            ax.semilogy(ranges, bw,
                        marker=_marker(beam), color=_colour(water, beam),
                        label=beam.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "3 dB Bandwidth (MHz)", water.name)
        ax.legend(fontsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig5_bandwidth.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: render all figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """Render and save all five figures."""
    os.makedirs(save_dir, exist_ok=True)
    plot_received_power(metrics, cfg, save_dir)
    plot_cir(metrics, cfg, save_dir)
    plot_frequency_response(metrics, cfg, save_dir)
    plot_delay_spread(metrics, cfg, save_dir)
    plot_bandwidth(metrics, cfg, save_dir)
