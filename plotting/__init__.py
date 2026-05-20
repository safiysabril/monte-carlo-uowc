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
  Homogeneous sweep (plot_all):
    fig1 — Normalised received power vs link range
    fig2 — Channel impulse response  (5 m and 25 m)
    fig3 — Frequency response        (5 m and 25 m)
    fig4 — RMS delay spread vs link range
    fig5 — 3 dB channel bandwidth vs link range

  Inhomogeneous sweep (plot_all_inhomogeneous):
    fig_inh1 — Received power comparison (medium × beam × range)
    fig_inh2 — CIR comparison at shortest and longest range
    fig_inh3 — Delay spread vs range for each medium
"""

from __future__ import annotations
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import (
    WaterParams, BeamParams,
    CLEAR_WATER, COASTAL_WATER, COLLIMATED, DIFFUSED,
    ALL_WATERS, ALL_BEAMS, SimConfig,
)
from simulation import RunKey


# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
_COLOUR: Dict[str, Dict[str, str]] = {
    CLEAR_WATER.name  : {COLLIMATED.name: "#1f77b4", DIFFUSED.name: "#7ab8e8"},
    COASTAL_WATER.name: {COLLIMATED.name: "#d62728", DIFFUSED.name: "#f4a261"},
}
_MARKER: Dict[str, str] = {
    COLLIMATED.name: "o",
    DIFFUSED.name:   "s",
}

# Colour palette for inhomogeneous media (up to 6 media)
_INH_COLOURS = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
_INH_MARKERS = ["D", "^", "v", "<", ">", "p"]

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
    print(f"    Saved → {path}")


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
        r_arr = np.array(ranges)
        bl    = 10.0 * np.log10(np.exp(-water.c * r_arr) + 1e-300)
        ax.plot(r_arr, bl, **_BEER_KW, label="Beer-Lambert")
        _style_ax(ax, "Link Range (m)", "Normalised Power (dB)", water.name)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig1_received_power.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Channel impulse response
# ─────────────────────────────────────────────────────────────────────────────

def plot_cir(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    ranges   = list(cfg.link_ranges_m)
    z_show   = [ranges[0], ranges[-1]]
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
                ax.plot(t_ns, m["cir"], color=_colour(water, beam), linewidth=1.2)
                _style_ax(ax, "Time (ns)", "Norm. amplitude",
                          f"{water.name}\n{beam.name} — {Z} m")
                ax.set_xlim([0, t_ns[-1]])
                col += 1
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig2_cir.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Frequency response
# ─────────────────────────────────────────────────────────────────────────────

def plot_frequency_response(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    ranges   = list(cfg.link_ranges_m)
    z_show   = [ranges[0], ranges[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Channel Frequency Response |H(f)|  |  Homogeneous Medium",
        fontsize=11, fontweight="bold",
    )
    for row, water in enumerate(ALL_WATERS):
        col = 0
        for beam in ALL_BEAMS:
            for Z in z_show:
                key = RunKey(water.name, beam.name, float(Z))
                m   = metrics[key]
                ax  = axes[row, col]
                f_MHz = m["freqs"] / 1e6
                bw    = m["bandwidth_hz"] / 1e6
                ax.plot(f_MHz, m["fr"], color=_colour(water, beam), linewidth=1.2)
                ax.axvline(bw, color="red", linestyle=":", linewidth=1.0,
                           label=f"BW={bw:.1f} MHz")
                ax.axhline(1/np.sqrt(2), color="grey", linestyle="--",
                           linewidth=0.8, label="−3 dB")
                _style_ax(ax, "Frequency (MHz)", "|H(f)|",
                          f"{water.name}\n{beam.name} — {Z} m")
                ax.set_xlim([0, min(f_MHz[-1], bw * 5)])
                ax.legend(fontsize=7)
                col += 1
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig3_frequency_response.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Delay spread vs link range
# ─────────────────────────────────────────────────────────────────────────────

def plot_delay_spread(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
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
# Inhomogeneous figure: power comparison across media
# ─────────────────────────────────────────────────────────────────────────────

def plot_inhomogeneous_power(
    metrics:    Dict[RunKey, dict],
    cfg:        SimConfig,
    media:      Tuple,        # tuple of MediumProfile instances
    save_dir:   str,
) -> None:
    """
    One panel per beam type; lines = media; x-axis = link range.

    Shows how stratified and gradient profiles change received power relative
    to the equivalent homogeneous bounds.
    """
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, len(ALL_BEAMS), figsize=(7 * len(ALL_BEAMS), 5))
    if len(ALL_BEAMS) == 1:
        axes = [axes]
    fig.suptitle(
        "Received Power vs Link Range  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for ax, beam in zip(axes, ALL_BEAMS):
        for i, medium in enumerate(media):
            colour = _INH_COLOURS[i % len(_INH_COLOURS)]
            marker = _INH_MARKERS[i % len(_INH_MARKERS)]
            pwr = []
            for Z in ranges:
                key = RunKey(medium.name, beam.name, float(Z))
                pwr.append(metrics[key]["power_dB"])
            ax.plot(ranges, pwr, marker=marker, color=colour,
                    label=medium.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "Normalised Power (dB)", beam.name)
        ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh1_power.png"))


def plot_inhomogeneous_cir(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Tuple,
    save_dir: str,
) -> None:
    """CIR at the shortest and longest range for each medium × beam combination."""
    ranges = list(cfg.link_ranges_m)
    z_show = [ranges[0], ranges[-1]]
    n_media = len(media)
    n_beams = len(ALL_BEAMS)
    fig, axes = plt.subplots(
        n_media, n_beams * 2,
        figsize=(6 * n_beams * 2, 4 * n_media),
        squeeze=False,
    )
    fig.suptitle(
        "Channel Impulse Response  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for row, medium in enumerate(media):
        col = 0
        for beam in ALL_BEAMS:
            for Z in z_show:
                key = RunKey(medium.name, beam.name, float(Z))
                m   = metrics[key]
                ax  = axes[row, col]
                colour = _INH_COLOURS[row % len(_INH_COLOURS)]
                t_ns = m["t_axis"] * 1e9
                ax.plot(t_ns, m["cir"], color=colour, linewidth=1.2)
                _style_ax(ax, "Time (ns)", "Norm. amplitude",
                          f"{medium.name}\n{beam.name} — {Z} m")
                ax.set_xlim([0, t_ns[-1]])
                col += 1
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh2_cir.png"))


def plot_inhomogeneous_delay_spread(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Tuple,
    save_dir: str,
) -> None:
    """Semi-log delay spread vs range for all inhomogeneous media on one panel."""
    ranges = list(cfg.link_ranges_m)
    fig, axes = plt.subplots(1, len(ALL_BEAMS), figsize=(7 * len(ALL_BEAMS), 5))
    if len(ALL_BEAMS) == 1:
        axes = [axes]
    fig.suptitle(
        "RMS Delay Spread vs Link Range  |  Inhomogeneous Media",
        fontsize=12, fontweight="bold",
    )
    for ax, beam in zip(axes, ALL_BEAMS):
        for i, medium in enumerate(media):
            colour = _INH_COLOURS[i % len(_INH_COLOURS)]
            marker = _INH_MARKERS[i % len(_INH_MARKERS)]
            ds = [metrics[RunKey(medium.name, beam.name, float(Z))]["delay_spread_s"]
                  for Z in ranges]
            ax.semilogy(ranges, ds, marker=marker, color=colour,
                        label=medium.name, **_LINE_KW)
        _style_ax(ax, "Link Range (m)", "RMS Delay Spread (s)", beam.name)
        ax.legend(fontsize=7)
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f"{y:.2e}")
        )
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "fig_inh3_delay_spread.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    save_dir: str,
) -> None:
    """Render and save all five homogeneous figures."""
    os.makedirs(save_dir, exist_ok=True)
    plot_received_power(metrics, cfg, save_dir)
    plot_cir(metrics, cfg, save_dir)
    plot_frequency_response(metrics, cfg, save_dir)
    plot_delay_spread(metrics, cfg, save_dir)
    plot_bandwidth(metrics, cfg, save_dir)


def plot_all_inhomogeneous(
    metrics:  Dict[RunKey, dict],
    cfg:      SimConfig,
    media:    Tuple,
    save_dir: str,
) -> None:
    """Render and save all three inhomogeneous figures."""
    os.makedirs(save_dir, exist_ok=True)
    plot_inhomogeneous_power(metrics, cfg, media, save_dir)
    plot_inhomogeneous_cir(metrics, cfg, media, save_dir)
    plot_inhomogeneous_delay_spread(metrics, cfg, media, save_dir)