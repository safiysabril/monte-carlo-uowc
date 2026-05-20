"""
uowc.config
===========
Single source of truth for every configurable value in the simulation.
Nothing in this module performs computation — it only declares data.

Separation-of-Concern role
---------------------------
  All magic numbers, physical constants, and simulation knobs live here.
  Every other module imports from here; nothing hard-codes values locally.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT:  float = 3.0e8       # speed of light in vacuum (m/s)
N_WATER:  float = 1.33        # seawater refractive index at 530 nm
C_MEDIUM: float = C_LIGHT / N_WATER   # propagation speed in seawater (~2.256 × 10⁸ m/s)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter dataclasses  (frozen = hashable, immutable, accidental-write-safe)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class WaterParams:
    """Inherent optical properties for one water type at the simulation wavelength."""
    name: str
    c:    float   # beam-attenuation coefficient  (m⁻¹)  c = a + b
    a:    float   # absorption coefficient         (m⁻¹)
    b:    float   # scattering coefficient         (m⁻¹)
    g:    float   # Henyey-Greenstein asymmetry parameter  (dimensionless, 0 < g < 1)

    @property
    def omega(self) -> float:
        """Single-scattering albedo  ω = b / c."""
        return self.b / self.c


@dataclass(frozen=True)
class BeamParams:
    """Transmitter beam geometry."""
    name:           str
    divergence_rad: float   # half-angle of the beam cone (rad)
    waist_m:        float   # 1/e² Gaussian beam-waist radius (m)


@dataclass(frozen=True)
class ReceiverParams:
    """Optical receiver geometry."""
    aperture_radius_m: float   # physical aperture radius (m)
    fov_rad:           float   # half-angle field-of-view (rad);  π ≡ 180°


@dataclass(frozen=True)
class SimConfig:
    """
    All tunable knobs for one simulation run.
    Changing any value here propagates automatically to transport, metrics,
    and plotting without touching those modules.
    """
    n_photons:        int            # total photons launched
    link_ranges_m:    Tuple[float, ...]   # evaluation depths (m)
    dt_bin_s:         float          # CIR histogram bin width (s)
    n_time_bins:      int            # number of CIR time bins
    weight_threshold: float          # Russian-roulette kill threshold
    roulette_m:       int            # Russian-roulette multiplier
    n_workers:        int            # parallel worker processes
    master_seed:      int            # top-level RNG seed (reproducibility)
    chunk_size:       int            # photon mini-batch size per iteration


# ─────────────────────────────────────────────────────────────────────────────
# Preset instances  (values from Sabril et al. 2021 + Gabriel et al. 2013)
# ─────────────────────────────────────────────────────────────────────────────

# --- Water types  (530 nm, homogeneous medium) --------------------------------
CLEAR_WATER = WaterParams(
    name="Clear Water",
    c=0.241, a=0.151, b=0.090, g=0.924,
)
COASTAL_WATER = WaterParams(
    name="Coastal Water",
    c=0.775, a=0.220, b=0.555, g=0.924,
)

# --- Beam types ---------------------------------------------------------------
COLLIMATED = BeamParams(
    name="Collimated (Laser)",
    divergence_rad=1.5e-3,           # 1.5 mrad half-angle
    waist_m=1e-3,
)
DIFFUSED = BeamParams(
    name="Diffused (LED)",
    divergence_rad=np.deg2rad(15.0), # 15°  half-angle
    waist_m=1e-3,
)

# --- Receiver -----------------------------------------------------------------
RECEIVER = ReceiverParams(
    aperture_radius_m=0.1016 / 2.0,  # 10.16 cm diameter
    fov_rad=np.pi,                   # 180° → accept all arrival angles
)

# --- Simulation run configuration --------------------------------------------
SIM = SimConfig(
    n_photons        = 1_000_000,
    link_ranges_m    = (5, 10, 15, 20, 25),
    dt_bin_s         = 1e-11,        # 10 ps bins
    n_time_bins      = 3000,
    weight_threshold = 1e-4,
    roulette_m       = 10,
    n_workers        = max(1, os.cpu_count() or 2),
    master_seed      = 20260519,
    chunk_size       = 10_000,
)

# --- Convenience collections (used by sweep loops) ---------------------------
ALL_WATERS: Tuple[WaterParams, ...] = (CLEAR_WATER, COASTAL_WATER)
ALL_BEAMS:  Tuple[BeamParams,  ...] = (COLLIMATED, DIFFUSED)
