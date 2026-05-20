# monte-carlo-uowc

A Python reference implementation of a Monte Carlo simulation for under-
water optical wireless communication (UOWC) in a homogeneous medium.

The project models photon propagation through seawater, computes channel
metrics, prints summary tables, and saves publication-ready figures.

## Key features

- Homogeneous underwater optical channel simulation using Monte Carlo photon
  transport
- Two water types: Clear Water and Coastal Water
- Two transmitter beam geometries: Collimated (laser-like) and Diffused
  (LED-like)
- Parallel sweep over multiple link ranges using `ProcessPoolExecutor`
- Channel metrics: received power, RMS delay spread, CIR, frequency response,
  and 3 dB bandwidth
- Five output figures saved to disk for post-run analysis

## Requirements

- Python 3.12 or newer
- NumPy
- Matplotlib

> Note: `pyproject.toml` currently declares `requires-python = ">=3.12"
>`, but does not list runtime dependencies. Install `numpy` and `matplotlib`
> manually if needed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

If you prefer, install packages in your own environment, then run the project
from the repository root.

## Usage

Run the simulation from the repository root:

```bash
python main.py
```

Optionally specify an output directory:

```bash
python main.py /path/to/outputs
```

The default output directory is `/mnt/user-data/outputs`.

## What the simulation does

The entry point is `main.py`, which performs the following pipeline:

1. Print a run header summarising the simulation configuration
2. Execute a parameter sweep across water types, beam types, and link ranges
3. Compute channel metrics from captured photon weights and time-of-flight data
4. Print summary tables for delay spread, received power, and bandwidth
5. Save figures to the output directory

## Project structure

- `main.py` — top-level orchestration and CLI entry point
- `config/` — physical constants, optical presets, and simulation parameters
- `transport/` — Monte Carlo photon propagation engine and receiver crossing logic
- `simulation/` — parallel sweep orchestration and worker dispatch
- `metrics/` — channel metric computation, CIR and frequency response analysis
- `plotting/` — figure generation and file saving
- `reporting.py` — console output and summary table formatting
- `physics/` — pure optical physics utilities used by transport and metrics

## Preset configurations

- Water types:
  - Clear Water: `c=0.241 m⁻¹`, `a=0.151 m⁻¹`, `b=0.090 m⁻¹`, `g=0.924`
  - Coastal Water: `c=0.775 m⁻¹`, `a=0.220 m⁻¹`, `b=0.555 m⁻¹`, `g=0.924`
- Beam types:
  - Collimated: 1.5 mrad half-angle
  - Diffused: 15° half-angle
- Receiver:
  - 10.16 cm diameter aperture
  - 180° field of view
- Link ranges: 5, 10, 15, 20, 25 meters
- Photon count per run: 1,000,000

## Output

The simulation produces:

- console summary tables for delay spread, received power, and bandwidth
- figure files saved in the output directory:
  - `fig1_received_power.png`
  - `fig2_cir.png`
  - `fig3_frequency_response.png`
  - `fig4_delay_spread.png`
  - `fig5_bandwidth.png`

## Notes

- The transport engine uses a Monte Carlo approach with free-flight sampling,
  Henyey-Greenstein scattering, implicit absorption, and Russian roulette.
- The simulation is designed to separate concerns: configuration, transport,
  metrics, plotting, and reporting are all independent modules.
- The current implementation assumes a homogeneous medium and a fixed
  receiver plane at the target link range.
