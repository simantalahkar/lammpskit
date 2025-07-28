# Electrochemical Cell Analysis Example

This directory contains a complete example workflow for analyzing LAMMPS simulation data from electrochemical cell simulations using LAMMPSKit.

## Files Structure

- `run_analysis.py` - Main analysis script demonstrating comprehensive workflow
- `usage/ecellmodel/data/` - Example simulation data directory
- `usage/ecellmodel/output/` - Generated plots and results directory

## Usage

```bash
cd usage/ecellmodel
python run_analysis.py
```

## Analysis Workflow

The `run_analysis.py` script demonstrates four main analysis types:

1. **Filament Evolution Tracking** - Tracks connectivity, gap, and separation over time
2. **Displacement Analysis** - Compares atomic displacements for different species
3. **Charge Distribution Analysis** - Analyzes atomic charge distributions
4. **Atomic Distribution Analysis** - Studies atomic distributions under different voltages

## Simulation Parameters

The example data corresponds to various simulation setups:

- **TIME_STEP**: 0.001 ps (filament tracking), 0.0002 ps (displacement analysis)
- **DUMP_INTERVAL_STEPS**: 500 (filament tracking), 5000 (displacement analysis)
- **Data Types**: LAMMPS trajectory files (.lammpstrj) and thermodynamic output containing running chunk-wise displacement timeseries (.dat)

## Requirements

**Installation:**
```bash
# For development (if working from repository)
pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)

# For usage only (from PyPI)
pip install lammpskit
```

**Other requirements:**
- LAMMPSKit package installed
- Example data files in the `data/` subdirectory
- Python environment with numpy, matplotlib, and ovito

## Customization

The script can be easily modified to:
- Use different simulation parameters
- Analyze custom data files
- Use different column configurations according to data and analysis.
- Adjust analysis parameters

See the script comments for detailed guidance on customization options.
