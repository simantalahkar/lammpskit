#!/usr/bin/env python3
"""
LAMMPSKit Example: Comprehensive Analysis Workflow

This script demonstrates how to use LAMMPSKit for analyzing LAMMPS 
molecular dynamics simulation data. It includes examples of:

- Filament evolution tracking
- Atomic distribution analysis  
- Charge distribution analysis
- Displacement comparison

Data Requirements:
- LAMMPS trajectory and displacement files in usage/ecellmodel/data/ subdirectory
- Simulation metadata (TIME_STEP, DUMP_INTERVAL_STEPS)

Usage:
    python run_analysis.py

Author: Simanta Lahkar
"""

import os
import sys
import glob
from pathlib import Path

# Add package to path if running from repository
if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))

from lammpskit.ecellmodel.filament_layer_analysis import (
    track_filament_evolution,
    plot_atomic_distribution, 
    plot_atomic_charge_distribution,
    plot_displacement_comparison
)
from lammpskit.config import EXTENDED_COLUMNS_TO_READ

def main():
    """
    Main analysis workflow demonstrating LAMMPSKit capabilities.
    
    This function orchestrates a comprehensive analysis workflow including:
    1. Filament evolution tracking over time
    2. Displacement analysis for different atomic species
    3. Atomic charge distribution analysis
    4. Atomic distribution analysis under different voltages
    
    Each analysis block demonstrates different simulation parameters and 
    data processing approaches suitable for various research scenarios.
    """
    # Set output directory for generated plots and analysis results
    output_dir = os.path.join(".", "usage", "ecellmodel", "output")

    # Use EXTENDED_COLUMNS_TO_READ for comprehensive analysis in main script
    columns_to_read = EXTENDED_COLUMNS_TO_READ

    ## Can also use custom columns for specific analysis
    # custom_columns = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    # columns_to_read = custom_columns
    
    ## ANALYSIS BLOCK 1: Filament Evolution Tracking
    ## The following code block generates plots that track the evolution of the 
    # filament connectivity state, gap, and separation over time for each 
    # timeseries trajectory file in the file_list.

    print("="*50)
    print("ANALYSIS BLOCK 1: Filament Evolution Tracking")
    print("="*50)

    ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.001
    DUMP_INTERVAL_STEPS = 500

    ###################################

    data_path = os.path.join(".", "usage", "ecellmodel", "data", "trajectory_series", "*.lammpstrj")
    analysis_name = 'track_'
    # data_path = "*.lammpstrj"
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print('The data_path is', data_path)
    print(analysis_name, file_list)

    if file_list:
        track_filament_evolution(file_list, analysis_name, TIME_STEP, DUMP_INTERVAL_STEPS, output_dir=output_dir)
        print("Filament evolution tracking completed.")
    else:
        print("No trajectory files found for filament evolution analysis.")

    #####################################

    ## ANALYSIS BLOCK 2: Displacement Analysis
    ## The following code block generates plots of atomic charge distributions
    # and compares the displacements of Hf, O, and Ta for different temperatures
    
    print("\n" + "="*50)
    print("ANALYSIS BLOCK 2: Displacement Analysis")
    print("="*50)
    
    ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.0002
    DUMP_INTERVAL_STEPS = 5000

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 2500000
    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)
    
    SKIP_ROWS_COORD = 9   
    HISTOGRAM_BINS = 15
    ###################################

    data_path = os.path.join(".", "usage", "ecellmodel", "data", "[1-9][A-Z][A-Za-z]mobilestc1.dat")
    analysis_name = 'displacements_atom_type'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name, file_list)
    labels = ['Hf', 'O', 'Ta']
    
    if file_list:
        plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0, output_dir=output_dir)
        print("Displacement comparison analysis completed.")
    else:
        print("No displacement data files found.")

    ## ANALYSIS BLOCK 3: Charge Distribution Analysis
    print("\n" + "="*50)
    print("ANALYSIS BLOCK 3: Charge Distribution Analysis")
    print("="*50)

    analysis_name = f'local_charge_{HISTOGRAM_BINS}'
    data_path = os.path.join(".", "usage", "ecellmodel", "data", "local2*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name, file_list)
    labels = ['initial', 'final']
    
    if file_list:
        plot_atomic_charge_distribution(file_list, labels, SKIP_ROWS_COORD, HISTOGRAM_BINS, analysis_name, output_dir=output_dir, columns_to_read=columns_to_read)
        print("Atomic charge distribution analysis completed.")
    else:
        print("No charge distribution data files found.")

    ## ANALYSIS BLOCK 4: Atomic Distribution Analysis
    ## The following code block generates plots of atomic distributions 
    #  of Hf, O, and Ta for different end states corresponding to simulations
    #  under different applied voltages. 
    #  
    ## Simulation parameters corresponding to the respective raw data
    ###################################

    print("\n" + "="*50)
    print("ANALYSIS BLOCK 4: Atomic Distribution Analysis")
    print("="*50)

    TIME_STEP = 0.001
    DUMP_INTERVAL_STEPS = 500

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 500000

    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    SKIP_ROWS_COORD = 9   
    HISTOGRAM_BINS = 15
    ###################################

    analysis_name = f'break_{HISTOGRAM_BINS}'
    data_path = os.path.join(".", "usage", "ecellmodel", "data", "[1-9]break*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name, file_list)
    labels = ['set-0.1V', 'break-0.5V']
    
    if file_list:
        plot_atomic_distribution(file_list, labels, SKIP_ROWS_COORD, HISTOGRAM_BINS, analysis_name, output_dir=output_dir, columns_to_read=columns_to_read)
        print("Atomic distribution analysis completed.")
    else:
        print("No atomic distribution data files found.")

    print("\n" + "="*50)
    print("All analyses completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
