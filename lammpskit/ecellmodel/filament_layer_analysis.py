

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ovito.io import import_file
import ovito.modifiers as om
from typing import Any, Dict, List, Optional

# Import configuration settings
from ..config import (
    DEFAULT_TIMESERIES_CONFIG, 
    DEFAULT_PLOT_CONFIG,
    TimeSeriesConfig,
    PlotConfig,
    process_displacement_timeseries_data,
    plot_timeseries_grid,
    save_figure
)

plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# mpl.rcParams['pdf.fonttype'] = 42  #this would allow us to edit the fonts in acrobat illustrator

"""
filament_layer_analysis.py

Module for post-processing and analyzing LAMMPS molecular dynamics simulation data.

This file contains functions for:
- Reading LAMMPS output data
- Analyzing filament layers and clusters
- Plotting results using matplotlib
- Orchestrating analysis workflows

Refactored for modularity, maintainability, and Sphinx documentation.
"""

# =========================
# Global Configuration (To be centralized)
# =========================
# TODO: Identify and move global settings here for future modularization
# Example:
# DEFAULT_PLOT_STYLE = 'ggplot'
# DATA_DIR = 'data/'

# =========================
# Data Reading Functions
# =========================


def read_lammps_data(filename: str) -> Dict[str, Any]:
    """
    Reads a LAMMPS data file and returns structured data.

    Parameters
    ----------
    filename : str
        Path to the LAMMPS data file.

    Returns
    -------
    data : dict
        Parsed data from the file.
    """
    # TODO: Implement actual reading logic based on file format
    # Example: parse columns, handle headers, etc.
    data = {}
    # ...parsing logic...
    return data

# =========================
# Analysis Functions
# =========================
def analyze_filament_layers(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyzes filament layers from structured data.

    Parameters
    ----------
    data : dict
        Structured data from LAMMPS output.

    Returns
    -------
    layers : list of dict
        Analysis results for each filament layer.
    """
    layers = []
    # TODO: Implement analysis logic
    # Example: identify layers, compute properties
    return layers

# =========================
# Plotting Functions
# =========================
def plot_filament_layers(layers: List[Dict[str, Any]], save_path: Optional[str] = None) -> None:
    """
    Plots filament layers using matplotlib.

    Parameters
    ----------
    layers : list of dict
        Analysis results for each filament layer.
    save_path : str, optional
        If provided, saves the plot to this path.
    """
    fig, ax = plt.subplots()
    # TODO: Implement plotting logic
    # Example: plot layer positions, colors, etc.
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# =========================
# Orchestration Function
# =========================
def main_analysis(filename: str, save_plot: bool = False, plot_path: Optional[str] = None) -> None:
    """
    Orchestrates the analysis workflow for filament layers.

    Parameters
    ----------
    filename : str
        Path to the LAMMPS data file.
    save_plot : bool, optional
        Whether to save the plot to disk.
    plot_path : str, optional
        Path to save the plot if save_plot is True.
    """
    # =========================
    # Main orchestration logic for analysis workflow
    # =========================
    # Read data
    data = read_lammps_data(filename)
    # Analyze layers
    layers = analyze_filament_layers(data)
    # Plot results
    if save_plot:
        plot_filament_layers(layers, save_path=plot_path)
    else:
        plot_filament_layers(layers)
    # TODO: Expand orchestration to handle additional analysis and plotting modules in future modularization.

# =========================
# Legacy/Redundant Code (Candidates for deprecation)
# =========================
# TODO: Mark unused or legacy functions for removal
# Example:
# def old_analysis_method(...):
#     """Deprecated: Use analyze_filament_layers instead."""
#     pass
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------

COLUMNS_TO_READ = (0,1,2,3,4,5,9,10,11,12) #,13,14,15,16
# =========================
# Structure Info Reading Function (Restored and Refactored)
# =========================
def read_structure_info(filepath: str) -> tuple[int, int, float, float, float, float, float, float]:
    """
    Reads the structure file and returns the timestep, total number of atoms, and the box dimensions.

    Parameters
    ----------
    filepath : str
        Path to the structure file.

    Returns
    -------
    tuple
        (timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi)
        timestep : int
            Simulation timestep.
        total_atoms : int
            Total number of atoms in the simulation.
        xlo, xhi : float
            Lower and upper bounds of the simulation box in x.
        ylo, yhi : float
            Lower and upper bounds of the simulation box in y.
        zlo, zhi : float
            Lower and upper bounds of the simulation box in z.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    EOFError
        If expected data is missing.
    ValueError
        If data is malformed.
    OSError
        If there is an error opening the file.
    """
    skip = 1
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            # Skip header lines before timestep
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"File is empty: {filepath}")
            # Read timestep
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for Timestep: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                timestep = int(c[0])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed timestep line in file: {filepath} (got: {line.strip()})")

            # Skip header lines before total atoms
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"Missing section for total atoms: {filepath}")
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for total atoms: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                total_atoms = int(c[0])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed total atoms line in file: {filepath} (got: {line.strip()})")

            # Skip header lines before box bounds
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"Missing section for box bounds: {filepath}")
            # Read x bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for x bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                xlo = float(c[0])
                xhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed x bounds line in file: {filepath} (got: {line.strip()})")

            # Read y bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for y bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                ylo = float(c[0])
                yhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed y bounds line in file: {filepath} (got: {line.strip()})")

            # Read z bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for z bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                zlo = float(c[0])
                zhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed z bounds line in file: {filepath} (got: {line.strip()})")
        return timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:
        raise OSError(f"Error opening file {filepath}: {e}")


def read_coordinates(
    file_list: list[str],
    skip_rows: int,
    columns_to_read: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, int, float, float, float, float, float, float]:
    """
    Reads multiple LAMMPS structure files and extracts simulation cell parameters, coordinates, and timesteps.

    Parameters
    ----------
    file_list : list of str
        List of file paths to structure files.
    skip_rows : int
        Number of header rows to skip before atomic coordinates.
    columns_to_read : tuple of int
        Indices of columns to read from each file.

    Returns
    -------
    tuple
        (coordinates, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi)
        coordinates : np.ndarray
            Array of atomic coordinates for all files.
        timestep_arr : np.ndarray
            Array of timesteps for all files.
        total_atoms : int
            Number of atoms (should be the same for all files).
        xlo, xhi, ylo, yhi, zlo, zhi : float
            Simulation box bounds (should be the same for all files).

    Raises
    ------
    ValueError
        If file_list is empty or atomic data is malformed.
    EOFError
        If a file has fewer atom lines than expected.
    """
    print(file_list)
    if not file_list:
        raise ValueError("file_list is empty. No files to process.")
    timestep_arr: list[int] = []
    coordinates: list[np.ndarray] = []
    for filepath in file_list:
        # Extract simulation parameters from each file
        timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info(filepath)
        timestep_arr.append(timestep)
        try:
            # Read atomic coordinates
            coords = np.loadtxt(
                filepath,
                delimiter=' ',
                comments='#',
                skiprows=skip_rows,
                max_rows=total_atoms,
                usecols=columns_to_read
            )
        except ValueError as e:
            raise ValueError(
                f"Column index out of range or Non-float atomic data in file: {filepath} (error: {e})\n (column indices provided to read = {columns_to_read})"
            )
        if coords.shape[0] != total_atoms:
            raise EOFError(f"File {filepath} has fewer atom lines ({coords.shape[0]}) than expected ({total_atoms})")
        coordinates.append(coords)
    # Return arrays and simulation parameters
    return (
        np.array(coordinates),
        np.array(timestep_arr),
        total_atoms,
        xlo, xhi, ylo, yhi, zlo, zhi
    )


def read_displacement_data(
    filepath: str,
    loop_start: int,
    loop_end: int,
    repeat_count: int = 0
) -> list[np.ndarray]:
    """
    Reads binwise averaged displacement data from a file and returns data for specified loops.

    Parameters
    ----------
    filepath : str
        Path to the binwise averaged output data file.
    loop_start : int
        Starting loop index (inclusive).
    loop_end : int
        Ending loop index (inclusive).
    repeat_count : int, optional
        Number of times the first timestep is repeated in the data file (default: 0).

    Returns
    -------
    thermo : list of np.ndarray
        List of displacement data arrays for each loop.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If loop_start > loop_end or column index is out of range.
    EOFError
        If expected data is missing or malformed.
    TypeError
        If Nchunks line is malformed.
    """
    print(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if loop_start > loop_end:
        raise ValueError(f"loop_start ({loop_start}) is greater than loop_end ({loop_end})")
    try:
        tmp = np.loadtxt(filepath, comments='#', skiprows=3, max_rows=1)
    except ValueError:
        raise TypeError(f"Malformed Nchunks line in file: {filepath}")

    try:
        Nchunks = int(tmp[1])
    except (IndexError, ValueError) as e:
        if isinstance(e, IndexError):
            raise EOFError(f"Missing Nchunks line in file: {filepath}") from e
        elif isinstance(e, ValueError):
            raise TypeError(f"Malformed Nchunks line in file: {filepath}") from e

    thermo: list[np.ndarray] = []
    for n in range(loop_start, loop_end + 1):
        try:
            chunk = np.loadtxt(
                filepath,
                comments='#',
                skiprows=3 + 1 + (n - loop_start) * (Nchunks + 4),
                max_rows=Nchunks
            )
        except ValueError:
            raise EOFError(f"Missing or malformed chunk data for loop {n} in file: {filepath}")
        if chunk.shape[0] != Nchunks:
            raise EOFError(f"Not enough data for chunk {n} in file: {filepath}")
        thermo.append(chunk)
    return thermo
    

def plot_multiple_cases(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    labels: list,
    xlabel: str,
    ylabel: str,
    output_filename: str,
    xsize: float,
    ysize: float,
    output_dir: str = os.getcwd(),
    **kwargs
) -> plt.Figure:
    """
    Plots multiple cases with the given x and y arrays, labels, and saves the figure.

    Parameters
    ----------
    x_arr : np.ndarray
        Array(s) of x values for each case.
    y_arr : np.ndarray
        Array(s) of y values for each case.
    labels : list
        List of labels for each case.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    output_filename : str
        Base filename for saving the figure (without extension).
    xsize : float
        Width of the figure in inches.
    ysize : float
        Height of the figure in inches.
    output_dir : str, optional
        Directory to save the figure. Defaults to current working directory.
    **kwargs
        Additional keyword arguments for customizing the plot (e.g., axis limits, marker style).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object for further use if needed.
    """
    nrows = 1
    ncolumns = 1
    xsize = 1.6
    ysize = 3.2
    print('before subplots')
    plt.ioff()
    fig, axes = plt.subplots(nrows, ncolumns, squeeze=False, constrained_layout=False, figsize=(xsize, ysize))
    print('before axes flatten')
    axes = axes.flatten()
    print('before tight layout')
    fig.tight_layout()
    #plt.rcParams['xtick.labelsize'] = 6
    #plt.rcParams['ytick.labelsize'] = 6
    colorlist = ['b', 'r', 'g', 'k']
    linestylelist = ['--', '-.', ':', '-']
    markerlist = ['o', '^', 's', '*']
    print('reached now plotting point')

    # Plot each case depending on array dimensions
    if x_arr.ndim > 1 and y_arr.ndim > 1:
        for i in range(len(x_arr)):
            j = kwargs.get('markerindex', i)
            axes[0].plot(x_arr[i], y_arr[i], label=labels[i], color=colorlist[j], linestyle=linestylelist[j], marker=markerlist[j], markersize=5, linewidth=1.2, alpha=0.75)
    elif x_arr.ndim > 1 and y_arr.ndim == 1:
        for i in range(len(x_arr)):
            j = kwargs.get('markerindex', i)
            axes[0].plot(x_arr[i], y_arr, label=labels[i], color=colorlist[j], linestyle=linestylelist[j], marker=markerlist[j], markersize=5, linewidth=1.2, alpha=0.75)
    elif x_arr.ndim == 1 and y_arr.ndim > 1:
        for i in range(len(y_arr)):
            j = kwargs.get('markerindex', i)
            axes[0].plot(x_arr, y_arr[i], label=labels[i], color=colorlist[j], linestyle=linestylelist[j], marker=markerlist[j], markersize=5, linewidth=1.2, alpha=0.75)
    else:
        j = kwargs.get('markerindex', 0)
        axes[0].plot(x_arr, y_arr, label=labels, color=colorlist[j], linestyle=linestylelist[j], marker=markerlist[j], markersize=5, linewidth=1.2, alpha=0.75)

    # Optionally plot atom bin counts and print averages
    if 'ncount' in kwargs:
        atoms_per_bin_count = kwargs['ncount']
        for i in range(len(x_arr)):
            Ncount_temp = atoms_per_bin_count[i, atoms_per_bin_count[i] > 0]
            x_arr_temp = x_arr[i, atoms_per_bin_count[i] > 0]
            average = np.sum(x_arr[i] * atoms_per_bin_count[i]) / np.sum(atoms_per_bin_count[i])
            print(f'\n The average for {labels[i]} in {output_filename} is {average} \n')

    # Axis limits and lines
    if 'xlimit' in kwargs:
        print('x axis is limited')
        axes[0].set_xlim(kwargs['xlimit'])
    if 'ylimit' in kwargs:
        print('y axis is limited')
        axes[0].set_ylim(kwargs['ylimit'])
    if 'xlimithi' in kwargs:
        print('x hi axis is limited')
        axes[0].set_xlim(right=kwargs['xlimithi'])
    if 'ylimithi' in kwargs:
        print('y hi axis is limited')
        axes[0].set_ylim(top=kwargs['ylimithi'])
    if 'xlimitlo' in kwargs:
        print('x lo axis is limited')
        axes[0].set_xlim(left=kwargs['xlimitlo'])
    if 'ylimitlo' in kwargs:
        print('y lo axis is limited')
        axes[0].set_ylim(bottom=kwargs['ylimitlo'])

    if 'xaxis' in kwargs:
        axes[0].axhline(y=0, color=colorlist[-1], linestyle=linestylelist[-1], linewidth=1, label='y=0')
    if 'yaxis' in kwargs:
        axes[0].axvline(x=0, color=colorlist[-1], linestyle=linestylelist[-1], linewidth=1, label='x=0')

    print('reached axes labelling point')
    axes[0].set_ylabel(ylabel, fontsize=8)
    axes[0].legend(loc='upper center', fontsize=7)
    axes[0].adjustable = 'datalim'
    axes[0].set_aspect('auto')
    axes[0].tick_params(axis='both', which='major', labelsize=7)
    axes[0].set_aspect('auto')
    axes[0].set_xlabel(xlabel, fontsize=8)
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)

    #plt.suptitle(f'{datatype} {dataindexname[dataindex]}', fontsize=8)
    #plt.show()

    plt.ioff()
    print('reached file saving point')
    output_filename_pdf = output_filename + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename_pdf)
    fig.savefig(savepath, bbox_inches='tight', format='pdf')
    output_filename_svg = output_filename + '.svg'
    savepath = os.path.join(output_dir, output_filename_svg)
    fig.savefig(savepath, bbox_inches='tight', format='svg')
    plt.close()
    return fig
 
def plot_atomic_distribution(
    file_list: list[str],
    labels: list[str],
    skip_rows: int,
    z_bins: int,
    analysis_name: str,
    output_dir: str = os.getcwd(),
    **kwargs
) -> dict[str, plt.Figure]:
    """
    Reads the coordinates from the file_list, calculates the atomic distributions,
    and plots the distributions for O, Hf, Ta, and all M atoms.

    Parameters
    ----------
    file_list : list of str
        List of file paths to structure files.
    labels : list of str
        List of labels for each case.
    skip_rows : int
        Number of header rows to skip before atomic coordinates.
    z_bins : int
        Number of bins along the z-axis for histogramming.
    analysis_name : str
        Base name for output files.
    output_dir : str, optional
        Directory to save output figures. Defaults to current working directory.
    **kwargs
        Additional keyword arguments for customizing the plots.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.
    """
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
    z_bin_width = (zhi - zlo) / z_bins
    z_bin_centers = np.linspace(zlo + z_bin_width / 2, zhi - z_bin_width / 2, z_bins)

    # Initialize distributions
    oxygen_distribution: list[np.ndarray] = []
    hafnium_distribution: list[np.ndarray] = []
    tantalum_distribution: list[np.ndarray] = []

    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")
    for coordinates in coordinates_arr:
        # Sort by z position
        coordinates = coordinates[coordinates[:, 5].argsort()]

        # Select atom types
        hf_atoms = coordinates[coordinates[:, 1] == 2]
        ta_atoms = coordinates[np.logical_or(coordinates[:, 1] == 4, np.logical_or(coordinates[:, 1] == 6, coordinates[:, 1] == 10))]
        o_atoms = coordinates[np.logical_or(coordinates[:, 1] == 1, np.logical_or(coordinates[:, 1] == 3, np.logical_or(coordinates[:, 1] == 5, coordinates[:, 1] == 9)))]

        # Histogram distributions
        hafnium_distribution.append(np.histogram(hf_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])
        oxygen_distribution.append(np.histogram(o_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])
        tantalum_distribution.append(np.histogram(ta_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])

    # Convert to arrays
    hafnium_distribution = np.array(hafnium_distribution)
    tantalum_distribution = np.array(tantalum_distribution)
    oxygen_distribution = np.array(oxygen_distribution)
    metal_distribution = hafnium_distribution + tantalum_distribution
    total_distribution = metal_distribution + oxygen_distribution

    # Avoid division by zero
    total_distribution_divide = total_distribution.copy()
    total_distribution_divide[total_distribution_divide == 0] = 1

    # Calculate stoichiometry for last and first trajectory
    O_stoich = 3.5 * oxygen_distribution[-1] / total_distribution_divide[-1]
    Ta_stoich = 3.5 * tantalum_distribution[-1] / total_distribution_divide[-1]
    Hf_stoich = 3.5 * hafnium_distribution[-1] / total_distribution_divide[-1]
    stoichiometry = np.array([Hf_stoich, O_stoich, Ta_stoich])
    proportion_labels = np.array(['a (of Hf$_a$)', 'b (of O$_b$)', 'c (of Ta$_c$)'])

    O_stoich_in = 3.5 * oxygen_distribution[0] / total_distribution_divide[0]
    Ta_stoich_in = 3.5 * tantalum_distribution[0] / total_distribution_divide[0]
    Hf_stoich_in = 3.5 * hafnium_distribution[0] / total_distribution_divide[0]
    initial_stoichiometry = np.array([Hf_stoich_in, O_stoich_in, Ta_stoich_in])

    figure_size = [2.5, 5]

    # Plot stoichiometry
    output_filename = f"{analysis_name}_stoichiometry_{z_bins}" + ''.join(f"_{i}" for i in labels)
    fig_stoich = plot_multiple_cases(stoichiometry, z_bin_centers, proportion_labels, 'Atoms # ratio', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)
    print('stoichiometry plotted')

    # Plot initial stoichiometry
    output_filename = f"{analysis_name}_initial_stoichiometry_{z_bins}" + ''.join(f"_{i}" for i in labels)
    fig_init_stoich = plot_multiple_cases(initial_stoichiometry, z_bin_centers, proportion_labels, 'Atoms # ratio', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)
    print('initial stoichiometry plotted')

    # Plot metal atoms
    output_filename = f"{analysis_name}_M" + ''.join(f"_{i}" for i in labels)
    fig_metal = plot_multiple_cases(metal_distribution, z_bin_centers, labels, 'Metal atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot Hf atoms
    output_filename = f"{analysis_name}_Hf" + ''.join(f"_{i}" for i in labels)
    fig_hf = plot_multiple_cases(hafnium_distribution, z_bin_centers, labels, 'Hf atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot Ta atoms
    output_filename = f"{analysis_name}_Ta" + ''.join(f"_{i}" for i in labels)
    fig_ta = plot_multiple_cases(tantalum_distribution, z_bin_centers, labels, 'Ta atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot O atoms
    output_filename = f"{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(oxygen_distribution, z_bin_centers, labels, 'O atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    return {
        "stoichiometry": fig_stoich,
        "initial_stoichiometry": fig_init_stoich,
        "metal": fig_metal,
        "Hf": fig_hf,
        "Ta": fig_ta,
        "O": fig_o,
    }

def plot_atomic_charge_distribution(
    file_list: list[str],
    labels: list[str],
    skip_rows: int,
    z_bins: int,
    analysis_name: str,
    output_dir: str = os.getcwd(),
    **kwargs
) -> dict[str, plt.Figure]:
    """
    Reads the coordinates from the file_list, calculates the atomic charge distributions,
    and plots the charge distributions for O, Hf, Ta, and all M atoms.

    Parameters
    ----------
    file_list : list of str
        List of file paths to structure files.
    labels : list of str
        List of labels for each case.
    skip_rows : int
        Number of header rows to skip before atomic coordinates.
    z_bins : int
        Number of bins along the z-axis for histogramming.
    analysis_name : str
        Base name for output files.
    output_dir : str, optional
        Directory to save output figures. Defaults to current working directory.
    **kwargs
        Additional keyword arguments for customizing the plots.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.
    """
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
    z_bin_width = (zhi - zlo) / z_bins
    z_bin_centers = np.linspace(zlo + z_bin_width / 2, zhi - z_bin_width / 2, z_bins)

    # Initialize distributions
    oxygen_charge_distribution: list[np.ndarray] = []
    hafnium_charge_distribution: list[np.ndarray] = []
    tantalum_charge_distribution: list[np.ndarray] = []
    total_charge_distribution: list[np.ndarray] = []
    hafnium_distribution: list[np.ndarray] = []
    tantalum_distribution: list[np.ndarray] = []
    oxygen_distribution: list[np.ndarray] = []

    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")
    for coordinates in coordinates_arr:
        # Sort by z position
        coordinates = coordinates[coordinates[:, 5].argsort()]

        # Select atom types
        hf_atoms = coordinates[coordinates[:, 1] == 2]
        ta_atoms = coordinates[np.logical_or(coordinates[:, 1] == 4, np.logical_or(coordinates[:, 1] == 6, coordinates[:, 1] == 10))]
        o_atoms = coordinates[np.logical_or(coordinates[:, 1] == 1, np.logical_or(coordinates[:, 1] == 3, np.logical_or(coordinates[:, 1] == 5, coordinates[:, 1] == 9)))]

        # Histogram distributions
        hafnium_distribution.append(np.histogram(hf_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])
        oxygen_distribution.append(np.histogram(o_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])
        tantalum_distribution.append(np.histogram(ta_atoms[:, 5], bins=z_bins, range=(zlo, zhi))[0])

        # Histogram charge distributions
        total_charge_distribution.append(np.histogram(coordinates[:, 5], bins=z_bins, range=(zlo, zhi), weights=coordinates[:, 2])[0])
        hafnium_charge_distribution.append(np.histogram(hf_atoms[:, 5], bins=z_bins, range=(zlo, zhi), weights=hf_atoms[:, 2])[0])
        oxygen_charge_distribution.append(np.histogram(o_atoms[:, 5], bins=z_bins, range=(zlo, zhi), weights=o_atoms[:, 2])[0])
        tantalum_charge_distribution.append(np.histogram(ta_atoms[:, 5], bins=z_bins, range=(zlo, zhi), weights=ta_atoms[:, 2])[0])

    # Convert to arrays
    hafnium_distribution = np.array(hafnium_distribution)
    tantalum_distribution = np.array(tantalum_distribution)
    oxygen_distribution = np.array(oxygen_distribution)
    metal_distribution = hafnium_distribution + tantalum_distribution
    total_distribution = metal_distribution + oxygen_distribution

    total_charge_distribution = np.array(total_charge_distribution)
    hafnium_charge_distribution = np.array(hafnium_charge_distribution)
    tantalum_charge_distribution = np.array(tantalum_charge_distribution)
    metal_charge_distribution = hafnium_charge_distribution + tantalum_charge_distribution
    oxygen_charge_distribution = np.array(oxygen_charge_distribution)

    # Avoid division by zero
    total_distribution_divide = total_distribution.copy()
    total_distribution_divide[total_distribution_divide == 0] = 1
    hafnium_distribution_divide = hafnium_distribution.copy()
    hafnium_distribution_divide[hafnium_distribution_divide == 0] = 1
    tantalum_distribution_divide = tantalum_distribution.copy()
    tantalum_distribution_divide[tantalum_distribution_divide == 0] = 1
    metal_distribution_divide = metal_distribution.copy()
    metal_distribution_divide[metal_distribution_divide == 0] = 1
    oxygen_distribution_divide = oxygen_distribution.copy()
    oxygen_distribution_divide[oxygen_distribution_divide == 0] = 1

    # Calculate mean charge distributions
    total_mean_charge_distribution = total_charge_distribution / total_distribution_divide
    hafnium_mean_charge_distribution = hafnium_charge_distribution / hafnium_distribution_divide
    tantalum_mean_charge_distribution = tantalum_charge_distribution / tantalum_distribution_divide
    metal_mean_charge_distribution = metal_charge_distribution / metal_distribution_divide
    O_mean_charge_dist = oxygen_charge_distribution / oxygen_distribution_divide

    figure_size = [2.5, 5]

    # Plot net charge
    output_filename = f"{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net = plot_multiple_cases(total_charge_distribution, z_bin_centers, labels, 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0)

    # Plot metal mean charge
    output_filename = f"{analysis_name}_M" + ''.join(f"_{i}" for i in labels)
    fig_metal = plot_multiple_cases(metal_mean_charge_distribution, z_bin_centers, labels, 'Metal atoms mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimitlo=0.7, xlimithi=1.2)

    # Plot oxygen mean charge
    output_filename = f"{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(O_mean_charge_dist, z_bin_centers, labels, 'O mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=0, xlimitlo=-0.7)

    # Plot final net charge
    output_filename = f"final_{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net_end = plot_multiple_cases(total_charge_distribution[-1], z_bin_centers, labels[-1], 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0, markerindex=1)

    # Plot initial net charge
    output_filename = f"initial_{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net_start = plot_multiple_cases(total_charge_distribution[0], z_bin_centers, labels[0], 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0)

    # Plot initial metal mean charge
    output_filename = f"initial_{analysis_name}_M" + ''.join(f"_{i}" for i in labels)
    fig_metal_start = plot_multiple_cases(metal_mean_charge_distribution[0], z_bin_centers, labels[0], 'Metal atoms mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimitlo=0.7, xlimithi=1.2)

    # Plot initial oxygen mean charge
    output_filename = f"initial_{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o_start = plot_multiple_cases(O_mean_charge_dist[0], z_bin_centers, labels[0], 'O mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=0, xlimitlo=-0.7)

    return {
        "net_charge": fig_net,
        "initial_net_charge": fig_net_start,
        "final_net_charge": fig_net_end,
        "metal_charge": fig_metal,
        "initial_metal_charge": fig_metal_start,
        "oxygen_charge": fig_o,
        "initial_oxygen_charge": fig_o_start,
    }


def plot_displacement_comparison(
    file_list: list[str],
    loop_start: int,
    loop_end: int,
    labels: list[str],
    analysis_name: str,
    repeat_count: int = 0,
    output_dir: str = os.getcwd()
) -> dict[str, plt.Figure]:
    """
    Reads the averaged thermodynamic output data for each case from the corresponding files in file_list,
    and plots the final displacements (z and lateral displacements) versus the z-bin group positions
    for the data row indices (1st index) corresponding to each case read from a file.

    Parameters
    ----------
    file_list : list of str
        List of file paths to thermodynamic output data files.
    loop_start : int
        Starting loop index (inclusive).
    loop_end : int
        Ending loop index (inclusive).
    labels : list of str
        List of labels for each case.
    analysis_name : str
        Base name for output files.
    repeat_count : int, optional
        Number of times the first timestep is repeated in the data file (default: 0).
    output_dir : str, optional
        Directory to save output figures. Defaults to current working directory.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.
    """
    # Read thermodynamic data for each file
    all_thermo_data: list[list[np.ndarray]] = []
    for filename in file_list:
        all_thermo_data.append(read_displacement_data(filename, loop_start, loop_end, repeat_count))

    displacements = np.array(all_thermo_data)
    print(f"\nshape of all_thermo_data array= {np.shape(all_thermo_data)}, length of all_thermo_data array= {len(all_thermo_data)}")

    # Initialize arrays for plotting
    zdisp: list[np.ndarray] = []
    lateraldisp: list[np.ndarray] = []
    binposition: list[np.ndarray] = []
    atoms_per_bin_count: list[np.ndarray] = []

    all_thermo_data = np.array(all_thermo_data)
    for i in range(len(displacements)):
        # Extract z displacement, lateral displacement, bin position, and atom count per bin
        zdisp_temp = all_thermo_data[i, -1, :, -3]
        lateraldisp_temp = all_thermo_data[i, -1, :, -2]
        binposition_temp = all_thermo_data[i, -1, :, 1]
        Ncount_temp = all_thermo_data[i, -1, :, 2]

        zdisp.append(zdisp_temp)
        lateraldisp.append(lateraldisp_temp)
        binposition.append(binposition_temp)
        atoms_per_bin_count.append(Ncount_temp)

    # Convert lists to arrays
    zdisp = np.array(zdisp)
    lateraldisp = np.array(lateraldisp)
    binposition = np.array(binposition)
    atoms_per_bin_count = np.array(atoms_per_bin_count)

    figure_size = [2.5, 5]

    # Plot z displacement
    output_filename = f"{analysis_name}_z" + ''.join(f"_{i}" for i in labels)
    fig_z = plot_multiple_cases(zdisp, binposition, labels, 'z displacement (A)', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, yaxis=0)

    # Plot z displacement magnitude
    output_filename = f"{analysis_name}_z_magnitude" + ''.join(f"_{i}" for i in labels)
    fig_zmag = plot_multiple_cases(np.abs(zdisp), binposition, labels, 'z displacement (A)', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir)

    # Plot lateral displacement
    output_filename = f"{analysis_name}_lateral" + ''.join(f"_{i}" for i in labels)
    fig_lateral = plot_multiple_cases(lateraldisp, binposition, labels, 'lateral displacement (A)', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir)

    return {
        "z_displacement": fig_z,
        "z_magnitude": fig_zmag,
        "lateral_displacement": fig_lateral,
    }


def analyze_clusters(
    filepath: str,
    z_filament_lower_limit: float = 5,
    z_filament_upper_limit: float = 23,
    thickness: float = 21
) -> tuple[int, int, int, float, np.ndarray, int, float, np.ndarray, float, float]:
    """
    Performs cluster analysis on the given file:
    - Computes the coordination number, selects metallic atoms, clusters them.
    - Deletes the non-filamentary atoms, separates the top and bottom part of filament.
    - Analyzes filament connectivities and RDF of filamentary atoms.

    Parameters
    ----------
    filepath : str
        Path to the input file for OVITO analysis.
    z_filament_lower_limit : float, optional
        Lower z-bound for filament connection (default: 5).
    z_filament_upper_limit : float, optional
        Upper z-bound for filament connection (default: 23).
    thickness : float, optional
        Filament thickness parameter (default: 21).

    Returns
    -------
    tuple
        (
            timestep: int,
            connection: int,
            fil_size_down: int,
            fil_height: float,
            rdf_down: np.ndarray,
            fil_size_up: int,
            fil_depth: float,
            rdf_up: np.ndarray,
            separation: float,
            gap: float
        )

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no clusters are found or file is malformed for OVITO.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        # Import file for OVITO pipelines
        pipeline1 = import_file(filepath)
        pipeline2 = import_file(filepath)
        pipeline_fil = import_file(filepath)
        pipeline_fil_up = import_file(filepath)
    except Exception as e:
        raise ValueError(f"Malformed or unreadable file for OVITO: {filepath} (error: {e})")

    # Pipeline 1: Analyze clusters (main)
    pipeline1.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline1.modifiers.append(om.ExpressionSelectionModifier(expression='((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10 || ParticleType==9 ) && Position.Z < 28 '))
    pipeline1.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline1.modifiers.append(om.ExpressionSelectionModifier(expression='Cluster !=1'))
    pipeline1.modifiers.append(om.DeleteSelectedModifier())
    data1 = pipeline1.compute()
    if data1.particles.count == 0:
        raise ValueError(f"No clusters found in file: {filepath}")

    timestep = data1.attributes['Timestep']
    xyz1 = np.array(data1.particles['Position'])

    # Pipeline 2: Analyze clusters (secondary)
    pipeline2.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline2.modifiers.append(om.ExpressionSelectionModifier(expression='((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28 '))
    pipeline2.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline2.modifiers.append(om.ExpressionSelectionModifier(expression='Cluster !=2'))
    pipeline2.modifiers.append(om.DeleteSelectedModifier())
    data2 = pipeline2.compute()
    xyz2 = np.array(data2.particles['Position'])

    # Determine filament connection and separation
    z1_min, z1_max = np.min(xyz1[:, 2]), np.max(xyz1[:, 2])
    print(np.shape(xyz1), len(xyz2))
    if len(xyz2) != 0:
        # z2_min, z2_max = np.min(xyz2[:, 2]), np.max(xyz2[:, 2])  # Unused
        pass

    if z1_min < z_filament_lower_limit and z1_max > z_filament_upper_limit:
        connection = 1
        separation = 0.0
        gap = 0.0
    else:
        connection = 0
        if z1_min < z_filament_lower_limit:
            group_a = xyz1
            group_b = xyz2
        else:
            group_a = xyz2
            group_b = xyz1
        separation = float('inf')
        gap = float('inf')
        for point1 in group_a:
            for point2 in group_b:
                distance = np.linalg.norm(point1 - point2)
                if distance < separation:
                    separation = distance
                    gap = abs(point1[2] - point2[2])
        separation -= 3.9  # Subtract cutoff

    # Pipeline for lower filament
    pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline_fil.modifiers.append(om.ExpressionSelectionModifier(expression='((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6)'))
    pipeline_fil.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline_fil.modifiers.append(om.ExpressionSelectionModifier(expression='Cluster !=1'))
    pipeline_fil.modifiers.append(om.DeleteSelectedModifier())
    pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff=3.9, number_of_bins=200))
    data_fil = pipeline_fil.compute()
    xyz_fil_down = np.array(data_fil.particles['Position'])
    fil_height = np.max(xyz_fil_down[:, 2])
    rdf_down = data_fil.tables['coordination-rdf'].xy()
    fil_size_down = data_fil.particles.count

    # Pipeline for upper filament
    pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline_fil_up.modifiers.append(om.ExpressionSelectionModifier(expression='((ParticleType==8 && Coordination<6) || ( ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28))'))
    pipeline_fil_up.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline_fil_up.modifiers.append(om.ExpressionSelectionModifier(expression='Cluster !=1'))
    pipeline_fil_up.modifiers.append(om.DeleteSelectedModifier())
    pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff=3.9, number_of_bins=200))
    data_fil_up = pipeline_fil_up.compute()
    xyz_fil_up = np.array(data_fil_up.particles['Position'])
    fil_depth = np.min(xyz_fil_up[:, 2])
    rdf_up = data_fil_up.tables['coordination-rdf'].xy()
    fil_size_up = data_fil_up.particles.count

    return (
        timestep,
        connection,
        fil_size_down,
        fil_height,
        rdf_down,
        fil_size_up,
        fil_depth,
        rdf_up,
        separation,
        gap
    )
    
def track_filament_evolution(
    file_list: list[str],
    analysis_name: str,
    time_step: float,
    dump_interval_steps: int,
    output_dir: str = os.getcwd()
) -> dict[str, plt.Figure]:
    """
    Tracks and plots the evolution of the filament connectivity state, gap, and separation over time for each timeseries trajectory file in the file_list.
    Plots key results including connectivity, gap, separation, filament size, and filament height/depth.

    Parameters
    ----------
    file_list : list of str
        List of file paths to timeseries trajectory files.
    analysis_name : str
        Base name for output files.
    time_step : float
        Simulation time step (ps).
    dump_interval_steps : int
        Number of steps between dumps.
    output_dir : str, optional
        Directory to save output figures. Defaults to current working directory.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.

    Raises
    ------
    FileNotFoundError
        If any file in file_list does not exist.
    ValueError
        If cluster analysis fails for any file.
    """
    # Initialize arrays to collect results
    step_arr: list[int] = []
    connection: list[int] = []
    fil_size_down: list[int] = []
    fil_height: list[float] = []
    rdf_down: list[np.ndarray] = []
    fil_size_up: list[int] = []
    fil_depth: list[float] = []
    rdf_up: list[np.ndarray] = []
    gap: list[float] = []
    separation: list[float] = []

    # Analyze each file
    for filepath in file_list:
        (
            step_temp,
            connection_temp,
            fil_size_down_temp,
            fil_height_temp,
            rdf_down_temp,
            fil_size_up_temp,
            fil_depth_temp,
            rdf_up_temp,
            separation_temp,
            gap_temp
        ) = analyze_clusters(filepath)
        step_arr.append(step_temp)
        connection.append(connection_temp)
        fil_size_down.append(fil_size_down_temp)
        fil_size_up.append(fil_size_up_temp)
        fil_height.append(fil_height_temp)
        fil_depth.append(fil_depth_temp)
        rdf_down.append(rdf_down_temp)
        rdf_up.append(rdf_up_temp)
        gap.append(gap_temp)
        separation.append(separation_temp)

    # Convert lists to numpy arrays for analysis
    step_arr = np.array(step_arr)
    connection = np.array(connection)
    gap = np.array(gap)
    separation = np.array(separation)
    fil_size_down = np.array(fil_size_down)
    fil_size_up = np.array(fil_size_up)
    fil_height = np.array(fil_height)
    fil_depth = np.array(fil_depth)
    rdf_down = np.array(rdf_down)
    rdf_up = np.array(rdf_up)

    print('shape of connections array', np.shape(connection)[0])

    # Calculate time axis
    time_switch = step_arr * time_step * dump_interval_steps

    # Plotting parameters
    ln = 0.1
    mrkr = 5
    alph = 0.55

    # Plot filament connectivity state
    on_frequency = np.sum(connection == 1) / len(connection)
    fig_conn, ax_conn = plt.subplots()
    ax_conn.plot(time_switch, connection, alpha=alph, linewidth=ln, markersize=mrkr)
    ax_conn.scatter(
        time_switch,
        connection,
        alpha=alph,
        linewidth=ln,
        s=mrkr,
        marker="^",
        label=f"filament is in connected state {on_frequency*100: .2f}% of the time",
    )
    ax_conn.set_xlabel("Time (ps)")
    ax_conn.set_ylabel("Filament connectivity state (1: connected, 0: broken)")
    ax_conn.set_title("Filament connectivity state (1: connected, 0: broken)")
    ax_conn.legend()
    output_filename = analysis_name + "OnOff" + ".pdf"
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close(fig_conn)

    # Plot filament gap
    average_filament_gap, sd_gap = np.mean(gap), np.std(gap)
    fig_gap, ax_gap = plt.subplots()
    ax_gap.plot(time_switch, gap, alpha=alph, linewidth=ln, markersize=mrkr)
    ax_gap.scatter(time_switch, gap, alpha=alph, linewidth=ln, s=mrkr, marker='^', label=f'average_filament_gap = {average_filament_gap: .2f} +/- {sd_gap: .2f}')
    ax_gap.set_xlabel('Time (ps)')
    ax_gap.set_ylabel('Filament gap (A)')
    ax_gap.set_title('Filament gap')
    ax_gap.legend()
    output_filename = analysis_name + 'fil_gap' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament separation
    average_filament_separation, sd_separation = np.mean(separation), np.std(separation)
    fig_sep, ax_sep = plt.subplots()
    ax_sep.plot(time_switch, separation, alpha=alph, linewidth=ln, markersize=mrkr)
    ax_sep.scatter(time_switch, separation, alpha=alph, linewidth=ln, s=mrkr, marker='^', label=f'average_filament_separation = {average_filament_separation: .2f} +/- {sd_separation: .2f}')
    ax_sep.set_xlabel('Time (ps)')
    ax_sep.set_ylabel('Filament separation (A)')
    ax_sep.set_title('Filament separation')
    ax_sep.legend(fontsize=8)
    output_filename = analysis_name + 'fil_separation' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament gap & number of conductive atoms
    fig_size_gap, ax1_size_gap = plt.subplots()
    color = 'tab:red'
    ax1_size_gap.set_xlabel('Time (ps)')
    ax1_size_gap.set_ylabel('Filament gap (A)', color=color)
    ax1_size_gap.scatter(time_switch, gap, alpha=alph, linewidth=ln, s=mrkr, color=color, label=f'average_filament_gap = {average_filament_gap: .2f} +/- {sd_gap: .2f}')
    ax1_size_gap.tick_params(axis='y', labelcolor=color)
    ax1_size_gap.set_ylim(-0.5, 8.5)

    ax2_size_gap = ax1_size_gap.twinx()
    color = 'tab:blue'
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    ax2_size_gap.set_ylabel('# of vacancies in filament (A.U.)', color=color)
    ax2_size_gap.scatter(time_switch, fil_size_down, alpha=alph, linewidth=ln, s=mrkr, marker='^', color=color, label=f'average # of vacancies in filament = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax2_size_gap.tick_params(axis='y', labelcolor=color)
    ax2_size_gap.set_ylim(0, 350)

    plt.title('Gap & no. of conductive atoms in Filament')
    fig_size_gap.tight_layout()
    ax1_size_gap.legend(loc='upper right', framealpha=0.8)
    ax2_size_gap.legend(loc='lower right', framealpha=0.8)
    output_filename = analysis_name + 'fil_state' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament lower part
    fig_lowfil, ax1_lowfil = plt.subplots()
    color = 'tab:red'
    average_filament_height, sd_height = np.mean(fil_height), np.std(fil_height)
    ax1_lowfil.set_xlabel('Timestep (ps)')
    ax1_lowfil.set_ylabel('Filament length-lower end (A)', color=color)
    ax1_lowfil.scatter(time_switch, fil_height, alpha=alph, linewidth=ln, s=mrkr, color=color, label=f'average_filament_height = {average_filament_height: .2f} +/- {sd_height: .2f}')
    ax1_lowfil.tick_params(axis='y', labelcolor=color)
    ax1_lowfil.set_ylim(3, 25)
    plt.legend(loc='upper right', framealpha=0.75)

    ax2_lowfil = ax1_lowfil.twinx()
    color = 'tab:blue'
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    ax2_lowfil.set_ylabel('# of vacancies in filament-lower end (A.U.)', color=color)
    ax2_lowfil.scatter(time_switch, fil_size_down, alpha=alph, linewidth=ln, s=mrkr, marker='^', color=color, label=f'average # of vacancies in filament (bottom half) = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax2_lowfil.tick_params(axis='y', labelcolor=color)
    ax2_lowfil.set_ylim(0, 350)

    plt.title('Filament lower part near cathode')
    fig_lowfil.tight_layout()
    plt.legend(loc='lower right', framealpha=0.75)
    output_filename = analysis_name + 'fil_lower' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament upper part
    fig_upfil, ax1_upfil = plt.subplots()
    color = 'tab:red'
    average_filament_depth, sd_depth = np.mean(fil_depth), np.std(fil_depth)
    ax1_upfil.set_xlabel('Timestep (ps)')
    ax1_upfil.set_ylabel('Filament length-upper end (A)', color=color)
    ax1_upfil.scatter(time_switch, fil_depth, alpha=alph, linewidth=ln, s=mrkr, color=color, label=f'average_filament_depth = {average_filament_depth: .2f} +/- {sd_depth}')
    ax1_upfil.tick_params(axis='y', labelcolor=color)
    plt.legend(loc='upper right', framealpha=0.75)

    ax2_upfil = ax1_upfil.twinx()
    color = 'tab:blue'
    average_filament_size_up, sd_size_up = np.mean(fil_size_up), np.std(fil_size_up)
    ax2_upfil.set_ylabel('# of vacancies in filament-upper end (A.U.)', color=color)
    ax2_upfil.scatter(time_switch, fil_size_up, alpha=alph, linewidth=ln, s=mrkr, marker='^', color=color, label=f'average # of vacancies in filament (top half) = {average_filament_size_up: .2f} +/- {sd_size_up: .2f}')
    ax2_upfil.tick_params(axis='y', labelcolor=color)

    plt.title('Filament upper part near anode')
    fig_upfil.tight_layout()
    plt.legend(loc='lower right', framealpha=0.75)
    output_filename = analysis_name + 'upper' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament height (lower end)
    average_filament_height, sd_height = np.mean(fil_height), np.std(fil_height)
    fig_height, ax_height = plt.subplots()
    ax_height.scatter(time_switch, fil_height, alpha=alph, linewidth=ln, s=mrkr, label=f'average_filament_height = {average_filament_height: .2f} +/- {sd_height: .2f}')
    ax_height.set_xlabel('Timestep (ps)')
    ax_height.set_ylabel('Filament length-lower end (A)')
    ax_height.set_title('Filament length-lower end')
    ax_height.legend()
    ax_height.set_ylim(3, 25)
    output_filename = analysis_name + 'fil_height' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament depth (upper end)
    average_filament_depth, sd_depth = np.mean(fil_depth), np.std(fil_depth)
    fig_depth, ax_depth = plt.subplots()
    ax_depth.scatter(time_switch, fil_depth, alpha=alph, linewidth=ln, s=mrkr, label=f'average_filament_depth = {average_filament_depth: .2f} +/- {sd_depth}')
    ax_depth.set_xlabel('Timestep (ps)')
    ax_depth.set_ylabel('Filament length-upper end (A)')
    ax_depth.set_title('Filament length-upper end')
    ax_depth.legend()
    output_filename = analysis_name + 'fil_depth' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament size (upper end)
    average_filament_size_up, sd_size_up = np.mean(fil_size_up), np.std(fil_size_up)
    fig_size_up, ax_size_up = plt.subplots()
    ax_size_up.scatter(time_switch, fil_size_up, alpha=alph, linewidth=ln, s=mrkr, label=f'average # of vacancies in filament (top half) = {average_filament_size_up: .2f} +/- {sd_size_up: .2f}')
    ax_size_up.set_xlabel('Timestep (ps)')
    ax_size_up.set_ylabel('# of vacancies in filament-upper end (A.U.)')
    ax_size_up.set_title('# of vacancies in filament-upper end')
    ax_size_up.legend()
    output_filename = analysis_name + 'fil_size_up' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Plot filament size (lower end)
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    fig_size_down, ax_size_down = plt.subplots()
    ax_size_down.scatter(time_switch, fil_size_down, alpha=alph, linewidth=ln, s=mrkr, label=f'average # of vacancies in filament (bottom half) = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax_size_down.set_xlabel('Timestep (ps)')
    ax_size_down.set_ylabel('# of vacancies in filament-lower end (A.U.)')
    ax_size_down.set_title('# of vacancies in filament-lower end (A.U.)')
    ax_size_down.set_ylim(0, 350)
    ax_size_down.legend()
    output_filename = analysis_name + 'fil_size_down' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    # Return all figure objects for further use if needed
    return {
        "connection": fig_conn,
        "gap": fig_gap,
        "separation": fig_sep,
        "filament_gap_and_size": fig_size_gap,
        "filament_lower_part": fig_lowfil,
        "filament_upper_part": fig_upfil,
        "filament_height": fig_height,
        "filament_depth": fig_depth,
        "filament_size_up": fig_size_up,
        "filament_size_down": fig_size_down
    }



def plot_displacement_timeseries(
    file_list: list[str],
    datatype: str,
    dataindex: int,
    Nchunks: int,
    loop_start: int,
    loop_end: int,
    output_dir: str = os.getcwd(),
    timeseries_config: Optional['TimeSeriesConfig'] = None,
    plot_config: Optional['PlotConfig'] = None
) -> dict[str, plt.Figure]:
    """
    Create time series plots showing displacement data across spatial bins and files.
    
    This function generates a grid of subplots where each row represents a spatial bin
    and each column represents a different file/element. The configuration parameters
    control the appearance and layout of the plots.

    Parameters
    ----------
    file_list : list of str
        List of file paths to thermodynamic output data files.
    datatype : str
        Type of data being plotted (used in labels and filenames).
    dataindex : int
        Index of the data type to plot (4th dimension index).
        Corresponds to: 0='abs total disp', 1='density - mass', 2='temp (K)',
        3='z disp (A)', 4='lateral disp (A)', 5='outward disp vector (A)'.
    Nchunks : int
        Number of spatial bins/chunks along z-axis.
    loop_start : int
        Starting loop index (inclusive).
    loop_end : int
        Ending loop index (inclusive).
    output_dir : str, optional
        Directory to save output figures. Defaults to current working directory.
    timeseries_config : TimeSeriesConfig, optional
        Configuration for time series plotting parameters. Uses defaults if None.
    plot_config : PlotConfig, optional
        Configuration for plot appearance and styling. Uses defaults if None.

    Returns
    -------
    dict
        Dictionary containing the figure object with key "displacement_timeseries".

    Raises
    ------
    FileNotFoundError
        If any file in file_list does not exist.
    ValueError
        If dataindex is out of valid range or data is malformed.
    """    # Initialize configurations with defaults if not provided
    if timeseries_config is None:
        timeseries_config = DEFAULT_TIMESERIES_CONFIG
    if plot_config is None:
        plot_config = DEFAULT_PLOT_CONFIG
    
    # Validate input parameters
    from ..config import validate_dataindex, validate_file_list, validate_loop_parameters, validate_chunks_parameter
    
    validate_file_list(file_list)
    validate_dataindex(dataindex)
    validate_loop_parameters(loop_start, loop_end)
    validate_chunks_parameter(Nchunks)
    
    # Process displacement data using extracted function
    all_thermo_data, element_labels, dump_steps = process_displacement_timeseries_data(
        file_list, loop_start, loop_end, timeseries_config.time_points, read_displacement_data
    )
    
    print(file_list)
    print(element_labels)
    print(np.shape(all_thermo_data))
    print('dump_steps=', dump_steps)

    # Setup subplot grid using configuration
    nrows = Nchunks
    ncolumns = timeseries_config.ncolumns
    figsize = timeseries_config.calculate_figsize(nrows)
    
    # Create the time series plot using extracted function
    fig = plot_timeseries_grid(
        data=all_thermo_data,
        x_values=dump_steps,
        element_labels=element_labels,
        datatype=datatype,
        data_labels=timeseries_config.data_labels,
        dataindex=dataindex,
        nrows=nrows,
        ncolumns=ncolumns,
        figsize=figsize,
        config=plot_config
    )

    # Generate output filename and save using extracted function
    output_filename = f"{datatype}-{timeseries_config.data_labels[dataindex]}"
    save_figure(
        fig=fig,
        output_dir=output_dir,
        filename=output_filename,
        config=plot_config,
        close_after_save=True
    )
    return {
        "displacement_timeseries": fig,
    }


def main():
    output_dir =  os.path.join("..", "..", "output", "ecellmodel")

    global COLUMNS_TO_READ
    COLUMNS_TO_READ = (0,1,2,3,4,5,9,10,11,12,13,14,15,16) 

    ## The following code block generates plots that track the evolution of the 
    # filament connectivity state, gap, and separation over time for each 
    # timeseries trajectory file in the file_list.

    ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.001
    DUMP_INTERVAL_STEPS = 500

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 500000
    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
    print(np.shape(time_points),'\n',time_points[-1])
    ###################################

    data_path = os.path.join("..", "..", "data","ecellmodel", "processed","trajectory_series", "*.lammpstrj")
    analysis_name = 'track_'
    # data_path = "*.lammpstrj"
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print('The data_path is ',data_path)
    print(analysis_name,file_list)

    track_filament_evolution(file_list, analysis_name,TIME_STEP,DUMP_INTERVAL_STEPS,output_dir=output_dir)

    ## The following code block generates plots of atomic distributions
    # and compares the displacements of Hf, O, and Ta for different temperatures

    ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.0002
    DUMP_INTERVAL_STEPS = 5000

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 2500000
    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
    print(np.shape(time_points),'\n',time_points[-1])
    SKIP_ROWS_COORD= 9   
    HISTOGRAM_BINS = 15
    ###################################

    analysis_name = f'temp_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "temp*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['300 K','900 K', '1300 K']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "*K_Hfmobilestc1.dat")
    analysis_name = f'displacements_temp_Hf'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['300 K','900 K', '1300 K']
    plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "*K_Omobilestc1.dat")
    analysis_name = f'displacements_temp_O'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['300 K','900 K', '1300 K']
    plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "*K_Tamobilestc1.dat")
    analysis_name = f'displacements_temp_Ta'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['300 K','900 K', '1300 K']
    plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

    
    ## The following code block generates plots of atomic and charge distributions 
    # and compares the displacements of Hf, O, and Ta for different temperatures   
        ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.0002
    DUMP_INTERVAL_STEPS = 5000

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 2500000
    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
    print(np.shape(time_points),'\n',time_points[-1])
    SKIP_ROWS_COORD= 9   
    HISTOGRAM_BINS = 15
    ###################################

    analysis_name = f'local_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "local2*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['initial','final']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    data_path = os.path.join("..", "..", "data","ecellmodel", "raw", "[1-9][A-Z][A-Za-z]mobilestc1.dat")
    analysis_name = f'displacements_atom_type'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['Hf','O', 'Ta']
    plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)


    analysis_name = f'local_charge_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..", "data","ecellmodel", "raw", "local2*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['initial','final']
    plot_atomic_charge_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    
    analysis_name = f'local_charge_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "local2*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['initial','final']
    plot_atomic_charge_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    data_path = os.path.join("..", "..","data","ecellmodel", "raw", "[1-9][A-Z][A-Za-z]mobilestc1.dat")
    analysis_name = f'displacements_atom_type'
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['Hf','O', 'Ta']
    plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=output_dir)

    ## The following code block generates plots of atomic and charge distributions 
    # and compares the displacements of Hf, O, and Ta for different temperatures   
        ## Simulation parameters corresponding to the respective raw data
    TIME_STEP = 0.001
    DUMP_INTERVAL_STEPS = 500

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 500000

    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
    print(np.shape(time_points),'\n',time_points[-1])
    SKIP_ROWS_COORD= 9   
    HISTOGRAM_BINS = 15
    ###################################

    analysis_name = f'forming_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]forming*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['relaxed','formed']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    #####################################
    # Following code block corresponds to analysis on new data files
    #####################################
    #####################################

    ## The following code block generates plots of atomic and charge distributions 
# and compares the displacements of Hf, O, and Ta for different temperatures  
#  
## Simulation parameters corresponding to the respective raw data
###################################

    TIME_STEP = 0.001
    DUMP_INTERVAL_STEPS = 500

    MIN_SIM_STEP = 0
    MAX_SIM_STEP = 500000

    loop_start = int(MIN_SIM_STEP / DUMP_INTERVAL_STEPS)
    loop_end = int(MAX_SIM_STEP / DUMP_INTERVAL_STEPS)

    time_points = np.linspace(loop_start*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end*DUMP_INTERVAL_STEPS*TIME_STEP,loop_end-loop_start+1)
    print(np.shape(time_points),'\n',time_points[-1])
    SKIP_ROWS_COORD= 9   
    HISTOGRAM_BINS = 15
    ###################################

    analysis_name = f'forming_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]forming*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['relaxed0V','formed2V']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    analysis_name = f'post_forming_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]formed*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['formed2V','formed0V']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    analysis_name = f'set_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]set*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['formed0V','set-0.1V']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    analysis_name = f'break_{HISTOGRAM_BINS}'
    data_path =  os.path.join("..", "..","data","ecellmodel", "raw", "[1-9]break*.lammpstrj")
    unsorted_file_list = glob.glob(data_path)
    file_list = sorted(unsorted_file_list)
    print(analysis_name,file_list)
    labels = ['set-0.1V','break-0.5V']
    plot_atomic_distribution(file_list,labels,SKIP_ROWS_COORD,HISTOGRAM_BINS,analysis_name,output_dir=output_dir)

    
    exit()

if __name__ == "__main__":
    main()



