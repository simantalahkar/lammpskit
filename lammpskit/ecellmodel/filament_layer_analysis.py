import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ovito.io import import_file
import ovito.modifiers as om

# Import general utilities
from ..io import read_coordinates
from ..plotting import plot_multiple_cases

# Import centralized plotting functions
from ..plotting import (
    create_time_series_plot,
    create_dual_axis_plot,
    TimeSeriesPlotConfig,
    DualAxisPlotConfig,
    save_and_close_figure,
    calculate_mean_std_label,
    calculate_frequency_label
)

# Import validation functions from config
from ..config import (
    validate_file_list,
    validate_dataindex, 
    validate_loop_parameters,
    validate_chunks_parameter,
    validate_filepath,
    validate_cluster_parameters,
    DEFAULT_COLUMNS_TO_READ
)

# Import configuration settings (simplified - inline values directly)

# Import ecellmodel-specific functions (inlined to reduce dependencies)

# Set global matplotlib parameters for consistent styling across all plots
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
# Data Reading Functions
# =========================

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
    # Validate input parameters using centralized functions
    validate_filepath(filepath)
    validate_loop_parameters(loop_start, loop_end)
    
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

# =========================
# Plotting  and Auxiliary Analysis Functions
# =========================

def plot_atomic_distribution(
    file_list: list[str],
    labels: list[str],
    skip_rows: int,
    z_bins: int,
    analysis_name: str,
    output_dir: str = os.getcwd(),
    columns_to_read: tuple = None,
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
    columns_to_read : tuple, optional
        Columns to read from the coordinate files. If None, uses DEFAULT_COLUMNS_TO_READ.
    **kwargs
        Additional keyword arguments for customizing the plots.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.
    """
    from .data_processing import calculate_atomic_distributions, calculate_z_bins_setup
    
    # Use default columns if not specified
    if columns_to_read is None:
        columns_to_read = DEFAULT_COLUMNS_TO_READ
    
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, columns_to_read)
    z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)

    # Calculate atomic distributions using modular function
    distributions = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)
    
    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")

    # Avoid division by zero for stoichiometry calculations
    total_distribution_divide = distributions['total'].copy()
    total_distribution_divide[total_distribution_divide == 0] = 1

    # Calculate stoichiometry for last and first trajectory
    O_stoich = 3.5 * distributions['oxygen'][-1] / total_distribution_divide[-1]
    Ta_stoich = 3.5 * distributions['tantalum'][-1] / total_distribution_divide[-1]
    Hf_stoich = 3.5 * distributions['hafnium'][-1] / total_distribution_divide[-1]
    stoichiometry = np.array([Hf_stoich, O_stoich, Ta_stoich])
    proportion_labels = np.array(['a (of Hf$_a$)', 'b (of O$_b$)', 'c (of Ta$_c$)'])

    O_stoich_in = 3.5 * distributions['oxygen'][0] / total_distribution_divide[0]
    Ta_stoich_in = 3.5 * distributions['tantalum'][0] / total_distribution_divide[0]
    Hf_stoich_in = 3.5 * distributions['hafnium'][0] / total_distribution_divide[0]
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
    fig_metal = plot_multiple_cases(distributions['metal'], z_bin_centers, labels, 'Metal atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot Hf atoms
    output_filename = f"{analysis_name}_Hf" + ''.join(f"_{i}" for i in labels)
    fig_hf = plot_multiple_cases(distributions['hafnium'], z_bin_centers, labels, 'Hf atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot Ta atoms
    output_filename = f"{analysis_name}_Ta" + ''.join(f"_{i}" for i in labels)
    fig_ta = plot_multiple_cases(distributions['tantalum'], z_bin_centers, labels, 'Ta atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

    # Plot O atoms
    output_filename = f"{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(distributions['oxygen'], z_bin_centers, labels, 'O atoms #', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)

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
    columns_to_read: tuple = None,
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
    columns_to_read : tuple, optional
        Columns to read from the coordinate files. If None, uses DEFAULT_COLUMNS_TO_READ.
    **kwargs
        Additional keyword arguments for customizing the plots.

    Returns
    -------
    dict
        Dictionary of figure objects for each plot type.
    """
    from .data_processing import calculate_atomic_distributions, calculate_charge_distributions, calculate_z_bins_setup
    
    # Use default columns if not specified
    if columns_to_read is None:
        columns_to_read = DEFAULT_COLUMNS_TO_READ
    
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, columns_to_read)
    z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)

    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")

    # Calculate atomic distributions (needed for charge distribution normalization)
    atomic_distributions = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)
    
    # Calculate charge distributions using modular function
    charge_distributions = calculate_charge_distributions(coordinates_arr, z_bins, zlo, zhi, atomic_distributions)

    figure_size = [2.5, 5]

    # Plot net charge
    output_filename = f"{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net = plot_multiple_cases(charge_distributions['total_charge'], z_bin_centers, labels, 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0)

    # Plot metal mean charge
    output_filename = f"{analysis_name}_M" + ''.join(f"_{i}" for i in labels)
    fig_metal = plot_multiple_cases(charge_distributions['metal_mean_charge'], z_bin_centers, labels, 'Metal atoms mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimitlo=0.7, xlimithi=1.2)

    # Plot oxygen mean charge
    output_filename = f"{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(charge_distributions['oxygen_mean_charge'], z_bin_centers, labels, 'O mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=0, xlimitlo=-0.7)

    # Plot final net charge
    output_filename = f"final_{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net_end = plot_multiple_cases(charge_distributions['total_charge'][-1], z_bin_centers, labels[-1], 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0, markerindex=1)

    # Plot initial net charge
    output_filename = f"initial_{analysis_name}_all" + ''.join(f"_{i}" for i in labels)
    fig_net_start = plot_multiple_cases(charge_distributions['total_charge'][0], z_bin_centers, labels[0], 'Net charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=15, xlimitlo=-20, yaxis=0)

    # Plot initial metal mean charge
    output_filename = f"initial_{analysis_name}_M" + ''.join(f"_{i}" for i in labels)
    fig_metal_start = plot_multiple_cases(charge_distributions['metal_mean_charge'][0], z_bin_centers, labels[0], 'Metal atoms mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimitlo=0.7, xlimithi=1.2)

    # Plot initial oxygen mean charge
    output_filename = f"initial_{analysis_name}_O" + ''.join(f"_{i}" for i in labels)
    fig_o_start = plot_multiple_cases(charge_distributions['oxygen_mean_charge'][0], z_bin_centers, labels[0], 'O mean charge', 'z position (A)', output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi=70, xlimithi=0, xlimitlo=-0.7)

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
    TypeError
        If parameters have invalid types.
    """
    # Validate input parameters using centralized functions
    validate_filepath(filepath)
    validate_cluster_parameters(z_filament_lower_limit, z_filament_upper_limit, thickness)
    
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

    # Create plotting configuration matching original parameters
    timeseries_config = TimeSeriesPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker="^",
        include_line=True,
        include_scatter=True
    )

    # Plot filament connectivity state
    freq_label = calculate_frequency_label(
        connection, 1, 
        "filament is in connected state {frequency: .2f}% of the time"
    )
    fig_conn, ax_conn = create_time_series_plot(
        time_switch, connection,
        title="Filament connectivity state (1: connected, 0: broken)",
        xlabel="Time (ps)",
        ylabel="Filament connectivity state (1: connected, 0: broken)",
        stats_label=freq_label,
        config=timeseries_config
    )
    save_and_close_figure(fig_conn, output_dir, analysis_name + "OnOff")

    # Plot filament gap
    gap_label = calculate_mean_std_label(gap, "average_filament_gap")
    fig_gap, ax_gap = create_time_series_plot(
        time_switch, gap,
        title="Filament gap",
        xlabel="Time (ps)",
        ylabel="Filament gap (A)",
        stats_label=gap_label,
        config=timeseries_config
    )
    save_and_close_figure(fig_gap, output_dir, analysis_name + 'fil_gap')

    # Plot filament separation
    separation_label = calculate_mean_std_label(separation, "average_filament_separation")
    fig_sep, ax_sep = create_time_series_plot(
        time_switch, separation,
        title="Filament separation",
        xlabel="Time (ps)",
        ylabel="Filament separation (A)",
        stats_label=separation_label,
        config=timeseries_config,
        fontsize_legend=8  # Override legend font size for this specific plot
    )
    save_and_close_figure(fig_sep, output_dir, analysis_name + 'fil_separation')

    # Create dual-axis plotting configuration matching original parameters
    dual_config = DualAxisPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker='^',
        primary_color='tab:red',
        secondary_color='tab:blue',
        primary_legend_loc='upper right',
        secondary_legend_loc='lower right',
        legend_framealpha=0.8,
        tight_layout=True
    )

    # Plot filament gap & number of conductive atoms
    gap_label = calculate_mean_std_label(gap, "average_filament_gap")
    size_down_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament")
    fig_size_gap, ax1_size_gap, ax2_size_gap = create_dual_axis_plot(
        time_switch, gap, fil_size_down,
        title="Gap & no. of conductive atoms in Filament",
        xlabel="Time (ps)",
        primary_ylabel="Filament gap (A)",
        secondary_ylabel="# of vacancies in filament (A.U.)",
        primary_stats_label=gap_label,
        secondary_stats_label=size_down_label,
        config=dual_config,
        primary_ylim=(-0.5, 8.5),
        secondary_ylim=(0, 350)
    )
    save_and_close_figure(fig_size_gap, output_dir, analysis_name + 'fil_state')

    # Plot filament lower part
    lower_dual_config = DualAxisPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker='^',
        primary_color='tab:red',
        secondary_color='tab:blue',
        primary_legend_loc='upper right',
        secondary_legend_loc='lower right',
        legend_framealpha=0.75,
        tight_layout=True
    )
    
    height_label = calculate_mean_std_label(fil_height, "average_filament_height")
    size_down_lower_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament (bottom half)")
    fig_lowfil, ax1_lowfil, ax2_lowfil = create_dual_axis_plot(
        time_switch, fil_height, fil_size_down,
        title="Filament lower part near cathode",
        xlabel="Timestep (ps)",
        primary_ylabel="Filament length-lower end (A)",
        secondary_ylabel="# of vacancies in filament-lower end (A.U.)",
        primary_stats_label=height_label,
        secondary_stats_label=size_down_lower_label,
        config=lower_dual_config,
        primary_ylim=(3, 25),
        secondary_ylim=(0, 350)
    )
    save_and_close_figure(fig_lowfil, output_dir, analysis_name + 'fil_lower')

    # Plot filament upper part
    depth_label = calculate_mean_std_label(fil_depth, "average_filament_depth")
    size_up_label = calculate_mean_std_label(fil_size_up, "average # of vacancies in filament (top half)")
    fig_upfil, ax1_upfil, ax2_upfil = create_dual_axis_plot(
        time_switch, fil_depth, fil_size_up,
        title="Filament upper part near anode",
        xlabel="Timestep (ps)",
        primary_ylabel="Filament length-upper end (A)",
        secondary_ylabel="# of vacancies in filament-upper end (A.U.)",
        primary_stats_label=depth_label,
        secondary_stats_label=size_up_label,
        config=lower_dual_config  # Use same config as lower
    )
    save_and_close_figure(fig_upfil, output_dir, analysis_name + 'upper')

    # Create scatter-only configuration for simple plots (no line)
    scatter_config = TimeSeriesPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker='^',
        include_line=False,  # Scatter only
        include_scatter=True
    )

    # Plot filament height (lower end)
    height_simple_label = calculate_mean_std_label(fil_height, "average_filament_height")
    fig_height, ax_height = create_time_series_plot(
        time_switch, fil_height,
        title="Filament length-lower end",
        xlabel="Timestep (ps)",
        ylabel="Filament length-lower end (A)",
        stats_label=height_simple_label,
        config=scatter_config,
        ylim=(3, 25)
    )
    save_and_close_figure(fig_height, output_dir, analysis_name + 'fil_height')

    # Plot filament depth (upper end)
    depth_simple_label = calculate_mean_std_label(fil_depth, "average_filament_depth")
    fig_depth, ax_depth = create_time_series_plot(
        time_switch, fil_depth,
        title="Filament length-upper end",
        xlabel="Timestep (ps)",
        ylabel="Filament length-upper end (A)",
        stats_label=depth_simple_label,
        config=scatter_config
    )
    save_and_close_figure(fig_depth, output_dir, analysis_name + 'fil_depth')

    # Plot filament size (upper end)
    size_up_simple_label = calculate_mean_std_label(fil_size_up, "average # of vacancies in filament (top half)")
    fig_size_up, ax_size_up = create_time_series_plot(
        time_switch, fil_size_up,
        title="# of vacancies in filament-upper end",
        xlabel="Timestep (ps)",
        ylabel="# of vacancies in filament-upper end (A.U.)",
        stats_label=size_up_simple_label,
        config=scatter_config
    )
    save_and_close_figure(fig_size_up, output_dir, analysis_name + 'fil_size_up')

    # Plot filament size (lower end)
    size_down_simple_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament (bottom half)")
    fig_size_down, ax_size_down = create_time_series_plot(
        time_switch, fil_size_down,
        title="# of vacancies in filament-lower end (A.U.)",
        xlabel="Timestep (ps)",
        ylabel="# of vacancies in filament-lower end (A.U.)",
        stats_label=size_down_simple_label,
        config=scatter_config,
        ylim=(0, 350)
    )
    save_and_close_figure(fig_size_down, output_dir, analysis_name + 'fil_size_down')

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
    output_dir: str = os.getcwd()
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
    """    # Initialize configurations with inlined defaults
    # Timeseries configuration defaults (inlined)
    data_labels = [
        'abs total disp', 'density - mass', 'temp (K)', 
        'z disp (A)', 'lateral disp (A)', 'outward disp vector (A)'
    ]
    ncolumns = 4  # Number of columns for subplot grid
    
    # Plot configuration defaults (inlined)
    colors = ['b', 'r', 'g', 'k']
    linewidth = 1.2
    alpha = 0.75
    title_fontsize = 12  # Increased from 8
    label_fontsize = 10  # Increased from 8 
    tick_fontsize = 8    # Increased from 6
    legend_fontsize = 7.5  # Smaller legend font size
    grid = True
    
    # Validate input parameters using centralized functions
    validate_file_list(file_list)
    validate_dataindex(dataindex)
    validate_loop_parameters(loop_start, loop_end)
    validate_chunks_parameter(Nchunks, min_chunks=1, max_chunks=100)
    
    # Process displacement data (inlined from plotting.py)
    from .data_processing import extract_element_label_from_filename
    
    all_thermo_data = []
    element_labels = []
    
    for filename in file_list:
        # Extract element label from filename
        element_label = extract_element_label_from_filename(filename)
        element_labels.append(element_label)
        
        # Read displacement data with error handling
        try:
            thermo_data = read_displacement_data(filename, loop_start, loop_end)
            all_thermo_data.append(thermo_data)
        except Exception as e:
            raise ValueError(f"Failed to process file {filename}: {str(e)}")
    
    # Create dump steps array
    dump_steps = np.arange(loop_start, loop_end + 1)
    
    print(file_list)
    print(element_labels)
    print(np.shape(all_thermo_data))
    print('dump_steps=', dump_steps)

    # Setup subplot grid using inlined configuration
    nrows = Nchunks
    # ncolumns = 4 (already defined above)
    figsize = (ncolumns * 3.0, nrows * 0.65)  # Calculate figsize with original multipliers
    
    # Create the time series plot (inlined from plotting.py)
    plt.ioff()
    fig, axes = plt.subplots(nrows, ncolumns, figsize=figsize, squeeze=False)
    
    # Convert data to numpy array for easier indexing
    data_array = np.array(all_thermo_data)
    
    # Plot data for each subplot
    for row in range(nrows):
        for col in range(min(ncolumns, len(element_labels))):
            ax = axes[row, col]
            
            # Extract data for this bin and element
            if row < data_array.shape[1] and col < data_array.shape[0]:
                # Reverse the data order: bottom row gets highest data index, top row gets lowest
                data_row_index = nrows - 1 - row
                y_data = data_array[col, :, data_row_index, dataindex]
                
                # Create legend label with 1-based chunk numbering (bottom=1, top=highest)
                chunk_id = data_row_index + 1
                legend_label = f"{element_labels[col]} in Chunk {chunk_id}"
                ax.plot(dump_steps, y_data, 
                       linewidth=linewidth,
                       alpha=alpha,
                       color=colors[col % len(colors)],
                       label=legend_label)
                
                # Configure subplot appearance
                if row == 0:  # Top row gets column titles
                    ax.set_title(f"{element_labels[col]}", fontsize=title_fontsize)
                
                if row == nrows - 1:  # Bottom row gets x-label
                    ax.set_xlabel("Time step", fontsize=label_fontsize)
                
                # Set minimal tick labels for scale reference
                ax.tick_params(labelsize=tick_fontsize)
                
                # Add legend to all subplots
                ax.legend(fontsize=legend_fontsize, loc='best')
                
                # Add grid if enabled
                if grid:
                    ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for col in range(len(element_labels), ncolumns):
        for row in range(nrows):
            fig.delaxes(axes[row, col])
    
    # Add shared y-label for the leftmost column
    shared_ylabel = f"{datatype} {data_labels[dataindex]}"
    fig.text(0.025, 0.5, shared_ylabel, fontsize=label_fontsize, 
             rotation=90, va='center', ha='center')
    
    # Set overall title
    fig.suptitle(f'{datatype} {data_labels[dataindex]}', fontsize=title_fontsize)
    
    # Adjust layout to create continuous/joined subplots vertically
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.05, hspace=0)  # hspace=0 creates continuous vertical layout

    # Generate output filename and save (inlined save_figure)
    output_filename = f"{datatype}-{data_labels[dataindex]}"
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in SVG format (simplified)
    filename = f"{output_filename}.svg"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='svg')
    
    plt.close(fig)
    return {
        "displacement_timeseries": fig,
    }


def main():
    """
    Minimal demonstration of LAMMPSKit basic functionality.
    
    For complete examples with real data, see:
    https://github.com/simantalahkar/lammpskit/tree/main/usage/ecellmodel
    """
    try:
        from ... import __version__
    except ImportError:
        # Fallback if relative import fails
        import lammpskit
        __version__ = lammpskit.__version__
    
    print("LAMMPSKit - Toolkit for MD simulations and analysis with LAMMPS")
    print(f"Version: {__version__}")
    print("")
    print("Example usage:")
    print("  from lammpskit.ecellmodel import plot_atomic_distribution")
    print("  from lammpskit.config import DEFAULT_COLUMNS_TO_READ")
    print("")
    print("For complete examples, visit:")
    print("  https://github.com/simantalahkar/lammpskit/tree/main/usage")

if __name__ == "__main__":
    main()



