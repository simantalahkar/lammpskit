import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ovito.io import import_file
import ovito.modifiers as om
# Remove unused typing import since config classes are inlined

# Import general utilities
from ..io import read_coordinates
from ..plotting import plot_multiple_cases

# Import configuration settings (simplified - inline values directly)

# Import ecellmodel-specific functions (inlined to reduce dependencies)

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

COLUMNS_TO_READ = (0,1,2,3,4,5,9,10,11,12) #,13,14,15,16

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

# =========================
# Plotting Functions  
# =========================

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
    from .data_processing import calculate_atomic_distributions, calculate_z_bins_setup
    
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
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
    from .data_processing import calculate_atomic_distributions, calculate_charge_distributions, calculate_z_bins_setup
    
    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
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
    time_points = 100  # Number of points for time series analysis
    
    # Plot configuration defaults (inlined)
    colors = ['b', 'r', 'g', 'k']
    linewidth = 1.2
    alpha = 0.75
    title_fontsize = 12  # Increased from 8
    label_fontsize = 10  # Increased from 8 
    tick_fontsize = 8    # Increased from 6
    legend_fontsize = 7.5  # Smaller legend font size
    grid = True
    
    # Validate input parameters (inlined from config.py)
    if not isinstance(file_list, list):
        raise ValueError("file_list must be a list")
    if not file_list:
        raise ValueError("file_list cannot be empty")
    
    # Check if files exist
    for filepath in file_list:
        if not isinstance(filepath, str):
            raise ValueError("All items in file_list must be strings")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
    
    # Validate dataindex
    if not isinstance(dataindex, int):
        raise ValueError("dataindex must be an integer")
    
    # Validate loop parameters  
    if not isinstance(loop_start, int) or not isinstance(loop_end, int):
        raise ValueError("loop_start and loop_end must be integers")
    if loop_start < 0:
        raise ValueError("loop_start must be non-negative")
    if loop_end < 0:
        raise ValueError("loop_end must be non-negative")
    if loop_start > loop_end:
        raise ValueError("loop_start must be less than or equal to loop_end")
    
    # Validate chunks parameter
    if not isinstance(Nchunks, int):
        raise ValueError("Nchunks must be an integer")
    if Nchunks < 1 or Nchunks > 1000:
        raise ValueError("Nchunks must be between 1 and 1000")
    
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



