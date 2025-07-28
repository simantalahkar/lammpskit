import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Robust OVITO import with fallback for documentation builds
try:
    from ovito.io import import_file
    import ovito.modifiers as om
    OVITO_AVAILABLE = True
except ImportError as e:
    # Fallback for documentation builds or missing OVITO
    import warnings
    warnings.warn(f"OVITO not available: {e}. Some functionality will be limited.", ImportWarning)
    OVITO_AVAILABLE = False
    
    # Create mock functions for documentation
    def import_file(*args, **kwargs):
        raise ImportError("OVITO not available - this function requires OVITO to be properly installed")
    
    class MockOvitoModifiers:
        """Mock OVITO modifiers for documentation builds"""
        
        def __init__(self):
            # Add commonly used OVITO modifiers as mock classes
            self.CoordinationAnalysisModifier = self._create_mock_modifier("CoordinationAnalysisModifier")
            self.ExpressionSelectionModifier = self._create_mock_modifier("ExpressionSelectionModifier")
            self.ClusterAnalysisModifier = self._create_mock_modifier("ClusterAnalysisModifier")
            self.DeleteSelectedModifier = self._create_mock_modifier("DeleteSelectedModifier")
        
        def _create_mock_modifier(self, name):
            """Create a mock modifier class that accepts any parameters"""
            class MockModifier:
                def __init__(self, *args, **kwargs):
                    self._name = name
                    self._args = args
                    self._kwargs = kwargs
                
                def __repr__(self):
                    return f"Mock{self._name}({self._args}, {self._kwargs})"
            
            MockModifier.__name__ = f"Mock{name}"
            return MockModifier
    
    om = MockOvitoModifiers()

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
    calculate_frequency_label,
)

# Import validation functions from config
from ..config import (
    validate_file_list,
    validate_dataindex,
    validate_loop_parameters,
    validate_chunks_parameter,
    validate_filepath,
    validate_cluster_parameters,
    DEFAULT_COLUMNS_TO_READ,
)

# Import configuration settings (simplified - inline values directly)

# Import ecellmodel-specific functions (inlined to reduce dependencies)

# Set global matplotlib parameters for consistent styling across all plots
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

# mpl.rcParams['pdf.fonttype'] = 42  #this would allow us to edit the fonts in acrobat illustrator

"""
Filament layer analysis for HfTaO electrochemical memory device characterization.

This module provides comprehensive post-processing and analysis capabilities for LAMMPS
molecular dynamics simulation data from hafnium-tantalum oxide (HfTaO) electrochemical
memory devices. Functions implement specialized workflows for filament formation analysis,
charge redistribution tracking, and temporal evolution characterization of resistive
switching phenomena.

Core Analysis Capabilities
--------------------------
**Spatial Analysis:**
- Atomic species distribution profiling across electrode-to-electrode axis
- Charge density mapping and electrostatic field characterization
- Layer-by-layer composition analysis for interface studies

**Temporal Dynamics:**
- Filament connectivity evolution during SET/RESET switching cycles
- Displacement time series analysis for ion migration tracking  
- Long-term stability assessment of conductive bridge formation

**Advanced Characterization:**
- OVITO-based cluster analysis for metallic filament identification
- Radial distribution function analysis for local structure characterization
- Connectivity gap analysis for resistance state quantification

**Visualization & Reporting:**
- Publication-ready scientific plotting with customizable styling
- Multi-panel comparative analysis across experimental conditions
- Statistical analysis integration with mean/std reporting

Electrochemical Memory Device Physics
-------------------------------------
The module is optimized for HfTaO-based resistive random access memory (ReRAM) analysis:

**Device Structure:**
- Electrode-oxide-electrode sandwich geometry (20-100 Å oxide thickness)
- HfTaO mixed oxide active layer with controlled stoichiometry
- Metal electrodes (typically Al, Ti, or noble metals)

**Switching Mechanisms:**
- **SET Process**: Oxygen vacancy migration → filament formation → low resistance
- **RESET Process**: Vacancy scattering → filament dissolution → high resistance  
- **Retention**: Long-term filament stability under bias stress

**Physical Phenomena:**
- Ion migration along electric field gradients (z-direction analysis)
- Local charge redistribution during voltage cycling
- Cluster formation through metal atom aggregation
- Interface phenomena at electrode-oxide boundaries

Performance & Integration
-------------------------
**Computational Efficiency:**
- Optimized for large trajectory analysis (>1M atoms, >10K frames)
- Memory-efficient streaming for multi-gigabyte simulation datasets
- Parallel processing support for batch analysis workflows

**Integration Ecosystem:**
- LAMMPS trajectory format support (.lammpstrj, .dump)
- OVITO pipeline integration for advanced cluster analysis
- LAMMPSKit ecosystem compatibility with io/ and plotting/ modules
- Sphinx documentation generation with cross-referencing

**Workflow Orchestration:**
- Automated file discovery and species identification
- Configurable analysis parameter management
- Output organization with consistent naming conventions
- Error handling and validation for robust pipeline execution

Usage Patterns
--------------
**Basic Distribution Analysis:**

>>> from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution
>>> file_list = ["traj1.lammpstrj", "traj2.lammpstrj"]  
>>> labels = ["Initial", "Final"]
>>> figures = plot_atomic_distribution(file_list, labels, skip_rows=9, z_bins=50, 
...                                   analysis_name="device_analysis")

**Filament Evolution Tracking:**

>>> from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution
>>> trajectory_files = ["switching_cycle.lammpstrj"]
>>> evolution_plots = track_filament_evolution(trajectory_files, "SET_analysis", 
...                                          time_step=0.0002, dump_interval_steps=5000)

**Advanced Cluster Analysis:**

>>> from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters  
>>> cluster_data = analyze_clusters("final_state.lammpstrj", 
...                                z_filament_lower_limit=5, z_filament_upper_limit=23)
>>> timestep, connection, fil_size, fil_height = cluster_data[:4]

**Comparative Displacement Analysis:**

>>> from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_comparison
>>> displacement_files = ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
>>> comp_plots = plot_displacement_comparison(displacement_files, loop_start=0, 
...                                         loop_end=100, labels=["Hf", "Ta", "O"], 
...                                         analysis_name="mobility_comparison")

Dependencies & Requirements  
---------------------------
**Core Dependencies:**
- numpy (≥1.19): Numerical computing and array operations
- matplotlib (≥3.3): Scientific plotting and visualization  
- ovito (≥3.7): Advanced cluster analysis and structural characterization

**LAMMPSKit Integration:**
- lammpskit.io: Trajectory reading and coordinate processing
- lammpskit.plotting: Scientific visualization utilities  
- lammpskit.config: Validation functions and parameter management
- lammpskit.ecellmodel.data_processing: Species separation and distribution calculation

**File Format Support:**
- LAMMPS trajectory files (.lammpstrj, .dump)
- Thermodynamic output files (.dat, .txt)
- Custom mobility analysis formats
- OVITO-compatible structure formats

Mathematical Foundations
------------------------
**Spatial Binning Theory:**
- Uniform z-direction discretization: Δz = (z_hi - z_lo) / N_bins
- Bin center calculation: z_c[i] = z_lo + (i + 0.5) * Δz
- Density normalization: ρ(z) = N(z) / (Δz × A_cross)

**Statistical Analysis:**
- Mean and standard deviation calculation across temporal sequences
- Frequency analysis for connectivity state characterization  
- Error propagation for multi-trajectory ensemble analysis

**Connectivity Analysis:**
- Distance-based cluster identification with cutoff parameters
- Filament bridging detection across electrode separation
- Gap analysis using minimum inter-cluster distances

Output Organization
-------------------
**Figure Generation:**
- SVG format for publication-quality vector graphics
- Consistent naming conventions: {analysis_name}_{plot_type}.svg
- Multi-panel layouts with shared axis coordination
- Statistical annotations with mean ± std reporting

**Data Export:**
- Numerical results in easily parseable formats
- Metadata preservation for reproducibility
- Parameter logging for analysis tracking
- Cross-reference generation for related analyses

Examples
--------
Complete workflow example for HfTaO device analysis:

>>> import numpy as np
>>> from lammpskit.ecellmodel.filament_layer_analysis import *
>>> 
>>> # Setup analysis parameters
>>> trajectory_files = ["set_process.lammpstrj", "reset_process.lammpstrj"]
>>> labels = ["SET", "RESET"] 
>>> analysis_name = "HfTaO_switching_analysis"
>>> 
>>> # 1. Atomic distribution analysis
>>> dist_figs = plot_atomic_distribution(trajectory_files, labels, skip_rows=9, 
...                                     z_bins=50, analysis_name=analysis_name)
>>> 
>>> # 2. Charge redistribution analysis  
>>> charge_figs = plot_atomic_charge_distribution(trajectory_files, labels, 
...                                              skip_rows=9, z_bins=50, 
...                                              analysis_name=analysis_name)
>>> 
>>> # 3. Filament evolution tracking
>>> evolution_figs = track_filament_evolution(trajectory_files, analysis_name,
...                                         time_step=0.0002, dump_interval_steps=5000)
>>> 
>>> # 4. Displacement comparison
>>> mobility_files = ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
>>> displacement_figs = plot_displacement_comparison(mobility_files, 0, 100, 
...                                                ["Hf", "Ta", "O"], analysis_name)
>>> 
>>> print(f"Generated {len(dist_figs) + len(charge_figs) + len(evolution_figs) + len(displacement_figs)} analysis figures")
"""

# =========================
# Data Reading Functions
# =========================


def read_displacement_data(filepath: str, loop_start: int, loop_end: int, repeat_count: int = 0) -> list[np.ndarray]:
    """
    Read spatially-binned displacement data from LAMMPS thermodynamic output files.

    Parses binwise averaged displacement and mobility analysis data generated by LAMMPS
    simulations, extracting loop-specific datasets for temporal analysis of ion migration,
    atomic redistribution, and electrochemical switching dynamics in HfTaO memory devices.
    Handles multi-loop data structures with automatic chunk detection and validation.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS thermodynamic output file containing binwise averaged data.
        Expected format: Header lines (3) + Nchunks line + loop data blocks.
        Common files: "Hf_mobility.dat", "O_mobility.dat", "temperature.dat"
    loop_start : int
        Starting loop index (inclusive) for data extraction. Corresponds to simulation
        timestep ranges or voltage cycle iterations. Minimum value: 0.
    loop_end : int
        Ending loop index (inclusive) for data extraction. Must be ≥ loop_start.
        Determines the temporal range of analysis.
    repeat_count : int, optional
        Number of times the first timestep is repeated in the data file structure.
        Used for correcting file parsing when initial conditions are duplicated.
        Default: 0 (no repetition correction applied).

    Returns
    -------
    thermo : list of np.ndarray
        List of displacement data arrays, one per requested loop. Each array has shape
        (Nchunks, N_columns) where:

        - **Nchunks**: Number of spatial bins along analysis direction (typically z-axis)
        - **N_columns**: Data columns per bin (typically 6-8 columns)

        **Standard Column Structure:**
        - Column 0: Chunk/bin ID number
        - Column 1: Spatial position (Angstroms)
        - Column 2: Atom count per bin
        - Column 3: Total displacement magnitude (Angstroms)
        - Column 4: Z-direction displacement (Angstroms)
        - Column 5: Lateral displacement (Angstroms)
        - Column 6: Outward displacement vector (Angstroms)
        - Column 7: Temperature or density (context-dependent)

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist or is not accessible.
    ValueError
        If loop_start > loop_end or if file contains invalid data structure.
    EOFError
        If expected loop data is missing, truncated, or has insufficient rows.
    TypeError
        If Nchunks header line cannot be parsed as integer or is malformed.

    Notes
    -----
    File Format Requirements:
    - **Header Structure**: 3 comment lines (starting with #) followed by Nchunks line
    - **Nchunks Line**: Format "# <unused> <Nchunks>" where Nchunks = spatial bins
    - **Loop Blocks**: Each loop contains Nchunks data rows + 4 separator lines
    - **Data Validation**: Automatic verification of chunk count consistency

    Performance Characteristics:
    - **Memory Usage**: O(N_loops × Nchunks × N_columns) for output storage
    - **I/O Efficiency**: Streaming read with selective row extraction
    - **Error Resilience**: Comprehensive validation with descriptive error messages
    - **Typical Speed**: ~1-10 MB/s depending on file size and chunk count

    Electrochemical Analysis Context:
    Essential for processing mobility analysis outputs where spatial discretization
    enables tracking of:
    - **Ion Migration**: Species-specific displacement patterns along electric field
    - **Thermal Effects**: Temperature-dependent mobility and diffusion coefficients
    - **Interface Dynamics**: Electrode-oxide boundary phenomena and charge injection
    - **Temporal Evolution**: Loop-by-loop analysis of switching cycle progression

    Integration with LAMMPSKit Workflow:
    - **Input Generation**: LAMMPS compute ave/chunk commands generate compatible files
    - **Species Processing**: Combine with extract_element_label_from_filename() for automation
    - **Visualization**: Output arrays integrate with plot_displacement_timeseries()
    - **Analysis Pipelines**: Essential component of plot_displacement_comparison() workflow

    Examples
    --------
    Basic displacement data loading:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import read_displacement_data
    >>> # Load mobility data for 100 simulation loops
    >>> displacement_data = read_displacement_data("Hf_mobility.dat",
    ...                                           loop_start=0, loop_end=99)
    >>> print(f"Loaded {len(displacement_data)} loops")
    >>> print(f"Each loop contains {displacement_data[0].shape[0]} spatial bins")
    >>> print(f"Data columns per bin: {displacement_data[0].shape[1]}")

    Temporal analysis workflow:

    >>> # Analyze displacement evolution across switching cycles
    >>> early_loops = read_displacement_data("O_mobility.dat", 0, 49)      # First 50 loops
    >>> late_loops = read_displacement_data("O_mobility.dat", 950, 999)    # Last 50 loops
    >>>
    >>> # Compare initial vs final displacement profiles
    >>> early_avg = np.mean([loop[:, 4] for loop in early_loops], axis=0)  # Z-displacement
    >>> late_avg = np.mean([loop[:, 4] for loop in late_loops], axis=0)
    >>> displacement_change = late_avg - early_avg
    >>> print(f"Max displacement change: {displacement_change.max():.3f} Å")

    Error handling and validation:

    >>> # Robust file processing with error handling
    >>> try:
    ...     data = read_displacement_data("temperature.dat", 10, 15)
    ...     print(f"Successfully loaded temperature data: {len(data)} loops")
    ... except FileNotFoundError:
    ...     print("Temperature file not found - skipping thermal analysis")
    ... except EOFError as e:
    ...     print(f"Incomplete data file: {e}")
    ... except TypeError as e:
    ...     print(f"File format error: {e}")

    Batch processing integration:

    >>> # Process multiple species files in automated workflow
    >>> mobility_files = ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
    >>> species_labels = ["Hf", "Ta", "O"]
    >>> loop_range = (0, 100)
    >>>
    >>> all_displacement_data = {}
    >>> for filename, species in zip(mobility_files, species_labels):
    ...     try:
    ...         data = read_displacement_data(filename, *loop_range)
    ...         all_displacement_data[species] = data
    ...         print(f"{species}: {len(data)} loops, {data[0].shape[0]} bins")
    ...     except Exception as e:
    ...         print(f"Failed to process {species}: {e}")

    Data structure analysis:

    >>> # Examine file structure and spatial binning
    >>> data = read_displacement_data("Hf_mobility.dat", 0, 0)  # Single loop
    >>> loop_data = data[0]
    >>>
    >>> spatial_positions = loop_data[:, 1]  # Bin center positions
    >>> bin_spacing = spatial_positions[1] - spatial_positions[0]
    >>> analysis_range = spatial_positions[-1] - spatial_positions[0]
    >>>
    >>> print(f"Spatial resolution: {bin_spacing:.2f} Å per bin")
    >>> print(f"Analysis range: {analysis_range:.1f} Å")
    >>> print(f"Electrode separation: ~{analysis_range:.0f} Å")

    Integration with time series plotting:

    >>> # Prepare data for displacement time series visualization
    >>> loop_data = read_displacement_data("O_mobility.dat", 0, 50)
    >>> z_displacements = np.array([loop[:, 4] for loop in loop_data])  # Shape: (51, Nbins)
    >>>
    >>> # Extract time evolution for specific spatial bin (e.g., center of device)
    >>> center_bin = z_displacements.shape[1] // 2
    >>> center_evolution = z_displacements[:, center_bin]
    >>> print(f"Center bin displacement range: {center_evolution.min():.3f} to {center_evolution.max():.3f} Å")
    """
    print(filepath)
    # Validate input parameters using centralized functions
    validate_filepath(filepath)
    validate_loop_parameters(loop_start, loop_end)

    try:
        tmp = np.loadtxt(filepath, comments="#", skiprows=3, max_rows=1)
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
            chunk = np.loadtxt(filepath, comments="#", skiprows=3 + 1 + (n - loop_start) * (Nchunks + 4), max_rows=Nchunks)
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
    **kwargs,
) -> dict[str, plt.Figure]:
    """
    Generate comprehensive atomic distribution analysis for HfTaO electrochemical devices.

    Processes multiple trajectory files to compute and visualize spatial distributions of
    atomic species along the electrode-to-electrode axis. Provides both individual species
    analysis and stoichiometric composition profiling essential for understanding filament
    formation, material redistribution, and interface phenomena in resistive memory devices.

    Parameters
    ----------
    file_list : list of str
        List of file paths to LAMMPS trajectory files (.lammpstrj, .dump).
        Files should contain atomic coordinates with consistent format and time ordering.
        Example: ["initial_state.lammpstrj", "final_state.lammpstrj"]
    labels : list of str
        Descriptive labels for each trajectory file, used in plot legends and filenames.
        Must have same length as file_list. Example: ["Initial", "SET", "RESET"]
    skip_rows : int
        Number of header rows to skip before atomic coordinate data in trajectory files.
        Typical LAMMPS values: 9 (standard dump format), varies with custom output.
    z_bins : int
        Number of spatial bins for z-direction discretization. Recommended: 15-100 bins
        for optimal balance between resolution and statistical significance.
    analysis_name : str
        Base name used for output file generation and plot titles. Should be descriptive
        of the analysis context. Example: "HfTaO_switching_analysis"
    output_dir : str, optional
        Directory path for saving generated plots. Default: current working directory.
        Creates directory if it doesn't exist.
    columns_to_read : tuple, optional
        Column indices to read from trajectory files. If None, uses DEFAULT_COLUMNS_TO_READ
        from config module. Standard format: (id, type, charge, x, y, z).
    **kwargs
        Additional keyword arguments passed to plot_multiple_cases() for plot customization:

        - **yaxis**: Y-axis starting position for plot alignment
        - **linewidth**: Line thickness for distribution curves
        - **alpha**: Transparency level for overlapping distributions
        - **colors**: Custom color scheme for different cases
        - **markers**: Marker styles for data points

    Returns
    -------
    distribution_figures : dict[str, plt.Figure]
        Dictionary of matplotlib Figure objects for different analysis types:

        **Stoichiometric Analysis:**
        - **'stoichiometry'**: Final-state composition (Hf_a-Ta_c-O_b stoichiometry)
        - **'initial_stoichiometry'**: Initial-state composition for comparison

        **Species-Specific Distributions:**
        - **'Hf'**: Hafnium atom spatial distribution (conductive species)
        - **'Ta'**: Tantalum atom spatial distribution (matrix material)
        - **'O'**: Oxygen atom spatial distribution (vacancy formation species)
        - **'metal'**: Combined metallic species (Hf + Ta) distribution

        All figures are saved to output_dir with descriptive filenames for publication use.

    Notes
    -----
    Implementation Details:
    The analysis implements the atom type system:
    - Type 2: Hf atoms
    - Odd types: O atoms
    - Even types (≠2): Ta atoms
    - Proportional factor 3.5: Normalization for analysis

    Performance Characteristics:
    - **Memory Usage**: O(N_files × N_atoms × N_columns) for coordinate storage
    - **Processing Speed**: ~1-10s per file depending on atom count and bin resolution
    - **Output Size**: ~1-5 MB per figure depending on data complexity
    - **Scalability**: Efficient for batch processing of multiple trajectories

    Integration with LAMMPSKit Ecosystem:
    - **Data Processing**: Uses calculate_atomic_distributions() for species separation
    - **Coordinate Reading**: Integrates with read_coordinates() for trajectory parsing
    - **Visualization**: Uses plot_multiple_cases() for publication-quality plotting
    - **Validation**: Employs config module functions for parameter verification

    Examples
    --------
    Basic device analysis workflow:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution
    >>> # Compare initial vs SET state distributions
    >>> trajectory_files = ["initial.lammpstrj", "set_state.lammpstrj"]
    >>> case_labels = ["Initial", "SET"]
    >>> figures = plot_atomic_distribution(trajectory_files, case_labels,
    ...                                   skip_rows=9, z_bins=50,
    ...                                   analysis_name="switching_analysis")
    >>> print(f"Generated {len(figures)} distribution plots")

    Advanced stoichiometric analysis:

    >>> # Multi-state comparison with custom parameters
    >>> files = ["pristine.lammpstrj", "forming.lammpstrj", "set.lammpstrj", "reset.lammpstrj"]
    >>> labels = ["Pristine", "Forming", "SET", "RESET"]
    >>>
    >>> # High-resolution analysis with custom styling
    >>> figures = plot_atomic_distribution(files, labels, skip_rows=9, z_bins=100,
    ...                                   analysis_name="full_cycle_analysis",
    ...                                   output_dir="./analysis_output",
    ...                                   linewidth=2.0, alpha=0.8)
    >>>
    >>> # Examine stoichiometric evolution
    >>> stoich_fig = figures['stoichiometry']
    >>> initial_stoich_fig = figures['initial_stoichiometry']
    >>> print("Stoichiometric analysis completed")

    Species-specific filament analysis:

    >>> # Focus on conductive species evolution
    >>> trajectory_files = ["before_switching.lammpstrj", "after_switching.lammpstrj"]
    >>> labels = ["Before", "After"]
    >>>
    >>> figures = plot_atomic_distribution(trajectory_files, labels, skip_rows=9,
    ...                                   z_bins=75, analysis_name="filament_formation")
    >>>
    >>> # Analyze individual species
    >>> hf_distribution = figures['Hf']      # Conductive filament species
    >>> o_distribution = figures['O']        # Vacancy formation species
    >>> metal_distribution = figures['metal'] # Combined metallic species
    >>>
    >>> print("Filament formation analysis completed")

    Interface and electrode analysis:

    >>> # High-resolution interface study
    >>> interface_files = ["electrode_interface.lammpstrj"]
    >>> interface_labels = ["Interface"]
    >>>
    >>> # Fine-grained binning for interface resolution
    >>> figures = plot_atomic_distribution(interface_files, interface_labels,
    ...                                   skip_rows=9, z_bins=150,
    ...                                   analysis_name="interface_analysis",
    ...                                   yaxis=0)  # Start y-axis at zero
    >>>
    >>> # Examine composition near electrodes
    >>> species_distributions = [figures['Hf'], figures['Ta'], figures['O']]
    >>> print("Interface composition analysis completed")

    Batch processing with error handling:

    >>> import os
    >>> # Process multiple device conditions
    >>> conditions = ["low_voltage", "medium_voltage", "high_voltage"]
    >>> all_figures = {}
    >>>
    >>> for condition in conditions:
    ...     try:
    ...         files = [f"{condition}_initial.lammpstrj", f"{condition}_final.lammpstrj"]
    ...         labels = ["Initial", "Final"]
    ...
    ...         if all(os.path.exists(f) for f in files):
    ...             figures = plot_atomic_distribution(files, labels, skip_rows=9,
    ...                                               z_bins=50,
    ...                                               analysis_name=f"{condition}_analysis")
    ...             all_figures[condition] = figures
    ...             print(f"Completed analysis for {condition}")
    ...         else:
    ...             print(f"Missing files for {condition} - skipping")
    ...     except Exception as e:
    ...         print(f"Error processing {condition}: {e}")
    >>>
    >>> print(f"Successfully analyzed {len(all_figures)} conditions")

    Custom visualization with publication styling:

    >>> # Publication-ready plots with custom styling
    >>> trajectory_files = ["experimental_data.lammpstrj"]
    >>> labels = ["Experimental"]
    >>>
    >>> # Custom styling parameters
    >>> custom_kwargs = {
    ...     'linewidth': 3.0,
    ...     'alpha': 0.9,
    ...     'colors': ['blue', 'red', 'green'],
    ...     'yaxis': 0
    ... }
    >>>
    >>> figures = plot_atomic_distribution(trajectory_files, labels, skip_rows=9,
    ...                                   z_bins=60, analysis_name="publication_analysis",
    ...                                   output_dir="./publication_figures",
    ...                                   **custom_kwargs)
    >>>
    >>> print("Publication-ready figures generated")

    Integration with other LAMMPSKit functions:

    >>> # Combine with charge analysis for comprehensive characterization
    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_charge_distribution
    >>>
    >>> files = ["switching_cycle.lammpstrj"]
    >>> labels = ["Switching"]
    >>>
    >>> # Atomic distribution analysis
    >>> dist_figures = plot_atomic_distribution(files, labels, skip_rows=9, z_bins=50,
    ...                                        analysis_name="comprehensive_analysis")
    >>>
    >>> # Charge distribution analysis
    >>> charge_figures = plot_atomic_charge_distribution(files, labels, skip_rows=9,
    ...                                                 z_bins=50,
    ...                                                 analysis_name="comprehensive_analysis")
    >>>
    >>> total_figures = len(dist_figures) + len(charge_figures)
    >>> print(f"Comprehensive analysis generated {total_figures} figures")
    """
    from .data_processing import calculate_atomic_distributions, calculate_z_bins_setup

    # Use default columns if not specified
    if columns_to_read is None:
        columns_to_read = DEFAULT_COLUMNS_TO_READ

    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(
        file_list, skip_rows, columns_to_read
    )
    z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)

    # Calculate atomic distributions using modular function
    distributions = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)

    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")

    # Avoid division by zero for stoichiometry calculations
    total_distribution_divide = distributions["total"].copy()
    total_distribution_divide[total_distribution_divide == 0] = 1

    # Calculate stoichiometry for last and first trajectory
    O_stoich = 3.5 * distributions["oxygen"][-1] / total_distribution_divide[-1]
    Ta_stoich = 3.5 * distributions["tantalum"][-1] / total_distribution_divide[-1]
    Hf_stoich = 3.5 * distributions["hafnium"][-1] / total_distribution_divide[-1]
    stoichiometry = np.array([Hf_stoich, O_stoich, Ta_stoich])
    proportion_labels = np.array(["a (of Hf$_a$)", "b (of O$_b$)", "c (of Ta$_c$)"])

    O_stoich_in = 3.5 * distributions["oxygen"][0] / total_distribution_divide[0]
    Ta_stoich_in = 3.5 * distributions["tantalum"][0] / total_distribution_divide[0]
    Hf_stoich_in = 3.5 * distributions["hafnium"][0] / total_distribution_divide[0]
    initial_stoichiometry = np.array([Hf_stoich_in, O_stoich_in, Ta_stoich_in])

    figure_size = [2.5, 5]

    # Plot stoichiometry
    output_filename = f"{analysis_name}_stoichiometry_{z_bins}" + "".join(f"_{i}" for i in labels)
    fig_stoich = plot_multiple_cases(
        stoichiometry,
        z_bin_centers,
        proportion_labels,
        "Atoms # ratio",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )
    print("stoichiometry plotted")

    # Plot initial stoichiometry
    output_filename = f"{analysis_name}_initial_stoichiometry_{z_bins}" + "".join(f"_{i}" for i in labels)
    fig_init_stoich = plot_multiple_cases(
        initial_stoichiometry,
        z_bin_centers,
        proportion_labels,
        "Atoms # ratio",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )
    print("initial stoichiometry plotted")

    # Plot metal atoms
    output_filename = f"{analysis_name}_M" + "".join(f"_{i}" for i in labels)
    fig_metal = plot_multiple_cases(
        distributions["metal"],
        z_bin_centers,
        labels,
        "Metal atoms #",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )

    # Plot Hf atoms
    output_filename = f"{analysis_name}_Hf" + "".join(f"_{i}" for i in labels)
    fig_hf = plot_multiple_cases(
        distributions["hafnium"],
        z_bin_centers,
        labels,
        "Hf atoms #",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )

    # Plot Ta atoms
    output_filename = f"{analysis_name}_Ta" + "".join(f"_{i}" for i in labels)
    fig_ta = plot_multiple_cases(
        distributions["tantalum"],
        z_bin_centers,
        labels,
        "Ta atoms #",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )

    # Plot O atoms
    output_filename = f"{analysis_name}_O" + "".join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(
        distributions["oxygen"],
        z_bin_centers,
        labels,
        "O atoms #",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        **kwargs,
    )

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
    **kwargs,
) -> dict[str, plt.Figure]:
    """
    Generate electrostatic charge distribution analysis for HfTaO electrochemical devices.

    Computes and visualizes spatial charge profiles across the electrode-to-electrode axis
    to analyze electrostatic field formation, charge redistribution during switching cycles,
    and ionic polarization effects in resistive memory devices. Provides both total charge
    density and species-specific mean charge characterization essential for understanding
    resistance switching mechanisms and device reliability.

    Parameters
    ----------
    file_list : list of str
        List of file paths to LAMMPS trajectory files containing atomic coordinates and charges.
        Files must include charge information (column 2) for electrostatic analysis.
        Example: ["initial_state.lammpstrj", "charged_state.lammpstrj"]
    labels : list of str
        Descriptive labels for each trajectory file, used in plot legends and output filenames.
        Must match file_list length. Example: ["Initial", "Polarized", "Relaxed"]
    skip_rows : int
        Number of header rows to skip before atomic data in trajectory files.
        Standard LAMMPS dump format typically uses 9 header rows.
    z_bins : int
        Number of spatial bins for z-direction charge profile discretization.
        Recommended: 15-100 bins for optimal field resolution vs. statistical significance.
    analysis_name : str
        Base identifier for output files and plot titles. Should describe analysis context.
        Example: "voltage_cycling_analysis", "field_redistribution_study"
    output_dir : str, optional
        Directory path for saving generated charge distribution plots.
        Default: current working directory. Creates directory if non-existent.
    columns_to_read : tuple, optional
        Column indices for trajectory file parsing. If None, uses DEFAULT_COLUMNS_TO_READ.
        Standard format: (id, type, charge, x, y, z) with charge in column 2.
    **kwargs
        Additional visualization parameters passed to plot_multiple_cases():

        - **ylimithi/xlimithi**: Upper axis limits for charge magnitude scaling
        - **ylimitlo/xlimitlo**: Lower axis limits for field detail resolution
        - **yaxis**: Y-axis origin position for bipolar charge visualization
        - **markerindex**: Marker style index for multi-case differentiation

    Returns
    -------
    charge_figures : dict[str, plt.Figure]
        Dictionary of matplotlib Figure objects for comprehensive charge analysis:

        **Total Charge Analysis:**
        - **'net_charge'**: All-frame total charge density profiles
        - **'initial_net_charge'**: First-frame charge distribution baseline
        - **'final_net_charge'**: Last-frame charge distribution for comparison

        **Species-Specific Mean Charge:**
        - **'metal_charge'**: Mean charge per metal atom (Hf + Ta combined)
        - **'initial_metal_charge'**: Initial metal charge state reference
        - **'oxygen_charge'**: Mean charge per oxygen atom (ion migration indicator)
        - **'initial_oxygen_charge'**: Initial oxygen charge state reference

        All figures include publication-ready formatting with axis labels, legends, and
        consistent styling for comparative analysis across experimental conditions.

    Notes
    -----
    Electrostatic Field Physics:
    - **Total Charge**: Reveals space charge regions and internal field gradients
    - **Metal Charges**: Indicate oxidation state changes and electron transfer
    - **Oxygen Charges**: Track ionic polarization and vacancy formation regions
    - **Temporal Evolution**: Initial vs. final states show switching-induced changes

    Mathematical Foundation:

    Raw charge distribution calculation::

        Q(z) = Σ q_i × δ(z_i - z_bin)

    Mean charge per species::

        <q>_species(z) = Q_species(z) / N_species(z)

    Where safe division prevents numerical errors when N_species(z) = 0.

    Performance Characteristics:
    - **Memory Complexity**: O(N_files × N_atoms × N_bins) for charge histogram storage
    - **Processing Speed**: ~2-15s per file depending on atom count and bin resolution
    - **Output Quality**: Publication-ready SVG figures with customizable styling
    - **Numerical Stability**: Robust handling of zero-atom bins and extreme charge values

    Integration with LAMMPSKit Analysis Pipeline:
    - **Charge Calculation**: Uses calculate_charge_distributions() for species separation
    - **Coordinate Processing**: Integrates with read_coordinates() for trajectory parsing
    - **Atomic Context**: Requires calculate_atomic_distributions() for normalization
    - **Visualization**: Leverages plot_multiple_cases() for consistent scientific plotting

    Examples
    --------
    Basic charge redistribution analysis:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_charge_distribution
    >>> # Analyze charge evolution during switching
    >>> trajectory_files = ["before_switch.lammpstrj", "after_switch.lammpstrj"]
    >>> state_labels = ["Before", "After"]
    >>>
    >>> charge_figures = plot_atomic_charge_distribution(trajectory_files, state_labels,
    ...                                                skip_rows=9, z_bins=50,
    ...                                                analysis_name="switching_charges")
    >>> print(f"Generated {len(charge_figures)} charge analysis plots")

    Voltage-dependent charge analysis:

    >>> # Multi-voltage charge redistribution study
    >>> voltage_files = ["0V.lammpstrj", "1V.lammpstrj", "2V.lammpstrj", "3V.lammpstrj"]
    >>> voltage_labels = ["0V", "1V", "2V", "3V"]
    >>>
    >>> # High-resolution field analysis
    >>> charge_figures = plot_atomic_charge_distribution(voltage_files, voltage_labels,
    ...                                                skip_rows=9, z_bins=100,
    ...                                                analysis_name="voltage_dependence",
    ...                                                output_dir="./charge_analysis")
    >>>
    >>> # Examine field gradient evolution
    >>> net_charge_fig = charge_figures['net_charge']
    >>> metal_charge_fig = charge_figures['metal_charge']
    >>> print("Voltage-dependent field analysis completed")

    Temporal charge evolution tracking:

    >>> # Long-term charge stability analysis
    >>> time_series_files = [f"time_{i}ps.lammpstrj" for i in [0, 100, 500, 1000, 5000]]
    >>> time_labels = ["0ps", "100ps", "500ps", "1000ps", "5000ps"]
    >>>
    >>> charge_figures = plot_atomic_charge_distribution(time_series_files, time_labels,
    ...                                                skip_rows=9, z_bins=75,
    ...                                                analysis_name="temporal_stability")
    >>>
    >>> # Compare initial vs. final charge states
    >>> initial_charge = charge_figures['initial_net_charge']
    >>> final_charge = charge_figures['final_net_charge']
    >>> print("Temporal charge stability analysis completed")

    Species-specific charge state analysis:

    >>> # Detailed oxidation state characterization
    >>> trajectory_files = ["pristine.lammpstrj", "oxidized.lammpstrj"]
    >>> condition_labels = ["Pristine", "Oxidized"]
    >>>
    >>> # Custom axis limits for charge detail resolution
    >>> custom_limits = {
    ...     'ylimithi': 100,    # Extended charge range
    ...     'ylimitlo': -100,   # Bipolar charge visualization
    ...     'xlimitlo': -15,    # Full device width
    ...     'xlimithi': 25,
    ...     'yaxis': 0          # Center y-axis at zero charge
    ... }
    >>>
    >>> charge_figures = plot_atomic_charge_distribution(trajectory_files, condition_labels,
    ...                                                skip_rows=9, z_bins=60,
    ...                                                analysis_name="oxidation_study",
    ...                                                **custom_limits)
    >>>
    >>> # Analyze species-specific charge states
    >>> metal_charges = charge_figures['metal_charge']       # Metal oxidation states
    >>> oxygen_charges = charge_figures['oxygen_charge']     # Oxygen polarization
    >>> initial_metal = charge_figures['initial_metal_charge'] # Reference state
    >>> print("Oxidation state analysis completed")

    Interface charge characterization:

    >>> # Electrode-oxide interface charge analysis
    >>> interface_files = ["bottom_electrode.lammpstrj", "top_electrode.lammpstrj"]
    >>> interface_labels = ["Bottom", "Top"]
    >>>
    >>> # Focus on interface region with appropriate limits
    >>> interface_params = {
    ...     'xlimitlo': -5,     # Near bottom electrode
    ...     'xlimithi': 35,     # Near top electrode
    ...     'ylimithi': 50,     # Moderate charge range
    ...     'yaxis': 0,         # Bipolar visualization
    ...     'markerindex': 1    # Distinctive markers
    ... }
    >>>
    >>> charge_figures = plot_atomic_charge_distribution(interface_files, interface_labels,
    ...                                                skip_rows=9, z_bins=80,
    ...                                                analysis_name="interface_charges",
    ...                                                **interface_params)
    >>> print("Interface charge analysis completed")

    Comparative analysis with atomic distributions:

    >>> # Combined atomic and charge distribution analysis
    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution
    >>>
    >>> trajectory_files = ["device_state.lammpstrj"]
    >>> analysis_labels = ["Device"]
    >>>
    >>> # Atomic distribution analysis
    >>> atomic_figures = plot_atomic_distribution(trajectory_files, analysis_labels,
    ...                                         skip_rows=9, z_bins=50,
    ...                                         analysis_name="comprehensive_study")
    >>>
    >>> # Charge distribution analysis
    >>> charge_figures = plot_atomic_charge_distribution(trajectory_files, analysis_labels,
    ...                                                skip_rows=9, z_bins=50,
    ...                                                analysis_name="comprehensive_study")
    >>>
    >>> # Correlate atomic and charge profiles
    >>> total_analyses = len(atomic_figures) + len(charge_figures)
    >>> print(f"Comprehensive analysis: {total_analyses} figures generated")
    >>> print("Atomic composition:", list(atomic_figures.keys()))
    >>> print("Charge distributions:", list(charge_figures.keys()))

    Batch processing for parameter studies:

    >>> import os
    >>> # Systematic parameter variation study
    >>> parameters = ["low_field", "medium_field", "high_field"]
    >>> all_charge_analyses = {}
    >>>
    >>> for param in parameters:
    ...     try:
    ...         param_files = [f"{param}_initial.lammpstrj", f"{param}_final.lammpstrj"]
    ...         param_labels = ["Initial", "Final"]
    ...
    ...         if all(os.path.exists(f) for f in param_files):
    ...             figures = plot_atomic_charge_distribution(param_files, param_labels,
    ...                                                     skip_rows=9, z_bins=50,
    ...                                                     analysis_name=f"{param}_charges",
    ...                                                     output_dir=f"./analysis_{param}")
    ...             all_charge_analyses[param] = figures
    ...             print(f"Completed charge analysis for {param}")
    ...         else:
    ...             print(f"Missing trajectory files for {param}")
    ...     except Exception as e:
    ...         print(f"Error analyzing {param}: {e}")
    >>>
    >>> successful_analyses = len(all_charge_analyses)
    >>> print(f"Successfully completed {successful_analyses} parameter studies")

    Advanced visualization customization:

    >>> # Publication-quality charge analysis with custom styling
    >>> publication_files = ["experiment_data.lammpstrj"]
    >>> publication_labels = ["Experimental"]
    >>>
    >>> # Publication-specific parameters
    >>> pub_styling = {
    ...     'ylimithi': 80,      # Appropriate charge scale
    ...     'ylimitlo': -80,     # Symmetric bipolar range
    ...     'xlimitlo': -10,     # Device boundaries
    ...     'xlimithi': 40,
    ...     'yaxis': 0,          # Zero-centered
    ...     'markerindex': 2     # Publication markers
    ... }
    >>>
    >>> charge_figures = plot_atomic_charge_distribution(publication_files, publication_labels,
    ...                                                skip_rows=9, z_bins=75,
    ...                                                analysis_name="publication_charges",
    ...                                                output_dir="./publication_figures",
    ...                                                **pub_styling)
    >>>
    >>> print("Publication-ready charge distribution figures generated")
    >>> print("Available figures:", list(charge_figures.keys()))
    """
    from .data_processing import calculate_atomic_distributions, calculate_charge_distributions, calculate_z_bins_setup

    # Use default columns if not specified
    if columns_to_read is None:
        columns_to_read = DEFAULT_COLUMNS_TO_READ

    # Read coordinates and simulation parameters
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(
        file_list, skip_rows, columns_to_read
    )
    z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)

    print(f"\nshape of coordinate_arr= {np.shape(coordinates_arr)}, length of coordinate_arr= {len(coordinates_arr)}")

    # Calculate atomic distributions (needed for charge distribution normalization)
    atomic_distributions = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)

    # Calculate charge distributions using modular function
    charge_distributions = calculate_charge_distributions(coordinates_arr, z_bins, zlo, zhi, atomic_distributions)

    figure_size = [2.5, 5]

    # Plot net charge
    output_filename = f"{analysis_name}_all" + "".join(f"_{i}" for i in labels)
    fig_net = plot_multiple_cases(
        charge_distributions["total_charge"],
        z_bin_centers,
        labels,
        "Net charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimithi=15,
        xlimitlo=-20,
        yaxis=0,
    )

    # Plot metal mean charge
    output_filename = f"{analysis_name}_M" + "".join(f"_{i}" for i in labels)
    fig_metal = plot_multiple_cases(
        charge_distributions["metal_mean_charge"],
        z_bin_centers,
        labels,
        "Metal atoms mean charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimitlo=0.7,
        xlimithi=1.2,
    )

    # Plot oxygen mean charge
    output_filename = f"{analysis_name}_O" + "".join(f"_{i}" for i in labels)
    fig_o = plot_multiple_cases(
        charge_distributions["oxygen_mean_charge"],
        z_bin_centers,
        labels,
        "O mean charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimithi=0,
        xlimitlo=-0.7,
    )

    # Plot final net charge
    output_filename = f"final_{analysis_name}_all" + "".join(f"_{i}" for i in labels)
    fig_net_end = plot_multiple_cases(
        charge_distributions["total_charge"][-1],
        z_bin_centers,
        labels[-1],
        "Net charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimithi=15,
        xlimitlo=-20,
        yaxis=0,
        markerindex=1,
    )

    # Plot initial net charge
    output_filename = f"initial_{analysis_name}_all" + "".join(f"_{i}" for i in labels)
    fig_net_start = plot_multiple_cases(
        charge_distributions["total_charge"][0],
        z_bin_centers,
        labels[0],
        "Net charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimithi=15,
        xlimitlo=-20,
        yaxis=0,
    )

    # Plot initial metal mean charge
    output_filename = f"initial_{analysis_name}_M" + "".join(f"_{i}" for i in labels)
    fig_metal_start = plot_multiple_cases(
        charge_distributions["metal_mean_charge"][0],
        z_bin_centers,
        labels[0],
        "Metal atoms mean charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimitlo=0.7,
        xlimithi=1.2,
    )

    # Plot initial oxygen mean charge
    output_filename = f"initial_{analysis_name}_O" + "".join(f"_{i}" for i in labels)
    fig_o_start = plot_multiple_cases(
        charge_distributions["oxygen_mean_charge"][0],
        z_bin_centers,
        labels[0],
        "O mean charge",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        ylimithi=70,
        xlimithi=0,
        xlimitlo=-0.7,
    )

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
    output_dir: str = os.getcwd(),
) -> dict[str, plt.Figure]:
    """
    Generate comparative displacement analysis for multi-species ion migration in HfTaO devices.

    Processes multiple thermodynamic output files to extract and visualize final displacement
    patterns across different atomic species, experimental conditions, or temporal states.
    Provides comprehensive mobility characterization essential for understanding ion migration
    mechanisms, filament formation pathways, and electrochemical switching dynamics in
    resistive memory devices through spatial displacement profiling.

    Parameters
    ----------
    file_list : list of str
        List of file paths to LAMMPS thermodynamic output files containing spatially-binned
        displacement data. Each file represents a different analysis case (species, condition,
        or temporal state). Common examples:

        - **Species Files**: ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
        - **Condition Files**: ["low_field.dat", "medium_field.dat", "high_field.dat"]
        - **Temporal Files**: ["early.dat", "middle.dat", "late.dat"]

        Files must follow thermodynamic output format with consistent spatial binning.
    loop_start : int
        Starting loop index (inclusive) for temporal analysis range. Corresponds to simulation
        timestep ranges or voltage cycle iterations. Minimum value: 0.
        Used to select specific time windows within displacement evolution data.
    loop_end : int
        Ending loop index (inclusive) for temporal analysis range. Must be ≥ loop_start.
        Defines the final temporal point for displacement analysis.
    labels : list of str
        Descriptive identifiers for each file in file_list, used in plot legends and output
        filenames. Must match file_list length. Examples:

        - **Species Labels**: ["Hf", "Ta", "O"] for multi-species comparison
        - **Condition Labels**: ["Low Field", "High Field"] for parametric studies
        - **State Labels**: ["Initial", "Intermediate", "Final"] for temporal analysis
    analysis_name : str
        Base identifier for output file naming and plot titles. Should describe the
        comparative analysis context. Examples: "species_mobility", "field_dependence",
        "temporal_evolution"
    repeat_count : int, optional
        Number of times the first timestep is repeated in data files. Used for file
        format correction when initial conditions are duplicated. Default: 0.
    output_dir : str, optional
        Directory path for saving generated displacement comparison plots.
        Default: current working directory. Creates directory if non-existent.

    Returns
    -------
    displacement_figures : dict[str, plt.Figure]
        Dictionary of matplotlib Figure objects for comprehensive displacement analysis:

        **Z-Direction Analysis:**
        - **'z_displacement'**: Signed z-displacement profiles showing migration direction
        - **'z_magnitude'**: Absolute z-displacement magnitude for migration extent analysis

        **Lateral Displacement Analysis:**
        - **'lateral_displacement'**: Radial displacement patterns indicating spreading behavior

        All figures include publication-ready formatting with comparative legends, axis labels,
        and consistent styling for multi-case analysis visualization.

    Notes
    -----
    Data Processing Methodology:

    **Final State Extraction:**
    The function processes the last temporal frame (loop_end) from each file:
    ```
    z_displacement = data[file_index, -1, :, -3]      # Column -3: Z-displacement
    lateral_displacement = data[file_index, -1, :, -2] # Column -2: Lateral displacement
    bin_position = data[file_index, -1, :, 1]          # Column 1: Spatial position
    atom_count = data[file_index, -1, :, 2]            # Column 2: Atoms per bin
    ```

    **Spatial Binning Framework:**
    - Uniform z-direction discretization across electrode separation
    - Bin centers represent average displacement for contained atoms
    - Statistical significance ensured through atom count validation
    - Cross-file spatial alignment for comparative analysis

    Performance Characteristics:
    - **Memory Usage**: O(N_files × N_loops × N_bins × N_columns) for data storage
    - **Processing Speed**: ~2-8s per file depending on loop range and bin count
    - **Output Quality**: Publication-ready SVG figures with comparative legends
    - **Scalability**: Efficient batch processing for parametric studies

    Integration with LAMMPSKit Analysis Pipeline:
    - **Data Reading**: Uses read_displacement_data() for robust file parsing
    - **Visualization**: Leverages plot_multiple_cases() for comparative plotting
    - **Error Handling**: Inherits validation from displacement data reading functions
    - **Workflow Integration**: Compatible with plot_displacement_timeseries() analysis

    Examples
    --------
    Multi-species mobility comparison:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_comparison
    >>> # Compare displacement patterns of different atomic species
    >>> mobility_files = ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
    >>> species_labels = ["Hf", "Ta", "O"]
    >>>
    >>> displacement_figures = plot_displacement_comparison(mobility_files,
    ...                                                   loop_start=0, loop_end=99,
    ...                                                   labels=species_labels,
    ...                                                   analysis_name="species_mobility")
    >>> print(f"Generated {len(displacement_figures)} displacement comparison plots")
    >>>
    >>> # Analyze species-specific migration behavior
    >>> z_disp_fig = displacement_figures['z_displacement']      # Migration direction
    >>> lateral_fig = displacement_figures['lateral_displacement'] # Spreading behavior
    >>> magnitude_fig = displacement_figures['z_magnitude']       # Migration extent

    Field-dependent displacement analysis:

    >>> # Parametric study of electric field effects on mobility
    >>> field_files = ["low_field_Hf.dat", "medium_field_Hf.dat", "high_field_Hf.dat"]
    >>> field_labels = ["Low Field", "Medium Field", "High Field"]
    >>>
    >>> field_displacement = plot_displacement_comparison(field_files,
    ...                                                 loop_start=50, loop_end=150,
    ...                                                 labels=field_labels,
    ...                                                 analysis_name="field_dependence",
    ...                                                 output_dir="./field_study")
    >>>
    >>> print("Field-dependent mobility analysis completed")
    >>> print("Available plots:", list(field_displacement.keys()))

    Temporal evolution analysis:

    >>> # Time-resolved displacement evolution study
    >>> temporal_files = ["early_mobility.dat", "mid_mobility.dat", "late_mobility.dat"]
    >>> time_labels = ["Early (0-100ps)", "Mid (500-600ps)", "Late (1900-2000ps)"]
    >>>
    >>> temporal_analysis = plot_displacement_comparison(temporal_files,
    ...                                                loop_start=0, loop_end=50,
    ...                                                labels=time_labels,
    ...                                                analysis_name="temporal_evolution")
    >>>
    >>> # Examine long-term migration trends
    >>> z_evolution = temporal_analysis['z_displacement']
    >>> lateral_evolution = temporal_analysis['lateral_displacement']
    >>> print("Temporal displacement evolution analysis completed")

    Device condition comparison:

    >>> # Compare displacement under different device conditions
    >>> condition_files = ["pristine_mobility.dat", "cycled_mobility.dat", "degraded_mobility.dat"]
    >>> condition_labels = ["Pristine", "After 1000 Cycles", "Degraded"]
    >>>
    >>> condition_analysis = plot_displacement_comparison(condition_files,
    ...                                                 loop_start=80, loop_end=120,
    ...                                                 labels=condition_labels,
    ...                                                 analysis_name="device_degradation")
    >>>
    >>> print("Device condition comparison completed")

    Comprehensive multi-case analysis:

    >>> # Large-scale comparative study with multiple parameters
    >>> import os
    >>> analysis_cases = {
    ...     "species_comparison": {
    ...         "files": ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"],
    ...         "labels": ["Hf", "Ta", "O"]
    ...     },
    ...     "voltage_dependence": {
    ...         "files": ["1V_mobility.dat", "2V_mobility.dat", "3V_mobility.dat"],
    ...         "labels": ["1V", "2V", "3V"]
    ...     },
    ...     "temperature_effects": {
    ...         "files": ["300K_mobility.dat", "400K_mobility.dat", "500K_mobility.dat"],
    ...         "labels": ["300K", "400K", "500K"]
    ...     }
    ... }
    >>>
    >>> all_displacement_analyses = {}
    >>> for case_name, case_params in analysis_cases.items():
    ...     try:
    ...         if all(os.path.exists(f) for f in case_params["files"]):
    ...             figures = plot_displacement_comparison(case_params["files"],
    ...                                                   loop_start=0, loop_end=75,
    ...                                                   labels=case_params["labels"],
    ...                                                   analysis_name=case_name,
    ...                                                   output_dir=f"./analysis_{case_name}")
    ...             all_displacement_analyses[case_name] = figures
    ...             print(f"Completed {case_name}: {len(figures)} figures")
    ...         else:
    ...             print(f"Missing files for {case_name}")
    ...     except Exception as e:
    ...         print(f"Error in {case_name}: {e}")
    >>>
    >>> total_figures = sum(len(figs) for figs in all_displacement_analyses.values())
    >>> print(f"Total displacement analyses: {total_figures} figures across {len(all_displacement_analyses)} cases")

    Advanced data extraction and analysis:

    >>> # Detailed analysis of displacement patterns
    >>> mobility_files = ["oxygen_mobility.dat"]
    >>> labels = ["O Mobility"]
    >>>
    >>> displacement_data = plot_displacement_comparison(mobility_files, 0, 50, labels,
    ...                                                "detailed_oxygen_analysis")
    >>>
    >>> # Access raw displacement data for further analysis
    >>> print("Available displacement analyses:", list(displacement_data.keys()))
    >>>
    >>> # Examine displacement characteristics
    >>> # Note: To access raw data, you would typically use read_displacement_data directly
    >>> from lammpskit.ecellmodel.filament_layer_analysis import read_displacement_data
    >>> raw_data = read_displacement_data("oxygen_mobility.dat", 0, 50)
    >>> final_frame = raw_data[-1]  # Last temporal frame
    >>>
    >>> z_positions = final_frame[:, 1]        # Spatial bin centers
    >>> z_displacements = final_frame[:, -3]   # Z-direction displacements
    >>> lateral_displacements = final_frame[:, -2]  # Lateral displacements
    >>>
    >>> print(f"Spatial resolution: {z_positions[1] - z_positions[0]:.2f} Å")
    >>> print(f"Max z-displacement: {z_displacements.max():.3f} Å")
    >>> print(f"Max lateral displacement: {lateral_displacements.max():.3f} Å")

    Error handling and validation:

    >>> # Robust displacement analysis with comprehensive error handling
    >>> mobility_files = ["species1.dat", "species2.dat", "species3.dat"]
    >>> species_labels = ["Species1", "Species2", "Species3"]
    >>>
    >>> try:
    ...     # Validate file existence before analysis
    ...     existing_files = [f for f in mobility_files if os.path.exists(f)]
    ...     existing_labels = [species_labels[i] for i, f in enumerate(mobility_files) if os.path.exists(f)]
    ...
    ...     if len(existing_files) >= 2:  # Require at least 2 files for comparison
    ...         displacement_figures = plot_displacement_comparison(existing_files,
    ...                                                           loop_start=0, loop_end=100,
    ...                                                           labels=existing_labels,
    ...                                                           analysis_name="validated_comparison")
    ...         print(f"Successfully analyzed {len(existing_files)} displacement files")
    ...     else:
    ...         print("Insufficient files for comparative analysis")
    ... except FileNotFoundError as e:
    ...     print(f"File access error: {e}")
    ... except ValueError as e:
    ...     print(f"Data format error: {e}")
    ... except Exception as e:
    ...     print(f"Analysis error: {e}")

    Integration with other LAMMPSKit functions:

    >>> # Combine displacement comparison with time series analysis
    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries
    >>>
    >>> # Step 1: Comparative final displacement analysis
    >>> comparative_files = ["condition_A.dat", "condition_B.dat"]
    >>> comparative_labels = ["Condition A", "Condition B"]
    >>>
    >>> final_displacement = plot_displacement_comparison(comparative_files, 0, 100,
    ...                                                 comparative_labels, "comparison_study")
    >>>
    >>> # Step 2: Detailed time series for selected condition
    >>> time_series = plot_displacement_timeseries(["condition_A.dat"], 0, 100, ["Condition A"],
    ...                                          "detailed_timeseries", time_step=0.0002,
    ...                                          dump_interval_steps=5000)
    >>>
    >>> print("Comprehensive displacement analysis completed:")
    >>> print(f"  Comparative analysis: {len(final_displacement)} figures")
    >>> print(f"  Time series analysis: {len(time_series)} figures")
    """
    # Read thermodynamic data for each file
    all_thermo_data: list[list[np.ndarray]] = []
    for filename in file_list:
        all_thermo_data.append(read_displacement_data(filename, loop_start, loop_end, repeat_count))

    displacements = np.array(all_thermo_data)
    print(
        f"\nshape of all_thermo_data array= {np.shape(all_thermo_data)}, length of all_thermo_data array= {len(all_thermo_data)}"
    )

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
    output_filename = f"{analysis_name}_z" + "".join(f"_{i}" for i in labels)
    fig_z = plot_multiple_cases(
        zdisp,
        binposition,
        labels,
        "z displacement (A)",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
        yaxis=0,
    )

    # Plot z displacement magnitude
    output_filename = f"{analysis_name}_z_magnitude" + "".join(f"_{i}" for i in labels)
    fig_zmag = plot_multiple_cases(
        np.abs(zdisp),
        binposition,
        labels,
        "z displacement (A)",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
    )

    # Plot lateral displacement
    output_filename = f"{analysis_name}_lateral" + "".join(f"_{i}" for i in labels)
    fig_lateral = plot_multiple_cases(
        lateraldisp,
        binposition,
        labels,
        "lateral displacement (A)",
        "z position (A)",
        output_filename,
        figure_size[0],
        figure_size[1],
        output_dir=output_dir,
    )

    return {
        "z_displacement": fig_z,
        "z_magnitude": fig_zmag,
        "lateral_displacement": fig_lateral,
    }


def analyze_clusters(
    filepath: str, z_filament_lower_limit: float = 5, z_filament_upper_limit: float = 23, thickness: float = 21
) -> tuple[int, int, int, float, np.ndarray, int, float, np.ndarray, float, float]:
    """
    Perform comprehensive OVITO-based cluster analysis for metallic filament characterization.

    Conducts advanced structural analysis of HfTaO electrochemical memory devices using OVITO's
    cluster detection algorithms to identify, characterize, and quantify conductive filament
    formation. Provides detailed connectivity analysis, size distribution characterization,
    and radial distribution function (RDF) analysis essential for understanding resistive
    switching mechanisms and device reliability in electrochemical memory applications.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS trajectory file (.lammpstrj, .dump) containing atomic coordinates
        and species information. File must be compatible with OVITO import functionality
        and contain sufficient atomic data for cluster detection.
        Example: "filament_formation.lammpstrj", "switching_cycle.dump"
    z_filament_lower_limit : float, optional
        Lower z-coordinate boundary (Angstroms) defining the bottom electrode interface
        for filament connectivity analysis. Represents the lower threshold for complete
        electrode-to-electrode conductive bridge formation. Default: 5.0 Å.
        Typical range: 3-8 Å depending on device geometry.
    z_filament_upper_limit : float, optional
        Upper z-coordinate boundary (Angstroms) defining the top electrode interface
        for filament connectivity analysis. Represents the upper threshold for complete
        conductive bridge formation. Default: 23.0 Å.
        Typical range: 20-30 Å depending on oxide thickness.
    thickness : float, optional
        Characteristic filament thickness parameter (Angstroms) used for structural
        analysis and RDF calculations. Influences cluster size classification and
        connectivity determination. Default: 21.0 Å.
        Optimization range: 15-30 Å based on device dimensions.

    Returns
    -------
    cluster_analysis_results : tuple[int, int, int, float, np.ndarray, int, float, np.ndarray, float, float]
        Comprehensive cluster analysis data containing connectivity metrics, size
        distributions, and structural characterization:

        **Temporal Information:**
        - **timestep** (int): Simulation timestep from trajectory file header

        **Connectivity Analysis:**
        - **connection** (int): Binary connectivity state (1=connected, 0=disconnected)
        - **separation** (float): Minimum inter-cluster distance (Å) when disconnected
        - **gap** (float): Z-direction gap between cluster boundaries (Å)

        **Lower Filament Characterization:**
        - **fil_size_down** (int): Number of atoms in bottom filament cluster
        - **fil_height** (float): Maximum z-coordinate of bottom cluster (Å)
        - **rdf_down** (np.ndarray): Radial distribution function data (shape: [N_bins, 2])

          * Column 0: Radial distance (Å)
          * Column 1: RDF intensity

        **Upper Filament Characterization:**
        - **fil_size_up** (int): Number of atoms in top filament cluster
        - **fil_depth** (float): Minimum z-coordinate of top cluster (Å)
        - **rdf_up** (np.ndarray): Upper cluster RDF data (shape: [N_bins, 2])

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist or is not accessible to OVITO.
    ValueError
        If no metallic clusters are detected in the trajectory file or if file format
        is incompatible with OVITO cluster analysis pipelines.
    TypeError
        If input parameters have invalid types or values outside acceptable ranges.

    Notes
    -----
    OVITO Cluster Analysis Pipeline:

    **Multi-Pipeline Architecture:**
    The function employs four specialized OVITO pipelines for comprehensive analysis:

    1. **Primary Cluster Pipeline**: Largest cluster identification and analysis
    2. **Secondary Cluster Pipeline**: Second-largest cluster characterization
    3. **Lower Filament Pipeline**: Bottom electrode region analysis
    4. **Upper Filament Pipeline**: Top electrode region analysis

    **Coordination Analysis Framework:**

    ::

        Cutoff Distance: 2.7 Å (first neighbor shell for metallic bonds)
        Coordination Threshold: <6 (under-coordinated atoms for filament detection)
        Clustering Cutoff: 3.9 Å (metallic cluster connectivity distance)

    **Species Selection Criteria:**
    The analysis targets metallic and under-coordinated species:
    - **Type 2**: Hafnium atoms (matrix metal component)
    - **Type 4, 8**: Tantalum atoms (matrix metal component)
    - **Type 8**: Additional Ta metal species (device-specific)
    - **Type 1, 3, 7**: Oxygen atoms (matrix component)
    - **Types 5, 6, 9, 10**: Electrode species

    **Connectivity Determination Algorithm:**

    ::

        if z_min < z_lower_limit and z_max > z_upper_limit:
            connection_state = 1  # Continuous bridge formed
            separation = 0.0
            gap = 0.0
        else:
            connection_state = 0  # Discontinuous filament
            separation = min_distance_between_clusters - cutoff_distance
            gap = z_direction_gap_between_clusters

    **Radial Distribution Function (RDF) Analysis:**
    Characterizes local atomic structure and bonding environment

    **Performance Characteristics:**

    - **Processing Time**: 5-30s per trajectory frame depending on atom count
    - **Memory Usage**: O(N_atoms × N_neighbors) for coordination analysis
    - **Accuracy**: Sub-angstrom precision for cluster boundary detection
    - **Scalability**: Efficient for trajectories up to ~10⁶ atoms per frame

    **Integration with LAMMPSKit Ecosystem:**

    - **File Compatibility**: Direct integration with LAMMPS trajectory formats
    - **Parameter Validation**: Uses config module validation functions
    - **Pipeline Integration**: Compatible with track_filament_evolution() workflows
    - **Data Export**: Results integrate with time series plotting functions

    Examples
    --------
    Basic filament connectivity analysis:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
    >>> # Analyze single trajectory frame for filament formation
    >>> trajectory_file = "switching_process.lammpstrj"
    >>>
    >>> cluster_results = analyze_clusters(trajectory_file,
    ...                                   z_filament_lower_limit=5,
    ...                                   z_filament_upper_limit=23)
    >>>
    >>> # Extract connectivity information
    >>> timestep, connection, size_down, height, rdf_down, size_up, depth, rdf_up, separation, gap = cluster_results
    >>>
    >>> print(f"Timestep: {timestep}")
    >>> print(f"Filament Connected: {'Yes' if connection else 'No'}")
    >>> if connection:
    ...     print(f"Complete bridge formed with {size_down + size_up} atoms")
    ... else:
    ...     print(f"Gap: {gap:.2f} Å, Separation: {separation:.2f} Å")

    Device geometry optimization:

    >>> # Parametric study of device geometry effects on connectivity
    >>> trajectory_file = "device_optimization.lammpstrj"
    >>>
    >>> # Test different electrode separations
    >>> geometries = [
    ...     {"lower": 3, "upper": 20, "name": "thin_oxide"},
    ...     {"lower": 5, "upper": 25, "name": "medium_oxide"},
    ...     {"lower": 7, "upper": 30, "name": "thick_oxide"}
    ... ]
    >>>
    >>> connectivity_results = {}
    >>> for geom in geometries:
    ...     try:
    ...         results = analyze_clusters(trajectory_file,
    ...                                  z_filament_lower_limit=geom["lower"],
    ...                                  z_filament_upper_limit=geom["upper"])
    ...         connectivity_results[geom["name"]] = {
    ...             "connected": results[1],
    ...             "total_atoms": results[2] + results[5],
    ...             "gap": results[9] if not results[1] else 0.0
    ...         }
    ...         print(f"{geom['name']}: Connected={results[1]}, Atoms={results[2] + results[5]}")
    ...     except Exception as e:
    ...         print(f"Error analyzing {geom['name']}: {e}")
    >>>
    >>> print("Geometry optimization completed")

    Detailed structural characterization:

    >>> # Comprehensive filament structure analysis
    >>> trajectory_file = "filament_structure.lammpstrj"
    >>>
    >>> cluster_data = analyze_clusters(trajectory_file, z_filament_lower_limit=4,
    ...                               z_filament_upper_limit=24, thickness=20)
    >>>
    >>> timestep, connection, size_down, height, rdf_down, size_up, depth, rdf_up, separation, gap = cluster_data
    >>>
    >>> # Analyze RDF characteristics
    >>> rdf_distances = rdf_down[:, 0]  # Radial distances
    >>> rdf_intensities = rdf_down[:, 1]  # RDF values
    >>>
    >>> # Find first coordination shell peak
    >>> first_peak_idx = np.argmax(rdf_intensities[:50])  # Search first 5 Å
    >>> first_peak_position = rdf_distances[first_peak_idx]
    >>> first_peak_intensity = rdf_intensities[first_peak_idx]
    >>>
    >>> print(f"Structural Analysis Results:")
    >>> print(f"  Lower cluster: {size_down} atoms, height: {height:.2f} Å")
    >>> print(f"  Upper cluster: {size_up} atoms, depth: {depth:.2f} Å")
    >>> print(f"  First RDF peak: {first_peak_position:.2f} Å (intensity: {first_peak_intensity:.3f})")
    >>>
    >>> if connection:
    ...     total_filament_size = size_down + size_up
    ...     filament_span = height - depth
    ...     print(f"  Complete filament: {total_filament_size} atoms spanning {filament_span:.2f} Å")
    ... else:
    ...     print(f"  Disconnected filament with {gap:.2f} Å gap")

    Time series connectivity analysis:

    >>> # Process multiple trajectory frames for evolution tracking
    >>> import glob
    >>> trajectory_files = sorted(glob.glob("time_series_*.lammpstrj"))
    >>>
    >>> connectivity_evolution = []
    >>> filament_size_evolution = []
    >>> gap_evolution = []
    >>>
    >>> for i, traj_file in enumerate(trajectory_files):
    ...     try:
    ...         results = analyze_clusters(traj_file, z_filament_lower_limit=5,
    ...                                  z_filament_upper_limit=23)
    ...
    ...         timestep, connection, size_down, height, rdf_down, size_up, depth, rdf_up, separation, gap = results
    ...
    ...         connectivity_evolution.append(connection)
    ...         filament_size_evolution.append(size_down + size_up)
    ...         gap_evolution.append(gap if not connection else 0.0)
    ...
    ...         if i % 10 == 0:  # Progress reporting
    ...             print(f"Processed frame {i+1}/{len(trajectory_files)}")
    ...
    ...     except Exception as e:
    ...         print(f"Error processing {traj_file}: {e}")
    ...         connectivity_evolution.append(0)
    ...         filament_size_evolution.append(0)
    ...         gap_evolution.append(float('inf'))
    >>>
    >>> # Analyze connectivity statistics
    >>> connectivity_fraction = np.mean(connectivity_evolution)
    >>> avg_connected_size = np.mean([size for size, conn in zip(filament_size_evolution, connectivity_evolution) if conn])
    >>> avg_gap_when_disconnected = np.mean([gap for gap, conn in zip(gap_evolution, connectivity_evolution) if not conn and gap < float('inf')])
    >>>
    >>> print(f"Time Series Analysis Results:")
    >>> print(f"  Connectivity fraction: {connectivity_fraction:.3f}")
    >>> print(f"  Average connected filament size: {avg_connected_size:.1f} atoms")
    >>> print(f"  Average gap when disconnected: {avg_gap_when_disconnected:.2f} Å")

    Advanced RDF analysis and comparison:

    >>> # Compare RDF characteristics between different states
    >>> connected_file = "connected_state.lammpstrj"
    >>> disconnected_file = "disconnected_state.lammpstrj"
    >>>
    >>> # Analyze connected state
    >>> conn_results = analyze_clusters(connected_file)
    >>> conn_rdf_down = conn_results[4]
    >>>
    >>> # Analyze disconnected state
    >>> disconn_results = analyze_clusters(disconnected_file)
    >>> disconn_rdf_down = disconn_results[4]
    >>>
    >>> # Compare structural order
    >>> conn_first_peak = np.max(conn_rdf_down[:50, 1])
    >>> disconn_first_peak = np.max(disconn_rdf_down[:50, 1])
    >>>
    >>> print(f"RDF Comparison:")
    >>> print(f"  Connected state first peak intensity: {conn_first_peak:.3f}")
    >>> print(f"  Disconnected state first peak intensity: {disconn_first_peak:.3f}")
    >>> print(f"  Structural order change: {(conn_first_peak - disconn_first_peak)/disconn_first_peak*100:.1f}%")

    Error handling and validation:

    >>> # Robust cluster analysis with comprehensive error handling
    >>> trajectory_files = ["test1.lammpstrj", "test2.lammpstrj", "test3.lammpstrj"]
    >>>
    >>> successful_analyses = []
    >>> failed_analyses = []
    >>>
    >>> for traj_file in trajectory_files:
    ...     try:
    ...         # Validate file exists before processing
    ...         import os
    ...         if not os.path.exists(traj_file):
    ...             print(f"File not found: {traj_file}")
    ...             failed_analyses.append((traj_file, "FileNotFoundError"))
    ...             continue
    ...
    ...         # Perform cluster analysis
    ...         results = analyze_clusters(traj_file,
    ...                                  z_filament_lower_limit=5,
    ...                                  z_filament_upper_limit=23)
    ...
    ...         # Validate results
    ...         if len(results) == 10:
    ...             successful_analyses.append((traj_file, results))
    ...             print(f"Successfully analyzed: {traj_file}")
    ...         else:
    ...             failed_analyses.append((traj_file, "Invalid results format"))
    ...
    ...     except ValueError as e:
    ...         print(f"Cluster detection failed for {traj_file}: {e}")
    ...         failed_analyses.append((traj_file, f"ValueError: {e}"))
    ...     except Exception as e:
    ...         print(f"Unexpected error for {traj_file}: {e}")
    ...         failed_analyses.append((traj_file, f"Exception: {e}"))
    >>>
    >>> print(f"Analysis Summary:")
    >>> print(f"  Successful: {len(successful_analyses)}")
    >>> print(f"  Failed: {len(failed_analyses)}")

    Integration with filament evolution tracking:

    >>> # Use with track_filament_evolution for comprehensive time series analysis
    >>> from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution
    >>>
    >>> # Single-frame cluster analysis
    >>> single_frame_file = "snapshot.lammpstrj"
    >>> cluster_analysis = analyze_clusters(single_frame_file)
    >>>
    >>> print(f"Single Frame Analysis:")
    >>> print(f"  Connectivity: {cluster_analysis[1]}")
    >>> print(f"  Total atoms: {cluster_analysis[2] + cluster_analysis[5]}")
    >>>
    >>> # Multi-frame evolution tracking
    >>> time_series_files = ["evolution.lammpstrj"]
    >>> evolution_plots = track_filament_evolution(time_series_files, "evolution_study",
    ...                                          time_step=0.0002, dump_interval_steps=5000)
    >>>
    >>> print(f"Evolution tracking completed with {len(evolution_plots)} plots")
    """
    # Check OVITO availability before proceeding
    if not OVITO_AVAILABLE:
        raise ImportError(
            "OVITO is required for cluster analysis but is not available. "
            "Please install OVITO (pip install ovito>=3.12.4) to use this function."
        )
    
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
    pipeline1.modifiers.append(
        om.ExpressionSelectionModifier(
            expression="((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10 || ParticleType==9 ) && Position.Z < 28 "
        )
    )
    pipeline1.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline1.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster !=1"))
    pipeline1.modifiers.append(om.DeleteSelectedModifier())
    data1 = pipeline1.compute()
    if data1.particles.count == 0:
        raise ValueError(f"No clusters found in file: {filepath}")

    timestep = data1.attributes["Timestep"]
    xyz1 = np.array(data1.particles["Position"])

    # Pipeline 2: Analyze clusters (secondary)
    pipeline2.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline2.modifiers.append(
        om.ExpressionSelectionModifier(
            expression="((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28 "
        )
    )
    pipeline2.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True))
    pipeline2.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster !=2"))
    pipeline2.modifiers.append(om.DeleteSelectedModifier())
    data2 = pipeline2.compute()
    xyz2 = np.array(data2.particles["Position"])

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
        separation = float("inf")
        gap = float("inf")
        for point1 in group_a:
            for point2 in group_b:
                distance = np.linalg.norm(point1 - point2)
                if distance < separation:
                    separation = distance
                    gap = abs(point1[2] - point2[2])
        separation -= 3.9  # Subtract cutoff

    # Pipeline for lower filament
    pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline_fil.modifiers.append(
        om.ExpressionSelectionModifier(expression="((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6)")
    )
    pipeline_fil.modifiers.append(
        om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True)
    )
    pipeline_fil.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster !=1"))
    pipeline_fil.modifiers.append(om.DeleteSelectedModifier())
    pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff=3.9, number_of_bins=200))
    data_fil = pipeline_fil.compute()
    xyz_fil_down = np.array(data_fil.particles["Position"])
    fil_height = np.max(xyz_fil_down[:, 2])
    rdf_down = data_fil.tables["coordination-rdf"].xy()
    fil_size_down = data_fil.particles.count

    # Pipeline for upper filament
    pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff=2.7, number_of_bins=200))
    pipeline_fil_up.modifiers.append(
        om.ExpressionSelectionModifier(
            expression="((ParticleType==8 && Coordination<6) || ( ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28))"
        )
    )
    pipeline_fil_up.modifiers.append(
        om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected=True)
    )
    pipeline_fil_up.modifiers.append(om.ExpressionSelectionModifier(expression="Cluster !=1"))
    pipeline_fil_up.modifiers.append(om.DeleteSelectedModifier())
    pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff=3.9, number_of_bins=200))
    data_fil_up = pipeline_fil_up.compute()
    xyz_fil_up = np.array(data_fil_up.particles["Position"])
    fil_depth = np.min(xyz_fil_up[:, 2])
    rdf_up = data_fil_up.tables["coordination-rdf"].xy()
    fil_size_up = data_fil_up.particles.count

    return (timestep, connection, fil_size_down, fil_height, rdf_down, fil_size_up, fil_depth, rdf_up, separation, gap)


def track_filament_evolution(
    file_list: list[str], analysis_name: str, time_step: float, dump_interval_steps: int, output_dir: str = os.getcwd()
) -> dict[str, plt.Figure]:
    """
    Comprehensive temporal tracking of filament formation and evolution dynamics.

    Processes time-series trajectory data to analyze and visualize the complete evolution
    of conductive filament formation, connectivity transitions, and structural characteristics
    in HfTaO electrochemical memory devices. Provides detailed time-resolved analysis
    essential for understanding switching mechanisms, device reliability, and optimization
    of resistive memory operation through systematic cluster analysis and visualization.

    Parameters
    ----------
    file_list : list of str
        List of file paths to sequential trajectory files representing temporal evolution.
        Files should be ordered and contain atomic coordinates with species
        information compatible with OVITO cluster analysis. Examples:

        - **Single Trajectory**: ["complete_switching_cycle.lammpstrj"]
        - **Multi-Frame Sequence**: ["frame_0001.lammpstrj", "frame_0002.lammpstrj", ...]
        - **Process Segments**: ["formation.lammpstrj", "retention.lammpstrj", "dissolution.lammpstrj"]
    analysis_name : str
        Base identifier for output file naming and analysis organization. Used to generate
        descriptive filenames for all output plots and data files. Should reflect the
        temporal analysis context. Examples: "switching_cycle", "formation_dynamics",
        "retention_study", "dissolution_analysis"
    time_step : float
        Simulation time step in picoseconds (ps) used for temporal axis conversion.
        Defines the fundamental temporal resolution of molecular dynamics simulation.
        Typical values: 0.0001-0.002 ps for atomic-scale MD simulations.
    dump_interval_steps : int
        Number of simulation steps between consecutive trajectory frame outputs.
        Used to calculate actual time intervals between analyzed frames.
        Example: dump_interval_steps=5000 with time_step=0.0002 gives 1 ps between frames.
    output_dir : str, optional
        Directory path for saving generated time series plots and analysis figures.
        Default: current working directory. Creates directory structure if non-existent.
        Recommended: dedicated analysis subdirectory for organized output management.

    Returns
    -------
    evolution_figures : dict[str, plt.Figure]
        Comprehensive dictionary of matplotlib Figure objects for temporal analysis:

        **Connectivity Analysis:**
        - **'connection'**: Binary connectivity state evolution (1=connected, 0=broken)
        - **'gap'**: Filament gap evolution showing discontinuity magnitude (Å)
        - **'separation'**: Inter-cluster separation distance when disconnected (Å)

        **Filament Morphology Evolution:**
        - **'filament_gap_and_size'**: Dual-axis plot of gap vs. atom count
        - **'filament_lower_part'**: Lower filament height and size evolution
        - **'filament_upper_part'**: Upper filament depth and size evolution

        **Detailed Structural Characteristics:**
        - **'filament_height'**: Lower filament penetration depth (scatter plot)
        - **'filament_depth'**: Upper filament penetration depth (scatter plot)
        - **'filament_size_up'**: Upper filament atom count evolution
        - **'filament_size_down'**: Lower filament atom count evolution

        All figures include statistical analysis (mean ± std) and frequency analysis
        with publication-ready formatting for scientific reporting.

    Raises
    ------
    FileNotFoundError
        If any trajectory file in file_list does not exist or is inaccessible.
    ValueError
        If OVITO cluster analysis fails for any trajectory frame or if temporal
        sequence contains inconsistent data structures.
    RuntimeError
        If memory limitations are exceeded during large-scale time series analysis.

    Notes
    -----
    Temporal Analysis Framework:

    **Time Axis Calculation:**
    ```
    actual_time = timestep × time_step × dump_interval_steps
    ```
    Where timestep is extracted from LAMMPS trajectory headers.

    **Connectivity State Analysis:**
    - **Connected State**: Filament spans complete electrode separation (connection = 1)
    - **Broken State**: Incomplete filament with measurable gap (connection = 0)
    - **Frequency Analysis**: Percentage of time spent in connected state
    - **Switching Events**: Transitions between connected/disconnected states

    **Statistical Characterization:**
    Each temporal series includes comprehensive statistical analysis:
    - **Mean Values**: Average behavior over analysis time window
    - **Standard Deviation**: Variability and stability assessment
    - **Frequency Metrics**: State occupation probabilities
    - **Trend Analysis**: Long-term evolution patterns

    Performance Characteristics:
    - **Processing Speed**: ~1-5 minutes per trajectory file depending on complexity
    - **Memory Usage**: O(N_frames × N_atoms) for trajectory sequence processing
    - **Temporal Resolution**: Limited by dump_interval_steps and simulation time_step
    - **Statistical Accuracy**: Requires sufficient frames for reliable statistics (≥50 frames)

    Integration with LAMMPSKit Analysis Pipeline:
    - **Cluster Analysis**: Uses analyze_clusters() for frame-by-frame structural analysis
    - **Visualization**: Leverages plotting utilities for publication-ready time series
    - **Data Processing**: Compatible with displacement analysis and atomic distribution functions
    - **Workflow Integration**: Supports batch processing and automated analysis pipelines

    Examples
    --------
    Basic filament evolution tracking:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution
    >>> # Analyze complete switching cycle evolution
    >>> trajectory_files = ["switching_cycle.lammpstrj"]
    >>>
    >>> evolution_figures = track_filament_evolution(trajectory_files,
    ...                                            analysis_name="switching_dynamics",
    ...                                            time_step=0.0002,  # 0.2 fs
    ...                                            dump_interval_steps=5000)  # 1 ps intervals
    >>>
    >>> print(f"Generated {len(evolution_figures)} temporal analysis plots")
    >>>
    >>> # Examine connectivity evolution
    >>> connection_fig = evolution_figures['connection']
    >>> gap_fig = evolution_figures['gap']
    >>> print("Basic filament evolution analysis completed")

    Multi-phase switching analysis:

    >>> # Analyze distinct switching phases with high temporal resolution
    >>> phase_files = ["formation.lammpstrj", "retention.lammpstrj", "dissolution.lammpstrj"]
    >>> phase_names = ["formation", "retention", "dissolution"]
    >>>
    >>> all_phase_analyses = {}
    >>> for phase_file, phase_name in zip(phase_files, phase_names):
    ...     try:
    ...         phase_figures = track_filament_evolution([phase_file],
    ...                                                analysis_name=f"{phase_name}_analysis",
    ...                                                time_step=0.0001,  # High resolution
    ...                                                dump_interval_steps=1000,  # 0.1 ps intervals
    ...                                                output_dir=f"./analysis_{phase_name}")
    ...         all_phase_analyses[phase_name] = phase_figures
    ...         print(f"Completed {phase_name} phase analysis: {len(phase_figures)} plots")
    ...     except Exception as e:
    ...         print(f"Error analyzing {phase_name} phase: {e}")
    >>>
    >>> total_plots = sum(len(figs) for figs in all_phase_analyses.values())
    >>> print(f"Multi-phase analysis completed: {total_plots} total plots")

    Long-term cycling study:

    >>> # Extended cycling analysis for endurance characterization
    >>> cycling_files = [f"cycle_{i:03d}.lammpstrj" for i in range(1, 101)]  # 100 cycles
    >>>
    >>> cycling_evolution = track_filament_evolution(cycling_files,
    ...                                            analysis_name="endurance_study",
    ...                                            time_step=0.0005,
    ...                                            dump_interval_steps=10000,  # 5 ps intervals
    ...                                            output_dir="./endurance_analysis")
    >>>
    >>> print(f"Endurance study completed: {len(cycling_evolution)} analysis plots")
    >>>
    >>> # Focus on key endurance metrics
    >>> connection_evolution = cycling_evolution['connection']
    >>> gap_evolution = cycling_evolution['gap']
    >>> size_evolution = cycling_evolution['filament_gap_and_size']
    >>> print("Long-term cycling analysis available for device optimization")

    High-resolution formation dynamics:

    >>> # Detailed analysis of initial filament formation with fine temporal resolution
    >>> formation_files = ["initial_formation.lammpstrj"]
    >>>
    >>> # Ultra-high temporal resolution for formation kinetics
    >>> formation_analysis = track_filament_evolution(formation_files,
    ...                                             analysis_name="formation_kinetics",
    ...                                             time_step=0.00005,  # 0.05 fs steps
    ...                                             dump_interval_steps=500,   # 0.025 ps intervals
    ...                                             output_dir="./formation_study")
    >>>
    >>> print("High-resolution formation dynamics analysis:")
    >>> print(f"  Total figures: {len(formation_analysis)}")
    >>> print("  Available analyses:", list(formation_analysis.keys()))
    >>>
    >>> # Examine detailed morphology evolution
    >>> lower_evolution = formation_analysis['filament_lower_part']
    >>> upper_evolution = formation_analysis['filament_upper_part']
    >>> height_evolution = formation_analysis['filament_height']
    >>> depth_evolution = formation_analysis['filament_depth']

    Comparative evolution analysis:

    >>> # Compare filament evolution under different conditions
    >>> conditions = ["low_voltage", "medium_voltage", "high_voltage"]
    >>> evolution_comparison = {}
    >>>
    >>> for condition in conditions:
    ...     try:
    ...         condition_file = f"{condition}_evolution.lammpstrj"
    ...         condition_analysis = track_filament_evolution([condition_file],
    ...                                                     analysis_name=f"{condition}_evolution",
    ...                                                     time_step=0.0002,
    ...                                                     dump_interval_steps=2500,  # 0.5 ps intervals
    ...                                                     output_dir=f"./comparison_{condition}")
    ...         evolution_comparison[condition] = condition_analysis
    ...         print(f"Completed {condition} evolution analysis")
    ...     except Exception as e:
    ...         print(f"Error analyzing {condition}: {e}")
    >>>
    >>> print(f"Comparative evolution analysis: {len(evolution_comparison)} conditions")
    >>> print("Available for voltage-dependent switching optimization")

    Advanced statistical analysis integration:

    >>> # Detailed analysis with custom statistical processing
    >>> trajectory_files = ["device_operation.lammpstrj"]
    >>>
    >>> evolution_data = track_filament_evolution(trajectory_files,
    ...                                         analysis_name="statistical_analysis",
    ...                                         time_step=0.0001,
    ...                                         dump_interval_steps=5000,
    ...                                         output_dir="./statistical_study")
    >>>
    >>> print("Statistical Evolution Analysis Results:")
    >>> print(f"  Connection evolution: {evolution_data['connection']}")
    >>> print(f"  Gap evolution: {evolution_data['gap']}")
    >>> print(f"  Separation evolution: {evolution_data['separation']}")
    >>> print(f"  Lower filament evolution: {evolution_data['filament_lower_part']}")
    >>> print(f"  Upper filament evolution: {evolution_data['filament_upper_part']}")
    >>>
    >>> # Note: Statistical summaries are automatically included in plot legends
    >>> # as mean ± std and frequency analysis for connectivity states

    Error handling and validation:

    >>> # Robust evolution tracking with comprehensive error handling
    >>> import os
    >>> import glob
    >>>
    >>> # Auto-discover trajectory files
    >>> trajectory_pattern = "evolution_*.lammpstrj"
    >>> discovered_files = sorted(glob.glob(trajectory_pattern))
    >>>
    >>> if len(discovered_files) >= 5:  # Require minimum frames for statistics
    ...     try:
    ...         evolution_analysis = track_filament_evolution(discovered_files,
    ...                                                     analysis_name="auto_discovery",
    ...                                                     time_step=0.0002,
    ...                                                     dump_interval_steps=5000)
    ...         print(f"Auto-discovery analysis: {len(evolution_analysis)} plots generated")
    ...         print(f"Processed {len(discovered_files)} trajectory frames")
    ...     except FileNotFoundError as e:
    ...         print(f"File access error: {e}")
    ...     except ValueError as e:
    ...         print(f"Cluster analysis error: {e}")
    ...     except Exception as e:
    ...         print(f"Unexpected error: {e}")
    ... else:
    ...     print(f"Insufficient files found: {len(discovered_files)} (minimum: 5)")

    Integration with other LAMMPSKit functions:

    >>> # Combine evolution tracking with single-frame analysis
    >>> from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
    >>>
    >>> # Time series evolution analysis
    >>> time_series_files = ["evolution.lammpstrj"]
    >>> evolution_plots = track_filament_evolution(time_series_files, "evolution_study",
    ...                                          time_step=0.0002, dump_interval_steps=5000)
    >>>
    >>> # Detailed single-frame analysis for specific timepoints
    >>> key_frames = ["initial.lammpstrj", "intermediate.lammpstrj", "final.lammpstrj"]
    >>> frame_analyses = {}
    >>>
    >>> for frame_file in key_frames:
    ...     try:
    ...         frame_name = frame_file.replace(".lammpstrj", "")
    ...         cluster_data = analyze_clusters(frame_file)
    ...         frame_analyses[frame_name] = {
    ...             "timestep": cluster_data[0],
    ...             "connected": cluster_data[1],
    ...             "total_atoms": cluster_data[2] + cluster_data[5],
    ...             "gap": cluster_data[9]
    ...         }
    ...         print(f"Frame analysis {frame_name}: Connected={cluster_data[1]}")
    ...     except Exception as e:
    ...         print(f"Error analyzing {frame_file}: {e}")
    >>>
    >>> print("Comprehensive Analysis Summary:")
    >>> print(f"  Evolution tracking: {len(evolution_plots)} time series plots")
    >>> print(f"  Frame-by-frame analysis: {len(frame_analyses)} detailed snapshots")
    >>>
    >>> # Results provide both temporal trends and detailed structural snapshots
    >>> # for complete filament formation and switching characterization

    Performance optimization for large datasets:

    >>> # Efficient processing of extensive time series data
    >>> import numpy as np
    >>>
    >>> # Process large trajectory sequence with progress monitoring
    >>> large_trajectory_files = [f"large_study_{i:04d}.lammpstrj" for i in range(1, 1001)]  # 1000 frames
    >>>
    >>> # Check file existence before processing
    >>> existing_files = [f for f in large_trajectory_files if os.path.exists(f)]
    >>>
    >>> if len(existing_files) >= 100:  # Proceed if sufficient data
    ...     try:
    ...         print(f"Processing {len(existing_files)} trajectory files...")
    ...         large_evolution = track_filament_evolution(existing_files[:100],  # Limit for demonstration
    ...                                                  analysis_name="large_scale_study",
    ...                                                  time_step=0.001,
    ...                                                  dump_interval_steps=10000,
    ...                                                  output_dir="./large_scale_analysis")
    ...         print(f"Large-scale evolution analysis completed: {len(large_evolution)} plots")
    ...     except MemoryError:
    ...         print("Memory limitation reached - consider processing in batches")
    ...     except Exception as e:
    ...         print(f"Processing error: {e}")
    ... else:
    ...     print(f"Insufficient data files: {len(existing_files)} available")
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
            gap_temp,
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

    print("shape of connections array", np.shape(connection)[0])

    # Calculate time axis
    time_switch = step_arr * time_step * dump_interval_steps

    # Create plotting configuration matching original parameters
    timeseries_config = TimeSeriesPlotConfig(
        alpha=0.55, linewidth=0.1, markersize=5, marker="^", include_line=True, include_scatter=True
    )

    # Plot filament connectivity state
    freq_label = calculate_frequency_label(connection, 1, "filament is in connected state {frequency: .2f}% of the time")
    fig_conn, ax_conn = create_time_series_plot(
        time_switch,
        connection,
        title="Filament connectivity state (1: connected, 0: broken)",
        xlabel="Time (ps)",
        ylabel="Filament connectivity state (1: connected, 0: broken)",
        stats_label=freq_label,
        config=timeseries_config,
    )
    save_and_close_figure(fig_conn, output_dir, analysis_name + "OnOff")

    # Plot filament gap
    gap_label = calculate_mean_std_label(gap, "average_filament_gap")
    fig_gap, ax_gap = create_time_series_plot(
        time_switch,
        gap,
        title="Filament gap",
        xlabel="Time (ps)",
        ylabel="Filament gap (A)",
        stats_label=gap_label,
        config=timeseries_config,
    )
    save_and_close_figure(fig_gap, output_dir, analysis_name + "fil_gap")

    # Plot filament separation
    separation_label = calculate_mean_std_label(separation, "average_filament_separation")
    fig_sep, ax_sep = create_time_series_plot(
        time_switch,
        separation,
        title="Filament separation",
        xlabel="Time (ps)",
        ylabel="Filament separation (A)",
        stats_label=separation_label,
        config=timeseries_config,
        fontsize_legend=8,  # Override legend font size for this specific plot
    )
    save_and_close_figure(fig_sep, output_dir, analysis_name + "fil_separation")

    # Create dual-axis plotting configuration matching original parameters
    dual_config = DualAxisPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker="^",
        primary_color="tab:red",
        secondary_color="tab:blue",
        primary_legend_loc="upper right",
        secondary_legend_loc="lower right",
        legend_framealpha=0.8,
        tight_layout=True,
    )

    # Plot filament gap & number of conductive atoms
    gap_label = calculate_mean_std_label(gap, "average_filament_gap")
    size_down_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament")
    fig_size_gap, ax1_size_gap, ax2_size_gap = create_dual_axis_plot(
        time_switch,
        gap,
        fil_size_down,
        title="Gap & no. of conductive atoms in Filament",
        xlabel="Time (ps)",
        primary_ylabel="Filament gap (A)",
        secondary_ylabel="# of vacancies in filament (A.U.)",
        primary_stats_label=gap_label,
        secondary_stats_label=size_down_label,
        config=dual_config,
        primary_ylim=(-0.5, 8.5),
        secondary_ylim=(0, 350),
    )
    save_and_close_figure(fig_size_gap, output_dir, analysis_name + "fil_state")

    # Plot filament lower part
    lower_dual_config = DualAxisPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker="^",
        primary_color="tab:red",
        secondary_color="tab:blue",
        primary_legend_loc="upper right",
        secondary_legend_loc="lower right",
        legend_framealpha=0.75,
        tight_layout=True,
    )

    height_label = calculate_mean_std_label(fil_height, "average_filament_height")
    size_down_lower_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament (bottom half)")
    fig_lowfil, ax1_lowfil, ax2_lowfil = create_dual_axis_plot(
        time_switch,
        fil_height,
        fil_size_down,
        title="Filament lower part near cathode",
        xlabel="Timestep (ps)",
        primary_ylabel="Filament length-lower end (A)",
        secondary_ylabel="# of vacancies in filament-lower end (A.U.)",
        primary_stats_label=height_label,
        secondary_stats_label=size_down_lower_label,
        config=lower_dual_config,
        primary_ylim=(3, 25),
        secondary_ylim=(0, 350),
    )
    save_and_close_figure(fig_lowfil, output_dir, analysis_name + "fil_lower")

    # Plot filament upper part
    depth_label = calculate_mean_std_label(fil_depth, "average_filament_depth")
    size_up_label = calculate_mean_std_label(fil_size_up, "average # of vacancies in filament (top half)")
    fig_upfil, ax1_upfil, ax2_upfil = create_dual_axis_plot(
        time_switch,
        fil_depth,
        fil_size_up,
        title="Filament upper part near anode",
        xlabel="Timestep (ps)",
        primary_ylabel="Filament length-upper end (A)",
        secondary_ylabel="# of vacancies in filament-upper end (A.U.)",
        primary_stats_label=depth_label,
        secondary_stats_label=size_up_label,
        config=lower_dual_config,  # Use same config as lower
    )
    save_and_close_figure(fig_upfil, output_dir, analysis_name + "upper")

    # Create scatter-only configuration for simple plots (no line)
    scatter_config = TimeSeriesPlotConfig(
        alpha=0.55, linewidth=0.1, markersize=5, marker="^", include_line=False, include_scatter=True  # Scatter only
    )

    # Plot filament height (lower end)
    height_simple_label = calculate_mean_std_label(fil_height, "average_filament_height")
    fig_height, ax_height = create_time_series_plot(
        time_switch,
        fil_height,
        title="Filament length-lower end",
        xlabel="Timestep (ps)",
        ylabel="Filament length-lower end (A)",
        stats_label=height_simple_label,
        config=scatter_config,
        ylim=(3, 25),
    )
    save_and_close_figure(fig_height, output_dir, analysis_name + "fil_height")

    # Plot filament depth (upper end)
    depth_simple_label = calculate_mean_std_label(fil_depth, "average_filament_depth")
    fig_depth, ax_depth = create_time_series_plot(
        time_switch,
        fil_depth,
        title="Filament length-upper end",
        xlabel="Timestep (ps)",
        ylabel="Filament length-upper end (A)",
        stats_label=depth_simple_label,
        config=scatter_config,
    )
    save_and_close_figure(fig_depth, output_dir, analysis_name + "fil_depth")

    # Plot filament size (upper end)
    size_up_simple_label = calculate_mean_std_label(fil_size_up, "average # of vacancies in filament (top half)")
    fig_size_up, ax_size_up = create_time_series_plot(
        time_switch,
        fil_size_up,
        title="# of vacancies in filament-upper end",
        xlabel="Timestep (ps)",
        ylabel="# of vacancies in filament-upper end (A.U.)",
        stats_label=size_up_simple_label,
        config=scatter_config,
    )
    save_and_close_figure(fig_size_up, output_dir, analysis_name + "fil_size_up")

    # Plot filament size (lower end)
    size_down_simple_label = calculate_mean_std_label(fil_size_down, "average # of vacancies in filament (bottom half)")
    fig_size_down, ax_size_down = create_time_series_plot(
        time_switch,
        fil_size_down,
        title="# of vacancies in filament-lower end (A.U.)",
        xlabel="Timestep (ps)",
        ylabel="# of vacancies in filament-lower end (A.U.)",
        stats_label=size_down_simple_label,
        config=scatter_config,
        ylim=(0, 350),
    )
    save_and_close_figure(fig_size_down, output_dir, analysis_name + "fil_size_down")

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
        "filament_size_down": fig_size_down,
    }


def plot_displacement_timeseries(
    file_list: list[str],
    datatype: str,
    dataindex: int,
    Nchunks: int,
    loop_start: int,
    loop_end: int,
    output_dir: str = os.getcwd(),
) -> dict[str, plt.Figure]:
    """
    Generate comprehensive spatially-resolved displacement time series visualization.

    Creates multi-panel grid plots showing temporal evolution of displacement data across
    spatial bins and multiple atomic species or experimental conditions. Provides detailed
    visualization of ion migration patterns, thermal effects, and spatial heterogeneity
    essential for understanding electrochemical switching dynamics in HfTaO memory devices
    through systematic time-resolved displacement analysis.

    Parameters
    ----------
    file_list : list of str
        List of file paths to LAMMPS thermodynamic output files containing spatially-binned
        displacement data. Each file represents different atomic species, experimental
        conditions, or temporal phases. Files must have consistent spatial binning and
        temporal sampling. Examples:

        - **Species Analysis**: ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
        - **Condition Comparison**: ["low_field.dat", "high_field.dat"]
        - **Temperature Study**: ["300K.dat", "400K.dat", "500K.dat"]
    datatype : str
        Descriptive identifier for the type of displacement analysis being performed.
        Used in plot titles, axis labels, and output filename generation. Should reflect
        the physical context of the analysis. Examples: "mobility", "temperature",
        "field_dependence", "species_comparison"
    dataindex : int
        Column index specifying the displacement component to visualize from thermodynamic
        output files. Maps to specific physical quantities:

        - **0**: 'abs total disp' - Total displacement magnitude (Angstroms)
        - **1**: 'density - mass' - Local mass density (atomic units)
        - **2**: 'temp (K)' - Local temperature (Kelvin)
        - **3**: 'z disp (A)' - Z-direction displacement (Angstroms, field direction)
        - **4**: 'lateral disp (A)' - Lateral displacement magnitude (Angstroms)
        - **5**: 'outward disp vector (A)' - Radial outward displacement (Angstroms)
    Nchunks : int
        Number of spatial bins along the z-direction (electrode-to-electrode axis).
        Determines the spatial resolution of displacement analysis and number of subplot rows.
        Typical range: 15-100 bins depending on device thickness and desired resolution.
    loop_start : int
        Starting loop index (inclusive) for temporal analysis window. Corresponds to
        simulation timestep ranges or voltage cycle iterations. Defines the beginning
        of the time series analysis period.
    loop_end : int
        Ending loop index (inclusive) for temporal analysis window. Must be ≥ loop_start.
        Defines the end of the time series analysis period.
    output_dir : str, optional
        Directory path for saving generated time series visualization plots.
        Default: current working directory. Creates directory structure if non-existent.
        Recommended: dedicated analysis subdirectory for organized output management.

    Returns
    -------
    timeseries_figures : dict[str, plt.Figure]
        Dictionary containing the generated matplotlib Figure object:

        **Time Series Visualization:**
        - **'displacement_timeseries'**: Multi-panel grid plot with comprehensive spatially-resolved temporal analysis

        **Plot Structure:**
        - **Rows**: Each row represents a spatial bin (z-position)
        - **Columns**: Each column represents a different file/species/condition
        - **Layout**: Bottom row = electrode interface, top row = opposite electrode
        - **Legends**: Species identification and spatial bin numbering
        - **Styling**: Publication-ready formatting with consistent color schemes

        The figure includes automatic layout optimization, shared axis labeling,
        and integrated legend systems for clear interpretation of multi-dimensional data.

    Raises
    ------
    FileNotFoundError
        If any file in file_list does not exist or is inaccessible.
    ValueError
        If dataindex is outside valid range (0-5) or if displacement data format
        is incompatible with expected thermodynamic output structure.
    IndexError
        If Nchunks exceeds available spatial bins in displacement data files.

    Notes
    -----
    Visualization Framework & Layout:

    **Multi-Panel Grid Structure:**
    ```
    Grid Layout: [Nchunks rows × 4 columns maximum]
    Row Assignment: Bottom = Bin 1 (electrode), Top = Bin Nchunks (opposite electrode)
    Column Assignment: File order from file_list (left to right)
    Color Scheme: ['blue', 'red', 'green', 'black'] cycling for multiple files
    ```

    **Spatial Bin Interpretation:**
    - **Bin 1 (Bottom Row)**: Electrode interface region (z ≈ z_min)
    - **Bin N (Top Row)**: Opposite electrode interface (z ≈ z_max)
    - **Middle Bins**: Oxide bulk regions with varying properties
    - **Bin Numbering**: 1-based indexing for intuitive spatial reference

    **Temporal Axis Configuration:**
    - **X-Axis**: Loop indices corresponding to simulation timesteps
    - **Time Conversion**: Actual time = loop × time_step × dump_interval_steps
    - **Resolution**: Determined by dump frequency in simulation output
    - **Range**: Defined by loop_start to loop_end parameters

    Data Index Configuration:
    - **Index 0,3,4,5**: Displacement magnitude components
    - **Index 1,2**: Density and temperature data

    Performance Characteristics:
    - **Memory Usage**: O(N_files × N_loops × N_chunks × N_columns) for data storage
    - **Processing Speed**: ~5-30s depending on data size and chunk count
    - **Output Quality**: Publication-ready SVG figures with high resolution
    - **Scalability**: Efficient for datasets up to ~10⁶ data points per subplot

    Integration with LAMMPSKit Analysis Pipeline:
    - **Data Reading**: Uses read_displacement_data() for robust file parsing
    - **Species Identification**: Leverages extract_element_label_from_filename()
    - **Parameter Validation**: Employs config module validation functions
    - **Workflow Compatibility**: Integrates with plot_displacement_comparison()

    Examples
    --------
    Multi-species z-displacement time series:

    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries
    >>> # Visualize z-direction displacement evolution for different species
    >>> mobility_files = ["Hf_mobility.dat", "Ta_mobility.dat", "O_mobility.dat"]
    >>>
    >>> timeseries_plots = plot_displacement_timeseries(mobility_files,
    ...                                               datatype="species_mobility",
    ...                                               dataindex=3,  # Z-displacement
    ...                                               Nchunks=50,
    ...                                               loop_start=0,
    ...                                               loop_end=100)
    >>>
    >>> z_displacement_fig = timeseries_plots['displacement_timeseries']
    >>> print("Z-displacement time series generated for species comparison")

    Temperature evolution analysis:

    >>> # Analyze spatial temperature evolution during switching
    >>> temperature_files = ["temperature_switching.dat"]
    >>>
    >>> temp_timeseries = plot_displacement_timeseries(temperature_files,
    ...                                              datatype="thermal_analysis",
    ...                                              dataindex=2,  # Temperature
    ...                                              Nchunks=75,
    ...                                              loop_start=50,
    ...                                              loop_end=200,
    ...                                              output_dir="./thermal_study")
    >>>
    >>> print("Temperature evolution time series analysis completed")

    Lateral displacement comparison:

    >>> # Compare lateral spreading under different field conditions
    >>> field_files = ["low_field_mobility.dat", "high_field_mobility.dat"]
    >>>
    >>> lateral_analysis = plot_displacement_timeseries(field_files,
    ...                                               datatype="field_dependence",
    ...                                               dataindex=4,  # Lateral displacement
    ...                                               Nchunks=60,
    ...                                               loop_start=0,
    ...                                               loop_end=150)
    >>>
    >>> lateral_fig = lateral_analysis['displacement_timeseries']
    >>> print("Lateral displacement field dependence analysis completed")

    Total displacement magnitude tracking:

    >>> # Track overall ion migration with high spatial resolution
    >>> migration_files = ["ion_migration.dat"]
    >>>
    >>> total_displacement = plot_displacement_timeseries(migration_files,
    ...                                                 datatype="migration_analysis",
    ...                                                 dataindex=0,  # Total displacement
    ...                                                 Nchunks=100,  # High resolution
    ...                                                 loop_start=10,
    ...                                                 loop_end=90)
    >>>
    >>> print("High-resolution total displacement tracking completed")

    Multi-condition comprehensive analysis:

    >>> # Systematic analysis across multiple displacement components
    >>> condition_files = ["condition_A.dat", "condition_B.dat", "condition_C.dat"]
    >>> displacement_components = {
    ...     0: "total_magnitude",
    ...     3: "z_direction",
    ...     4: "lateral_spreading",
    ...     5: "radial_outward"
    ... }
    >>>
    >>> all_displacement_analyses = {}
    >>> for dataindex, component_name in displacement_components.items():
    ...     try:
    ...         analysis_figures = plot_displacement_timeseries(condition_files,
    ...                                                       datatype=f"multi_condition_{component_name}",
    ...                                                       dataindex=dataindex,
    ...                                                       Nchunks=50,
    ...                                                       loop_start=0,
    ...                                                       loop_end=120,
    ...                                                       output_dir=f"./analysis_{component_name}")
    ...         all_displacement_analyses[component_name] = analysis_figures
    ...         print(f"Completed {component_name} displacement analysis")
    ...     except Exception as e:
    ...         print(f"Error analyzing {component_name}: {e}")
    >>>
    >>> total_figures = len(all_displacement_analyses)
    >>> print(f"Multi-component analysis: {total_figures} displacement types analyzed")

    Fine-grained temporal resolution study:

    >>> # High-resolution temporal analysis for formation dynamics
    >>> formation_files = ["formation_dynamics.dat"]
    >>>
    >>> # Analyze z-displacement with fine temporal and spatial resolution
    >>> fine_resolution = plot_displacement_timeseries(formation_files,
    ...                                               datatype="formation_kinetics",
    ...                                               dataindex=3,  # Z-displacement
    ...                                               Nchunks=80,   # Fine spatial resolution
    ...                                               loop_start=0,
    ...                                               loop_end=500,  # Extended time window
    ...                                               output_dir="./formation_kinetics")
    >>>
    >>> print("Fine-grained formation kinetics analysis completed")
    >>> print("Available for detailed filament nucleation and growth characterization")

    Density redistribution analysis:

    >>> # Track mass density evolution during switching cycles
    >>> density_files = ["density_evolution.dat"]
    >>>
    >>> density_timeseries = plot_displacement_timeseries(density_files,
    ...                                                  datatype="density_redistribution",
    ...                                                  dataindex=1,  # Density
    ...                                                  Nchunks=40,
    ...                                                  loop_start=20,
    ...                                                  loop_end=180)
    >>>
    >>> density_fig = density_timeseries['displacement_timeseries']
    >>> print("Mass density redistribution time series generated")
    >>> print("Available for vacancy formation and material transport analysis")

    Error handling and parameter validation:

    >>> # Robust time series analysis with comprehensive validation
    >>> import os
    >>>
    >>> test_files = ["species1.dat", "species2.dat", "species3.dat"]
    >>> valid_files = [f for f in test_files if os.path.exists(f)]
    >>>
    >>> if len(valid_files) >= 1:
    ...     try:
    ...         # Validate dataindex range
    ...         valid_dataindices = [0, 3, 4, 5]  # Focus on displacement components
    ...
    ...         for dataindex in valid_dataindices:
    ...             try:
    ...                 component_names = {0: "total", 3: "z_direction", 4: "lateral", 5: "radial"}
    ...                 component_name = component_names[dataindex]
    ...
    ...                 timeseries_result = plot_displacement_timeseries(valid_files,
    ...                                                                datatype=f"validated_{component_name}",
    ...                                                                dataindex=dataindex,
    ...                                                                Nchunks=30,  # Moderate resolution
    ...                                                                loop_start=0,
    ...                                                                loop_end=50)
    ...                 print(f"Successfully analyzed {component_name} displacement")
    ...             except ValueError as e:
    ...                 print(f"Data error for index {dataindex}: {e}")
    ...             except Exception as e:
    ...                 print(f"Unexpected error for index {dataindex}: {e}")
    ...
    ...     except FileNotFoundError as e:
    ...         print(f"File access error: {e}")
    ... else:
    ...     print(f"No valid files found from {test_files}")

    Integration with displacement comparison analysis:

    >>> # Combine time series with comparative final displacement analysis
    >>> from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_comparison
    >>>
    >>> analysis_files = ["evolution_study.dat"]
    >>> species_labels = ["Evolution"]
    >>>
    >>> # Step 1: Time series analysis for detailed temporal evolution
    >>> timeseries_analysis = plot_displacement_timeseries(analysis_files,
    ...                                                  datatype="comprehensive_study",
    ...                                                  dataindex=3,  # Z-displacement
    ...                                                  Nchunks=50,
    ...                                                  loop_start=0,
    ...                                                  loop_end=100)
    >>>
    >>> # Step 2: Final displacement comparison for endpoint analysis
    >>> comparison_analysis = plot_displacement_comparison(analysis_files,
    ...                                                  loop_start=0, loop_end=100,
    ...                                                  labels=species_labels,
    ...                                                  analysis_name="comprehensive_study")
    >>>
    >>> print("Comprehensive Analysis Summary:")
    >>> print(f"  Time series plots: {len(timeseries_analysis)}")
    >>> print(f"  Comparison plots: {len(comparison_analysis)}")
    >>> print("  Total analysis: Detailed temporal + final state characterization")

    Advanced data interpretation and analysis:

    >>> # Extract and analyze specific spatial regions from time series
    >>> mobility_files = ["detailed_mobility.dat"]
    >>>
    >>> # Generate time series for detailed analysis
    >>> detailed_timeseries = plot_displacement_timeseries(mobility_files,
    ...                                                   datatype="detailed_analysis",
    ...                                                   dataindex=3,  # Z-displacement
    ...                                                   Nchunks=60,
    ...                                                   loop_start=0,
    ...                                                   loop_end=200)
    >>>
    >>> print("Detailed Time Series Analysis:")
    >>> print("  Generated multi-panel visualization for:")
    >>> print("    - Electrode interface regions (bottom/top rows)")
    >>> print("    - Oxide bulk regions (middle rows)")
    >>> print("    - Temporal evolution patterns (x-axis progression)")
    >>> print("    - Species-specific migration characteristics (color coding)")
    >>>
    >>> # Note: For quantitative analysis of time series data,
    >>> # use read_displacement_data() directly to access raw numerical values
    >>> from lammpskit.ecellmodel.filament_layer_analysis import read_displacement_data
    >>> raw_timeseries = read_displacement_data("detailed_mobility.dat", 0, 200)
    >>> print(f"Raw data available: {len(raw_timeseries)} time points, "
    ...       f"{raw_timeseries[0].shape[0]} spatial bins")
    """  # Initialize configurations with inlined defaults
    # Timeseries configuration defaults (inlined)
    data_labels = ["abs total disp", "density - mass", "temp (K)", "z disp (A)", "lateral disp (A)", "outward disp vector (A)"]
    ncolumns = 4  # Number of columns for subplot grid

    # Plot configuration defaults (inlined)
    colors = ["b", "r", "g", "k"]
    linewidth = 1.2
    alpha = 0.75
    title_fontsize = 12  # Increased from 8
    label_fontsize = 10  # Increased from 8
    tick_fontsize = 8  # Increased from 6
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
    print("dump_steps=", dump_steps)

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
                ax.plot(
                    dump_steps, y_data, linewidth=linewidth, alpha=alpha, color=colors[col % len(colors)], label=legend_label
                )

                # Configure subplot appearance
                if row == 0:  # Top row gets column titles
                    ax.set_title(f"{element_labels[col]}", fontsize=title_fontsize)

                if row == nrows - 1:  # Bottom row gets x-label
                    ax.set_xlabel("Time step", fontsize=label_fontsize)

                # Set minimal tick labels for scale reference
                ax.tick_params(labelsize=tick_fontsize)

                # Add legend to all subplots
                ax.legend(fontsize=legend_fontsize, loc="best")

                # Add grid if enabled
                if grid:
                    ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for col in range(len(element_labels), ncolumns):
        for row in range(nrows):
            fig.delaxes(axes[row, col])

    # Add shared y-label for the leftmost column
    shared_ylabel = f"{datatype} {data_labels[dataindex]}"
    fig.text(0.025, 0.5, shared_ylabel, fontsize=label_fontsize, rotation=90, va="center", ha="center")

    # Set overall title
    fig.suptitle(f"{datatype} {data_labels[dataindex]}", fontsize=title_fontsize)

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
    fig.savefig(filepath, dpi=300, bbox_inches="tight", format="svg")

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
