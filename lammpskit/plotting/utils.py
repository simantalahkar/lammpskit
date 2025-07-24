"""
General-purpose plotting utilities for scientific visualization.

This module provides flexible plotting functions for creating publication-ready figures
across different analysis workflows. Functions are designed for reusability with
consistent styling and support for both simple and complex data visualizations.

Key Features
------------
- Multi-dimensional array handling for comparative analysis
- Automatic styling with scientific color schemes and markers
- Flexible axis control and customization options
- Publication-ready output in multiple formats (PDF, SVG)
- Memory-efficient plotting for large datasets

Design Philosophy
-----------------
Functions prioritize flexibility over rigid interfaces, using **kwargs for extensive
customization. This approach supports diverse scientific plotting needs while maintaining
consistent visual output across the LAMMPSKit ecosystem.

Styling Standards
-----------------
- Color palette: ['b', 'r', 'g', 'k'] (blue, red, green, black)
- Line styles: ['--', '-.', ':', '-'] (dashed, dash-dot, dotted, solid)  
- Markers: ['o', '^', 's', '*'] (circle, triangle, square, star)
- Font sizes: 8pt labels, 7pt legends, 7pt ticks for compact publication layout

Performance Notes
-----------------
Memory usage scales with data size and number of cases. For large datasets (>10^5 points),
consider data downsampling before plotting. Figure generation is optimized for batch
processing workflows.

Examples
--------
Simple comparative plot:

>>> import numpy as np
>>> from lammpskit.plotting import plot_multiple_cases
>>> x = np.array([1, 2, 3])
>>> y = np.array([[1, 4, 9], [1, 8, 27]])  # Two cases as an example
>>> labels = ['Case 1', 'Case 2']
>>> fig = plot_multiple_cases(x, y, labels, 'X values', 'Y values', 'comparison', 8, 6)

Electrochemical analysis plot:

>>> z_bins = np.linspace(-10, 40, 50)  # Electrode-to-electrode z positions
>>> atom_counts = np.array([[10, 15, 20], [5, 12, 18]])  # Hf, O, Ta counts
>>> labels = ['SET state', 'RESET state']
>>> fig = plot_multiple_cases(atom_counts, z_bins, labels, 
...                          'Atom count', 'Z position (Å)', 'distribution', 10, 8)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_multiple_cases(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    labels: List[str],
    xlabel: str,
    ylabel: str,
    output_filename: str,
    xsize: float,
    ysize: float,
    output_dir: str = os.getcwd(),
    **kwargs
) -> plt.Figure:
    """
    Create comparative plots for multiple datasets with publication-ready styling.

    Versatile plotting function for scientific data visualization supporting various array
    dimensions and comparison scenarios. Handles both single-case and multi-case analysis
    with automatic styling, customizable limits, and dual-format output. Optimized for
    electrochemical cell analysis and general MD simulation data visualization.

    Parameters
    ----------
    x_arr : np.ndarray
        X-axis data for plotting. Supports multiple dimensions:
        - 1D: Single x-series for all cases
        - 2D: Different x-series for each case (shape: n_cases, n_points)
    y_arr : np.ndarray  
        Y-axis data for plotting. Supports multiple dimensions:
        - 1D: Single y-series (used with single case or shared across cases)
        - 2D: Different y-series for each case (shape: n_cases, n_points)
    labels : List[str]
        Legend labels for each case. Length should match number of cases in data arrays.
    xlabel : str
        X-axis label with units. Example: 'Z position (Å)', 'Time (ps)'
    ylabel : str
        Y-axis label with units. Example: 'Atom count', 'Displacement (Å)'
    output_filename : str
        Base filename for saved figures (extensions added automatically).
        Example: 'atomic_distribution', 'filament_evolution'
    xsize : float
        Figure width in inches. Note: Function overrides with hardcoded value (1.6).
    ysize : float  
        Figure height in inches. Note: Function overrides with hardcoded value (3.2).
    output_dir : str, optional
        Output directory for saved figures. Created if doesn't exist (default: cwd).

    **kwargs : dict, optional
        Advanced customization options:

        Axis Limits:
            xlimit : tuple (xmin, xmax) - Set both x-axis limits
            ylimit : tuple (ymin, ymax) - Set both y-axis limits  
            xlimitlo : float - Set x-axis lower limit only
            xlimithi : float - Set x-axis upper limit only
            ylimitlo : float - Set y-axis lower limit only
            ylimithi : float - Set y-axis upper limit only

        Reference Lines:
            xaxis : bool - Add horizontal line at y=0
            yaxis : bool - Add vertical line at x=0

        Styling:
            markerindex : int - Override automatic color/marker cycling

        Statistical Analysis:
            ncount : np.ndarray - Atom counts per bin for average calculations
                   Shape: (n_cases, n_bins). Prints weighted averages.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for further customization or display.
        Note: Figure is automatically saved and closed for memory efficiency.

    Notes
    -----
    Array Dimension Handling:
    - x_arr.ndim=1, y_arr.ndim=1: Single case plot
    - x_arr.ndim=1, y_arr.ndim=2: Shared x-axis, multiple y-series  
    - x_arr.ndim=2, y_arr.ndim=1: Multiple x-series, shared y-axis
    - x_arr.ndim=2, y_arr.ndim=2: Full multi-case plot (most common)

    Performance Characteristics:
    - Memory usage: O(max(x_size, y_size)) 
    - Rendering time: O(n_cases * n_points)
    - File I/O: Dual output (PDF + SVG) for versatility

    Output Format:
    - PDF: Vector format for publications and presentations
    - SVG: Web-compatible vector format for interactive displays
    - Both saved with tight bounding boxes for clean appearance

    Common Usage Patterns in LAMMPSKit:
    -----------------------------------
    Electrochemical analysis (atom distributions):
    >>> plot_multiple_cases(distributions['hafnium'], z_bin_centers, labels, 
    ...                    'Hf atoms #', 'z position (A)', 'hf_distribution', 8, 6)

    Displacement analysis:
    >>> plot_multiple_cases(zdisp, binposition, labels, 
    ...                    'z displacement (A)', 'z position (A)', 'z_disp', 8, 6,
    ...                    yaxis=True)  # Add y=0 reference line

    Charge distribution with axis limits:
    >>> plot_multiple_cases(charge_data, z_positions, labels,
    ...                    'Net charge', 'z position (A)', 'charge_dist', 8, 6,
    ...                    ylimithi=70, xlimithi=15, xlimitlo=-20)

    Examples
    --------
    Basic multi-case comparison:

    >>> import numpy as np
    >>> z_pos = np.linspace(0, 30, 50)  # Electrode positions
    >>> hf_counts = np.array([[5, 10, 15], [8, 12, 18]])  # Two voltage states
    >>> labels = ['0.5V', '1.0V'] 
    >>> fig = plot_multiple_cases(hf_counts, z_pos, labels,
    ...                          'Hf atom count', 'Z position (Å)', 
    ...                          'hafnium_analysis', 10, 8)

    Single case with reference lines:

    >>> displacement = np.random.normal(0, 1, 100)
    >>> positions = np.linspace(-10, 40, 100)
    >>> fig = plot_multiple_cases(displacement, positions, ['Displacement'],
    ...                          'Displacement (Å)', 'Z position (Å)',
    ...                          'displacement_profile', 8, 6, 
    ...                          yaxis=True, xaxis=True)

    Multi-dimensional array example:

    >>> # 3 cases, 4 elements each
    >>> element_counts = np.random.randint(1, 20, (3, 4))  
    >>> elements = ['Hf', 'Ta', 'O', 'Electrode']
    >>> case_labels = ['SET', 'Intermediate', 'RESET']
    >>> fig = plot_multiple_cases(element_counts, elements, case_labels,
    ...                          'Element count', 'Element type',
    ...                          'element_comparison', 12, 8)
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
            # Calculate weighted average for statistical reporting
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
