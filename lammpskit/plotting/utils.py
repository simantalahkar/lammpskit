"""
General plotting utilities for LAMMPSKit.

This module contains general-purpose plotting functions that can be reused
across different analysis workflows and simulation types.
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
    Plots multiple cases with the given x and y arrays, labels, and saves the figure.

    This is a general-purpose plotting function that can handle various data dimensions
    and provides consistent styling across different analysis workflows.

    Parameters
    ----------
    x_arr : np.ndarray
        Array(s) of x values for each case.
    y_arr : np.ndarray
        Array(s) of y values for each case.
    labels : list of str
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
