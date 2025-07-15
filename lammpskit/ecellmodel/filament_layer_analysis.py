

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

def read_structure_info(filepath):
    """Reads the structure file and returns the 
    timestep, total number of atoms, and the box dimensions."""
    skip = 1
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            for j in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"File is empty: {filepath}")
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for Timestep: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                timestep = int(c[0])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed timestep line in file: {filepath} (got: {line.strip()})")
            
            for j in range(skip):
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
        
            for j in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"Missing section for box bounds: {filepath}")
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

def read_coordinates(file_list, skip_rows, columns_to_read):        ## Calls read_structure_info(...)
    """Reads the structure files and returns the 
    simulation cell parameters along with coordinates 
    and timestep array."""
    
    print(file_list)
    if not file_list:
        raise ValueError("file_list is empty. No files to process.")
    timestep_arr = []
    coordinates = []
    for filepath in file_list:
        timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info(filepath)       ## All these values are expected to be the same other than the timestep for this analysis
        timestep_arr.append(timestep)
        try:
            coords = np.loadtxt(filepath, delimiter=' ', comments='#', skiprows=skip_rows, max_rows=total_atoms, usecols=columns_to_read)
        except ValueError as e:
            raise ValueError(f"Column index out of range or Non-float atomic data in file: {filepath} (error: {e})\n (column indices provided to read = {columns_to_read})")
        if coords.shape[0] != total_atoms:
            raise EOFError(f"File {filepath} has fewer atom lines ({coords.shape[0]}) than expected ({total_atoms})")
        coordinates.append(coords)
    return np.array(coordinates), np.array(timestep_arr), total_atoms, xlo, xhi, ylo, yhi, zlo, zhi

def read_displacement_data(filepath, loop_start, loop_end, repeat_count=0):
    """Reads the displacement data from the binwise averaged output data file 
    and returns the displacement data for the specified loops."""
    print(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if loop_start > loop_end:
        raise ValueError(f"loop_start ({loop_start}) is greater than loop_end ({loop_end})")
    try:
        tmp = np.loadtxt(filepath, comments='#', skiprows=3, max_rows=1)
    except ValueError:
        raise TypeError("Malformed Nchunks line in file: {filepath}")
    
    #print(tmp)
    try:
        Nchunks = tmp[1].astype(int)
    except (IndexError, ValueError) as e:
        if isinstance(e, IndexError):
            raise EOFError(f"Missing Nchunks line in file: {filepath}") from e
        elif isinstance(e, ValueError):
            raise TypeError("Malformed Nchunks line in file: {filepath}") from e

    
    #print(Nchunks,loop_start,loop_end) 
    thermo = []
    for n in range(loop_start,loop_end+1):
      #step_time.append([n*DUMP_INTERVAL_STEPS,n*DUMP_INTERVAL_STEPS*TIME_STEP])
      #print(Nchunks,loop_start,loop_end,n) 
        try:
            chunk = np.loadtxt(filepath, comments='#', skiprows=3 + 1 + (n - loop_start) * (Nchunks + 4), max_rows=Nchunks)
        except ValueError:
            raise EOFError(f"Missing or malformed chunk data for loop {n} in file: {filepath}")
        if chunk.shape[0] != Nchunks:
            raise EOFError(f"Not enough data for chunk {n} in file: {filepath}")
        thermo.append(chunk)
#      print(n)
      #if n == 0:
        #print(np.shape(thermo),'\n',thermo[-1])   
    #print(np.shape(thermo),'\n',thermo[-1]) 
    return thermo
    #step_time = np.array(step_time)
    

def plot_multiple_cases(x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=os.getcwd(), **kwargs):  
    """Plots the cases with the given x and y arrays, 
    labels, and saves the figure.""" 
    nrows = 1
    ncolumns = 1
    xsize=1.6
    ysize=3.2
    print('before subplots')
    plt.ioff()
    fig,axes = plt.subplots(nrows,ncolumns,squeeze=False,constrained_layout=False,figsize=(xsize,ysize))
    print('before axes flatten')
    axes=axes.flatten()
    print('before tight layout')
    fig.tight_layout()
    #plt.rcParams['xtick.labelsize'] = 6
    #plt.rcParams['ytick.labelsize'] = 6
    #print(axes)
    colorlist = ['b', 'r', 'g','k']
    linestylelist = ['--', '-.', ':','-']
    markerlist = ['o', '^', 's', '*']
    print('reached now plotting point')
    if x_arr.ndim >1 and y_arr.ndim >1:
        for i in range(len(x_arr)):
            if 'markerindex' in kwargs:
                j = kwargs['markerindex']
            else:
                j = i
            axes[0].plot(x_arr[i], y_arr[i], label=labels[i], color = colorlist[j], linestyle=linestylelist[j], marker = markerlist[j], markersize=5, linewidth = 1.2, alpha = 0.75)
    elif x_arr.ndim >1 and y_arr.ndim ==1:
        for i in range(len(x_arr)):
            if 'markerindex' in kwargs:
                j = kwargs['markerindex']
            else:
                j = i
            axes[0].plot(x_arr[i], y_arr, label=labels[i], color = colorlist[j], linestyle=linestylelist[j], marker = markerlist[j], markersize=5, linewidth = 1.2, alpha = 0.75)
    elif x_arr.ndim ==1 and y_arr.ndim >1:
        for i in range(len(y_arr)):
            if 'markerindex' in kwargs:
                j = kwargs['markerindex']
            else:
                j = i
            axes[0].plot(x_arr, y_arr[i], label=labels[i], color = colorlist[j], linestyle=linestylelist[j], marker = markerlist[j], markersize=5, linewidth = 1.2, alpha = 0.75)
    else:
        if 'markerindex' in kwargs:
            j = kwargs['markerindex']
        else:
            j = 0        
        axes[0].plot(x_arr, y_arr, label=labels, color = colorlist[j], linestyle=linestylelist[j], marker = markerlist[j], markersize=5, linewidth = 1.2, alpha = 0.75)
    if 'ncount' in kwargs:
        atoms_per_bin_count = kwargs['ncount']
        for i in range(len(x_arr)):
            Ncount_temp = atoms_per_bin_count[i,atoms_per_bin_count[i]>0]
            x_arr_temp = x_arr[i,atoms_per_bin_count[i]>0]
            average = np.sum(x_arr[i]*atoms_per_bin_count[i])/np.sum(atoms_per_bin_count[i])
            print(f'\n The average for {labels[i]} in {output_filename} is {average} \n')
                      

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
    axes[0].adjustable='datalim'
    axes[0].set_aspect('auto')
    axes[0].tick_params(axis='both', which='major', labelsize=7)
    axes[0].set_aspect('auto')
    #axes.set_xticklabels(ax.get_xticks(), fontsize=6)
    axes[0].set_xlabel(xlabel, fontsize=8)
#    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)

      
    #plt.suptitle(f'{datatype} {dataindexname[dataindex]}', fontsize = 8)
    #plt.show()
    
    plt.ioff()
    print('reached file saving point')
    output_filename_pdf = output_filename + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename_pdf)
    fig.savefig(savepath, bbox_inches='tight', format='pdf')#,dpi=300)#, )
    output_filename_svg = output_filename + '.svg'
    savepath = os.path.join(output_dir, output_filename_svg)
    fig.savefig(savepath, bbox_inches='tight', format='svg')
    plt.close()  
    return fig  # Return the figure object for further use if needed
 
def plot_atomic_distribution(file_list,labels,skip_rows,z_bins,analysis_name,output_dir=os.getcwd(),**kwargs):     ## Calls read_coordinates(...) and plot_multiple_cases(...)
    """Reads the coordinates from the file_list, calculates the atomic distributions,
    and plots the distributions for O, Hf, Ta, and all M atoms."""
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
    z_bin_width = (zhi-zlo)/z_bins
    z_bin_centers = np.linspace(zlo+z_bin_width/2, zhi-z_bin_width/2, z_bins)
    oxygen_distribution = []
    hafnium_distribution = []
    tantalum_distribution = []
    print('\nshape of coordinate_arr=', np.shape(coordinates_arr), '\nlength of coordinate_arr=', len(coordinates_arr))
    for i in range(len(coordinates_arr[:])):
        coordinates = coordinates_arr[i]
        coordinates = coordinates[coordinates[:, 5].argsort()]
#        print(coordinates[:,5])

        hf_atoms = coordinates[coordinates[:, 1]==2]
        ta_atoms = coordinates[np.logical_or(coordinates[:, 1]==4, np.logical_or(coordinates[:, 1]==6, coordinates[:, 1]==10))]
#       print(np.shape(ta_atoms),ta_atoms[:,5])
        o_atoms = coordinates[np.logical_or(coordinates[:, 1]==1, np.logical_or(coordinates[:, 1]==3, np.logical_or(coordinates[:, 1]==5, coordinates[:, 1]==9)))]
        
        hafnium_distribution.append(np.histogram(hf_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
#        print(np.shape(hafnium_distribution))
        oxygen_distribution.append(np.histogram(o_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
        tantalum_distribution.append(np.histogram(ta_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
    hafnium_distribution = np.array(hafnium_distribution)
    tantalum_distribution = np.array(tantalum_distribution)
    oxygen_distribution = np.array(oxygen_distribution)
    metal_distribution = hafnium_distribution + tantalum_distribution
    total_distribution = metal_distribution + oxygen_distribution
    
    #print(oxygen_distribution,tantalum_distribution,hafnium_distribution,metal_distribution,total_distribution)

    total_distribution_divide = total_distribution
    total_distribution_divide[total_distribution_divide ==0]=1

    O_stoich = 3.5*oxygen_distribution[-1]/total_distribution_divide[-1] #this is for the last trajectory file supplied in the list
    Ta_stoich = 3.5*tantalum_distribution[-1]/total_distribution_divide[-1]
    Hf_stoich = 3.5*hafnium_distribution[-1]/total_distribution_divide[-1]
    stoichiometry = np.array([Hf_stoich, O_stoich, Ta_stoich])
    proportion_labels = np.array(['a (of Hf$_a$)', 'b (of O$_b$)', 'c (of Ta$_c$)'])
    
    
    O_stoich_in = 3.5*oxygen_distribution[0]/total_distribution_divide[0] #this is for the first trajectory file supplied in the list => comparison makes sense if more than 1 files supplied, else self comparison
    Ta_stoich_in = 3.5*tantalum_distribution[0]/total_distribution_divide[0]
    Hf_stoich_in = 3.5*hafnium_distribution[0]/total_distribution_divide[0]  
    initial_stoichiometry = np.array([Hf_stoich_in, O_stoich_in, Ta_stoich_in])
    
    
    figure_size = [2.5,5]

    output_filename = analysis_name  + '_' + 'stoichiometry' + '_' + f'{z_bins}'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_stoich = plot_multiple_cases(stoichiometry, z_bin_centers, proportion_labels, 'Atoms # ratio','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir,**kwargs)    #, ylimit = [0,45]
    print('stoichiometry plotted')
    
    
    output_filename = analysis_name  + '_' + 'initial_stoichiometry' + '_' + f'{z_bins}'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_init_stoich = plot_multiple_cases(initial_stoichiometry, z_bin_centers, proportion_labels, 'Atoms # ratio','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)    #, xlimit = 3.5
    print('stoichiometry plotted')
    
    
    output_filename = analysis_name + '_' + 'M'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_metal = plot_multiple_cases(metal_distribution, z_bin_centers, labels, 'Metal atoms #','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)  
    
    output_filename = analysis_name + '_' + 'Hf'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_hf = plot_multiple_cases(hafnium_distribution, z_bin_centers, labels, 'Hf atoms #','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs) 
    
    output_filename = analysis_name + '_' + 'Ta'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_ta = plot_multiple_cases(tantalum_distribution, z_bin_centers, labels, 'Ta atoms #','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)
    
    output_filename = analysis_name + '_' + 'O'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_o = plot_multiple_cases(oxygen_distribution, z_bin_centers, labels, 'O atoms #','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, **kwargs)    

    
    return {
        "stoichiometry": fig_stoich,
        "initial_stoichiometry": fig_init_stoich,
        "metal": fig_metal,
        "Hf": fig_hf,
        "Ta": fig_ta,
        "O": fig_o,
    }  # Return the figure objects for further use if needed 

def plot_atomic_charge_distribution(file_list,labels,skip_rows,z_bins,analysis_name,output_dir=os.getcwd()):     ## Calls read_coordinates(...) and plot_multiple_cases(...)
    """Reads the coordinates from the file_list, calculates the atomic charge distributions,
    and plots the charge distributions for O, Hf, Ta, and all M atoms."""
    coordinates_arr, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(file_list, skip_rows, COLUMNS_TO_READ)
    z_bin_width = (zhi-zlo)/z_bins
    z_bin_centers = np.linspace(zlo+z_bin_width/2, zhi-z_bin_width/2, z_bins)
    oxygen_charge_distribution = []
    hafnium_charge_distribution = []
    tantalum_charge_distribution = []
    total_charge_distribution = []
    
    hafnium_distribution = []
    tantalum_distribution = []
    oxygen_distribution = []

    print('\nshape of coordinate_arr=', np.shape(coordinates_arr), '\nlength of coordinate_arr=', len(coordinates_arr))
    for i in range(len(coordinates_arr[:])):
        coordinates = coordinates_arr[i]
        coordinates = coordinates[coordinates[:, 5].argsort()]
#        print(coordinates[:,5])

        hf_atoms = coordinates[coordinates[:, 1]==2]
        ta_atoms = coordinates[np.logical_or(coordinates[:, 1]==4, np.logical_or(coordinates[:, 1]==6, coordinates[:, 1]==10))]
#       print(np.shape(ta_atoms),ta_atoms[:,5])
        o_atoms = coordinates[np.logical_or(coordinates[:, 1]==1, np.logical_or(coordinates[:, 1]==3, np.logical_or(coordinates[:, 1]==5, coordinates[:, 1]==9)))]
        
        hafnium_distribution.append(np.histogram(hf_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
        oxygen_distribution.append(np.histogram(o_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
        tantalum_distribution.append(np.histogram(ta_atoms[:,5],bins=z_bins,range=(zlo,zhi))[0])
        
        total_charge_distribution.append(np.histogram(coordinates[:,5],bins=z_bins,range=(zlo,zhi), weights = coordinates[:,2])[0])
        hafnium_charge_distribution.append(np.histogram(hf_atoms[:,5],bins=z_bins,range=(zlo,zhi), weights = hf_atoms[:,2])[0])
#        print(np.shape(hafnium_distribution))
        oxygen_charge_distribution.append(np.histogram(o_atoms[:,5],bins=z_bins,range=(zlo,zhi), weights = o_atoms[:,2])[0])
        tantalum_charge_distribution.append(np.histogram(ta_atoms[:,5],bins=z_bins,range=(zlo,zhi), weights = ta_atoms[:,2])[0])
    
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

    total_distribution_divide = total_distribution
    total_distribution_divide[total_distribution_divide ==0]=1
    hafnium_distribution_divide = hafnium_distribution
    hafnium_distribution_divide[hafnium_distribution_divide ==0]=1
    tantalum_distribution_divide = tantalum_distribution
    tantalum_distribution_divide[tantalum_distribution_divide ==0]=1
    metal_distribution_divide = metal_distribution
    metal_distribution_divide[metal_distribution_divide ==0]=1
    oxygen_distribution_divide = oxygen_distribution
    oxygen_distribution_divide[oxygen_distribution_divide ==0]=1

    total_mean_charge_distribution = total_charge_distribution/total_distribution_divide
    hafnium_mean_charge_distribution = hafnium_charge_distribution/hafnium_distribution_divide
    tantalum_mean_charge_distribution = tantalum_charge_distribution/tantalum_distribution_divide
    metal_mean_charge_distribution = metal_charge_distribution/metal_distribution_divide
    O_mean_charge_dist = oxygen_charge_distribution/oxygen_distribution_divide

    figure_size = [2.5,5]


    output_filename = analysis_name + '_' + 'all'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_net = plot_multiple_cases(total_charge_distribution, z_bin_centers, labels, 'Net charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimithi = 15, xlimitlo = -20, yaxis=0)

    output_filename = analysis_name + '_' + 'M'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_metal = plot_multiple_cases(metal_mean_charge_distribution, z_bin_centers, labels, 'Metal atoms mean charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimitlo = 0.7, xlimithi = 1.2)  

    output_filename = analysis_name + '_' + 'O'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_o = plot_multiple_cases(O_mean_charge_dist, z_bin_centers, labels, 'O mean charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimithi = 0, xlimitlo = -0.7)   
   

    output_filename =  'final' + '_' + analysis_name + '_' + 'all'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_net_end = plot_multiple_cases(total_charge_distribution[-1], z_bin_centers, labels[-1], 'Net charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimithi = 15, xlimitlo = -20, yaxis=0, markerindex = 1)

   
    output_filename =  'initial' + '_' + analysis_name + '_' + 'all'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_net_start = plot_multiple_cases(total_charge_distribution[0], z_bin_centers, labels[0], 'Net charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimithi = 15, xlimitlo = -20, yaxis=0)

    output_filename =  'initial' + '_' + analysis_name + '_' + 'M'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_metal_start = plot_multiple_cases(metal_mean_charge_distribution[0], z_bin_centers, labels[0], 'Metal atoms mean charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimitlo = 0.7, xlimithi = 1.2)  


    output_filename = 'initial' + '_' + analysis_name + '_' + 'O'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_o_start = plot_multiple_cases(O_mean_charge_dist[0], z_bin_centers, labels[0], 'O mean charge','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, ylimithi = 70, xlimithi = 0, xlimitlo = -0.7)   
    
    return {
        "net_charge": fig_net,
        "initial_net_charge": fig_net_start,
        "final_net_charge": fig_net_end,
        "metal_charge": fig_metal,
        "initial_metal_charge": fig_metal_start,
        "oxygen_charge": fig_o,
        "initial_oxygen_charge": fig_o_start,
    }  # Return the figure objects for further use if needed 


def plot_displacement_comparison(file_list, loop_start, loop_end, labels, analysis_name, repeat_count=0,output_dir=os.getcwd()):     ## Calls read_displacement_data(...) and plot multiple cases(...)
    """Reads the averaged thermodynamic output data for each case
    from the correspinginging files in a file_list, and plots the final displacements
    (z and lateral displacements) versus the z-bin groups positions 
    for the data row indices (1st index) corresponding to each case read from a file."""

    # repeat_count is how many times the first timestep is repeated in data file
    all_thermo_data = []
    for filename in file_list:
        all_thermo_data.append(read_displacement_data(filename, loop_start, loop_end, repeat_count))      ## 1st, 2nd and 3rd indices correspond to file, timestep and bin number correspondingly
                                                                                ## 4th index corresponds to the type of data (z, lateral displacement...)
    displacements = np.array(all_thermo_data) 
    print('\nshape of all_thermo_data array=', np.shape(all_thermo_data), '\nlength of all_thermo_data array=', len(all_thermo_data))
    
    zdisp = []
    lateraldisp = []
    binposition = []
    atoms_per_bin_count = []                      
    
    all_thermo_data = np.array(all_thermo_data)
    
    for i in range(len(displacements)):
        zdisp_temp = all_thermo_data[i,-1,:,-3]
        lateraldisp_temp = all_thermo_data[i,-1,:,-2]
        binposition_temp = all_thermo_data[i,-1,:,1]
        Ncount_temp = all_thermo_data[i,-1,:,2]                                 
        
        zdisp.append(zdisp_temp)
        lateraldisp.append(lateraldisp_temp)
        binposition.append(binposition_temp)
        atoms_per_bin_count.append(Ncount_temp)                          
        
    zdisp = np.array(zdisp)
    lateraldisp = np.array(lateraldisp)
    binposition = np.array(binposition)
    atoms_per_bin_count = np.array(atoms_per_bin_count)                         
    
    #print(zdisp)
    figure_size = [2.5,5]
    
    output_filename = analysis_name + '_' + 'z'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_z = plot_multiple_cases(zdisp, binposition, labels, 'z displacement (A)','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir, yaxis = 0) 
    
    output_filename = analysis_name + '_' + 'z_magnitude'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_zmag = plot_multiple_cases(np.abs(zdisp), binposition, labels, 'z displacement (A)','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir) 
    
    output_filename = analysis_name + '_' + 'lateral'
    for i in labels:
        output_filename = output_filename + '_' + i

    fig_lateral = plot_multiple_cases(lateraldisp, binposition, labels, 'lateral displacement (A)','z position (A)',output_filename, figure_size[0], figure_size[1], output_dir=output_dir) 

    return {
        "z_displacement": fig_z,
        "z_magnitude": fig_zmag,
        "lateral_displacement": fig_lateral,
    }  # Return the figure objects for further use if needed


def analyze_clusters(filepath, z_filament_lower_limit=5,z_filament_upper_limit=23, thickness = 21):
    """Performs cluster analysis on the given file:
    computes the coordination number, selects metallic atoms, clusters them,
    deletes the non-filamentary atoms, separates the top and bottom part of filament,
    analyzes filament connectivities and rdf of filamentary atoms."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    try:
        pipeline1 = import_file(filepath)
        pipeline2 = import_file(filepath)
        pipeline_fil = import_file(filepath)
        pipeline_fil_up = import_file(filepath)
    except Exception as e:
        raise ValueError(f"Malformed or unreadable file for OVITO: {filepath} (error: {e})")
    #pipeline_cmo = import_file(filepath) 
    coord1 = pipeline1.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 2.7, number_of_bins = 200))
    select_metal1 = pipeline1.modifiers.append(om.ExpressionSelectionModifier(expression = '((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10 || ParticleType==9 ) && Position.Z < 28 '))
    cluster_metal1 = pipeline1.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected = True))
    select_no_metal1 = pipeline1.modifiers.append(om.ExpressionSelectionModifier(expression = 'Cluster !=1'))
    delete_no_metal1 = pipeline1.modifiers.append(om.DeleteSelectedModifier())
    data1 = pipeline1.compute()
    if data1.particles.count == 0:
        raise ValueError(f"No clusters found in file: {filepath}")

    timestep = data1.attributes['Timestep']

    xyz1 = np.array(data1.particles['Position'])
    
    coord2 = pipeline2.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 2.7, number_of_bins = 200))
    select_metal2 = pipeline2.modifiers.append(om.ExpressionSelectionModifier(expression = '((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6) || ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28 '))
    cluster_metal2 = pipeline2.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected = True))
    select_no_metal2 = pipeline2.modifiers.append(om.ExpressionSelectionModifier(expression = 'Cluster !=2'))
    delete_no_metal2 = pipeline2.modifiers.append(om.DeleteSelectedModifier())
    data2 = pipeline2.compute()
    xyz2 = np.array(data2.particles['Position'])
    
    z1_min, z1_max = np.min(xyz1[:,2]), np.max(xyz1[:,2])
    print(np.shape(xyz1),len(xyz2))
    if len(xyz2)!=0:
        z2_min, z2_max = np.min(xyz2[:,2]), np.max(xyz2[:,2])
    
    if z1_min < z_filament_lower_limit and z1_max > z_filament_upper_limit:
        connection = 1
        upper_filament = xyz1
        lower_filament = xyz1
        separation = 0
        gap = 0
    else:
        connection = 0
        if z1_min < z_filament_lower_limit:
            upper_filament = xyz1
            lower_filament = xyz2
        else:
            upper_filament = xyz2
            lower_filament = xyz1
        separation = float('inf')
        closest_pair = (None, None)
        for point1 in xyz1:
            for point2 in xyz2:
                distance = np.linalg.norm(point1 - point2)
                if distance < separation:
                    separation = distance
                    closest_pair = (point1, point2)
                    gap = abs(point1[2] - point2[2])
        separation -= 3.9

    coord_fil = pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 2.7, number_of_bins = 200))
    select_fil = pipeline_fil.modifiers.append(om.ExpressionSelectionModifier(expression = '((ParticleType==2 || ParticleType==4 ||ParticleType==8) && Coordination<6)'))
    cluster_fil = pipeline_fil.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected = True))
    select_no_fil = pipeline_fil.modifiers.append(om.ExpressionSelectionModifier(expression = 'Cluster !=1'))
    delete_no_fil = pipeline_fil.modifiers.append(om.DeleteSelectedModifier())
    rdf_mod = pipeline_fil.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 3.9, number_of_bins = 200))
    data_fil = pipeline_fil.compute()
    
    xyz_fil_down = np.array(data_fil.particles['Position'])
    fil_height = np.max(xyz_fil_down[:,2])
    rdf_down = data_fil.tables['coordination-rdf'].xy()  
    fil_size_down = data_fil.particles.count
    
    coord_fil_up = pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 2.7, number_of_bins = 200))
    select_fil_up = pipeline_fil_up.modifiers.append(om.ExpressionSelectionModifier(expression = '((ParticleType==8 && Coordination<6) || ( ( ParticleType==10  || ParticleType==9 ) && Position.Z < 28))'))
    cluster_fil_up = pipeline_fil_up.modifiers.append(om.ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected = True))
    select_no_fil_up = pipeline_fil_up.modifiers.append(om.ExpressionSelectionModifier(expression = 'Cluster !=1'))
    delete_no_fil_up = pipeline_fil_up.modifiers.append(om.DeleteSelectedModifier())
    rdf_mod_up = pipeline_fil_up.modifiers.append(om.CoordinationAnalysisModifier(cutoff = 3.9, number_of_bins = 200))
    data_fil_up = pipeline_fil_up.compute()
   

    xyz_fil_up = np.array(data_fil_up.particles['Position'])
    fil_depth = np.min(xyz_fil_up[:,2])
    rdf_up = data_fil_up.tables['coordination-rdf'].xy()  
    fil_size_up = data_fil_up.particles.count
    
    
    return timestep, connection, fil_size_down, fil_height, rdf_down, fil_size_up, fil_depth, rdf_up, separation, gap
    
    
    #select_metallic_cmo = pipeline_cmo.modifiers.append(ExpressionSelectionModifier(expression = 'ParticleType==10 || ParticleType==9'))
    #cluster_cmo = pipeline_cmo.modifiers.append(ClusterAnalysisModifier(cutoff=3.9, sort_by_size=True, compute_com=True, only_selected = True))
    #data_cmo = pipeline_cmo.compute()
    
def track_filament_evolution(file_list, analysis_name,time_step,dump_interval_steps,output_dir=os.getcwd()):     ## Calls analyze_clusters(...)
    """Tracks and plots the evolution of the filament connectivity state, 
    gap and separation over time for each timeseries trajectory file in the file_list, 
    and plots the key results."""
    step_arr, connection, fil_size_down, fil_height, rdf_down, fil_size_up, fil_depth, rdf_up, gap, separation = [], [],[],[],[], [], [], [], [], []
    for filepath in file_list:
        step_temp, connection_temp, fil_size_down_temp, fil_height_temp, rdf_down_temp, fil_size_up_temp, fil_depth_temp, rdf_up_temp, separation_temp, gap_temp = analyze_clusters(filepath)
        fil_size_down.append(fil_size_down_temp)
        fil_size_up.append(fil_size_up_temp)
        connection.append(connection_temp)
        fil_height.append(fil_height_temp)
        fil_depth.append(fil_depth_temp)
        rdf_down.append(rdf_down_temp)
        rdf_up.append(rdf_up_temp)
        gap.append(gap_temp)
        separation.append(separation_temp)
        step_arr.append(step_temp) 

        
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
    print('shape of connections array',np.shape(np.array(connection))[0])
    #step_arr = np.linspace(0,np.shape(np.array(connection))[0]-1,np.shape(np.array(connection))[0])
    
    time_switch = step_arr * time_step * dump_interval_steps
    
    #figure_size = [2.5,5]
    
    # max_height = 21  # Unused variable, candidate for removal
    
    ln = 0.1
    mrkr = 5
    alph = 0.55 

    
    on_frequency = np.sum(connection==1)/len(connection)
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
    
    
    average_filament_gap, sd_gap = np.mean(gap), np.std(gap)
    fig_gap, ax_gap = plt.subplots()
    ax_gap.plot(time_switch, gap, alpha=alph, linewidth=ln, markersize=mrkr)
    ax_gap.scatter(time_switch, gap, alpha=alph, linewidth=ln, s=mrkr, marker='^', label=f'average_filament_gap = {average_filament_gap: .2f} +/- {sd_gap: .2f}')
    ax_gap.set_xlabel('Time (ps)')
    ax_gap.set_ylabel('Filament gap (A)')
    ax_gap.set_title('Filament gap')
    ax_gap.legend()
    # ax_gap.set_ylim(heightmin, heightmax)  # Uncomment if you want to set y-limits
    output_filename = analysis_name + 'fil_gap' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    average_filament_separation, sd_separation = np.mean(separation), np.std(separation)
    fig_sep, ax_sep = plt.subplots()
    ax_sep.plot(time_switch, separation, alpha=alph, linewidth=ln, markersize=mrkr)
    ax_sep.scatter(time_switch, separation, alpha=alph, linewidth=ln, s=mrkr, marker='^',
                   label=f'average_filament_separation = {average_filament_separation: .2f} +/- {sd_separation: .2f}')
    ax_sep.set_xlabel('Time (ps)')
    ax_sep.set_ylabel('Filament separation (A)')
    ax_sep.set_title('Filament separation')
    ax_sep.legend(fontsize=8)
    # ax_sep.set_ylim(heightmin, heightmax)  # Uncomment if you want to set y-limits
    output_filename = analysis_name + 'fil_separation' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    ###### filament gap & # of conductive atoms
    
    fig_size_gap, ax1_size_gap = plt.subplots()
    
    color = 'tab:red'
    average_filament_gap, sd_gap = np.mean(gap), np.std(gap)
    gap_max = 8.5
    gap_min = -0.5
    
    ax1_size_gap.set_xlabel('Time (ps)')
    ax1_size_gap.set_ylabel('Filament gap (A)', color=color)
    ax1_size_gap.scatter(time_switch, gap, alpha = alph, linewidth = ln, s=mrkr, color=color, label=f'average_filament_gap = {average_filament_gap: .2f} +/- {sd_gap: .2f}')
    ax1_size_gap.tick_params(axis='y', labelcolor=color)
    ax1_size_gap.set_ylim(gap_min,gap_max)
   
   
    ax2_size_gap = ax1_size_gap.twinx()
    
    sizemax_down = 350
    sizemin_down = 0
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    color = 'tab:blue'
    
    ax2_size_gap.set_ylabel('# of vacancies in filament (A.U.)', color=color)
    ax2_size_gap.scatter(time_switch, fil_size_down, alpha = alph, linewidth = ln, s=mrkr,  marker='^',  color=color, label=f'average # of vacancies in filament = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax2_size_gap.tick_params(axis='y', labelcolor=color)
    ax2_size_gap.set_ylim(sizemin_down,sizemax_down)
    
    plt.title('Gap & no. of conductive atoms in Filament')
    fig_size_gap.tight_layout()
    ax1_size_gap.legend(loc = 'upper right', framealpha = 0.8)
    ax2_size_gap.legend(loc = 'lower right', framealpha = 0.8)
    output_filename = analysis_name + 'fil_state' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    ###### filament lower
    
    fig_lowfil, ax1_lowfil = plt.subplots()
    
    heightmax = 25
    heightmin = 3
    average_filament_height, sd_height = np.mean(fil_height), np.std(fil_height)
    color = 'tab:red'
    
    ax1_lowfil.set_xlabel('Timestep (ps)')
    ax1_lowfil.set_ylabel('Filament length-lower end (A)', color=color)
    ax1_lowfil.scatter(time_switch, fil_height, alpha = alph, linewidth = ln, s=mrkr, color=color, label=f'average_filament_height = {average_filament_height: .2f} +/- {sd_height: .2f}')
    ax1_lowfil.tick_params(axis='y', labelcolor=color)
    ax1_lowfil.set_ylim(heightmin,heightmax)
    plt.legend(loc = 'upper right', framealpha = 0.75)
    
    
    ax2_lowfil = ax1_lowfil.twinx()
    
    sizemax_down = 350
    sizemin_down = 0
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    color = 'tab:blue'
    
    ax2_lowfil.set_ylabel('# of vacancies in filament-lower end (A.U.)', color=color)
    ax2_lowfil.scatter(time_switch, fil_size_down, alpha = alph, linewidth = ln, s=mrkr,  marker='^',  color=color, label=f'average # of vacancies in filament (bottom half) = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax2_lowfil.tick_params(axis='y', labelcolor=color)
    ax2_lowfil.set_ylim(sizemin_down,sizemax_down)
    
    plt.title('Filament lower part near cathode')
    fig_lowfil.tight_layout()
    plt.legend(loc = 'lower right', framealpha = 0.75)
    output_filename = analysis_name + 'fil_lower' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    ###### filament upper
      
    # depthmax = 25  # Unused variable, candidate for removal
    # depthmin = 3   # Unused variable, candidate for removal
    
    # sizemax_up = 700  # Unused variable, candidate for removal
    # sizemin_up = 400  # Unused variable, candidate for removal
       
    fig_upfil, ax1_upfil = plt.subplots()
    
    average_filament_depth, sd_depth = np.mean(fil_depth), np.std(fil_depth)
    color = 'tab:red'
    
    ax1_upfil.set_xlabel('Timestep (ps)')
    ax1_upfil.set_ylabel('Filament length-upper end (A)', color=color)
    ax1_upfil.scatter(time_switch, fil_depth, alpha = alph, linewidth = ln, s=mrkr, color=color, label=f'average_filament_depth = {average_filament_depth: .2f} +/- {sd_depth}')
    ax1_upfil.tick_params(axis='y', labelcolor=color)
    plt.legend(loc = 'upper right', framealpha = 0.75)
#    ax1_upfil.set_ylim(heightmin,heightmax)
    
    
    ax2_upfil = ax1_upfil.twinx()
    

    average_filament_size_up, sd_size_up = np.mean(fil_size_up), np.std(fil_size_up)
    color = 'tab:blue'
    
    ax2_upfil.set_ylabel('# of vacancies in filament-upper end (A.U.)', color=color)
    ax2_upfil.scatter(time_switch, fil_size_up, alpha = alph, linewidth = ln, s=mrkr,  marker='^', color=color, label=f'average # of vacancies in filament (top half) = {average_filament_size_up: .2f} +/- {sd_size_up: .2f}')
    ax2_upfil.tick_params(axis='y', labelcolor=color)
#    ax2_upfil.set_ylim(sizemin_down,sizemax_down)
    
    plt.title('Filament upper part near anode')
    fig_upfil.tight_layout()
    plt.legend(loc = 'lower right', framealpha = 0.75)
    output_filename = analysis_name + 'upper' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    ###############
    
    # heightmax = 25  # Unused variable, candidate for removal
    # heightmin = 3   # Unused variable, candidate for removal
    
    average_filament_height, sd_height = np.mean(fil_height), np.std(fil_height)
    fig_height, ax_height = plt.subplots()
    ax_height.scatter(time_switch, fil_height, alpha=alph, linewidth=ln, s=mrkr, label=f'average_filament_height = {average_filament_height: .2f} +/- {sd_height: .2f}')
    ax_height.set_xlabel('Timestep (ps)')
    ax_height.set_ylabel('Filament length-lower end (A)')
    ax_height.set_title('Filament length-lower end')
    ax_height.legend()
    ax_height.set_ylim(heightmin, heightmax)
    output_filename = analysis_name + 'fil_height' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    
    # depthmax = 25  # Unused variable, candidate for removal
    # depthmin = 3   # Unused variable, candidate for removal
    
    average_filament_depth, sd_depth = np.mean(fil_depth), np.std(fil_depth)
    fig_depth, ax_depth = plt.subplots()
    ax_depth.scatter(time_switch, fil_depth, alpha=alph, linewidth=ln, s=mrkr, label=f'average_filament_depth = {average_filament_depth: .2f} +/- {sd_depth}')
    ax_depth.set_xlabel('Timestep (ps)')
    ax_depth.set_ylabel('Filament length-upper end (A)')
    ax_depth.set_title('Filament length-upper end')
    ax_depth.legend()
    # ax_depth.set_ylim(depthmin, depthmax)
    output_filename = analysis_name + 'fil_depth' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()

    ###### filament size   
    # sizemax_up = 700  # Unused variable, candidate for removal
    # sizemin_up = 400  # Unused variable, candidate for removal
    
    average_filament_size_up, sd_size_up = np.mean(fil_size_up), np.std(fil_size_up)
    fig_size_up, ax_size_up = plt.subplots()
    ax_size_up.scatter(time_switch, fil_size_up, alpha=alph, linewidth=ln, s=mrkr,
                       label=f'average # of vacancies in filament (top half) = {average_filament_size_up: .2f} +/- {sd_size_up: .2f}')
    ax_size_up.set_xlabel('Timestep (ps)')
    ax_size_up.set_ylabel('# of vacancies in filament-upper end (A.U.)')
    ax_size_up.set_title('# of vacancies in filament-upper end')
    # ax_size_up.set_ylim(sizemin_up, sizemax_up)
    ax_size_up.legend()
    output_filename = analysis_name + 'fil_size_up' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    
    
    # sizemax_down = 350  # Unused variable, candidate for removal
    # sizemin_down = 0    # Unused variable, candidate for removal
    
    average_filament_size_down, sd_size_down = np.mean(fil_size_down), np.std(fil_size_down)
    fig_size_down, ax_size_down = plt.subplots()
    ax_size_down.scatter(time_switch, fil_size_down, alpha=alph, linewidth=ln, s=mrkr,
                        label=f'average # of vacancies in filament (bottom half) = {average_filament_size_down: .2f} +/- {sd_size_down: .2f}')
    ax_size_down.set_xlabel('Timestep (ps)')
    ax_size_down.set_ylabel('# of vacancies in filament-lower end (A.U.)')
    ax_size_down.set_title('# of vacancies in filament-lower end (A.U.)')
    ax_size_down.set_ylim(sizemin_down, sizemax_down)
    ax_size_down.legend()
    output_filename = analysis_name + 'fil_size_down' + '.pdf'
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    plt.savefig(savepath)
    plt.close()
    #    plt.scatter(rdf[-1,:,0], rdf[-1,:,1])

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
    }  # Return the figure objects for further use if needed



def plot_displacement_timeseries(file_list,datatype,dataindex, Nchunks, loop_start, loop_end, output_dir=os.getcwd()):     ## Calls read_displacement_data(...)
    """Reads the averaged thermodynamic output data for each case 
    from the correspinging files in a file_list, and plots the timeseries displacements 
    (one of the output data types selected by the dataindex as the 4th index) averaged 
    in groups (as the 3rd index) according to z-position of the atoms in the model
    for the data row indices (1st index) corresponding to each case read from a file."""
    all_thermo_data = []
    element_labels = []
    print(file_list)
    for filename in file_list:
        element_labels.append(filename[:2])
        all_thermo_data.append(read_displacement_data(filename, loop_start, loop_end))
    all_thermo_data = np.array(all_thermo_data)                                 ## 1st, 2nd and 3rd indices correspond to file, timestep and bin number correspondingly
                                                                    ## 4th index corresponds to the type of data (z, lateral displacement...)
    #print(filename)
    print(element_labels)
    print(np.shape(all_thermo_data))
    
    dump_steps = np.linspace(0,loop_end-1,100)
    print('dump_steps=',dump_steps)
    

    dataindexname = ['abs total disp','density - mass', 'temp (K)', 'z disp (A)', 'lateral disp (A)', 'outward disp vector (A)']

    nrows = Nchunks
    ncolumns = 4
    fig,axes = plt.subplots(nrows,ncolumns,squeeze=False,constrained_layout=False,figsize=(ncolumns*3,nrows*0.65))
    #axes=axes.flatten()
    fig.tight_layout()
    #plt.rcParams['xtick.labelsize'] = 6
    #plt.rcParams['ytick.labelsize'] = 6

    for j in range(ncolumns):
        for i in range(nrows):
          axes[nrows-1-i,j].plot(dump_steps, all_thermo_data[j,:,i,dataindex], label=f'{element_labels[j]} of region {i+1}', color = 'blue')
          if  j == 0:
            axes[nrows-1-i,j].set_ylabel(f'{datatype} \n {dataindexname[dataindex]}', fontsize=5)
          axes[nrows-1-i,j].legend(loc='upper center', fontsize=7)
          axes[nrows-1-i,j].adjustable='datalim'
          axes[nrows-1-i,j].set_aspect('auto')
          axes[nrows-1-i,j].tick_params(axis='both', which='major', labelsize=6)
          axes[nrows-1-i,j].set_aspect('auto')
          #axes[nrows-1-i,j].set_xticklabels(ax.get_xticks(), fontsize=6)
          if nrows-1-i != nrows-1:
            axes[nrows-1-i,j].set_xticklabels(())

        axes[nrows-1,j].set_xlabel('Dump steps', fontsize=8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0)

    output_filename= datatype + '-' + dataindexname[dataindex] #+'.pdf'
        
    #plt.suptitle(f'{datatype} {dataindexname[dataindex]}', fontsize = 8)
    #plt.show()
    os.makedirs(output_dir, exist_ok=True)
    savepath = os.path.join(output_dir, output_filename)
    fig.savefig(savepath, bbox_inches='tight', format='svg')#,dpi=300)#, )
    plt.close()

    return {
        "displacement_timeseries": fig,
    }  # Return the figure object for further use if needed


def run_analysis():
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
    run_analysis()



