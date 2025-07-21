"""
Main workflow orchestration for electrochemical cell model analysis.

This module contains the main analysis workflow function that coordinates
all aspects of HfTaO electrochemical cell simulation analysis, integrating
data processing, plotting, and analysis capabilities.
"""

import os
import numpy as np
from typing import List, Dict, Any
from ..config import (
    PlotConfig, TimeSeriesConfig, DataConfig,
    DEFAULT_PLOT_CONFIG, DEFAULT_TIMESERIES_CONFIG, DEFAULT_DATA_CONFIG
)


def run_complete_analysis(
    base_dir: str,
    simulation_files: List[str],
    output_dir: str,
    analysis_config: Dict[str, Any] = None,
    plot_config: PlotConfig = None,
    timeseries_config: TimeSeriesConfig = None,
    data_config: DataConfig = None
) -> Dict[str, Any]:
    """
    Run complete electrochemical cell model analysis workflow.
    
    This is the main orchestration function that coordinates all analysis
    steps for HfTaO simulation data including atomic distributions, charge
    analysis, displacement tracking, and visualization.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing simulation files.
    simulation_files : List[str]
        List of simulation filenames to analyze.
    output_dir : str
        Directory for saving analysis results and figures.
    analysis_config : Dict[str, Any], optional
        Configuration parameters for analysis steps.
    plot_config : PlotConfig, optional
        Configuration for plotting parameters.
    timeseries_config : TimeSeriesConfig, optional
        Configuration for time series analysis.
    data_config : DataConfig, optional
        Configuration for data processing.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing all analysis results including:
        - atomic_distributions: Spatial distribution analysis
        - charge_distributions: Charge analysis results
        - displacement_analysis: Time series displacement data
        - structure_analysis: Structural evolution metrics
        - cluster_analysis: Cluster identification results
        - figure_paths: Paths to generated figures
    """
    from .data_processing import (
        select_atom_types_from_coordinates,
        calculate_atomic_distributions,
        calculate_charge_distributions,
        calculate_z_bins_setup
    )
    from .plotting import (
        process_displacement_timeseries_data,
        plot_timeseries_grid,
        create_and_save_figure
    )
    from .analysis import (
        analyze_clusters_with_ovito,
        analyze_atomic_structure_evolution,
        calculate_displacement_statistics
    )
    from ..io import read_structure_info, read_coordinates
    from ..plotting import plot_multiple_cases
    
    # Set default configurations
    if plot_config is None:
        plot_config = DEFAULT_PLOT_CONFIG
    if timeseries_config is None:
        timeseries_config = DEFAULT_TIMESERIES_CONFIG
    if data_config is None:
        data_config = DEFAULT_DATA_CONFIG
    if analysis_config is None:
        analysis_config = {}
    
    # Initialize results dictionary
    results = {
        'atomic_distributions': {},
        'charge_distributions': {},
        'displacement_analysis': {},
        'structure_analysis': {},
        'cluster_analysis': {},
        'figure_paths': [],
        'analysis_summary': {}
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting complete electrochemical cell analysis workflow...")
    
    # Step 1: Process atomic distributions
    print("Step 1: Processing atomic distributions...")
    
    file_paths = [os.path.join(base_dir, filename) for filename in simulation_files]
    
    # Read structure information
    structure_info = {}
    coordinates_data = {}
    
    for i, file_path in enumerate(file_paths):
        print(f"  Reading file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")
        
        try:
            # Read structure and coordinates
            struct_info = read_structure_info(file_path)
            coords = read_coordinates(file_path)
            
            structure_info[file_path] = struct_info
            coordinates_data[file_path] = coords
            
        except Exception as e:
            print(f"  Warning: Could not read {file_path}: {e}")
            continue
    
    # Step 2: Calculate atomic distributions
    print("Step 2: Calculating atomic distributions...")
    
    if coordinates_data:
        # Set up z-bins for spatial analysis
        z_bins_info = calculate_z_bins_setup(
            list(coordinates_data.values())[0],  # Use first dataset for binning
            data_config.n_bins
        )
        
        # Calculate distributions for each file
        for file_path, coords in coordinates_data.items():
            try:
                # Select relevant atom types
                selected_atoms = select_atom_types_from_coordinates(
                    coords, 
                    atom_types=data_config.atom_types
                )
                
                # Calculate spatial distributions
                atomic_dist = calculate_atomic_distributions(
                    selected_atoms,
                    z_bins_info,
                    data_config.distribution_types
                )
                
                results['atomic_distributions'][file_path] = atomic_dist
                
            except Exception as e:
                print(f"  Warning: Could not process distributions for {file_path}: {e}")
                continue
    
    # Step 3: Calculate charge distributions
    print("Step 3: Calculating charge distributions...")
    
    if 'charge_analysis' in analysis_config and analysis_config['charge_analysis']:
        for file_path, coords in coordinates_data.items():
            try:
                charge_dist = calculate_charge_distributions(
                    coords,
                    z_bins_info,
                    analysis_config.get('charge_types', ['initial', 'final'])
                )
                
                results['charge_distributions'][file_path] = charge_dist
                
            except Exception as e:
                print(f"  Warning: Could not calculate charge distributions for {file_path}: {e}")
                continue
    
    # Step 4: Displacement time series analysis
    print("Step 4: Processing displacement time series...")
    
    if 'displacement_analysis' in analysis_config and analysis_config['displacement_analysis']:
        try:
            # Process displacement data
            displacement_data, element_labels, dump_steps = process_displacement_timeseries_data(
                file_paths,
                timeseries_config.loop_start,
                timeseries_config.loop_end,
                timeseries_config.time_points,
                analysis_config.get('read_displacement_func')
            )
            
            results['displacement_analysis'] = {
                'data': displacement_data,
                'element_labels': element_labels,
                'dump_steps': dump_steps
            }
            
            # Calculate displacement statistics
            if displacement_data:
                disp_stats = calculate_displacement_statistics(
                    np.array(displacement_data),
                    dump_steps,
                    analysis_config.get('time_windows')
                )
                results['displacement_analysis']['statistics'] = disp_stats
            
        except Exception as e:
            print(f"  Warning: Could not process displacement analysis: {e}")
    
    # Step 5: Structural evolution analysis
    print("Step 5: Analyzing structural evolution...")
    
    if 'structure_evolution' in analysis_config and analysis_config['structure_evolution']:
        try:
            struct_evolution = analyze_atomic_structure_evolution(
                file_paths,
                analysis_config.get('structure_analysis_type', 'radial_distribution'),
                **analysis_config.get('structure_params', {})
            )
            
            results['structure_analysis'] = struct_evolution
            
        except Exception as e:
            print(f"  Warning: Could not perform structural evolution analysis: {e}")
    
    # Step 6: Cluster analysis
    print("Step 6: Performing cluster analysis...")
    
    if 'cluster_analysis' in analysis_config and analysis_config['cluster_analysis']:
        cluster_results = {}
        
        for file_path in file_paths:
            try:
                clusters = analyze_clusters_with_ovito(
                    file_path,
                    analysis_config.get('cluster_params', {})
                )
                cluster_results[file_path] = clusters
                
            except Exception as e:
                print(f"  Warning: Could not perform cluster analysis for {file_path}: {e}")
                continue
        
        results['cluster_analysis'] = cluster_results
    
    # Step 7: Generate visualizations
    print("Step 7: Generating visualizations...")
    
    figure_paths = []
    
    # Plot atomic distributions
    if results['atomic_distributions']:
        try:
            fig_path = os.path.join(output_dir, "atomic_distributions")
            fig = plot_multiple_cases(
                results['atomic_distributions'],
                plot_type='distribution',
                config=plot_config
            )
            
            create_and_save_figure(
                lambda x: fig,
                None,
                fig_path,
                output_dir,
                plot_config
            )
            figure_paths.append(f"{fig_path}.{plot_config.save_formats[0]}")
            
        except Exception as e:
            print(f"  Warning: Could not generate atomic distribution plots: {e}")
    
    # Plot displacement time series
    if results['displacement_analysis'] and 'data' in results['displacement_analysis']:
        try:
            disp_data = results['displacement_analysis']
            
            fig = plot_timeseries_grid(
                disp_data['data'],
                disp_data['dump_steps'],
                disp_data['element_labels'],
                'Displacement',
                timeseries_config.data_labels,
                timeseries_config.dataindex,
                timeseries_config.nrows,
                timeseries_config.ncolumns,
                plot_config.figsize,
                plot_config
            )
            
            fig_path = os.path.join(output_dir, "displacement_timeseries")
            fig.savefig(f"{fig_path}.{plot_config.save_formats[0]}", 
                       dpi=plot_config.dpi, bbox_inches='tight')
            figure_paths.append(f"{fig_path}.{plot_config.save_formats[0]}")
            
        except Exception as e:
            print(f"  Warning: Could not generate displacement time series plots: {e}")
    
    results['figure_paths'] = figure_paths
    
    # Step 8: Generate analysis summary
    print("Step 8: Generating analysis summary...")
    
    summary = {
        'total_files_processed': len([k for k in coordinates_data.keys()]),
        'analysis_steps_completed': [],
        'output_directory': output_dir,
        'configuration_used': {
            'plot_config': plot_config.__dict__ if hasattr(plot_config, '__dict__') else str(plot_config),
            'timeseries_config': timeseries_config.__dict__ if hasattr(timeseries_config, '__dict__') else str(timeseries_config),
            'data_config': data_config.__dict__ if hasattr(data_config, '__dict__') else str(data_config),
            'analysis_config': analysis_config
        }
    }
    
    # Track completed analysis steps
    if results['atomic_distributions']:
        summary['analysis_steps_completed'].append('atomic_distributions')
    if results['charge_distributions']:
        summary['analysis_steps_completed'].append('charge_distributions')
    if results['displacement_analysis']:
        summary['analysis_steps_completed'].append('displacement_analysis')
    if results['structure_analysis']:
        summary['analysis_steps_completed'].append('structure_analysis')
    if results['cluster_analysis']:
        summary['analysis_steps_completed'].append('cluster_analysis')
    
    results['analysis_summary'] = summary
    
    print("Analysis workflow completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated {len(figure_paths)} figures")
    print(f"Completed analysis steps: {', '.join(summary['analysis_steps_completed'])}")
    
    return results
