"""Test the new centralized time series plotting functions."""

import numpy as np
from lammpskit.plotting import (
    create_time_series_plot, 
    create_dual_axis_plot,
    TimeSeriesPlotConfig,
    calculate_mean_std_label,
    calculate_frequency_label
)

def test_centralized_functions():
    """Test that our centralized functions work as expected."""
    
    # Create test data
    x_data = np.array([1, 2, 3, 4, 5])
    y_data1 = np.array([10, 15, 12, 18, 14])
    y_data2 = np.array([100, 110, 105, 120, 115])
    
    # Test time series plot with both line and scatter
    config_both = TimeSeriesPlotConfig(include_line=True, include_scatter=True)
    stats_label = calculate_mean_std_label(y_data1, "average_value")
    
    fig1, ax1 = create_time_series_plot(
        x_data, y_data1,
        title="Test Both Line and Scatter", 
        xlabel="Time", ylabel="Value",
        stats_label=stats_label,
        config=config_both
    )
    print("✓ Line + scatter plot created")
    
    # Test time series plot with scatter only
    config_scatter = TimeSeriesPlotConfig(include_line=False, include_scatter=True)
    
    fig2, ax2 = create_time_series_plot(
        x_data, y_data1,
        title="Test Scatter Only",
        xlabel="Time", ylabel="Value", 
        stats_label=stats_label,
        config=config_scatter
    )
    print("✓ Scatter-only plot created")
    
    # Test frequency calculation for connection-type data
    connection_data = np.array([1, 0, 1, 1, 0])
    freq_label = calculate_frequency_label(
        connection_data, 1, 
        "filament is in connected state {frequency:.2f}% of the time"
    )
    print(f"✓ Frequency label: {freq_label}")
    
    # Test dual axis plot
    primary_label = calculate_mean_std_label(y_data1, "primary_metric")
    secondary_label = calculate_mean_std_label(y_data2, "secondary_metric")
    
    fig3, ax3_1, ax3_2 = create_dual_axis_plot(
        x_data, y_data1, y_data2,
        title="Test Dual Axis",
        xlabel="Time",
        primary_ylabel="Primary (A)", 
        secondary_ylabel="Secondary (A.U.)",
        primary_stats_label=primary_label,
        secondary_stats_label=secondary_label
    )
    print("✓ Dual-axis plot created")
    
    print("All centralized functions working correctly!")

if __name__ == "__main__":
    test_centralized_functions()
