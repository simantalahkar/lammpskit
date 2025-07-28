"""
Tests for centralized time series plotting functions.

This module tests the general-purpose time series plotting functions
to ensure they produce consistent and correct plots across different
configurations and use cases.
"""

import numpy as np
import pytest
from lammpskit.plotting import (
    create_time_series_plot,
    create_dual_axis_plot,
    TimeSeriesPlotConfig,
    DualAxisPlotConfig,
    calculate_mean_std_label,
    calculate_frequency_label
)

# Get baseline directory relative to tests root (works from any execution context)
# Since this test file is directly in tests/, the baseline is in the same directory level  
BASELINE_DIR_RELATIVE = "baseline"


@pytest.fixture
def sample_time_data():
    """Provide sample time series data for testing."""
    return {
        'x_data': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        'y_data1': np.array([10.5, 15.2, 12.8, 18.1, 14.6]),
        'y_data2': np.array([100, 110, 105, 120, 115]),
        'connection_data': np.array([1, 0, 1, 1, 0])
    }


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_time_series_plot_line_and_scatter(sample_time_data):
    """Test time series plot with both line and scatter elements."""
    config = TimeSeriesPlotConfig(include_line=True, include_scatter=True)
    stats_label = calculate_mean_std_label(sample_time_data['y_data1'], "average_value")
    
    fig, ax = create_time_series_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        title="Time Series with Line and Scatter",
        xlabel="Time (ps)",
        ylabel="Value (units)",
        stats_label=stats_label,
        config=config
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_time_series_plot_scatter_only(sample_time_data):
    """Test time series plot with scatter only."""
    config = TimeSeriesPlotConfig(include_line=False, include_scatter=True)
    stats_label = calculate_mean_std_label(sample_time_data['y_data1'], "average_measurement")
    
    fig, ax = create_time_series_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        title="Scatter Only Time Series",
        xlabel="Time (ps)",
        ylabel="Measurement (A)",
        stats_label=stats_label,
        config=config,
        ylim=(8, 20)
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_time_series_plot_line_only(sample_time_data):
    """Test time series plot with line only."""
    config = TimeSeriesPlotConfig(include_line=True, include_scatter=False)
    
    fig, ax = create_time_series_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        title="Line Only Time Series",
        xlabel="Time (ps)",
        ylabel="Signal (V)",
        stats_label="",  # No label since no scatter points
        config=config
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_time_series_plot_with_frequency_label(sample_time_data):
    """Test time series plot with frequency-based statistics label."""
    config = TimeSeriesPlotConfig(include_line=True, include_scatter=True)
    freq_label = calculate_frequency_label(
        sample_time_data['connection_data'], 1,
        "connected {frequency:.1f}% of the time"
    )
    
    fig, ax = create_time_series_plot(
        sample_time_data['x_data'],
        sample_time_data['connection_data'],
        title="Connection State Time Series",
        xlabel="Time (ps)",
        ylabel="Connection State (1: connected, 0: broken)",
        stats_label=freq_label,
        config=config,
        fontsize_legend=10  # Use new parameter name
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_dual_axis_plot_default_config(sample_time_data):
    """Test dual axis plot with default configuration."""
    primary_label = calculate_mean_std_label(sample_time_data['y_data1'], "primary_metric")
    secondary_label = calculate_mean_std_label(sample_time_data['y_data2'], "secondary_metric")
    
    fig, ax1, ax2 = create_dual_axis_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        sample_time_data['y_data2'],
        title="Dual Axis Plot - Default Config",
        xlabel="Time (ps)",
        primary_ylabel="Primary Measurement (A)",
        secondary_ylabel="Secondary Measurement (A.U.)",
        primary_stats_label=primary_label,
        secondary_stats_label=secondary_label
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_dual_axis_plot_custom_config(sample_time_data):
    """Test dual axis plot with custom configuration."""
    config = DualAxisPlotConfig(
        primary_color='green',
        secondary_color='orange',
        primary_legend_loc='lower left',
        secondary_legend_loc='upper left',
        legend_framealpha=0.9
    )
    
    primary_label = calculate_mean_std_label(sample_time_data['y_data1'], "gap_measurement")
    secondary_label = calculate_mean_std_label(sample_time_data['y_data2'], "size_measurement")
    
    fig, ax1, ax2 = create_dual_axis_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        sample_time_data['y_data2'],
        title="Gap & Size Analysis",
        xlabel="Time (ps)",
        primary_ylabel="Gap (A)",
        secondary_ylabel="# of vacancies (A.U.)",
        primary_stats_label=primary_label,
        secondary_stats_label=secondary_label,
        config=config,
        primary_ylim=(5, 25),
        secondary_ylim=(90, 130)
    )
    
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_dual_axis_plot_filament_style(sample_time_data):
    """Test dual axis plot mimicking filament analysis style."""
    # Use exact same configuration as track_filament_evolution
    config = DualAxisPlotConfig(
        alpha=0.55,
        linewidth=0.1,
        markersize=5,
        marker='^',
        primary_color='tab:red',
        secondary_color='tab:blue',
        primary_legend_loc='upper right',
        secondary_legend_loc='lower right',
        legend_framealpha=0.75
    )
    
    # Simulate filament data statistics
    primary_label = f"average_filament_gap = {np.mean(sample_time_data['y_data1']):.2f} +/- {np.std(sample_time_data['y_data1']):.2f}"
    secondary_label = f"average # of vacancies in filament = {np.mean(sample_time_data['y_data2']):.2f} +/- {np.std(sample_time_data['y_data2']):.2f}"
    
    fig, ax1, ax2 = create_dual_axis_plot(
        sample_time_data['x_data'],
        sample_time_data['y_data1'],
        sample_time_data['y_data2'],
        title="Gap & no. of conductive atoms in Filament",
        xlabel="Time (ps)",
        primary_ylabel="Filament gap (A)",
        secondary_ylabel="# of vacancies in filament (A.U.)",
        primary_stats_label=primary_label,
        secondary_stats_label=secondary_label,
        config=config,
        primary_ylim=(-0.5, 20.5),
        secondary_ylim=(90, 130)
    )
    
    return fig


def test_calculate_mean_std_label():
    """Test the mean and standard deviation label calculation."""
    data = np.array([10.0, 12.0, 11.0, 13.0, 9.0])
    
    # Test default precision
    label = calculate_mean_std_label(data, "test_metric")
    expected = f"test_metric = {np.mean(data):.2f} +/- {np.std(data):.2f}"
    assert label == expected
    
    # Test custom precision
    label = calculate_mean_std_label(data, "precise_metric", precision=3)
    expected = f"precise_metric = {np.mean(data):.3f} +/- {np.std(data):.3f}"
    assert label == expected


def test_calculate_frequency_label():
    """Test the frequency label calculation."""
    data = np.array([1, 0, 1, 1, 0])
    
    # Test frequency calculation
    label = calculate_frequency_label(data, 1, "connected {frequency:.1f}% of time")
    expected_freq = np.sum(data == 1) / len(data) * 100  # 60.0%
    expected = f"connected {expected_freq:.1f}% of time"
    assert label == expected
    
    # Test with different template
    label = calculate_frequency_label(data, 0, "broken state occurs {frequency:.2f}% of simulation")
    expected_freq = np.sum(data == 0) / len(data) * 100  # 40.0%
    expected = f"broken state occurs {expected_freq:.2f}% of simulation"
    assert label == expected


def test_timeseries_plot_config_defaults():
    """Test that TimeSeriesPlotConfig has correct defaults."""
    config = TimeSeriesPlotConfig()
    
    assert config.alpha == 0.55
    assert config.linewidth == 0.1
    assert config.markersize == 5
    assert config.marker == '^'
    assert config.include_line is True
    assert config.include_scatter is True
    assert config.format == 'pdf'
    # Test centralized font size defaults
    assert config.fontsize_title == 8
    assert config.fontsize_labels == 8
    assert config.fontsize_ticks == 8
    assert config.fontsize_legend == 8


def test_dual_axis_plot_config_defaults():
    """Test that DualAxisPlotConfig has correct defaults."""
    config = DualAxisPlotConfig()
    
    assert config.alpha == 0.55
    assert config.linewidth == 0.1
    assert config.markersize == 5
    assert config.marker == '^'
    assert config.primary_color == 'tab:red'
    assert config.secondary_color == 'tab:blue'
    assert config.format == 'pdf'
    assert config.primary_legend_loc == 'upper right'
    assert config.secondary_legend_loc == 'lower right'
    assert config.legend_framealpha == 0.75
    assert config.tight_layout is True
    # Test centralized font size defaults
    assert config.fontsize_title == 8
    assert config.fontsize_labels == 8
    assert config.fontsize_ticks == 8
    assert config.fontsize_legend == 8
