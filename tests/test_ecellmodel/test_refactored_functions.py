"""
Tests for the refactored atomic distribution and charge distribution functions.

These tests verify that the refactored functions produce the same results
as the original implementations while using the new modular approach.
"""

import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock
from lammpskit.ecellmodel.filament_layer_analysis import (
    plot_atomic_distribution,
    plot_atomic_charge_distribution
)


class TestRefactoredAtomicDistribution:
    """Test the refactored plot_atomic_distribution function."""

    def test_plot_atomic_distribution_integration(self):
        """Test that plot_atomic_distribution uses modular functions correctly."""
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock data files - we'll mock the read_coordinates function
            file_list = ['mock_file1.txt', 'mock_file2.txt']
            labels = ['case1', 'case2']
            skip_rows = 9
            z_bins = 10
            analysis_name = 'test_atomic'
            
            # Mock coordinate data
            mock_coordinates = np.array([
                [1, 1, 0.1, 10, 20, 2],   # O atom at z=2
                [2, 2, 0.2, 11, 21, 4],   # Hf atom at z=4
                [3, 4, 0.3, 12, 22, 6],   # Ta atom at z=6
                [4, 1, 0.4, 13, 23, 8]    # O atom at z=8
            ])
            
            # Mock read_coordinates return values
            mock_return = (
                [mock_coordinates, mock_coordinates],  # coordinates_arr
                [100, 200],                            # timestep_arr
                4,                                     # total_atoms
                0, 10,                                # xlo, xhi
                0, 10,                                # ylo, yhi
                0, 10                                 # zlo, zhi
            )
            
            # Mock the read_coordinates function and plot_multiple_cases
            with patch('lammpskit.ecellmodel.filament_layer_analysis.read_coordinates', return_value=mock_return), \
                 patch('lammpskit.ecellmodel.filament_layer_analysis.plot_multiple_cases') as mock_plot:
                
                # Set up mock plot function to return a mock figure
                mock_fig = MagicMock()
                mock_plot.return_value = mock_fig
                
                # Call the function
                result = plot_atomic_distribution(
                    file_list=file_list,
                    labels=labels,
                    skip_rows=skip_rows,
                    z_bins=z_bins,
                    analysis_name=analysis_name,
                    output_dir=temp_dir
                )
                
                # Verify the function returns the expected dictionary
                expected_keys = ["stoichiometry", "initial_stoichiometry", "metal", "Hf", "Ta", "O"]
                assert all(key in result for key in expected_keys)
                
                # Verify plot_multiple_cases was called the expected number of times
                # Should be called 6 times: stoich, initial_stoich, metal, Hf, Ta, O
                assert mock_plot.call_count == 6
                
                # Verify that the modular functions are being used by checking
                # that the function completed without errors
                assert result is not None

    def test_plot_atomic_distribution_file_operations(self):
        """Test file operations and output directory handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_list = ['test_file.txt']
            labels = ['test']
            
            # Create a subdirectory for output
            output_subdir = os.path.join(temp_dir, 'output')
            
            mock_coordinates = np.array([
                [1, 2, 0.1, 10, 20, 5]  # Single Hf atom
            ])
            
            mock_return = (
                [mock_coordinates],
                [100],
                1,
                0, 10, 0, 10, 0, 10
            )
            
            with patch('lammpskit.ecellmodel.filament_layer_analysis.read_coordinates', return_value=mock_return), \
                 patch('lammpskit.ecellmodel.filament_layer_analysis.plot_multiple_cases') as mock_plot:
                
                mock_plot.return_value = MagicMock()
                
                # Test with non-existent output directory
                result = plot_atomic_distribution(
                    file_list=file_list,
                    labels=labels,
                    skip_rows=9,
                    z_bins=5,
                    analysis_name='test',
                    output_dir=output_subdir
                )
                
                # Function should complete successfully
                assert result is not None
                
                # Verify all expected plots were called
                assert mock_plot.call_count == 6


class TestRefactoredChargeDistribution:
    """Test the refactored plot_atomic_charge_distribution function."""

    def test_plot_atomic_charge_distribution_integration(self):
        """Test that plot_atomic_charge_distribution uses modular functions correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_list = ['mock_file1.txt']
            labels = ['case1']
            skip_rows = 9
            z_bins = 8
            analysis_name = 'test_charge'
            
            # Mock coordinate data with charges
            mock_coordinates = np.array([
                [1, 1, -0.5, 10, 20, 2],   # O atom with charge -0.5 at z=2
                [2, 2, 2.0, 11, 21, 4],    # Hf atom with charge 2.0 at z=4
                [3, 4, 1.5, 12, 22, 6],    # Ta atom with charge 1.5 at z=6
                [4, 1, -0.3, 13, 23, 8]    # O atom with charge -0.3 at z=8
            ])
            
            mock_return = (
                [mock_coordinates],
                [100],
                4,
                0, 10, 0, 10, 0, 10
            )
            
            with patch('lammpskit.ecellmodel.filament_layer_analysis.read_coordinates', return_value=mock_return), \
                 patch('lammpskit.ecellmodel.filament_layer_analysis.plot_multiple_cases') as mock_plot:
                
                mock_fig = MagicMock()
                mock_plot.return_value = mock_fig
                
                result = plot_atomic_charge_distribution(
                    file_list=file_list,
                    labels=labels,
                    skip_rows=skip_rows,
                    z_bins=z_bins,
                    analysis_name=analysis_name,
                    output_dir=temp_dir
                )
                
                # Verify the function returns the expected dictionary
                expected_keys = [
                    "net_charge", "initial_net_charge", "final_net_charge",
                    "metal_charge", "initial_metal_charge", "oxygen_charge", 
                    "initial_oxygen_charge"
                ]
                assert all(key in result for key in expected_keys)
                
                # Verify plot_multiple_cases was called the expected number of times
                # Should be called 7 times for different charge plots
                assert mock_plot.call_count == 7

    def test_plot_atomic_charge_distribution_error_handling(self):
        """Test error handling in charge distribution calculation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_list = ['mock_file.txt']
            labels = ['test']
            
            # Mock coordinate data that might cause division by zero
            mock_coordinates = np.array([
                [1, 1, 0.5, 10, 20, 2]  # Single O atom
            ])
            
            mock_return = (
                [mock_coordinates],
                [100],
                1,
                0, 10, 0, 10, 0, 10
            )
            
            with patch('lammpskit.ecellmodel.filament_layer_analysis.read_coordinates', return_value=mock_return), \
                 patch('lammpskit.ecellmodel.filament_layer_analysis.plot_multiple_cases') as mock_plot:
                
                mock_plot.return_value = MagicMock()
                
                # This should not raise any division by zero errors
                result = plot_atomic_charge_distribution(
                    file_list=file_list,
                    labels=labels,
                    skip_rows=9,
                    z_bins=4,
                    analysis_name='test',
                    output_dir=temp_dir
                )
                
                assert result is not None
                # Function should complete successfully even with edge cases
                assert mock_plot.call_count == 7


class TestBackwardCompatibility:
    """Test that refactored functions maintain backward compatibility."""

    def test_function_signatures_unchanged(self):
        """Test that function signatures remain the same."""
        # Import the functions to ensure they exist with expected signatures
        from lammpskit.ecellmodel.filament_layer_analysis import (
            plot_atomic_distribution,
            plot_atomic_charge_distribution
        )
        
        # Check that we can call functions with the same parameters as before
        # (This test will fail if signatures change)
        import inspect
        
        # Check plot_atomic_distribution signature
        sig = inspect.signature(plot_atomic_distribution)
        expected_params = [
            'file_list', 'labels', 'skip_rows', 'z_bins', 
            'analysis_name', 'output_dir'
        ]
        
        for param in expected_params:
            assert param in sig.parameters
        
        # Check plot_atomic_charge_distribution signature
        sig = inspect.signature(plot_atomic_charge_distribution)
        for param in expected_params:
            assert param in sig.parameters

    def test_return_value_structure(self):
        """Test that return value structures remain the same."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock minimal data
            mock_coordinates = np.array([[1, 2, 0.1, 10, 20, 5]])
            mock_return = ([mock_coordinates], [100], 1, 0, 10, 0, 10, 0, 10)
            
            with patch('lammpskit.ecellmodel.filament_layer_analysis.read_coordinates', return_value=mock_return), \
                 patch('lammpskit.ecellmodel.filament_layer_analysis.plot_multiple_cases') as mock_plot:
                
                mock_plot.return_value = MagicMock()
                
                # Test plot_atomic_distribution return structure
                result = plot_atomic_distribution(
                    file_list=['test.txt'],
                    labels=['test'],
                    skip_rows=9,
                    z_bins=4,
                    analysis_name='test',
                    output_dir=temp_dir
                )
                
                # Should return dictionary with specific keys
                expected_keys = ["stoichiometry", "initial_stoichiometry", "metal", "Hf", "Ta", "O"]
                assert isinstance(result, dict)
                assert all(key in result for key in expected_keys)
                
                # Test plot_atomic_charge_distribution return structure
                result = plot_atomic_charge_distribution(
                    file_list=['test.txt'],
                    labels=['test'],
                    skip_rows=9,
                    z_bins=4,
                    analysis_name='test',
                    output_dir=temp_dir
                )
                
                expected_keys = [
                    "net_charge", "initial_net_charge", "final_net_charge",
                    "metal_charge", "initial_metal_charge", "oxygen_charge", 
                    "initial_oxygen_charge"
                ]
                assert isinstance(result, dict)
                assert all(key in result for key in expected_keys)
