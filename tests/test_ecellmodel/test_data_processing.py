"""
Tests for atomic processing functions in data_processing.py

These tests validate the new modular functions for atom type selection,
distribution calculations, and charge distribution processing.
"""

import numpy as np
from lammpskit.ecellmodel.data_processing import (
    select_atom_types_from_coordinates,
    calculate_z_bins_setup,
    calculate_atomic_distributions,
    calculate_charge_distributions
)


class TestAtomTypeSelection:
    """Test atom type selection from coordinates."""

    def test_select_atom_types_basic(self):
        """Test basic atom type selection."""
        # Create test coordinate data with different atom types
        coordinates = np.array([
            [1, 1, 0.1, 10, 20, 5],   # O atom (type 1)
            [2, 2, 0.2, 11, 21, 6],   # Hf atom (type 2)
            [3, 3, 0.3, 12, 22, 7],   # O atom (type 3)
            [4, 4, 0.4, 13, 23, 8],   # Ta atom (type 4)
            [5, 5, 0.5, 14, 24, 9],   # O atom (type 5)
            [6, 6, 0.6, 15, 25, 10],  # Ta atom (type 6)
            [7, 7, 0.7, 16, 26, 11],  # O atom (type 7)
            [8, 8, 0.8, 17, 27, 12],  # Ta atom (type 8)
            [9, 9, 0.9, 18, 28, 13],  # O atom (type 9)
            [10, 10, 1.0, 19, 29, 14] # Ta atom (type 10)
        ])
        
        result = select_atom_types_from_coordinates(coordinates)
        
        # Verify we get the expected keys
        assert 'hf' in result
        assert 'ta' in result
        assert 'o' in result
        
        # Verify Hf atoms (type 2)
        assert len(result['hf']) == 1
        assert result['hf'][0, 1] == 2
        
        # Verify Ta atoms (types 4, 6, 8, 10)
        assert len(result['ta']) == 4
        ta_types = result['ta'][:, 1]
        assert 4 in ta_types
        assert 6 in ta_types
        assert 8 in ta_types
        assert 10 in ta_types
        
        # Verify O atoms (types 1, 3, 5, 7, 9)
        assert len(result['o']) == 5
        o_types = result['o'][:, 1]
        assert 1 in o_types
        assert 3 in o_types
        assert 5 in o_types
        assert 7 in o_types
        assert 9 in o_types

    def test_select_atom_types_sorted_by_z(self):
        """Test that coordinates are sorted by z position."""
        coordinates = np.array([
            [1, 1, 0.1, 10, 20, 10],   # O atom at z=10
            [2, 2, 0.2, 11, 21, 5],    # Hf atom at z=5
            [3, 1, 0.3, 12, 22, 15]    # O atom at z=15
        ])
        
        result = select_atom_types_from_coordinates(coordinates)
        
        # Verify O atoms are sorted by z position
        o_z_positions = result['o'][:, 5]
        assert o_z_positions[0] < o_z_positions[1]  # Should be 10, 15

    def test_select_atom_types_empty_types(self):
        """Test behavior when some atom types are missing."""
        coordinates = np.array([
            [1, 1, 0.1, 10, 20, 5],   # Only O atom (type 1)
        ])
        
        result = select_atom_types_from_coordinates(coordinates)
        
        # Verify empty arrays for missing types
        assert len(result['hf']) == 0
        assert len(result['ta']) == 0
        assert len(result['o']) == 1


class TestZBinsSetup:
    """Test z-bin calculation functions."""

    def test_calculate_z_bins_setup_basic(self):
        """Test basic z-bin setup calculation."""
        zlo, zhi, z_bins = -10.0, 10.0, 4
        
        z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)
        
        # Verify bin width
        expected_width = (zhi - zlo) / z_bins
        assert z_bin_width == expected_width
        
        # Verify bin centers
        expected_centers = np.array([-7.5, -2.5, 2.5, 7.5])
        np.testing.assert_array_almost_equal(z_bin_centers, expected_centers)

    def test_calculate_z_bins_setup_single_bin(self):
        """Test z-bin setup with single bin."""
        zlo, zhi, z_bins = 0.0, 20.0, 1
        
        z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)
        
        assert z_bin_width == 20.0
        assert len(z_bin_centers) == 1
        assert z_bin_centers[0] == 10.0


class TestAtomicDistributions:
    """Test atomic distribution calculations."""

    def test_calculate_atomic_distributions_basic(self):
        """Test basic atomic distribution calculation."""
        # Create test coordinates
        coordinates1 = np.array([
            [1, 1, 0.1, 10, 20, 5],   # O atom at z=5
            [2, 2, 0.2, 11, 21, 6],   # Hf atom at z=6
            [3, 4, 0.3, 12, 22, 7]    # Ta atom at z=7
        ])
        coordinates2 = np.array([
            [4, 1, 0.4, 13, 23, 5],   # O atom at z=5
            [5, 2, 0.5, 14, 24, 8]    # Hf atom at z=8
        ])
        
        coordinates_arr = [coordinates1, coordinates2]
        z_bins = 4
        zlo, zhi = 0.0, 10.0
        
        result = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)
        
        # Verify we get all expected keys
        expected_keys = ['hafnium', 'tantalum', 'oxygen', 'metal', 'total']
        for key in expected_keys:
            assert key in result
        
        # Verify array shapes
        for key in ['hafnium', 'tantalum', 'oxygen']:
            assert result[key].shape == (2, z_bins)  # 2 coordinate sets, z_bins bins
        
        # Verify composite distributions
        np.testing.assert_array_equal(
            result['metal'], result['hafnium'] + result['tantalum']
        )
        np.testing.assert_array_equal(
            result['total'], result['metal'] + result['oxygen']
        )

    def test_calculate_atomic_distributions_empty_coordinates(self):
        """Test handling of empty coordinate arrays."""
        coordinates_arr = []
        z_bins = 4
        zlo, zhi = 0.0, 10.0
        
        result = calculate_atomic_distributions(coordinates_arr, z_bins, zlo, zhi)
        
        # Should return arrays with shape (0, z_bins)
        for key in ['hafnium', 'tantalum', 'oxygen', 'metal', 'total']:
            assert result[key].shape == (0, z_bins)


class TestChargeDistributions:
    """Test charge distribution calculations."""

    def test_calculate_charge_distributions_basic(self):
        """Test basic charge distribution calculation."""
        # Create test coordinates with charges
        coordinates1 = np.array([
            [1, 1, 0.5, 10, 20, 5],   # O atom with charge 0.5 at z=5
            [2, 2, 1.0, 11, 21, 6],   # Hf atom with charge 1.0 at z=6
            [3, 4, 0.8, 12, 22, 7]    # Ta atom with charge 0.8 at z=7
        ])
        
        coordinates_arr = [coordinates1]
        z_bins = 4
        zlo, zhi = 0.0, 10.0
        
        # Create mock atomic distributions for normalization
        atomic_distributions = {
            'hafnium': np.array([[1, 0, 0, 0]]),  # 1 Hf atom in first bin
            'tantalum': np.array([[0, 0, 1, 0]]), # 1 Ta atom in third bin
            'oxygen': np.array([[1, 0, 0, 0]]),   # 1 O atom in first bin
            'metal': np.array([[1, 0, 1, 0]]),    # Metal = Hf + Ta
            'total': np.array([[2, 0, 1, 0]])     # Total = Metal + O
        }
        
        result = calculate_charge_distributions(
            coordinates_arr, z_bins, zlo, zhi, atomic_distributions
        )
        
        # Verify we get all expected keys
        expected_keys = [
            'hafnium_charge', 'tantalum_charge', 'oxygen_charge', 'total_charge',
            'metal_charge', 'total_mean_charge', 'hafnium_mean_charge',
            'tantalum_mean_charge', 'metal_mean_charge', 'oxygen_mean_charge'
        ]
        for key in expected_keys:
            assert key in result
        
        # Verify charge array shapes
        for key in ['hafnium_charge', 'tantalum_charge', 'oxygen_charge', 'total_charge']:
            assert result[key].shape == (1, z_bins)

    def test_calculate_charge_distributions_division_by_zero(self):
        """Test safe division handling when atom counts are zero."""
        coordinates1 = np.array([
            [1, 1, 0.5, 10, 20, 5]    # Single O atom
        ])
        
        coordinates_arr = [coordinates1]
        z_bins = 4
        zlo, zhi = 0.0, 10.0
        
        # Create atomic distributions with some zeros
        atomic_distributions = {
            'hafnium': np.array([[0, 0, 0, 0]]),  # No Hf atoms
            'tantalum': np.array([[0, 0, 0, 0]]), # No Ta atoms
            'oxygen': np.array([[1, 0, 0, 0]]),   # 1 O atom in first bin
            'metal': np.array([[0, 0, 0, 0]]),    # No metal atoms
            'total': np.array([[1, 0, 0, 0]])     # Only O atom
        }
        
        result = calculate_charge_distributions(
            coordinates_arr, z_bins, zlo, zhi, atomic_distributions
        )
        
        # Verify no division by zero errors occur
        # Mean charges should be 0 where atom counts are 0
        assert not np.any(np.isnan(result['hafnium_mean_charge']))
        assert not np.any(np.isnan(result['tantalum_mean_charge']))
        assert not np.any(np.isnan(result['metal_mean_charge']))


class TestIntegrationTests:
    """Integration tests combining multiple functions."""

    def test_complete_workflow(self):
        """Test complete workflow from coordinates to distributions."""
        # Create realistic test data
        coordinates1 = np.array([
            [1, 1, -0.5, 10, 20, 2],   # O atom at z=2
            [2, 2, 2.0, 11, 21, 3],    # Hf atom at z=3
            [3, 4, 1.5, 12, 22, 4],    # Ta atom at z=4
            [4, 1, -0.3, 13, 23, 6],   # O atom at z=6
            [5, 2, 1.8, 14, 24, 7]     # Hf atom at z=7
        ])
        
        coordinates2 = np.array([
            [6, 1, -0.4, 15, 25, 1],   # O atom at z=1
            [7, 2, 1.9, 16, 26, 5],    # Hf atom at z=5
            [8, 4, 1.6, 17, 27, 8]     # Ta atom at z=8
        ])
        
        coordinates_arr = [coordinates1, coordinates2]
        z_bins = 8
        zlo, zhi = 0.0, 8.0
        
        # Test z-bin setup
        z_bin_width, z_bin_centers = calculate_z_bins_setup(zlo, zhi, z_bins)
        assert z_bin_width == 1.0
        assert len(z_bin_centers) == z_bins
        
        # Test atomic distributions
        atomic_distributions = calculate_atomic_distributions(
            coordinates_arr, z_bins, zlo, zhi
        )
        
        assert atomic_distributions['hafnium'].shape == (2, z_bins)
        assert atomic_distributions['total'].shape == (2, z_bins)
        
        # Test charge distributions
        charge_distributions = calculate_charge_distributions(
            coordinates_arr, z_bins, zlo, zhi, atomic_distributions
        )
        
        assert charge_distributions['total_charge'].shape == (2, z_bins)
        assert charge_distributions['total_mean_charge'].shape == (2, z_bins)
        
        # Verify conservation: metal + oxygen = total
        np.testing.assert_array_equal(
            charge_distributions['metal_charge'] + charge_distributions['oxygen_charge'],
            charge_distributions['total_charge']
        )