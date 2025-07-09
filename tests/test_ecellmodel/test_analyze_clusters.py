import numpy as np
import pytest
import tempfile
from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters

def test_analyze_clusters_output_values():
    filepath = "tests/test_ecellmodel/test_data/data_for_layer_analysis/3set-0.1V.500000.lammpstrj"
    result = analyze_clusters(filepath)
    assert isinstance(result, tuple)
    assert len(result) == 10

    timestep, connection, filament_size_down, filament_height, rdf_down, filament_size_up, filament_depth, rdf_up, separation, gap = result

    assert timestep == 500000
    assert connection == 1
    assert filament_size_down == 214
    assert round(filament_height, 4) == 17.8633
    assert isinstance(rdf_down, np.ndarray)
    assert rdf_down.ndim == 2 and rdf_down.shape[0] > 0 and rdf_down.shape[1] > 0
    assert filament_size_up == 552
    assert round(filament_depth, 4) == 12.5072
    assert isinstance(rdf_up, np.ndarray)
    assert rdf_up.ndim == 2 and rdf_up.shape[0] > 0 and rdf_up.shape[1] > 0
    assert separation == 0
    assert gap == 0

def test_analyze_clusters_missing_file():
    missing_path = "tests/test_ecellmodel/test_data/data_for_layer_analysis/nonexistent_file.lammpstrj"
    with pytest.raises(FileNotFoundError) as excinfo:
        analyze_clusters(missing_path)
    assert "File not found" in str(excinfo.value)

def test_analyze_clusters_malformed_file():
    # Create a temporary file with invalid content
    filepath = "tests/test_ecellmodel/test_data/data_for_layer_analysis/malformed-0.1V.500000.lammpstrj"
    with pytest.raises(ValueError) as excinfo:    
        result = analyze_clusters(filepath)
    assert "Malformed or unreadable file for OVITO" in str(excinfo.value)

def test_analyze_clusters_no_clusters():
    # This file should be a valid LAMMPS trajectory file with no atoms,
    # or only atoms that do not match the selection criteria.
    filepath = "tests/test_ecellmodel/test_data/data_for_layer_analysis/no_clusters-0.1V.500000.lammpstrj"
    with pytest.raises(ValueError) as excinfo:
        analyze_clusters(filepath)
    assert "No clusters found in file" in str(excinfo.value)



