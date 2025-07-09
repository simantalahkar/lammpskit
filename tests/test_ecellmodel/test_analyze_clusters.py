import numpy as np
import pytest
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