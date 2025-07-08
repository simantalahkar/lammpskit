import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_comparison

@pytest.mark.parametrize("fig_key", [
    "z_displacement",
    "z_magnitude",
    "lateral_displacement",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_displacement_comparison_atom_type(tmp_path, fig_key):
    # First case: [1-9][A-Z][a-z]mobilestc1.dat, labels Hf, O, Ta
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[1-9][A-Z][a-z]mobilestc1.dat")))
    labels = ["Hf", "O", "Ta"]
    loop_start = 1
    loop_end = 3
    repeat_count = 1
    analysis_name = "test_displacement_atom_type"

    figs = plot_displacement_comparison(
        file_list=file_list,
        loop_start=loop_start,
        loop_end=loop_end,
        labels=labels,
        analysis_name=analysis_name,
        repeat_count=repeat_count,
        output_dir=tmp_path
    )
    return figs[fig_key]

@pytest.mark.parametrize("fig_key", [
    "z_displacement",
    "z_magnitude",
    "lateral_displacement",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_displacement_comparison_temperature(tmp_path, fig_key):
    # Second case: *_Hfmobilestc1.dat, labels 300K, 900K, 1300K
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "*_Hfmobilestc1.dat")))
    labels = ["300K", "900K", "1300K"]
    loop_start = 1
    loop_end = 3
    repeat_count = 1
    analysis_name = "test_displacement_temperature"

    figs = plot_displacement_comparison(
        file_list=file_list,
        loop_start=loop_start,
        loop_end=loop_end,
        labels=labels,
        analysis_name=analysis_name,
        repeat_count=repeat_count,
        output_dir=tmp_path
    )
    return figs[fig_key]