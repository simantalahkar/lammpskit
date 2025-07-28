import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import track_filament_evolution

# Get baseline directory relative to tests root (works from any execution context)
# Since this test file is in tests/test_ecellmodel/, the baseline is one level up
BASELINE_DIR_RELATIVE = "../baseline"

@pytest.mark.parametrize("fig_key", [
    "connection",
    "gap",
    "separation",
    "filament_gap_and_size",
    "filament_lower_part",
    "filament_upper_part",
    "filament_height",
    "filament_depth",
    "filament_size_up",
    "filament_size_down",
])
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_track_filament_evolution_figures(tmp_path, fig_key):
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:4]
    analysis_name = "test_filament_evolution"
    time_step = 0.001
    dump_interval_steps = 500

    figs = track_filament_evolution(
        file_list=file_list,
        analysis_name=analysis_name,
        time_step=time_step,
        dump_interval_steps=dump_interval_steps,
        output_dir=tmp_path
    )
    return figs[fig_key]

@pytest.mark.parametrize("fig_key", [
    "connection",
    "gap",
    "separation",
    "filament_gap_and_size",
    "filament_lower_part",
    "filament_upper_part",
    "filament_height",
    "filament_depth",
    "filament_size_up",
    "filament_size_down",
])
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_track_filament_evolution_single_file(tmp_path, fig_key):
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:1]
    analysis_name = "test_single_file"
    time_step = 0.001
    dump_interval_steps = 500

    figs = track_filament_evolution(
        file_list=file_list,
        analysis_name=analysis_name,
        time_step=time_step,
        dump_interval_steps=dump_interval_steps,
        output_dir=tmp_path
    )
    return figs[fig_key]