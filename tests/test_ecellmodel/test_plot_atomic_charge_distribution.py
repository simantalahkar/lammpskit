import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_charge_distribution

@pytest.mark.parametrize("nfiles,labels,analysis_name", [
    (4, ["traj1", "traj2", "traj3", "traj4"], "test_atomic_charge_4files"),
    (3, ["traj1", "traj2", "traj3"], "test_atomic_charge_3files"),
    (2, ["traj1", "traj2"], "test_atomic_charge_2files"),
    (1, ["traj1"], "test_atomic_charge_1file"),
])
@pytest.mark.parametrize("fig_key", [
    "net_charge",
    "initial_net_charge",
    "final_net_charge",
    "metal_charge",
    "initial_metal_charge",
    "oxygen_charge",
    "initial_oxygen_charge",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_atomic_charge_distribution_figures(tmp_path, nfiles, labels, analysis_name, fig_key):
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:nfiles]
    skip_rows = 9
    z_bins = 15

    figs = plot_atomic_charge_distribution(
        file_list=file_list,
        labels=labels,
        skip_rows=skip_rows,
        z_bins=z_bins,
        analysis_name=analysis_name,
        output_dir=tmp_path
    )
    return figs[fig_key]