import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution

@pytest.mark.parametrize("fig_key", [
    "stoichiometry",
    "initial_stoichiometry",
    "metal",
    "Hf",
    "Ta",
    "O",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_atomic_distribution_figures(tmp_path, fig_key):
    # Prepare file list as before
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:4]
    labels = ["traj1", "traj2", "traj3", "traj4"]
    skip_rows = 9
    z_bins = 15
    analysis_name = "test_atomic_dist_4files"

    # Call the function and get all figures
    figs = plot_atomic_distribution(
        file_list=file_list,
        labels=labels,
        skip_rows=skip_rows,
        z_bins=z_bins,
        analysis_name=analysis_name,
        output_dir=tmp_path
    )
    # Return the requested figure for pytest-mpl
    return figs[fig_key]


@pytest.mark.parametrize("fig_key", [
    "stoichiometry",
    "initial_stoichiometry",
    "metal",
    "Hf",
    "Ta",
    "O",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_atomic_distribution_figures_3files(tmp_path, fig_key):
    # Prepare file list for 3 files starting with "temp"
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "temp*.lammpstrj")))[:3]
    labels = ["traj1", "traj2", "traj3"]
    skip_rows = 9
    z_bins = 15
    analysis_name = "test_atomic_dist_3files"

    figs = plot_atomic_distribution(
        file_list=file_list,
        labels=labels,
        skip_rows=skip_rows,
        z_bins=z_bins,
        analysis_name=analysis_name,
        output_dir=tmp_path
    )
    return figs[fig_key]

@pytest.mark.parametrize("nfiles,labels,analysis_name", [
    (2, ["traj1", "traj2"], "test_atomic_dist_2files"),
    (1, ["traj1"], "test_atomic_dist_1file"),
])
@pytest.mark.parametrize("fig_key", [
    "stoichiometry",
    "initial_stoichiometry",
    "metal",
    "Hf",
    "Ta",
    "O",
])
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_atomic_distribution_figures_nfiles(tmp_path, nfiles, labels, analysis_name, fig_key):
    data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:nfiles]
    skip_rows = 9
    z_bins = 15

    figs = plot_atomic_distribution(
        file_list=file_list,
        labels=labels,
        skip_rows=skip_rows,
        z_bins=z_bins,
        analysis_name=analysis_name,
        output_dir=tmp_path
    )
    return figs[fig_key]
