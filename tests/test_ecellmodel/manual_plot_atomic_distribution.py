import numpy as np
import pytest
import tempfile
from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters

filepath = "tests/test_ecellmodel/test_data/data_for_layer_analysis/malformed-3set-0.1V.500000.lammpstrj"
with pytest.raises(ValueError) as excinfo:    
    result = analyze_clusters(filepath)
assert "Malformed or unreadable file for OVITO" in str(excinfo.value)

exit()

# Create a temporary file with invalid content
with tempfile.NamedTemporaryFile("w", suffix=".lammpstrj", delete=True) as tmp:
    print('test')
    tmp.write("this is not a valid LAMMPS trajectory file\n")
    tmp.flush()
    with pytest.raises(ValueError) as excinfo:
        print('test')
        analyze_clusters(tmp.name)
        print('test')
    assert "Malformed or unreadable file for OVITO" in str(excinfo.value)

exit()

import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_atomic_distribution

def test_plot_atomic_distribution_figures_3files(tmp_path = os.getcwd()):
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
        output_dir=tmp_path,
        ylimit=[0,60]
    )
    #return figs[fig_key]
if __name__ == "__main__":
    test_plot_atomic_distribution_figures_3files()


# @pytest.mark.parametrize("fig_key", [
#     "stoichiometry",
#     "initial_stoichiometry",
#     "metal",
#     "Hf",
#     "Ta",
#     "O",
# ])
# @pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
# def test_plot_atomic_distribution_figures(tmp_path, fig_key):
    # # Prepare file list as before
    # data_dir = os.path.join(os.path.dirname(__file__), "test_data")
    # file_list = sorted(glob.glob(os.path.join(data_dir, "[0-9]*.lammpstrj")))[:4]
    # labels = ["traj1", "traj2", "traj3", "traj4"]
    # skip_rows = 9
    # z_bins = 15
    # analysis_name = "test_atomic_dist_4files"

    # # Call the function and get all figures
    # figs = plot_atomic_distribution(
    #     file_list=file_list,
    #     labels=labels,
    #     skip_rows=skip_rows,
    #     z_bins=z_bins,
    #     analysis_name=analysis_name,
    #     output_dir=tmp_path
    # )
    # # Return the requested figure for pytest-mpl
    # return figs[fig_key]


# @pytest.mark.parametrize("fig_key", [
#     "stoichiometry",
#     "initial_stoichiometry",
#     "metal",
#     "Hf",
#     "Ta",
#     "O",
# ])
#@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)

