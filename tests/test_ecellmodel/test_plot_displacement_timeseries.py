import os
import glob
import pytest
from lammpskit.ecellmodel.filament_layer_analysis import plot_displacement_timeseries

@pytest.mark.parametrize("dataindex", range(-6,0))
@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_displacement_timeseries(tmp_path, dataindex):
    # Prepare file list and labels
    data_dir = os.path.join(os.path.dirname(__file__), "test_data", "data_for_timeseries")
    file_list = sorted(glob.glob(os.path.join(data_dir, "[1-9][A-Z][a-z]mobilestc1.dat")))
    print('data directory path to file_list',data_dir)
    datatype = "mobility"
    Nchunks = 12
    loop_start = 1
    loop_end = 100

    # Call the function
    figs = plot_displacement_timeseries(
        file_list=file_list,
        datatype=datatype,
        dataindex=dataindex,
        Nchunks=Nchunks,
        loop_start=loop_start,
        loop_end=loop_end,
        output_dir=tmp_path
    )
    return figs["displacement_timeseries"]