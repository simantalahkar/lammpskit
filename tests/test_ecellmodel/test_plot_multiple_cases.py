# test_plot_multiple_cases.py
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lammpskit.ecellmodel.filament_layer_analysis import plot_multiple_cases

import pytest

@pytest.mark.mpl_image_compare(baseline_dir="baseline", remove_text=True)
def test_plot_multiple_cases_single_1d(tmp_path):
    # Synthetic data: single case
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    labels = "sin(x)"
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_single_1d"
    xsize, ysize = 2, 2

    # # Call the plotting function
    # plot_multiple_cases(x, y, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=tmp_path)

    # Call the plotting function and get the figure
    fig = plot_multiple_cases(x, y, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=tmp_path)

    # Return the figure for pytest-mpl to compare
    # fig = plt.figure()
    # img_path = os.path.join(tmp_path, output_filename + ".svg")#'output','ecellmodel','test_output', output_filename + ".svg")
    # fig = matplotlib.pyplot.figure()
    # img = matplotlib.image.imread(img_path)
    # plt.imshow(img)
    # plt.axis('off')
    # return plt.gcf()

    return fig