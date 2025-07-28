# test_plot_multiple_cases.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from lammpskit.plotting import plot_multiple_cases

import pytest

# Get baseline directory relative to tests root (works from any execution context)
# Since this test file is directly in tests/, the baseline is in the same directory level
BASELINE_DIR_RELATIVE = "baseline"

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
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

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_multi_2d(tmp_path):
    # Synthetic data: two cases
    x = np.linspace(0, 10, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    x_arr = np.stack([x, x])  # shape (2, 20)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_multi_2d"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=tmp_path)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_2d_x_1d_y(tmp_path):
    # Synthetic data: two x cases, one y
    x1 = np.linspace(0, 10, 20)
    x2 = np.linspace(1, 11, 20)
    y = np.sin(np.linspace(0, 10, 20))
    x_arr = np.stack([x1, x2])  # shape (2, 20)
    labels = ["x1", "x2"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_2d_x_1d_y"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(x_arr, y, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=tmp_path)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_1d_x_2d_y(tmp_path):
    # Synthetic data: one x, two y cases
    x = np.linspace(0, 10, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_1d_x_2d_y"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(x, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize, output_dir=tmp_path)
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_with_kwargs(tmp_path):
    # Synthetic data: two cases
    x = np.linspace(0, 10, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    x_arr = np.stack([x, x])  # shape (2, 20)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_kwargs"
    xsize, ysize = 2, 2

    # Test with axis limits and markerindex
    fig = plot_multiple_cases(
        x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize,
        output_dir=tmp_path,
        xlimit=(2, 8),
        ylimit=(-1, 1),
        markerindex=1
    )
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_with_xaxis_yaxis(tmp_path):
    # Synthetic data: two cases
    x = np.linspace(-5, 5, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    x_arr = np.stack([x, x])  # shape (2, 20)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_xaxis_yaxis"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(
        x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize,
        output_dir=tmp_path,
        xaxis=0.5,   # horizontal line at y=0.5
        yaxis=-2     # vertical line at x=-2
    )
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_with_limit_hilo_kwargs(tmp_path):
    # Synthetic data: two cases
    x = np.linspace(-5, 5, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    x_arr = np.stack([x, x])  # shape (2, 20)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_limit_hilo_kwargs"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(
        x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize,
        output_dir=tmp_path,
        xlimithi=3,
        xlimitlo=-3,
        ylimithi=1,
        ylimitlo=-1
    )
    return fig

@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
def test_plot_multiple_cases_with_limit_kwargs(tmp_path):
    # Synthetic data: two cases
    x = np.linspace(-5, 5, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)
    x_arr = np.stack([x, x])  # shape (2, 20)
    y_arr = np.stack([y1, y2])  # shape (2, 20)
    labels = ["sin(x)", "cos(x)"]
    xlabel = "x"
    ylabel = "y"
    output_filename = "test_limit_kwargs"
    xsize, ysize = 2, 2

    fig = plot_multiple_cases(
        x_arr, y_arr, labels, xlabel, ylabel, output_filename, xsize, ysize,
        output_dir=tmp_path,
        xlimit=[-3,3],
        ylimit=[-1,1]
    )
    return fig


