import os
import tempfile
import numpy as np
from lammpskit.ecellmodel.filament_layer_analysis import read_structure_info
from lammpskit.ecellmodel.filament_layer_analysis import read_coordinates


def test_read_structure_info_minimal():
    # Create a minimal .lammpstrj file with 13 columns in the atom section
    content = """\
ITEM: TIMESTEP
1200000
ITEM: NUMBER OF ATOMS
5
ITEM: BOX BOUNDS pp pp pp
0.0 50.0
0.0 50.0
0.0 50.0
ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
1 2 0.1 1.0 2.0 3.0 0 0 0 0 0 0 0
2 1 -0.2 2.0 3.0 4.0 0 0 0 0 0 0 0
3 2 0.3 3.0 4.0 5.0 0 0 0 0 0 0 0
4 1 -0.4 4.0 5.0 6.0 0 0 0 0 0 0 0
5 2 0.5 5.0 6.0 7.0 0 0 0 0 0 0 0
"""
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    # The function expects to skip the first line, so we simulate that
    timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info(tmp_path)

    assert timestep == 1200000
    assert total_atoms == 5
    assert np.isclose(xlo, 0.0)
    assert np.isclose(xhi, 50.0)
    assert np.isclose(ylo, 0.0)
    assert np.isclose(yhi, 50.0)
    assert np.isclose(zlo, 0.0)
    assert np.isclose(zhi, 50.0)

    os.remove(tmp_path)


def test_read_coordinates_minimal():
    content1 = """\
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
1 2 0.1 1.0 2.0 3.0 0 0 0 0 0 0 0
2 1 -0.2 2.0 3.0 4.0 0 0 0 0 0 0 0
"""
    content2 = """\
ITEM: TIMESTEP
200
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
1 2 0.3 3.0 4.0 5.0 0 0 0 0 0 0 0
2 1 -0.4 4.0 5.0 6.0 0 0 0 0 0 0 0
"""
    files = []
    for content in [content1, content2]:
        tmp = tempfile.NamedTemporaryFile('w+', delete=False)
        tmp.write(content)
        tmp.flush()
        files.append(tmp.name)
        tmp.close()

    skip_rows = 9  # skip header lines before atom data
    columns_to_read = (0,1,2,3,4,5,9,10,11,12)  # as in your code

    coords, timesteps, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_coordinates(files, skip_rows, columns_to_read)

    assert coords.shape == (2, 2, len(columns_to_read))
    assert np.allclose(timesteps, [100, 200])
    assert total_atoms == 2
    assert np.isclose(xlo, 0.0)
    assert np.isclose(xhi, 10.0)
    assert np.isclose(ylo, 0.0)
    assert np.isclose(yhi, 10.0)
    assert np.isclose(zlo, 0.0)
    assert np.isclose(zhi, 10.0)

    for f in files:
        os.remove(f)

