import os
import tempfile
import numpy as np
import pytest
from lammpskit.io import read_structure_info, read_coordinates
from lammpskit.ecellmodel.filament_layer_analysis import read_displacement_data

####################
# Test cases for edge cases and exceptions of data reader functions
####################

def test_read_structure_info_empty_file():
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp_path = tmp.name
    with pytest.raises(StopIteration) as excinfo:
        read_structure_info(tmp_path)
    assert "File is empty" in str(excinfo.value)

def test_read_structure_info_missing_timestep():
    content = "ITEM: TIMESTEP\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(EOFError) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing data for Timestep" in str(excinfo.value)

def test_read_structure_info_malformed_timestep():
    content = "ITEM: TIMESTEP\nnot_an_int\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0.0 50.0\n0.0 50.0\n0.0 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(ValueError) as excinfo:
        read_structure_info(tmp_path)
    assert "Malformed timestep line" in str(excinfo.value)

def test_read_structure_info_missing_total_atoms_section():
    content = "ITEM: TIMESTEP\n100"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(StopIteration) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing section for total atoms" in str(excinfo.value)

def test_read_structure_info_missing_total_atoms_data():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(EOFError) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing data for total atoms" in str(excinfo.value)

def test_read_structure_info_malformed_total_atoms():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\nnot_an_int"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(ValueError) as excinfo:
        read_structure_info(tmp_path)
    assert "Malformed total atoms line" in str(excinfo.value)

def test_read_structure_info_missing_box_bounds_section():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(StopIteration) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing section for box bounds" in str(excinfo.value)

def test_read_structure_info_missing_x_bounds_data():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(EOFError) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing data for x bounds" in str(excinfo.value)

def test_read_structure_info_malformed_x_bounds():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\nnot_a_float 50.0\n0.0 50.0\n0.0 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(ValueError) as excinfo:
        read_structure_info(tmp_path)
    assert "Malformed x bounds line" in str(excinfo.value)

def test_read_structure_info_missing_y_bounds_data():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0.0 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(EOFError) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing data for y bounds" in str(excinfo.value)

def test_read_structure_info_malformed_y_bounds():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0.0 50.0\nnot_float 50.0\n0.0 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(ValueError) as excinfo:
        read_structure_info(tmp_path)
    assert "Malformed y bounds line" in str(excinfo.value)

def test_read_structure_info_missing_z_bounds_data():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0.0 50.0\n0.0 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(EOFError) as excinfo:
        read_structure_info(tmp_path)
    assert "Missing data for z bounds" in str(excinfo.value)

def test_read_structure_info_malformed_z_bounds():
    content = "ITEM: TIMESTEP\n100\nITEM: NUMBER OF ATOMS\n5\nITEM: BOX BOUNDS pp pp pp\n0.0 50.0\n0.0 50.0\nnot_float 50.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    with pytest.raises(ValueError) as excinfo:
        read_structure_info(tmp_path)
    assert "Malformed z bounds line" in str(excinfo.value)

def test_read_structure_info_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        read_structure_info("nonexistent_file.lammpstrj")
    assert "File not found" in str(excinfo.value)


def test_read_coordinates_empty_file_list():
    with pytest.raises(ValueError) as excinfo:
        read_coordinates([], skip_rows=9, columns_to_read=(0,1,2,3,4,5,9,10,11,12))
    assert "file_list is empty" in str(excinfo.value)

def test_read_coordinates_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        read_coordinates(["nonexistent_file.lammpstrj"], skip_rows=9, columns_to_read=(0,1,2,3,4,5,9,10,11,12))
    assert "File not found" in str(excinfo.value)

def test_read_coordinates_file_fewer_atoms_than_expected():
    # total_atoms says 3, but only 2 atom lines provided
    content = """\
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp pp
0.0 10.0
0.0 10.0
0.0 10.0
ITEM: ATOMS id type q x y z ix iy iz vx vy vz c_eng
1 2 0.1 1.0 2.0 3.0 0 0 0 0 0 0 0
2 1 -0.2 2.0 3.0 4.0 0 0 0 0 0 0 0
"""
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_coordinates([tmp_path], skip_rows=9, columns_to_read=(0,1,2,3,4,5,9,10,11,12))
        assert "fewer atom lines" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_coordinates_file_non_numeric_atom_data():
    # Atom section contains a non-numeric value
    content = """\
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
2 X -0.2 2.0 3.0 4.0 0 0 0 0 0 0 0
"""
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError) as excinfo:
            read_coordinates([tmp_path], skip_rows=9, columns_to_read=(0,1,2,3,4,5,9,10,11,12))
        assert "Non-float atomic data" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_coordinates_columns_out_of_range():
    # Only 13 columns in atom section, but columns_to_read requests a non-existent column (e.g., index 20)
    content = """\
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
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        # columns_to_read includes an out-of-range index (20)
        with pytest.raises(ValueError) as excinfo:
            read_coordinates([tmp_path], skip_rows=9, columns_to_read=(0,1,2,3,4,5,9,10,11,12,20))
        assert "Column index out of range" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_coordinates_skip_rows_too_large():
    # Only 2 atom lines, but skip_rows skips past them
    content = """\
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
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        # skip_rows is set too high, so no atom data is read
        with pytest.raises(EOFError) as excinfo:
            read_coordinates([tmp_path], skip_rows=10, columns_to_read=(0,1,2,3,4,5,9,10,11,12))
        assert "fewer atom lines" in str(excinfo.value)
    finally:
        os.remove(tmp_path)


def test_read_displacement_data_file_not_found():
    with pytest.raises(FileNotFoundError) as excinfo:
        read_displacement_data("nonexistent_file.dat", 0, 1)
    assert "File not found" in str(excinfo.value)

def test_read_displacement_data_missing_nchunks_line():
    # Only headers, no Nchunks line
    content = "# header1\n# header2"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_displacement_data(tmp_path, 0, 0)
        assert "Missing Nchunks line" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_displacement_data_malformed_nchunks_line():
    # Nchunks line is malformed (not enough columns or not an int)
    content = "# header1\n# header2\n# header3\n0 not_an_int\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(TypeError) as excinfo:
            read_displacement_data(tmp_path, 0, 0)
        assert "Malformed Nchunks line" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_displacement_data_missing_chunk_data():
    # Nchunks says 3, but only 2 lines of data provided
    content = "# header1\n# header2\n# header3\n0 3\n1.0 2.0\n2.0 3.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_displacement_data(tmp_path, 0, 0)
        assert "Not enough data for chunk" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_displacement_data_malformed_chunk_data():
    # Nchunks says 3, but only 2 lines of data provided
    content = "# header1\n# header2\n# header3\n0 3\nnot_float 2.0\n2.0 3.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_displacement_data(tmp_path, 0, 0)
        assert "Missing or malformed chunk data" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

    content = "# header1\n# header2\n# header3\n0 3\n 2.0\n2.0 3.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_displacement_data(tmp_path, 0, 0)
        assert "Missing or malformed chunk data" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_displacement_data_loop_start_greater_than_loop_end():
    # Minimal valid file content
    content = "# header1\n# header2\n# header3\n0 1\n1.0 2.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(ValueError) as excinfo:
            read_displacement_data(tmp_path, 2, 1)
        assert "loop_start" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

def test_read_displacement_data_loop_end_out_of_range():
    # Only one loop present, but loop_end requests two
    content = "# header1\n# header2\n# header3\n0 2\n1.0 3.0\n2.0 4.0\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name
    try:
        with pytest.raises(EOFError) as excinfo:
            read_displacement_data(tmp_path, 0, 1)
        assert "Not enough data for chunk" in str(excinfo.value)
    finally:
        os.remove(tmp_path)

####################
# Test cases for typical usage of data reader functions
####################

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

def test_read_displacement_data_minimal():
    # Create a minimal fake displacement data file
    content = """\
# header1
# header2
# header3
0 2
1.0 3.0
2.0 4.0
# header1
# header2
# header3
0 2
1.0 6.0
2.0 8.0
# end loop 2
"""
    with tempfile.NamedTemporaryFile('w+', delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    # loop_start=0, loop_end=1, so two loops, Nchunks=2
    result = read_displacement_data(tmp_path, loop_start=0, loop_end=1)
    assert len(result) == 2
    assert np.allclose(result[0], [[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(result[1], [[1.0, 6.0], [2.0, 8.0]])

    os.remove(tmp_path)




