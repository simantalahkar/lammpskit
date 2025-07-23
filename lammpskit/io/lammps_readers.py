"""
LAMMPS file readers for general-purpose I/O operations.

This module contains functions for reading LAMMPS trajectory files,
structure files, and simulation metadata that can be used across
different types of molecular dynamics analysis.
"""

import re
import numpy as np
from typing import Tuple

# Import validation functions from config
from ..config import validate_file_list


def read_structure_info(filepath: str) -> Tuple[int, int, float, float, float, float, float, float]:
    """
    Reads the structure file and returns the timestep, total number of atoms, and the box dimensions.

    This is a general-purpose function for reading LAMMPS trajectory metadata that can be used
    across different analysis workflows.

    Parameters
    ----------
    filepath : str
        Path to the structure file.

    Returns
    -------
    tuple
        (timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi)
        timestep : int
            Simulation timestep.
        total_atoms : int
            Total number of atoms in the simulation.
        xlo, xhi : float
            Lower and upper bounds of the simulation box in x.
        ylo, yhi : float
            Lower and upper bounds of the simulation box in y.
        zlo, zhi : float
            Lower and upper bounds of the simulation box in z.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    EOFError
        If expected data is missing.
    ValueError
        If data is malformed.
    OSError
        If there is an error opening the file.
    """
    skip = 1
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            # Skip header lines before timestep
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"File is empty: {filepath}")
            # Read timestep
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for Timestep: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                timestep = int(c[0])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed timestep line in file: {filepath} (got: {line.strip()})")

            # Skip header lines before total atoms
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"Missing section for total atoms: {filepath}")
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for total atoms: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                total_atoms = int(c[0])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed total atoms line in file: {filepath} (got: {line.strip()})")

            # Skip header lines before box bounds
            for _ in range(skip):
                try:
                    next(f)
                except StopIteration:
                    raise StopIteration(f"Missing section for box bounds: {filepath}")
            # Read x bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for x bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                xlo = float(c[0])
                xhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed x bounds line in file: {filepath} (got: {line.strip()})")

            # Read y bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for y bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                ylo = float(c[0])
                yhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed y bounds line in file: {filepath} (got: {line.strip()})")

            # Read z bounds
            line = f.readline()
            if line == "":
                raise EOFError(f"Missing data for z bounds: {filepath}")
            c0 = re.split(r'\s+|\s|, |,', line)
            c = [ele for ele in c0 if ele.strip()]
            try:
                zlo = float(c[0])
                zhi = float(c[1])
            except (IndexError, ValueError):
                raise ValueError(f"Malformed z bounds line in file: {filepath} (got: {line.strip()})")
        return timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except OSError as e:
        raise OSError(f"Error opening file {filepath}: {e}")


def read_coordinates(
    file_list: list[str],
    skip_rows: int,
    columns_to_read: Tuple[int, ...]
) -> Tuple[np.ndarray, np.ndarray, int, float, float, float, float, float, float]:
    """
    Reads multiple LAMMPS structure files and extracts simulation cell parameters, coordinates, and timesteps.

    This is a general-purpose function for reading LAMMPS trajectory coordinates that can be used
    across different analysis workflows.

    Parameters
    ----------
    file_list : list of str
        List of file paths to structure files.
    skip_rows : int
        Number of header rows to skip before atomic coordinates.
    columns_to_read : tuple of int
        Indices of columns to read from each file.

    Returns
    -------
    tuple
        (coordinates, timestep_arr, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi)
        coordinates : np.ndarray
            Array of atomic coordinates for all files.
        timestep_arr : np.ndarray
            Array of timesteps for all files.
        total_atoms : int
            Number of atoms (should be the same for all files).
        xlo, xhi, ylo, yhi, zlo, zhi : float
            Simulation box bounds (should be the same for all files).

    Raises
    ------
    ValueError
        If file_list is empty or atomic data is malformed.
    EOFError
        If a file has fewer atom lines than expected.
    """
    print(file_list)
    # Validate input parameters using centralized functions
    validate_file_list(file_list)
    
    timestep_arr: list[int] = []
    coordinates: list[np.ndarray] = []
    for filepath in file_list:
        # Extract simulation parameters from each file
        timestep, total_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_structure_info(filepath)
        timestep_arr.append(timestep)
        try:
            # Read atomic coordinates
            coords = np.loadtxt(
                filepath,
                delimiter=' ',
                comments='#',
                skiprows=skip_rows,
                max_rows=total_atoms,
                usecols=columns_to_read
            )
        except ValueError as e:
            raise ValueError(
                f"Column index out of range or Non-float atomic data in file: {filepath} (error: {e})\n (column indices provided to read = {columns_to_read})"
            )
        if coords.shape[0] != total_atoms:
            raise EOFError(f"File {filepath} has fewer atom lines ({coords.shape[0]}) than expected ({total_atoms})")
        coordinates.append(coords)
    # Return arrays and simulation parameters
    return (
        np.array(coordinates),
        np.array(timestep_arr),
        total_atoms,
        xlo, xhi, ylo, yhi, zlo, zhi
    )
