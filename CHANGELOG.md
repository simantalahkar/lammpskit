# CHANGELOG


## [0.2.2] - 2025-07-15

### Changed
- Updated version number in project files for new release.
- Added absolute link to CHANGELOG.md in README for PyPI compatibility.
- Improved README documentation for Docker usage and project instructions.

### Fixed
- Minor documentation and deployment workflow improvements.

---

## [0.2.1] - 2025-07-14

### Added
- Added `.travis.yml` for automated testing and deployment of Docker image and PyPI package using Travis CI.
- README updated to include Docker image usage and deployment instructions.

### Changed
- Renamed `.travis.yml` correctly for CI integration.
- Updated project files for CI/CD compatibility.

---


## [0.2.0] - 2025-07-14
## lammpskit/ecellmodel/filament_layer_analysis.py

### Added
- New function: `plot_displacement_timeseries` for plotting time series of displacement data.
- Additional error handling in data reader functions (`read_structure_info`, `read_coordinates`, `read_displacement_data`).
- More robust handling of malformed or missing data in all major data reader functions.

### Changed
- Improved plotting customization in `plot_multiple_cases` (more kwargs, better axis handling).
- Refactored cluster analysis logic in `analyze_clusters` for clearer separation of filament connectivity states.
- `run_analysis` orchestrates a wider range of analyses, including forming, post-forming, set, break, and temperature-dependent studies.
- `COLUMNS_TO_READ` is now set globally and updated in `run_analysis` for different analysis scenarios.

### Fixed
- Numerous bug fixes for edge cases in file reading (as verified by expanded tests).
- Improved handling of empty arrays and zero divisions in distribution calculations.

---

## tests/test_ecellmodel/test_plot_atomic_distribution.py

### Added
- Parametrized tests for plotting atomic distributions with different numbers of files and labels.
- `pytest-mpl` image comparison for all plot types.

---

## tests/test_ecellmodel/test_plot_atomic_charge_distribution.py

### Added
- Parametrized tests for charge distribution plots for various scenarios.

---

## tests/test_ecellmodel/test_plot_displacement_comparison.py

### Added
- Parametrized tests for displacement comparison plots (atom type, temperature).

---

## tests/test_ecellmodel/test_plot_displacement_timeseries.py

### Added
- Parametrized tests for time series displacement plots.

---

## tests/test_ecellmodel/test_plot_multiple_cases.py

### Added
- Tests for all plot configurations supported by `plot_multiple_cases`.

---

## tests/test_ecellmodel/test_track_filament_evolution.py

### Added
- Parametrized tests for filament evolution plots and single-file scenarios.

---

## tests/test_ecellmodel/test_data_readers.py

### Added
- New tests for minimal file formats and typical usage scenarios.
- Tests for all error cases (missing sections, malformed lines, file not found).
- Tests for correct parsing of box bounds and atom data.

### Changed
- Expanded coverage for `read_structure_info`, `read_coordinates`, and `read_displacement_data`.
- Improved assertions for error messages and output values.

---

## Other Changes

- `.gitignore` updated to include `supporting_docs/`, `examples/`, and `data/`.

---

**Summary:**  
This release adds new analysis and plotting capabilities, improves error handling and robustness, and greatly expands automated test coverage for all major functions. The module is now more reliable and easier to maintain, with comprehensive tests for both typical and edge-case scenarios.

---
