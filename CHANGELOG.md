# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Enhanced
- **Visual Regression Testing**: Modernized pytest-mpl infrastructure with centralized baseline directory approach
- **Cross-Platform Compatibility**: Implemented relative path resolution for consistent test behavior across Windows, Linux, macOS, and Docker environments
- **Test Infrastructure**: Added comprehensive test package documentation (`tests/__init__.py`) and shared pytest configuration (`tests/conftest.py`)
- **Documentation Coverage**: Updated README.md, CONTRIBUTING.md, and PROJECT_STRUCTURE.md with detailed visual regression testing guidelines

### Fixed
- **Docker Build**: Removed unnecessary `COPY tests` from Dockerfile to prevent GitHub Actions build failures; tests now exclusively use volume mounts for better development workflow
- **CI/CD Python Indentation**: Fixed IndentationError in GitHub Actions workflow by correcting Python command indentation in docker exec statements
- **CI/CD pytest Command**: Fixed "pytest: command not found" error by using `python -m pytest` instead of direct `pytest` command in Docker containers
- **Documentation Build Permissions**: Fixed "Permission denied" error in Sphinx documentation build by running docs container as root user for volume mount compatibility

## [1.1.0] - 2025-07-28

### Added
- **CI/CD Infrastructure**: Complete GitHub Actions workflows for automated testing, documentation building, and code quality checks
- **Professional Documentation**: Sphinx-based documentation with Read the Docs deployment and hybrid autosummary structure
- **Code Quality Tools**: Black formatting, flake8 linting, isort import sorting, and mypy type checking
- **Collaboration Framework**: Issue templates, PR templates, contributing guidelines, and CODEOWNERS configuration
- **Enhanced Dependencies**: Updated all development dependencies to latest versions with comprehensive code quality toolchain

### Enhanced
- **Project Metadata**: Added comprehensive package classifiers, keywords, and enhanced project URLs
- **Development Workflow**: Local CI/CD testing script for pre-deployment validation
- **Documentation Links**: Updated all references to point to https://lammpskit.readthedocs.io/

### Changed
- **Version**: Upgraded to v1.1.0 to reflect comprehensive project modernization
- **Dependency Management**: Reorganized and updated all development dependencies with clear categorization

## [1.0.0] - 2025-07-23

### Added
- **New timeseries plotting module** (`lammpskit.plotting.timeseries_plots`) with centralized plotting logic and configuration
- **Comprehensive example workflow** (`usage/ecellmodel/run_analysis.py`) demonstrating 4 main analysis types:
  - Filament evolution tracking
  - Displacement analysis for different atomic species  
  - Atomic charge distribution analysis
  - Atomic distribution analysis under different voltages
- **Configuration management system** (`lammpskit.config`) with centralized settings including `COLUMNS_TO_READ` and `EXTENDED_COLUMNS_TO_READ`
- **Modular data processing module** (`lammpskit.ecellmodel.data_processing`) extracting reusable analysis functions
- **Enhanced validation logic** across I/O functions with centralized error handling and descriptive error messages
- **Comprehensive test suite** with 272 new test functions and 205 baseline images for visual regression testing
- **Project structure documentation** (`PROJECT_STRUCTURE.md`, `PROJECT_STRUCTURE.txt`) with complete package organization
- **Usage examples and output** in `usage/ecellmodel/` with real analysis results demonstrating package capabilities

### Changed
- **Complete package restructuring** from monolithic design to modular architecture:
  - Separated I/O functions into `lammpskit.io.lammps_readers`
  - Extracted plotting utilities to `lammpskit.plotting.utils` and `lammpskit.plotting.timeseries_plots`
  - Modularized analysis functions in `lammpskit.ecellmodel.data_processing`
- **Enhanced function signatures** with explicit parameters for `columns_to_read` defaulting to configuration values
- **Improved documentation** with comprehensive docstrings, type hints, and scientific context across all modules
- **Updated plotting functions** with better output formatting, font control, and visualization quality
- **Standardized validation logic** with consistent error messages and fail-fast philosophy throughout the package

### Removed
- **Redundant modules and functions** including unused plotting code and duplicate functionality
- **Unused imports and variables** throughout codebase for improved clarity and performance
- **Over-engineered validation functions** replaced with streamlined, focused implementations

### Fixed
- **Documentation formatting issues** that caused baseline image mismatches in tests
- **Atom type ID and element categorization** inconsistencies in package code and tests
- **Plot formatting issues** in displacement timeseries functions affecting visual output quality
- **Import structure** and dependency management across modules

### Security
- **Enhanced input validation** with comprehensive error checking and descriptive failure messages
- **Improved file handling** with better error recovery and validation across I/O operations

---

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
