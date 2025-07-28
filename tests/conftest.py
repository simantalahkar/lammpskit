"""
Shared pytest configuration and fixtures for LAMMPSKit test suite.

This module provides common pytest configuration, fixtures, and utilities
used across all test modules in the LAMMPSKit test suite.

Centralized Baseline Testing
----------------------------
All pytest-mpl baseline images are stored in the centralized `tests/baseline/`
directory regardless of test file location. Test modules use relative paths
to access this directory:

- Root level tests (tests/test_*.py): baseline_dir="baseline"
- Subdirectory tests (tests/test_*/test_*.py): baseline_dir="../baseline"

This approach ensures consistent behavior across development environments
and platforms (Windows, Linux, macOS, Docker).

Fixtures
--------
baseline_dir : fixture
    Provides the appropriate relative path to the baseline directory
    based on the test file location.

test_output_dir : fixture
    Creates and provides access to temporary output directories for tests.

Markers
-------
visual : Custom marker for visual regression tests
mpl_image_compare : Marker for matplotlib image comparison tests
slow : Marker for tests that take longer to execute
integration : Marker for integration tests

Example Usage
-------------
```python
def test_plotting_function(baseline_dir):
    # baseline_dir automatically resolves to correct relative path
    @pytest.mark.mpl_image_compare(baseline_dir=baseline_dir)
    def test_plot():
        # Test implementation
        pass
```
"""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def baseline_dir():
    """
    Determine the appropriate relative path to the baseline directory.
    
    Returns the relative path from the current test file to the centralized
    baseline directory at tests/baseline/. This ensures all tests use the
    same baseline images regardless of their location in the test directory.
    
    Returns
    -------
    str
        Relative path to baseline directory ("baseline" or "../baseline")
    """
    # For tests in root tests/ directory: use "baseline"
    # For tests in subdirectories: use "../baseline"
    current_file = Path(__file__).parent
    
    # Check if we're in a subdirectory of tests/
    if current_file.name != "tests":
        return "../baseline"
    else:
        return "baseline"


@pytest.fixture(scope="function")
def test_output_dir():
    """
    Create a temporary directory for test outputs.
    
    Provides a clean temporary directory for each test function.
    The directory is automatically cleaned up after the test completes.
    
    Yields
    ------
    Path
        Path to temporary output directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """
    Configure the test environment for consistent behavior.
    
    Sets up environment variables and configurations needed for
    reliable test execution across different platforms and CI environments.
    """
    # Ensure matplotlib uses non-interactive backend
    os.environ["MPLBACKEND"] = "Agg"
    
    # Configure Qt for headless operation
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    
    # Configure OVITO for headless operation
    if "OVITO_HEADLESS" not in os.environ:
        os.environ["OVITO_HEADLESS"] = "1"


# Custom test collection modifiers
def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add appropriate markers.
    
    Automatically adds markers to tests based on their characteristics:
    - Tests with 'mpl_image_compare' decorator get 'visual' marker
    - Tests in certain modules get 'slow' marker
    """
    for item in items:
        # Add visual marker to matplotlib image comparison tests
        if hasattr(item, 'pytestmark'):
            for mark in item.pytestmark:
                if mark.name == 'mpl_image_compare':
                    item.add_marker(pytest.mark.visual)
                    break
        
        # Add slow marker to computationally intensive tests
        if "analyze_clusters" in item.name or "track_filament" in item.name:
            item.add_marker(pytest.mark.slow)


# Cleanup utilities
@pytest.fixture(scope="session", autouse=True)
def cleanup_test_artifacts():
    """
    Clean up test artifacts after test session.
    
    Removes temporary files and directories created during testing
    to keep the workspace clean.
    """
    yield  # Run tests
    
    # Cleanup after all tests complete
    import glob
    
    # Remove temporary plot files
    for pattern in ["*.png", "*.pdf", "*.svg"]:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass  # File may have been already removed
