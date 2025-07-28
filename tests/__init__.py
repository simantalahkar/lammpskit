"""
LAMMPSKit Test Suite

This package contains comprehensive tests for LAMMPSKit, including:
- Unit tests for core functionality
- Integration tests for complex workflows  
- Visual regression tests for plotting functions

Visual Regression Testing with Centralized Baselines
===================================================

All visual regression tests use a centralized baseline directory approach:

Directory Structure:
    tests/
    ├── baseline/                 # Centralized baseline images (ALL tests)
    ├── test_*.py                # Tests at root level (use "baseline")
    ├── test_ecellmodel/         # Subdirectory tests (use "../baseline")
    │   └── test_*.py            # Tests in subdirs (use relative paths)
    └── other_subdirs/

Baseline Path Configuration:
- **Root level tests**: `BASELINE_DIR_RELATIVE = "baseline"`
- **Subdirectory tests**: `BASELINE_DIR_RELATIVE = "../baseline"`

Example Usage:
    ```python
    # For tests in tests/ directory
    BASELINE_DIR_RELATIVE = "baseline"
    
    # For tests in tests/test_ecellmodel/ directory  
    BASELINE_DIR_RELATIVE = "../baseline"
    
    @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
    def test_plotting_function():
        # Test implementation
        return figure
    ```

Why Centralized Baselines?
--------------------------
1. **Consistency**: All baseline images in single location
2. **Cross-platform compatibility**: Relative paths work everywhere
3. **Container compatibility**: Works in Docker and local environments
4. **Maintainability**: Easier baseline management and organization
5. **CI/CD integration**: Simplified path handling in workflows

Baseline Management:
-------------------
- Generate: `pytest --mpl-generate-path=tests/baseline tests/`
- Compare: `pytest --mpl --mpl-baseline-path=tests/baseline tests/`
- Update: Regenerate baselines when plot functions change intentionally

For detailed testing guidelines, see CONTRIBUTING.md and LOCAL_DOCKER_TEST.md
"""

__version__ = "1.2.0"
__all__ = []
