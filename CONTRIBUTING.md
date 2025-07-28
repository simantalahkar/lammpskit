# Contributing to LAMMPSKit

Thank you for your interest in contributing to LAMMPSKit! This document provides guidelines for contributing to the project.

## Code of Conduct

This project follows the [Python Community Code of Conduct](https://www.python.org/psf/conduct/). Please be respectful and inclusive in all interactions.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lammpskit.git
   cd lammpskit
   ```

3. Create a virtual environment and install in development mode:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

4. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```

### Development Workflow

1. Make your changes
2. Add/update tests for your changes
3. Update documentation if needed
4. Run tests locally:
   ```bash
   # Run all tests
   python -m pytest tests/
   
   # Run with visual regression testing
   python -m pytest tests/ --mpl
   
   # Run with coverage
   python -m pytest tests/ --cov=lammpskit --cov-report=html
   ```
5. Build documentation to check for errors:
   ```bash
   cd docs
   python -m sphinx -b html source build
   ```
6. Commit your changes with a descriptive message
7. Push to your fork and create a pull request

## Code Standards

### Code Style
- Follow PEP 8 for Python code style
- Use `black` for automatic code formatting
- Use `isort` for import sorting
- Maximum line length: 127 characters

### Documentation
- All public functions must have comprehensive docstrings
- Use NumPy/Google docstring format
- Include scientific context and examples
- Update user documentation for new features

### Testing Guidelines

#### General Testing Requirements
- Write tests for all new functionality
- Maintain or improve test coverage (target: >90%)
- Use pytest for testing framework
- Follow existing test patterns and naming conventions

#### Visual Regression Testing for Plotting Functions

LAMMPSKit uses a **centralized baseline approach** for visual regression testing. All baseline images are stored in `tests/baseline/` regardless of where the test files are located.

**For New Plotting Functions:**

1. **Add visual regression test** using the centralized baseline pattern:
   ```python
   # Determine relative path based on test file location
   # Root level (tests/test_*.py): 
   BASELINE_DIR_RELATIVE = "baseline"
   
   # Subdirectory (tests/test_ecellmodel/test_*.py):
   BASELINE_DIR_RELATIVE = "../baseline"
   
   @pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR_RELATIVE, remove_text=True)
   def test_your_plotting_function(tmp_path):
       # Your test implementation
       fig = your_plotting_function(test_data)
       return fig
   ```

2. **Generate baseline images** for your new tests:
   ```bash
   # Generate baselines for new tests
   pytest --mpl-generate-path=tests/baseline tests/test_your_file.py::test_your_function
   
   # Verify baselines were created correctly
   pytest --mpl tests/test_your_file.py::test_your_function
   ```

3. **Update baselines** when plot functions change intentionally:
   ```bash
   # Regenerate all baselines
   pytest --mpl-generate-path=tests/baseline tests/
   
   # Regenerate specific test baselines
   pytest --mpl-generate-path=tests/baseline tests/test_specific.py
   ```

**Directory Structure:**
```
tests/
├── baseline/                    # All baseline images (centralized)
├── test_plotting.py            # Root level → "baseline"
├── test_timeseries_plots.py    # Root level → "baseline" 
├── test_ecellmodel/            # Subdirectory tests
│   ├── test_plot_*.py          # Subdirectory → "../baseline"
│   └── test_data/              # Test data files
└── conftest.py                 # Shared configuration
```

**Why Centralized Baselines?**
- **Cross-platform compatibility**: Works on Windows, Linux, macOS
- **Container compatibility**: Identical behavior in Docker and local environments
- **Maintainability**: Single location for all visual regression references
- **CI/CD integration**: Simplified path handling in automated testing

**Testing Commands:**
```bash
# Test everything with visual regression
pytest --mpl

# Generate new baselines (after intentional plot changes)
pytest --mpl-generate-path=tests/baseline tests/

# Test specific plotting module
pytest --mpl tests/test_plotting.py

# Local development with coverage
pytest --mpl --cov=lammpskit --cov-report=html tests/
```

## Scientific Contributions

### Domain Expertise
We especially welcome contributions from researchers working with:
- LAMMPS molecular dynamics simulations
- Electrochemical memory devices (ReRAM, CBRAM)
- Materials science simulations
- Atomic-scale analysis methods

### Types of Contributions
- **Bug fixes**: Corrections to existing functionality
- **New analysis functions**: Additional post-processing capabilities
- **Performance improvements**: Optimization of existing algorithms
- **Documentation**: Improvements to user guides and examples
- **Examples**: Real-world usage demonstrations

## Pull Request Guidelines

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation builds without warnings
- [ ] Changes are well-documented
- [ ] Commit messages are descriptive

### PR Requirements
- Provide clear description of changes
- Link to related issues if applicable
- Include test results
- Request review from maintainers

## Issue Reporting

### Bug Reports
- Use the bug report template
- Include minimal reproducible example
- Provide environment details
- Include full error traceback

### Feature Requests
- Use the feature request template
- Explain scientific motivation
- Provide example use cases
- Consider implementation complexity

## Questions and Support

- **Documentation**: Check the [documentation](https://lammpskit.readthedocs.io/en/latest/) first
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact maintainers for sensitive issues

## License

By contributing to LAMMPSKit, you agree that your contributions will be licensed under the GPL-3.0-or-later license.
