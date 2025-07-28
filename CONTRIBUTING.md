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
   python -m pytest tests/
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

### Testing
- Write tests for all new functionality
- Maintain or improve test coverage
- Use pytest for testing framework
- Include visual regression tests for plotting functions

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

- **Documentation**: Check the [documentation](https://lammpskit.readthedocs.io/) first
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Email**: Contact maintainers for sensitive issues

## License

By contributing to LAMMPSKit, you agree that your contributions will be licensed under the GPL-3.0-or-later license.
