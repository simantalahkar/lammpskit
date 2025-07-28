Installation
============

Requirements
------------

**Runtime requirements (for using the package):**

- Python 3.12+
- numpy ≥ 2.3.1
- matplotlib ≥ 3.10.3
- ovito ≥ 3.12.4

**Development & testing requirements (for contributing, testing, or building):**

- All runtime requirements above
- pytest ≥ 8.4.1
- pytest-cov ≥ 6.2.1
- pytest-mpl ≥ 0.17.0
- sphinx ≥ 8.2.3
- sphinx-autodoc-typehints ≥ 3.2.0
- sphinx-rtd-theme ≥ 3.0.2

Install from PyPI
-----------------

For end users (runtime):

.. code-block:: bash

   pip install lammpskit

For development and testing:

.. code-block:: bash

   # Primary method (modern pip with optional dependencies)
   pip install lammpskit[dev]
   
   # Alternative method (if above fails with older pip versions)
   pip install lammpskit
   pip install -r https://raw.githubusercontent.com/simantalahkar/lammpskit/main/requirements-dev.txt

Install from Source
-------------------

Clone the repository and install:

.. code-block:: bash

   git clone https://github.com/simantalahkar/lammpskit.git
   cd lammpskit
   pip install -e .

For development:

.. code-block:: bash

   git clone https://github.com/simantalahkar/lammpskit.git
   cd lammpskit
   
   # Primary method (modern pip with optional dependencies)
   pip install -e .[dev]
   
   # Alternative method (if above fails with older pip versions)
   pip install -e . && pip install -r requirements-dev.txt

Docker Installation
-------------------

An official Docker image is available on Docker Hub:

.. code-block:: bash

   docker pull simantalahkar/lammpskit:latest

Run the container with your data mounted:

.. code-block:: bash

   docker run -it -v /path/to/your/data:/data simantalahkar/lammpskit:latest

Verification
------------

Verify your installation:

.. code-block:: python

   import lammpskit
   print(lammpskit.__version__)
   
   # Test basic functionality
   from lammpskit.config import DEFAULT_COLUMNS_TO_READ
   print("Installation successful!")

Development Setup
-----------------

For contributing to LAMMPSKit:

.. code-block:: bash

   git clone https://github.com/simantalahkar/lammpskit.git
   cd lammpskit
   
   # Recommended approach with automatic fallback
   pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)
   
   # Run tests to verify setup
   pytest

**Alternative Installation Methods:**

The package supports multiple dependency installation approaches for maximum compatibility:

1. **Modern approach** (Python 3.8+, pip 21.2+):

   .. code-block:: bash
   
      pip install -e .[dev]

2. **Traditional approach** (any pip version):

   .. code-block:: bash
   
      pip install -e .
      pip install -r requirements-dev.txt

3. **Hybrid approach** (automatic fallback):

   .. code-block:: bash
   
      pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)

The test suite includes 270+ test functions and 205 baseline images for visual regression testing.

Dependency Management
---------------------

**Automatic Dependency Detection in Documentation:**

When dependencies change in ``pyproject.toml`` or ``requirements-dev.txt``, documentation builds will **automatically detect and use the updated dependencies** because:

1. **CI/CD Integration**: GitHub Actions workflows install dependencies from the current repository state during each build
2. **Live Installation**: Documentation builds use ``pip install -e .[dev]`` which reads the current ``pyproject.toml`` 
3. **Read the Docs**: Automatically pulls the latest repository state and installs current dependencies
4. **Version Synchronization**: The ``conf.py`` imports the package to get the current version dynamically

**Dependency Configuration Files:**

The package maintains dependencies in multiple formats for compatibility:

- ``pyproject.toml`` - Modern Python packaging standard (``[project.optional-dependencies]``)
- ``requirements-dev.txt`` - Traditional pip requirements format
- ``setup.py`` - Legacy setuptools format (``extras_require``)

Changes to any of these files are automatically reflected in the next documentation build cycle.

Troubleshooting
---------------

**Common Installation Issues:**

1. **"No module named sphinx" Error:**
   
   Use the fallback installation method:
   
   .. code-block:: bash
   
      pip install -e . && pip install -r requirements-dev.txt

2. **Optional Dependencies Not Found:**
   
   Older pip versions may not support ``[project.optional-dependencies]``. Use:
   
   .. code-block:: bash
   
      pip install --upgrade pip
      pip install -e .[dev]

3. **CI/CD Build Failures:**
   
   GitHub Actions uses the hybrid approach automatically:
   
   .. code-block:: bash
   
      pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)

**Verify Installation:**

.. code-block:: bash

   python -c "import lammpskit; print(f'LAMMPSKit {lammpskit.__version__} installed successfully')"
   python -m sphinx --version  # For development installations
