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

   pip install lammpskit[dev]

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
   pip install -e .[dev]

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
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Run tests to verify setup
   pytest

The test suite includes 270+ test functions and 205 baseline images for visual regression testing.
