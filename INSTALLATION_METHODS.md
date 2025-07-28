# LAMMPSKit Installation Methods

## Overview

LAMMPSKit supports multiple installation methods to ensure compatibility across different environments and pip versions.

## Installation Methods

### 1. Standard PyPI Installation

```bash
# For end users
pip install lammpskit

# For development with optional dependencies
pip install lammpskit[dev]
```

### 2. From Source (Recommended for Development)

```bash
git clone https://github.com/simantalahkar/lammpskit.git
cd lammpskit

# Primary method (modern pip)
pip install -e .[dev]

# Fallback method (if above fails)
pip install -e . && pip install -r requirements-dev.txt
```

### 3. Hybrid Approach (Automatic Fallback)

Used in CI/CD and recommended for robust installations:

```bash
pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)
```

## Dependency Configuration

LAMMPSKit maintains dependencies in multiple formats for maximum compatibility:

1. **pyproject.toml** - `[project.optional-dependencies]` (modern standard)
2. **requirements-dev.txt** - Traditional pip requirements format  
3. **setup.py** - `extras_require` (legacy compatibility)

## Automatic Dependency Detection

**Documentation builds automatically detect dependency changes** because:

- ✅ CI/CD workflows install from current repository state
- ✅ Read the Docs pulls latest dependencies during build
- ✅ `conf.py` dynamically imports package version
- ✅ Live installation reads current `pyproject.toml`

**When you update dependencies in any configuration file, the next documentation build will automatically use the updated versions.**

## Troubleshooting

### "No module named sphinx" Error
```bash
pip install -e . && pip install -r requirements-dev.txt
```

### Optional Dependencies Not Found
```bash
pip install --upgrade pip
pip install -e .[dev]
```

### Verification
```bash
python -c "import lammpskit; print(f'LAMMPSKit {lammpskit.__version__} installed successfully')"
python -m sphinx --version  # For development installations
```

## Files Updated

- ✅ `README.md` - Enhanced installation sections with fallback methods
- ✅ `docs/source/installation.rst` - Comprehensive installation guide with troubleshooting
- ✅ `usage/ecellmodel/README.md` - Added installation instructions to usage example
- ✅ `setup.py` - Added comments explaining multiple dependency formats
- ✅ `.github/workflows/*.yml` - Already implemented hybrid installation approach

This ensures users have reliable installation methods regardless of their environment or pip version.
