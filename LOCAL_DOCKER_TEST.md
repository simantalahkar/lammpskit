# Local Docker Testing Guide for Windows PowerShell

## Quick Test Commands

To test the Docker-based CI locally before pushing (Windows PowerShell commands):

### 1. Build the Docker Image
```powershell
docker build -t simantalahkar/lammpskit:test .
```

### 2. Test Container with Volume Mounts
```powershell
# Create test container (OVITO dependencies pre-installed in Dockerfile)
docker run -d --name test_container `
  --user root `
  -v ${PWD}/tests:/app/tests `
  -v ${PWD}/supporting_docs:/app/supporting_docs `
  -v ${PWD}/usage:/app/usage `
  -e QT_QPA_PLATFORM=offscreen `
  -e OVITO_HEADLESS=1 `
  simantalahkar/lammpskit:test `
  bash -c "sleep infinity"

# Install testing dependencies
docker exec test_container pip install `
  pytest pytest-cov pytest-mpl coverage

# Verify OVITO and LAMMPSKit
docker exec test_container python -c "
import ovito; print('OVITO version:', ovito.version)
import lammpskit; print('LAMMPSKit version:', lammpskit.__version__)
from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters
print('analyze_clusters function imported successfully')
"

# Run tests (from project root directory within container)
docker exec test_container bash -c "
  export DISPLAY=:99
  Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
  cd /app && python -m pytest --mpl-generate-path=tests/baseline tests/ || echo 'Baseline generation complete'
  cd /app && python -m pytest --mpl --mpl-baseline-path=tests/baseline --cov=lammpskit --cov-report=xml tests/
"

# Clean up
docker rm -f test_container
```

### 3. Test Documentation Build
```powershell
# Create docs container (OVITO dependencies pre-installed in Dockerfile)
docker run -d --name docs_container `
  --user root `
  -v ${PWD}/docs:/app/docs `
  -e QT_QPA_PLATFORM=offscreen `
  -e OVITO_HEADLESS=1 `
  simantalahkar/lammpskit:test `
  bash -c "sleep infinity"

# Install documentation dependencies
docker exec docs_container pip install `
  sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Build documentation
docker exec docs_container bash -c "
  export DISPLAY=:99
  Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
  cd /app/docs && mkdir -p build && python -m sphinx -b html source build
"

# Clean up
docker rm -f docs_container
```

### 4. Alternative: One-Shot Testing
For quick verification without persistent containers:

```powershell
# Build and test in one command
docker run --rm `
  --user root `
  -v ${PWD}/tests:/app/tests `
  -v ${PWD}/supporting_docs:/app/supporting_docs `
  -v ${PWD}/usage:/app/usage `
  -e QT_QPA_PLATFORM=offscreen `
  -e OVITO_HEADLESS=1 `
  simantalahkar/lammpskit:test `
  bash -c "
    pip install pytest pytest-cov pytest-mpl coverage
    export DISPLAY=:99
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    python -c 'import ovito; import lammpskit; from lammpskit.ecellmodel.filament_layer_analysis import analyze_clusters; print(\"All imports successful\")'
    cd /app && python -m pytest --mpl-generate-path=tests/baseline tests/ || echo 'Baseline generation complete'
    cd /app && python -m pytest --mpl --mpl-baseline-path=tests/baseline --cov=lammpskit --cov-report=xml tests/
  "
```
```

## Key Improvements with Revised Dockerfile

### Pre-installed Dependencies
- **OVITO System Dependencies**: Qt5, OpenGL, X11 libraries included in base image
- **Virtual Display Support**: Xvfb pre-installed for headless operations
- **No Runtime Installation**: System dependencies already available

### Volume-Based Testing
- **Tests Not Copied**: Test files mounted as volumes for live updates
- **Baseline Management**: Tests/baseline directory accessible via volume mounts
- **Development Friendly**: Change tests locally, immediately available in container
- **Python Module Execution**: Uses `python -m pytest` instead of `pytest` command for reliable PATH resolution

### Simplified Commands
- **Removed**: Complex system dependency installation steps
- **Faster**: No need to update package lists or install system packages
- **Cleaner**: Focus on Python dependencies and testing

## Verification Checklist

- [ ] Docker image builds successfully (with OVITO dependencies)
- [ ] OVITO imports without errors in container
- [ ] LAMMPSKit imports and shows correct version (1.2.0)
- [ ] `analyze_clusters` function can be imported
- [ ] Tests run without ImportError exceptions
- [ ] Documentation builds successfully
- [ ] Volume mounts work (test files accessible in container)
- [ ] Baseline directory accessible at `/app/tests/baseline/` in container
- [ ] Virtual display (Xvfb) starts properly
- [ ] Coverage reports generate correctly

## Troubleshooting

### Common Issues
- **Volume Mount Paths**: Ensure you're in the project root directory
- **Docker Desktop**: Make sure Docker Desktop is running on Windows
- **PowerShell Execution**: Run PowerShell as Administrator if needed
- **Documentation Build Permissions**: Use `--user root` for docs container to avoid permission issues with volume mounts
- **Coverage Database Permissions**: Use `--user root` for test container to avoid SQLite database permission errors

### Verification Commands
```powershell
# Check if you're in the right directory
ls # Should show pyproject.toml, Dockerfile, tests/, docs/, etc.

# Verify Docker is running
docker --version

# Check available disk space
docker system df
```

This streamlined approach leverages the revised Dockerfile with pre-installed OVITO dependencies, making local testing much faster and more reliable.
