# OVITO Integration in GitHub Actions CI

## Overview

LAMMPSKit requires OVITO for cluster analysis functionality. This document explains how OVITO is integrated into the GitHub Actions CI environment to enable comprehensive testing.

## OVITO Requirements

OVITO requires several system dependencies and display capabilities to function in a headless CI environment:

### System Dependencies
- **Qt5 Libraries**: Core GUI framework
- **OpenGL Libraries**: 3D rendering capabilities  
- **X11 Libraries**: Display system components
- **Virtual Display**: Xvfb for headless operation

### Environment Configuration
- `QT_QPA_PLATFORM=offscreen`: Enables Qt headless mode
- `OVITO_HEADLESS=1`: Forces OVITO headless operation
- `DISPLAY=:99`: Virtual display for X11 applications

## CI Workflow Integration

### Test Job (`tests.yml`)
```yaml
- name: Install system dependencies for OVITO
  run: |
    sudo apt-get update
    sudo apt-get install -y \
      xvfb \
      libgl1-mesa-glx \
      libglu1-mesa \
      libxrender1 \
      libxrandr2 \
      libxss1 \
      libxcursor1 \
      libxcomposite1 \
      libasound2 \
      libxi6 \
      libxtst6 \
      qtbase5-dev \
      libqt5gui5 \
      libqt5widgets5 \
      libqt5opengl5-dev \
      libqt5core5a

- name: Set up environment for OVITO
  run: |
    echo "QT_QPA_PLATFORM=offscreen" >> $GITHUB_ENV
    echo "OVITO_HEADLESS=1" >> $GITHUB_ENV

- name: Run tests with virtual display
  run: |
    export DISPLAY=:99
    Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
    python -m pytest tests/ --cov=lammpskit --cov-report=xml --cov-report=html
```

### Documentation Job (`tests.yml` & `docs.yml`)
Same system dependencies and environment setup, with virtual display for Sphinx builds that import OVITO-dependent modules.

## Benefits

1. **Complete Test Coverage**: All cluster analysis functions are tested in CI
2. **Dependency Validation**: Ensures OVITO integration works across environments
3. **Documentation Builds**: Sphinx can import all modules without errors
4. **Production Readiness**: CI validates real-world usage scenarios

## Fallback Behavior

If OVITO is not available:
- Functions raise `ImportError` with clear installation instructions
- Tests fail as expected (no code changes to mask missing dependencies)
- User gets actionable error messages

## Testing Strategy

- ✅ **Install OVITO**: Full dependency installation in CI
- ✅ **Test Functions**: Complete cluster analysis testing
- ❌ **Don't Mock**: No code changes to mask missing dependencies
- ✅ **Clear Errors**: Informative error messages for users

This approach ensures that LAMMPSKit's OVITO-dependent functionality is thoroughly tested while maintaining clear dependency requirements for end users.
