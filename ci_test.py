#!/usr/bin/env python3
"""
LAMMPSKit CI/CD Local Testing Script
===================================

This script simulates the GitHub Actions workflow locally to verify
that all quality checks pass before deployment.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description}: PASSED")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: FAILED with exception: {e}")
        return False

def main():
    """Run all CI/CD checks."""
    print("=" * 50)
    print("LAMMPSKit CI/CD Local Testing")
    print("=" * 50)
    
    # Get the virtual environment python path
    venv_python = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
    
    checks = [
        (f'"{venv_python}" -m black --check lammpskit/', "Black code formatting"),
        (f'"{venv_python}" -m flake8 lammpskit/ --exclude=__pycache__ --statistics', "Flake8 linting"),
        (f'"{venv_python}" -m pytest tests/test_io.py --cov=lammpskit --quiet', "Unit tests with coverage"),
        (f'cd docs && "{venv_python}" -m sphinx -b html source build -W', "Documentation build"),
    ]
    
    results = []
    for cmd, desc in checks:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    
    all_passed = True
    for desc, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {desc}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("Your project is ready for GitHub deployment!")
        return 0
    else:
        print("⚠️ Some checks failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
