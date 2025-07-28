from setuptools import setup, find_packages

setup(
    name="lammpskit",
    version="1.2.0",
    description="Toolkit for MD simulations and analysis with LAMMPS - specialized for electrochemical memory device characterization.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Simanta Lahkar",
    author_email="simantalahkar@hotmail.com",
    url="https://github.com/simantalahkar/lammpskit",
    project_urls={
        "Documentation": "https://lammpskit.readthedocs.io/en/latest/",
        "Source": "https://github.com/simantalahkar/lammpskit",
        "Bug Tracker": "https://github.com/simantalahkar/lammpskit/issues",
        "Changelog": "https://github.com/simantalahkar/lammpskit/blob/main/CHANGELOG.md",
    },
    license="GPL-3.0-or-later",
    packages=find_packages(exclude=["tests", "tests.*", "supporting_docs", "supporting_docs.*"]),
    include_package_data=True,
    python_requires=">=3.12",
    keywords=["LAMMPS", "molecular-dynamics", "ReRAM", "electrochemical", "memory-devices", "filament-analysis", "HfTaO", "materials-science"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    install_requires=[
        "numpy>=2.3.1",
        "matplotlib>=3.10.3",
        "ovito>=3.12.4"
    ],
    # Development dependencies for legacy compatibility
    # Note: Also available in pyproject.toml [project.optional-dependencies] and requirements-dev.txt
    # Install with: pip install -e .[dev] || (pip install -e . && pip install -r requirements-dev.txt)
    extras_require={
        "dev": [
            # Testing framework
            "pytest>=8.4.1",
            "pytest-cov>=6.2.1",
            "pytest-mpl>=0.17.0",
            "coverage>=7.9.1",
            
            # Documentation tools
            "sphinx>=8.2.3",
            "sphinx-autodoc-typehints>=3.2.0",
            "sphinx-rtd-theme>=3.0.2",
            "Jinja2>=3.1.6",
            "MarkupSafe>=3.0.2",
            
            # Code quality and formatting
            "black>=25.1.0",
            "isort>=5.13.0",
            "flake8>=7.3.0",
            "mypy>=1.8.0",
            
            # Build and distribution
            "build>=1.2.2.post1",
            "setuptools>=80.9.0",
            "wheel>=0.45.1",
            "twine>=6.1.0",
            "readme_renderer>=44.0",
            
            # Development utilities
            "pillow>=11.2.1",
            "packaging>=25.0",
            "cycler>=0.12.1",
            "fonttools>=4.58.4",
            "pluggy>=1.6.0",
            "pyparsing>=3.2.3",
            "python-dateutil>=2.9.0.post0",
            "requests>=2.32.4",
            "requests-toolbelt>=1.0.0",
            "rich>=14.0.0",
        ]
    },
)