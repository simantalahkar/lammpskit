from setuptools import setup, find_packages

setup(
    name="lammpskit",
    version="0.1.0",
    description="Toolkit for MD simulations and analysis with LAMMPS.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Simanta Lahkar",
    author_email="simantalahkar@hotmail.com",
    url="https://github.com/simantalahkar/lammpskit",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=2.3.1",
        "matplotlib>=3.10.3",
        "ovito>=3.12.4"
    ],
    extras_require={
        "dev": [
            "pytest>=8.4.1",
            "pytest-cov>=6.2.1",
            "pytest-mpl>=0.17.0",
        ]
    },
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    entry_points={
        "console_scripts": [
            "lammpskit=lammpskit.cli:main",
        ],
    },
)