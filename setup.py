from setuptools import setup, find_packages

setup(
    name="lammpskit",
    version="1.0.0",
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
            "Jinja2>=3.1.6",
            "MarkupSafe>=3.0.2",
            "pillow>=11.2.1",
            "packaging>=25.0",
            "build>=1.2.2.post1",
            "coverage>=7.9.1",
            "cycler>=0.12.1",
            "fonttools>=4.58.4",
            "pluggy>=1.6.0",
            "pyparsing>=3.2.3",
            "python-dateutil>=2.9.0.post0",
            "readme_renderer>=44.0",
            "requests>=2.32.4",
            "requests-toolbelt>=1.0.0",
            "rich>=14.0.0",
            "setuptools>=80.9.0",
            "twine>=6.1.0"
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
)