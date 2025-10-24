.. _installation:

************
Installation
************

PyFVCOM2 can be installed using pip or conda. We recommend using conda for scientific Python environments as it handles complex dependencies more reliably.

Using Conda (Recommended)
=========================

Create a new conda environment with PyFVCOM2::

    conda env create -f environment.yml
    conda activate pyfvcom2

Or install into an existing environment::

    conda install -c conda-forge pyfvcom2

Using Pip
=========

Install from PyPI::

    pip install pyfvcom2

For development installation::

    git clone https://github.com/pmlmodelling/pyfvcom2.git
    cd pyfvcom2
    pip install -e .

Dependencies
============

PyFVCOM2 requires the following packages:

**Core Dependencies:**

- Python >= 3.8
- NumPy >= 1.19
- SciPy >= 1.5
- NetCDF4 >= 1.5
- xarray >= 0.16

**Visualization Dependencies:**

- Matplotlib >= 3.3
- Cartopy >= 0.18

**Optional Dependencies:**

- Jupyter (for notebook examples)

Verifying Installation
======================

Test your installation::

    python -c "import pyfvcom2; print(pyfvcom2.__version__)"

Run the test suite::

    python -m pytest tests/

Troubleshooting
===============

**Common Issues:**

1. **Cartopy installation problems**: Install via conda-forge channel
2. **NetCDF4 library not found**: Install libnetcdf development headers
3. **GEOS/PROJ errors**: Update to latest cartopy version

**Getting Help:**

- Check the GitHub Issues page
- Join the PyFVCOM2 discussions
- Contact the development team