.. _installation:

Installation
============

PyFVCOM2 is currently installed from source. Package-index installation is not
available yet, so commands such as ``pip install pyfvcom2`` or
``conda install -c conda-forge pyfvcom2`` are not documented as supported
installation methods.

Python Versions
---------------

The project metadata requires Python 3.8 or newer. The package classifiers
currently advertise Python 3.9, 3.10, and 3.11.

Normal User Installation
------------------------

Use this route if you want to install PyFVCOM2 from a local checkout without
installing development tools.

First, create and activate an environment:

.. code-block:: bash

   conda create -n pyfvcom2 -c conda-forge python=3.11 pip
   conda activate pyfvcom2

Then clone and install the package:

.. code-block:: bash

   git clone https://github.com/pmlmodelling/pyfvcom2.git
   cd pyfvcom2
   python -m pip install .

Development Installation
------------------------

Use this route if you want to edit the source code, run tests, or build the
documentation.

.. code-block:: bash

   git clone https://github.com/pmlmodelling/pyfvcom2.git
   cd pyfvcom2
   conda env create -f environment.yml
   conda activate pyfvcom2

The development environment installs PyFVCOM2 in editable mode through the
``environment.yml`` file.

To install the editable package manually in an existing environment:

.. code-block:: bash

   python -m pip install -e .

Installation Test
-----------------

Check that PyFVCOM2 imports:

.. code-block:: bash

   python -c "import pyfvcom2"

To print the installed version:

.. code-block:: bash

   python -c "import pyfvcom2; print(pyfvcom2.__version__)"

Dependencies
------------

Runtime dependencies are defined in ``pyproject.toml``. They currently include:

* ``numpy>=1.19.0``
* ``scipy>=1.5.0``
* ``matplotlib>=3.3.0``
* ``netCDF4>=1.5.0``
* ``xarray>=0.16.0``
* ``pyproj``
* ``cftime>=1.6.0``
* ``cartopy>=0.20.0``
* ``cmocean>=2.0``
* ``stripy>=0.6.0``
* ``utide``

For scientific Python environments, conda-forge is recommended for compiled
geospatial and NetCDF dependencies such as ``cartopy``, ``pyproj``, and
``netCDF4``.

Optional Development and Documentation Dependencies
---------------------------------------------------

Development dependencies, including ``pytest``, ``black``, ``flake8``, ``mypy``,
and ``pre-commit``, are listed in the ``dev`` optional dependency group in
``pyproject.toml``.

Documentation dependencies are listed in the ``docs`` optional dependency group
and in ``doc/requirements.txt``.

Troubleshooting
---------------

If installation fails while building compiled dependencies, create the
environment with conda-forge first, then install PyFVCOM2 into that environment.
This avoids many local compiler and system-library issues.
