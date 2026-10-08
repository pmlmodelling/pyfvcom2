.. _installation:

Installation
============

PyFVCOM2 is published on conda-forge and can also be installed from a source
checkout.

Python Versions
---------------

The project metadata requires Python 3.8 or newer. The package classifiers
currently advertise Python 3.9, 3.10, and 3.11.

Normal User Installation
------------------------

Install PyFVCOM2 from conda-forge with:

.. code-block:: bash

   conda create -n pyfvcom2 -c conda-forge pyfvcom2
   conda activate pyfvcom2

For a local checkout installation without development tools, use the source
installation route below.

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

Building the Conda Package Locally
----------------------------------

The recipe in ``conda/meta.yaml`` uses the tagged GitHub release for conda-forge.
To build from the current checkout instead, temporarily replace its ``source``
section with:

.. code-block:: yaml

   source:
     path: ..

Then build from the repository root:

.. code-block:: bash

   CONDA_BLD_PATH=/tmp/pyfvcom2-conda-bld conda build conda -c conda-forge

Create a test environment from the resulting local package:

.. code-block:: bash

   conda create -n pyfvcom2-conda-test -c conda-forge --use-local pyfvcom2=0.1.0
   conda activate pyfvcom2-conda-test
   python -c "import pyfvcom2; print(pyfvcom2.__version__)"

Restore the tagged-release ``source`` section before committing the recipe.

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
