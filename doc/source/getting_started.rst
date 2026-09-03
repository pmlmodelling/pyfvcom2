.. _getting_started:

Getting Started
===============

After installing PyFVCOM2, check that the package imports correctly:

.. code-block:: python

   import pyfvcom2

Most workflows start by reading a mesh, loading model or forcing data, and then
using one of the higher-level managers or interpolators to prepare FVCOM inputs
or analyse outputs.

For worked examples, start with the :doc:`cookbook/index`. The cookbook covers
common tasks such as:

* comparing TPXO tidal harmonics;
* creating tide-only FVCOM inputs;
* interpolating FVCOM and CMEMS data;
* generating restart and nesting files;
* applying forcing ramps; and
* smoothing bathymetry.

For details on individual functions and classes, see the :doc:`api`.
