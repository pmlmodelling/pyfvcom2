"""
PyFVCOM2: A Python package for processing FVCOM data.

This package provides tools for working with FVCOM (Finite Volume Community Ocean Model)
output data, including reading, processing, and analyzing unstructured grid ocean model results.
"""

# Import version information
from .version import version as __version__

# Package metadata
__author__ = "James Clark"
__email__ = "jcl@pml.ac.uk"
__license__ = "MIT"

# Import exceptions
from .exceptions import (
    PyFVCOM2Exception,
    PyFVCOM2RuntimeError,
    PyFVCOM2AttributeError,
    PyFVCOM2ValueError,
    PyFVCOM2TypeError,
)

# Define what gets imported with "from pyfvcom2 import *"
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "PyFVCOM2Exception",
    "PyFVCOM2RuntimeError",
    "PyFVCOM2AttributeError",
    "PyFVCOM2ValueError",
    "PyFVCOM2TypeError",
]
