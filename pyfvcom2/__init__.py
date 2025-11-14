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

# Import mesh reading functionality
from .mesh_reader import (
    MeshData,
    read_mesh_file,
    read_sms_mesh,
    read_fvcom_mesh,
    read_smesh_mesh,
    read_mike_mesh,
    read_gmsh_mesh,
    read_fvcom_obc,
    parse_obc_sections,
)

# Import sigma coordinate functionality
from .sigma_reader import (
    SigmaConfig,
    read_sigma_file,
    process_sigma_file,
    write_sigma_file,
    sigma_generalized,
    sigma_geometric,
    sigma_tanh,
    hybrid_sigma_coordinate,
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
    # Mesh reading functionality
    "MeshData",
    "read_mesh_file",
    "read_sms_mesh",
    "read_fvcom_mesh",
    "read_smesh_mesh",
    "read_mike_mesh",
    "read_gmsh_mesh",
    "read_fvcom_obc",
    "parse_obc_sections",
    # Sigma coordinate functionality
    "SigmaConfig",
    "read_sigma_file",
    "process_sigma_file",
    "write_sigma_file",
    "sigma_generalized",
    "sigma_geometric",
    "sigma_tanh",
    "hybrid_sigma_coordinate",
    # Subpackages
    "plotting",
]
