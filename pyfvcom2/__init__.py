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
    PyFVCOM2FileNotFoundError,
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
    SigmaData,
    SigmaConfig,
    read_sigma_file,
    process_sigma_config,
    write_sigma_file,
    sigma_generalized,
    sigma_geometric,
    sigma_tanh,
    hybrid_sigma_coordinate,
)

# Import coordinate transformation functionality
from .coordinates import (
    sigma_to_z_coords,
    z_to_sigma_coords,
    get_epsg_code,
    utm_from_lonlat,
    lonlat_from_utm,
    cart2pol,
    pol2cart,
)

# Import file utilities
from .file_utils import find_file, find_files

# Import readers
from .fvcom_reader import FVCOMReader
from .cmems_reader import CMEMSReader, default_fvcom_to_cmems_var_names

# Import interpolation functionality
from .interpolation_coordinates import InterpolationCoordinates
from .interpolation import (
    Interpolator,
    CMEMSInterpolator,
    FVCOMInterpolator,
)

# Import grid functionality
from .grid import Grid, OpenBoundary

# Import restart functionality
from .restart import write_restart

# Import nest functionality  
from .nest import (
    NestManager,
    Nest,
    GridBand,
)

# Import weights calculator
from .weights_calculator import (
    get_weights_calculator,
    LinearWeightsCalculator,
    ExponentialWeightsCalculator,
    WeightsCalculator,
)

# Import ocean functions
from .ocean import (
    pressure2depth, depth2pressure, dT_adiab_sw, theta_sw, cp_sw,
    sw_smow, sw_dens0, sw_seck, sw_dens, sw_svan, sw_sal78,
    sw_sal80, sw_salinity, dens_jackett, pea, simpsonhunter,
    mixedlayerdepth, stokes, dissipation, rhum, cfl,
    turbulent_kinetic_energy,
)

# Import grid functions
from .grid import (
    Grid, OpenBoundary, connectivity, nodes2elems, find_connected_elements,
)

# Import tide harmonics
from .tide_reader import (
    FVCOMHarmonicsNames, TPXOHarmonicsNames, TPXOComplexHarmonicsNames,
    HarmonicsData, HarmonicsReader, FVCOMHarmonicsReader,
    TPXOHarmonicsReader, TPXOComplexHarmonicsReader,
)

# Define what gets imported with "from pyfvcom2 import *"
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Exceptions
    "PyFVCOM2Exception",
    "PyFVCOM2RuntimeError",
    "PyFVCOM2AttributeError",
    "PyFVCOM2ValueError",
    "PyFVCOM2TypeError",
    "PyFVCOM2FileNotFoundError",
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
    "SigmaData",
    "SigmaConfig",
    "read_sigma_file",
    "process_sigma_config",
    "write_sigma_file",
    "sigma_generalized",
    "sigma_geometric",
    "sigma_tanh",
    "hybrid_sigma_coordinate",
    # Coordinate transformations
    "sigma_to_z_coords",
    "z_to_sigma_coords",
    "get_epsg_code",
    "utm_from_lonlat",
    "lonlat_from_utm",
    "cart2pol",
    "pol2cart",
    # File utilities
    "find_file",
    "find_files",
    # Readers
    "FVCOMReader",
    "CMEMSReader",
    "default_fvcom_to_cmems_var_names",
    # Interpolation
    "InterpolationCoordinates",
    "Interpolator",
    "CMEMSInterpolator",
    "FVCOMInterpolator",
    # Grid functionality
    "Grid",
    "OpenBoundary",
    "connectivity",
    "nodes2elems",
    "find_connected_elements",
    # Restart functionality
    "write_restart",
    # Nest functionality
    "NestManager",
    "Nest",
    "GridBand",
    # Weights calculator
    "WeightsCalculator",
    "get_weights_calculator",
    "LinearWeightsCalculator",
    "ExponentialWeightsCalculator",
    # Ocean functions
    "pressure2depth", "depth2pressure", "dT_adiab_sw", "theta_sw", "cp_sw",
    "sw_smow", "sw_dens0", "sw_seck", "sw_dens", "sw_svan", "sw_sal78",
    "sw_sal80", "sw_salinity", "dens_jackett", "pea", "simpsonhunter",
    "mixedlayerdepth", "stokes", "dissipation", "rhum", "cfl",
    "turbulent_kinetic_energy",
    # Tide harmonics
    "FVCOMHarmonicsNames", "TPXOHarmonicsNames", "TPXOComplexHarmonicsNames",
    "HarmonicsData", "HarmonicsReader", "FVCOMHarmonicsReader",
    "TPXOHarmonicsReader", "TPXOComplexHarmonicsReader",
    # Subpackages
    "plotting",
]
