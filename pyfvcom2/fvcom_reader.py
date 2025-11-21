"""FVCOM data reader for PyFVCOM2"""

__all__ = ["FVCOMReader"]

import numpy as np
from netCDF4 import Dataset

from .interpolation_coordinates import InterpolationCoordinates
from .coordinates import sigma_to_z_coords
from .grid import Grid
from .mesh_reader import MeshData
from .sigma_reader import SigmaData


class FVCOMReader:
    """A class to read FVCOM model output and restart files.

    This class provides methods to read FVCOM netCDF files and extract relevant data.

    Attributes:
        filepath (str): Path to the FVCOM netCDF file.
        dataset (xarray.Dataset): The loaded FVCOM dataset.
    """

    def __init__(self, filepath):
        """Initialize the FVCOMReader with the path to the netCDF file.

        Args:
            filepath (str): Path to the FVCOM netCDF file.
        """
        self.filepath = filepath
        self.dataset = None
        self._grid = None  # Lazy initialization

        # Load the dataset upon initialization
        self._load_data()

    @property
    def grid(self) -> Grid:
        """Get the Grid object (constructed lazily).
        
        Returns:
            Grid: The grid object containing mesh structure and open boundaries.
        """
        if self._grid is None:
            mesh_data = self._extract_mesh_data()
            sigma_data = self._extract_sigma_data()
            self._grid = Grid(mesh_data, sigma_data, "geographic")
        return self._grid

    def _load_data(self):
        """Load the FVCOM netCDF file into an xarray Dataset."""
        self.dataset = Dataset(self.filepath)

    @property
    def n_nodes(self):
        """Get the number of nodes in the FVCOM grid."""
        return self.grid.n_nodes

    @property
    def n_elements(self):
        """Get the number of elements in the FVCOM grid."""
        return self.grid.n_elements

    @property
    def n_sigma_layers(self):
        """Get the number of sigma layers in the FVCOM grid."""
        return self.grid.n_sigma_layers

    @property
    def n_sigma_levels(self):
        """Get the number of sigma levels in the FVCOM grid."""
        return self.grid.n_sigma_levels

    @property
    def lon_nodes(self):
        """Get the longitude values from the dataset."""
        return self.grid.lon

    @property
    def lat_nodes(self):
        """Get the latitude values from the dataset."""
        return self.grid.lat

    @property
    def lon_elements(self):
        """Get the longitude values of element centroids."""
        return self.grid.lonc

    @property
    def lat_elements(self):
        """Get the latitude values of element centroids."""
        return self.grid.latc

    @property
    def sigma_layers_nodes(self):
        """Get the sigma layer values at nodes."""
        return self.grid.sigma_layers

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids."""
        return self.grid.sigmac_layers

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes (transformed so to be positive up)"""
        return self.grid.bathy_nodes * -1.0

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids (transformed so to be positive up)"""
        return self.grid.bathy_elements * -1.0

    def get_var(self, var_name):
        """Return the data for a given variable name.

        Effectively a wrapper around Dataset. Warn if the variable
        contains masked data for any reason.

        Args:
            var_name (str): The name of the variable to retrieve.
        Returns:
            np.ndarray: The data array for the specified variable.
        """
        return self._return_variable_data(var_name)
    
    def get_interpolation_coordinates(self, grid_position: str) -> InterpolationCoordinates:
        """Get interpolation coordinates for a specific grid position.

        Wrapper for Grid.get_interpolation_coordinates.

        Args:
            grid_position: The grid position ('node' or 'element') for which to retrieve
            interpolation coordinates.

        Returns:
            InterpolationCoordinates: The interpolation coordinates for the specified grid position.
        """
        return self.grid.get_interpolation_coordinates(grid_position)

    def _extract_mesh_data(self) -> MeshData:
        """Extract mesh data from FVCOM output file.
        
        Returns:
            MeshData: Mesh data object compatible with Grid construction.
        """
        # Extract basic mesh components
        nodes = np.arange(1, self.dataset.dimensions['node'].size+1) # TBC zero based indexing kept here.
        triangles = self.dataset.variables['nv'][:].T - 1  # Convert to 0-based indexing, transpose to (n_elem, 3)
        x1 = self.dataset.variables['lon'][:]
        x2 = self.dataset.variables['lat'][:]
        x3 = self._return_variable_data('h')[:]

        open_bdy_node_lists = None
        bdy_types = None

        return MeshData(triangles, nodes, x1, x2, x3, bdy_types, open_bdy_node_lists)
    
    def _extract_sigma_data(self) -> SigmaData:
        """Extract sigma coordinate data from FVCOM output file.
        
        Returns:
            SigmaData: Sigma data object compatible with Grid construction.
        """
        # Generate "dummy" sigma configuration
        sigma_config = {
            'sigma_type': 'dummy',  # Assume generalised coordinates
            'sigma_power': np.nan,
            'sigma_theta': np.nan,
            'sigma_b': np.nan
        }
        
        # Extract sigma levels at nodes. Transpose from (levels, nodes) to (nodes, levels)
        sigma_levels = self._return_variable_data("siglev").T
        
        return SigmaData(sigma_config, sigma_levels)

    def _return_variable_data(self, var_name):
        """Return the data for a given variable name.

        Warn if the variable contains masked data for any reason.

        Args:
            var_name (str): The name of the variable to retrieve.
        Returns:
            np.ndarray: The data array for the specified variable.
        """
        if np.ma.is_masked(self.dataset[var_name][:]):
            print(
                f"Warning: {var_name} contains masked data. "
                "Masked values will be filled with default fill value."
            )
        return np.ma.getdata(self.dataset.variables[var_name][:])

    def close(self):
        """Close the dataset to free up resources."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
