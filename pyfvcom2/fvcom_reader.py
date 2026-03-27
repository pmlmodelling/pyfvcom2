"""FVCOM data reader for PyFVCOM2"""

__all__ = ["FVCOMReader"]

from datetime import datetime, timedelta
import numpy as np
from netCDF4 import Dataset
from typing import Union, List, Optional
from cftime import num2pydate

from .interpolation_coordinates import InterpolationCoordinates
from .coordinates import sigma_to_z_coords
from .grid import Grid
from .mesh_reader import MeshData
from .sigma_reader import SigmaData
from .date_utils import round_time
from .exceptions import PyFVCOM2ValueError


class FVCOMReader:
    """A class to read FVCOM model output and restart files.

    This class provides methods to read FVCOM netCDF files and extract relevant data.

    Attributes:
        filepath (str): Path to the FVCOM netCDF file.
        dataset (xarray.Dataset): The loaded FVCOM dataset.
    """
    # Class variable for time conversion
    DAYS_PER_MILLISECOND = 1.0 / (1000.0 * 60.0 * 60.0 * 24.0)

    def __init__(self,
                 file_paths: Union[str, List[str]]):
        """Initialize the FVCOMReader with the path to the netCDF file.

        Args:
            file_paths (str, list): Path to the FVCOM netCDF file.
        """
        # Handle single file path or list of file paths
        if isinstance(file_paths, str):
            self.file_paths = [file_paths]
        else:
            self.file_paths = file_paths

        # Load only the first file initially for metadata and time-independent data
        print(f'Accessing FVCOM metadata from: {self.file_paths[0]}')
        self._metadata_dataset = Dataset(self.file_paths[0])

        self._grid = None  # Lazy initialization

        # Build the time index mapping for multiple files
        self._build_time_index_mapping()

    def _build_time_index_mapping(self):
        """Build a mapping from datetime to (file_path, local_time_index)"""
        self._time_to_file_map = {}
        self._all_dates = []

        for file_path in self.file_paths:
            with Dataset(file_path) as ds:
                times = self._read_times(ds)
                for local_idx, time_val in enumerate(times):
                    self._time_to_file_map[time_val] = (file_path, local_idx)
                    self._all_dates.append(time_val)

        # Sometime FVCOM dates are repeated across files; ensure uniqueness
        self._all_dates = list(set(self._all_dates))

        # Sort dates for efficient searching
        self._all_dates.sort()

    def _read_times(self, dataset: Dataset) -> List[datetime]:
        """Read time variable from the dataset and convert to datetime objects.

        Args:
            dataset (Dataset): The netCDF dataset.
        Returns:
            List[datetime]: List of datetime objects corresponding to the time variable.
        """
        time_raw = (dataset.variables['Itime'][:] +
                    dataset.variables['Itime2'][:] * self.DAYS_PER_MILLISECOND)
        units = dataset.variables['Itime'].units

        datetime_raw = num2pydate(time_raw[:], units=units)
        return round_time(datetime_raw)

    def _load_dataset_for_datetime(self, target_datetime, tolerance=None):
        """Load the appropriate dataset for a given datetime

        Args:
            target_datetime: The target datetime to find data for
            tolerance: Maximum allowed time difference (as timedelta). If None, uses default bounds checking.
        """
        # Convert datetime to numpy datetime64 if needed
        if isinstance(target_datetime, datetime):
            target_datetime = np.datetime64(target_datetime)

        # Check bounds first
        if len(self._all_dates) == 0:
            raise PyFVCOM2ValueError("No dates available in the dataset(s)")

        start_date = self._all_dates[0]
        end_date = self._all_dates[-1]

        # Check if target is exactly in our time mapping
        if target_datetime in self._time_to_file_map:
            required_file_path, local_time_index = self._time_to_file_map[target_datetime]
        else:
            # Check if target is within reasonable bounds
            if target_datetime < start_date or target_datetime > end_date:
                raise PyFVCOM2ValueError(
                    f"Target datetime {target_datetime} is outside the available data range "
                    f"[{start_date} to {end_date}]"
                )

            # Find closest time within the valid range
            time_diffs = [abs(dt - target_datetime) for dt in self._all_dates]
            closest_idx = time_diffs.index(min(time_diffs))
            closest_time = self._all_dates[closest_idx]

            # Optional tolerance check
            if tolerance is not None:
                min_diff = min(time_diffs)
                if min_diff > np.timedelta64(tolerance):
                    raise PyFVCOM2ValueError(
                        f"Closest available time ({closest_time}) is {min_diff} away from target "
                        f"({target_datetime}), which exceeds tolerance ({tolerance})"
                    )

            required_file_path, local_time_index = self._time_to_file_map[closest_time]

        dataset = Dataset(required_file_path)

        return dataset, local_time_index

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
        """Get the sigma layer values at nodes (n_layers, n_nodes)."""
        return self.grid.sigma_layers.T

    @property
    def sigma_layers_elements(self):
        """Get the sigma layer values at element centroids (n_layers, n_elements)."""
        return self.grid.sigmac_layers.T

    @property
    def sigma_levels_nodes(self):
        """Get the sigma level values at nodes (n_levels, n_nodes)."""
        return self.grid.sigma_levels.T
    
    @property
    def sigma_levels_elements(self):
        """Get the sigma level values at element centroids (n_levels, n_elements)."""
        return self.grid.sigmac_levels.T

    @property
    def bathy_nodes(self):
        """Get the bathymetry values at nodes (transformed so to be positive up)"""
        return self.grid.bathy_nodes * -1.0

    @property
    def bathy_elements(self):
        """Get the bathymetry values at element centroids (transformed so to be positive up)"""
        return self.grid.bathy_elements * -1.0

    @property
    def dates(self) -> List[datetime]:
        """Get the available dates in the dataset(s).

        Returns:
            List[datetime]: List of available datetime objects.
        """
        return self._all_dates

    def get_var_dimensions(self, var_name: str) -> tuple:
        """Get the dimensions of a variable.

        Args:
            var_name (str): The name of the variable to check.
        Returns:
            tuple: The dimensions of the variable.
        """
        return self._metadata_dataset.variables[var_name].dimensions

    def var_is_node_based(self, var_name: str) -> bool:
        """Check if a variable is node-based.

        Args:
            var_name (str): The name of the variable to check.
        Returns:
            bool: True if the variable is node-based, False if element-based.
        """
        var_dims = self._metadata_dataset.variables[var_name].dimensions
        if 'node' in var_dims:
            return True
        elif 'nele' in var_dims:
            return False
        else:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} is neither node-based nor element-based."
            )

    def get_sigma_type(self, var_name: str) -> str:
        """Get the sigma coordinate type for a variable.

        Args:
            var_name (str): The name of the variable.
        Returns:
            str: 'siglay' if the variable uses sigma layers, 'siglev' if it uses sigma levels.
        """
        var_dims = self._metadata_dataset.variables[var_name].dimensions
        if 'siglay' in var_dims:
            return 'siglay'
        elif 'siglev' in var_dims:
            return 'siglev'
        else:
            return None

    def get_var(self, var_name: str, target_datetime: Optional[datetime]=None,
                tolerance: Optional[timedelta]=None) -> np.ndarray:
        """Get the data for a given variable at a specific time.

        Args:
            var_name (str): The name of the variable to retrieve.
            target_datetime (datetime): The target datetime for which to retrieve the data.
        Returns:
            np.ndarray: The data array for the specified variable at the given time.
        """
        if target_datetime is None:
            # Check if time is a dimension of the variable
            var_dims = self._metadata_dataset.variables[var_name].dimensions
            if 'time' not in var_dims:
                return self._return_grid_variable_data(var_name)
            else:
                raise PyFVCOM2ValueError(
                    f"Variable {var_name} is time-dependent; please provide a target_datetime."
                )
        
        dataset, local_time_index = self._load_dataset_for_datetime(target_datetime, tolerance=tolerance)

        # Make sure variable exists in the dataset
        if var_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} not found in dataset for time {target_datetime}."
            )
        
        # 2D spatial variable; no depth indexing needed
        var = dataset.variables[var_name][local_time_index, :]
        if np.ma.is_masked(var):
            print(
                f"Warning: {var_name} contains masked data at time {target_datetime}. "
                "Masked values will be filled with default fill value."
            )
        return np.ma.getdata(var)
        
    def get_z_levels(self, var_name: str) -> np.ndarray:
        """Get z levels/layers for the variable based on whether it is node or element based.
        
        Args:
            var_name (str): The name of the variable.
        
        Returns:
            np.ndarray: Sigma levels array.
        """
        var_is_node_based = self.var_is_node_based(var_name)

        sigma_type = self.get_sigma_type(var_name)

        if var_is_node_based:
            if sigma_type == 'siglay':
                return self.grid.sigma_layers_z.T
            else:
                return self.grid.sigma_levels_z.T
        else:
            if sigma_type == 'siglay':
                return self.grid.sigmac_layers_z.T
            else:
                return self.grid.sigmac_levels_z.T

    def get_n_z_levels(self, var_name: str) -> int:
        """Get number of z levels/layers
        
        Args:
            var_name (str): The name of the variable.
        
        Returns:
            int: Number of sigma levels/layers.
        """
        sigma_type = self.get_sigma_type(var_name)
        if sigma_type == 'siglay':
            return self.grid.n_sigma_layers
        else:
            return self.grid.n_sigma_levels

    def get_interpolation_coordinates(self, horizontal_position: str,
                                      vertical_position: Optional[str] = 'layer_centre',
                                      coordinate_system: Optional[str] = "geographic",
                                      dates: Optional[np.ndarray] = None) -> InterpolationCoordinates:
        """Get interpolation coordinates for a specific grid position.

        Wrapper for Grid.get_interpolation_coordinates.

        Args:
            horizontal_position: Whether coordinates are at mesh nodes or element centres ('node' or 'element').
            vertical_position: Whether depth coordinates are at layer centres or layer interfaces
                ('layer_centre' or 'layer_interface').
            coordinate_system: The coordinate system ("geographic" or "cartesian") for the interpolation
                coordinates.
            dates: Optional array of datetime objects for the interpolation coordinates.

        Returns:
            InterpolationCoordinates: The interpolation coordinates for the specified grid position.
        """
        return self.grid.get_interpolation_coordinates(horizontal_position, vertical_position=vertical_position,
                                                       coordinate_system=coordinate_system,
                                                       dates=dates)

    def _extract_mesh_data(self) -> MeshData:
        """Extract mesh data from FVCOM output file.
        
        Returns:
            MeshData: Mesh data object compatible with Grid construction.
        """
        # Extract basic mesh components
        nodes = np.arange(1, self._metadata_dataset.dimensions['node'].size+1) # TBC zero based indexing kept here.
        triangles = self._metadata_dataset.variables['nv'][:].T - 1  # Convert to 0-based indexing, transpose to (n_elem, 3)
        x1 = self._metadata_dataset.variables['lon'][:]
        x2 = self._metadata_dataset.variables['lat'][:]
        x3 = self._return_grid_variable_data('h')[:]

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
        sigma_levels = self._return_grid_variable_data("siglev").T
        
        return SigmaData(sigma_config, sigma_levels)

    def _return_grid_variable_data(self, var_name):
        """Return the data for a given variable name.

        Warn if the variable contains masked data for any reason.

        Args:
            var_name (str): The name of the variable to retrieve.
        Returns:
            np.ndarray: The data array for the specified variable.
        """
        if np.ma.is_masked(self._metadata_dataset.variables[var_name][:]):
            print(
                f"Warning: {var_name} contains masked data. "
                "Masked values will be filled with default fill value."
            )
        return np.ma.getdata(self._metadata_dataset.variables[var_name][:])
