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

        # Load the dataset upon initialization
        self._load_data()

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
