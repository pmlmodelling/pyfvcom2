"""Read CMEMS data"""

__all__ = ["CMEMSReader", "default_fvcom_to_cmems_var_names"]

import bisect
from datetime import datetime
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from typing import Optional, Union, List
from collections import namedtuple
from pyfvcom2.exceptions import PyFVCOM2ValueError


# Default mapping of FVCOM variable names to CMEMS variable names
default_fvcom_to_cmems_var_names = {'temp': 'thetao',
                                    'salinity': 'so',
                                    'u': 'uo',
                                    'v': 'vo',
                                    'zeta': 'zos'}


class CMEMSReader:
    """Class to read CMEMS Data files"""

    def __init__(
        self,
        file_path: Union[str, List[str]],
        reference_var_name: str,
        dimension_var_names: Optional[dict] = None,
    ):
        # Handle both single file and list of files
        if isinstance(file_path, str):
            self.file_paths = [file_path]
        else:
            self.file_paths = file_path

        # Load only the first file initially for metadata and time-independent data
        print(f'Accessing CMEMS metadata from: {self.file_paths[0]}')
        self._metadata_dataset = xr.open_dataset(self.file_paths[0])
        
        # Set dimension variable names
        self.time_dim_name = (
            dimension_var_names.get("time", "time") if dimension_var_names else "time"
        )
        self.depth_dim_name = (
            dimension_var_names.get("depth", "depth")
            if dimension_var_names
            else "depth"
        )
        self.lon_dim_name = (
            dimension_var_names.get("longitude", "longitude")
            if dimension_var_names
            else "longitude"
        )
        self.lat_dim_name = (
            dimension_var_names.get("latitude", "latitude")
            if dimension_var_names
            else "latitude"
        )

        # Confirm dimension variable names exist in dataset. Assumes dimension and variable have same name.
        for dim_name in [self.time_dim_name, self.lon_dim_name, self.lat_dim_name]:
            if (
                dim_name not in self._metadata_dataset.dims
                and dim_name not in self._metadata_dataset.variables
            ):
                raise PyFVCOM2ValueError(
                    f"Dimension variable name {dim_name} not found in CMEMS file {self.file_paths[0]}"
                )

        # If reading 3D variables, check depth dimension exists
        self.has_depth_dimension = True
        if (
            self.depth_dim_name not in self._metadata_dataset.dims
            and self.depth_dim_name not in self._metadata_dataset.variables
        ):
            print(
                f"Depth dimension variable name {self.depth_dim_name} not found in CMEMS file {self.file_paths[0]}."
            )
            print(f"Assuming the dataset includes 2D variables only.")
            self.has_depth_dimension = False

        print("Using dimension variable names:")
        print(f"  Time: {self.time_dim_name}")
        if self.has_depth_dimension:
            print(f"  Depth: {self.depth_dim_name}")
        print(f"  Longitude: {self.lon_dim_name}")
        print(f"  Latitude: {self.lat_dim_name}")

        # Check reference var exists
        self.reference_var_name = reference_var_name
        print(f"Using reference variable {self.reference_var_name}.")

        if self.reference_var_name not in self._metadata_dataset.variables:
            raise PyFVCOM2ValueError(
                f"Reference variable {self.reference_var_name} not found in dataset(s) {self.file_paths[0]}."
            )

        # Check reference var dimensions
        if self.has_depth_dimension:
            if (
                self.depth_dim_name
                not in self._metadata_dataset.variables[self.reference_var_name].dims
            ):
                raise PyFVCOM2ValueError(
                    f"Please provide a 3D reference variable so the depth mask can be inferred. "
                    f"The supplied reference variable {self.reference_var_name} does not have a depth axis."
                )

        # Build time index mapping from all files
        self._build_time_index_mapping()

        # Set masks using metadata dataset
        self._set_masks()

        # Determine unmasked lon/lat points
        self._set_unmasked_lons_lats()

        # Store variable for bottom indices. Only compute this if it is needed.
        self._bottom_indices = None
        
        # Cache triangulation for surface interpolation optimization
        self._surface_triangulation = None
        
    def _build_time_index_mapping(self):
        """Build a mapping from datetime to (file_path, local_time_index)"""
        self._time_to_file_map = {}
        self._all_dates = []
        
        for file_path in self.file_paths:
            with xr.open_dataset(file_path) as ds:
                times = ds[self.time_dim_name].data
                for local_idx, time_val in enumerate(times):
                    self._time_to_file_map[time_val] = (file_path, local_idx)
                    self._all_dates.append(time_val)
        
        # Sort dates for efficient searching
        self._all_dates.sort()
        
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

        dataset = xr.open_dataset(required_file_path)

        return dataset, local_time_index

    def _get_bracketing_times(self, target_datetime):
        """Find the two CMEMS time steps that bracket a target datetime.

        Used to support linear temporal interpolation when the requested output
        time step lies between two source time steps (e.g. hourly output from
        daily CMEMS data).

        Args:
            target_datetime: The target datetime (datetime or np.datetime64).

        Returns:
            tuple: (t0, t1, alpha) where t0 and t1 are the bounding CMEMS
            datetimes (np.datetime64) and alpha is the fractional weight for t1
            so that the interpolated value is ``(1 - alpha)*v(t0) + alpha*v(t1)``.
            When target_datetime exactly matches a CMEMS time step, t0 == t1
            and alpha == 0.0.
        """
        if isinstance(target_datetime, datetime):
            target_datetime = np.datetime64(target_datetime)

        if len(self._all_dates) == 0:
            raise PyFVCOM2ValueError("No dates available in the dataset(s)")

        start_date = self._all_dates[0]
        end_date = self._all_dates[-1]

        if target_datetime < start_date or target_datetime > end_date:
            raise PyFVCOM2ValueError(
                f"Target datetime {target_datetime} is outside the available data range "
                f"[{start_date} to {end_date}]"
            )

        # Exact match — no interpolation needed
        if target_datetime in self._time_to_file_map:
            return target_datetime, target_datetime, 0.0

        # Find the index of the first CMEMS time strictly greater than target
        idx = bisect.bisect_right(self._all_dates, target_datetime)
        t0 = self._all_dates[idx - 1]
        t1 = self._all_dates[idx]

        dt_total = (t1 - t0) / np.timedelta64(1, 's')
        dt_target = (target_datetime - t0) / np.timedelta64(1, 's')
        alpha = float(dt_target / dt_total)

        return t0, t1, alpha

    def _set_masks(self):
        """Use reference variable to infer the mask"""

        var = self._metadata_dataset[self.reference_var_name].isel({self.time_dim_name: 0})
        var_mask = self.get_mask(var)

        if not self.has_depth_dimension:
            reference_mask_3D = None
            reference_mask_2D = var_mask
        else:
            reference_mask_3D = var_mask

            # Set the 2D mask from the 3D mask (surface layer)
            reference_mask_2D = reference_mask_3D[0, :, :]

        # Save the mask
        self.mask_2D = reference_mask_2D
        self.mask_3D = reference_mask_3D

    def _set_unmasked_lons_lats(self):
        """Determine the unmasked longitude and latitude points.

        A 2D meshgrid is first formed from the 1D lon-lat variables. Unmasked
        lons and lats are then identified from this.
        """
        lons = self._metadata_dataset.variables[f"{self.lon_dim_name}"][:]
        lats = self._metadata_dataset.variables[f"{self.lat_dim_name}"][:]
        self._lon_grid, self._lat_grid = np.meshgrid(lons, lats)

        self._unmasked_lons = self._lon_grid[~self.mask_2D]
        self._unmasked_lats = self._lat_grid[~self.mask_2D]

    @property
    def n_files(self):
        """Get the number of files being used."""
        return len(self.file_paths)
        
    @property
    def time_span(self):
        """Get the time span covered by the dataset."""
        dates = self.dates
        if len(dates) == 0:
            return None
        return {'start': dates[0], 'end': dates[-1], 'count': len(dates)}

    @property
    def n_depths(self):
        if not self.has_depth_dimension:
            raise PyFVCOM2ValueError("The dataset does not have a depth dimension.")

        return self._metadata_dataset.sizes[self.depth_dim_name]

    @property
    def dates(self):
        return np.array(self._all_dates)

    @property
    def lons(self):
        return self._metadata_dataset.variables[f"{self.lon_dim_name}"][:]

    @property
    def lats(self):
        return self._metadata_dataset.variables[f"{self.lat_dim_name}"][:]

    @property
    def lons_2D(self):
        return self._lon_grid

    @property
    def lats_2D(self):
        return self._lat_grid

    @property
    def unmasked_lons(self):
        return self._unmasked_lons

    @property
    def unmasked_lats(self):
        return self._unmasked_lats

    @property
    def depth_levels(self):
        if not self.has_depth_dimension:
            raise PyFVCOM2ValueError("The dataset does not have a depth dimension.")

        return -self._metadata_dataset.variables[f"{self.depth_dim_name}"][:].values

    def contains_date(self, date_time: datetime) -> bool:
        """Check if the dataset contains the given date_time.

        Args:
            date_time (datetime): Date time to check.
        Returns:
            bool: True if date_time is within the dataset time range, False otherwise.
        """
        # If date_time is a of type dateime.dateime, convert to np.datetime64
        if isinstance(date_time, datetime):
            date_time = np.datetime64(date_time)

        dates = self.dates
        if len(dates) == 0:
            return False

        start_date = dates[0]
        end_date = dates[-1]

        return start_date <= date_time <= end_date

    def get_closest_date_index(self, date_time) -> int:
        """Get the index of the datetime variable closest to the given date_time.

        Args:
            date_time (datetime): Date time to find closest index for.
        Returns:
            int: Index of the closest time variable.
        """
        if isinstance(date_time, datetime):
            date_time = np.datetime64(date_time)

        time_diffs = [abs(dt - date_time) for dt in self.dates]
        closest_date_index = time_diffs.index(min(time_diffs))
        return closest_date_index

    def get_var_ndims(self, var_name: str) -> int:
        """Get the number of dimensions of a variable.

        Args:
            var_name (str): Variable name.

        Returns:
            int: Number of dimensions.
        """
        if var_name not in self._metadata_dataset.variables:
            raise PyFVCOM2ValueError(
                f"The supplied variable {var_name} is not in the dataset(s) {self.file_paths}"
            )

        var = self._metadata_dataset[var_name]
        return len(var.dims)

    def get_mask(self, var) -> np.ndarray:
        """Get the mask for a variable.
        Args:
            var (xarray.DataArray): Variable to get the mask for.
        Returns:
            np.ndarray: Boolean mask array, where True indicates a masked value.
        """
        arr = (
            var.values
        )  # materialise the array (may be numpy masked array, ndarray or dask array)

        # 1) If it's already a masked array
        if np.ma.is_masked(arr):
            var_mask = np.ma.getmaskarray(arr)
        else:
            # 2) If xarray decoded fill values to NaN (common default)
            var_mask = np.isnan(arr)

            # 3) If still no mask, try the _FillValue / missing_value fallback
            if not var_mask.any():
                fill = var.encoding.get("_FillValue", var.attrs.get("_FillValue", None))
                if fill is not None:
                    # use isclose for floats to avoid precision issues
                    var_mask = np.isclose(arr, fill)

        return var_mask

    def get_bottom_indices(self) -> np.ndarray:
        """Get indices of the deepest unmasked level for each horizontal point.

        Returns:
            np.ndarray: 2D array of bottom indices (lat, lon)
        """
        if self._bottom_indices is not None:
            return self._bottom_indices

        if self.mask_3D is None:
            raise PyFVCOM2ValueError(
                "3D variable mask not set. Does the output file contain 3D variables?"
            )

        # By setting it to zero, the surface level is always considered unmasked, even if that's a land point
        # TODO - use xarray to get lon/lat coords?
        bottom_indices = np.zeros(
            (self.mask_3D.shape[1], self.mask_3D.shape[2]), dtype=int
        )

        for j in range(self.mask_3D.shape[1]):
            for i in range(self.mask_3D.shape[2]):
                indices = np.where(self.mask_3D[:, j, i] == False)[0]

                # Applied to non-land points only
                if len(indices) != 0:
                    k = indices[-1]
                    bottom_indices[j, i] = k

        self._bottom_indices = bottom_indices

        return self._bottom_indices

    def get_var(
        self, var_name: str, target_datetime: datetime, depth_index: int = None, tolerance=None
    ) -> np.ndarray:
        """Get the values of a variable at a given datetime and depth index.

        Args:
            var_name (str): Variable name.
            target_datetime (datetime): Target datetime to retrieve data for.
            depth_index (int, optional): Depth index for 3D variables. Defaults to None.
            tolerance (timedelta, optional): Maximum allowed time difference. Defaults to None.
        Returns:
            np.ndarray: Variable values.
        """
        dataset, local_time_index = self._load_dataset_for_datetime(target_datetime, tolerance)
        
        if var_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"The supplied variable {var_name} is not in the dataset"
            )

        var_has_depth = self.has_depth_dimension and self.depth_dim_name in dataset[var_name].dims

        if not var_has_depth:
            var = dataset[var_name].isel({self.time_dim_name: local_time_index})
            var_data = var.values
            return var_data

        else:
            if depth_index is None:
                raise PyFVCOM2ValueError(
                    "depth_index must be provided for 3D variables"
                )
            var = dataset[var_name].isel(
                {self.time_dim_name: local_time_index, self.depth_dim_name: depth_index}
            )
            var_data = var.values
            return var_data

    def get_unmasked_variable(
        self, var_name: str, target_datetime: datetime, depth_index: int = None, tolerance=None
    ) -> np.ndarray:
        """Get the unmasked values of a variable at a given datetime and depth index.

        Args:
            var_name (str): Variable name.
            target_datetime (datetime): Target datetime to retrieve data for.
            depth_index (int, optional): Depth index for 3D variables. Defaults to None.
            tolerance (timedelta, optional): Maximum allowed time difference. Defaults to None.
        Returns:
            np.ndarray: Unmasked variable values.
        """
        dataset, local_time_index = self._load_dataset_for_datetime(target_datetime, tolerance)
        
        if var_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"The supplied variable {var_name} is not in the dataset"
            )

        var_has_depth = self.has_depth_dimension and self.depth_dim_name in dataset[var_name].dims

        if not var_has_depth:
            var = dataset[var_name].isel({self.time_dim_name: local_time_index})
            var_data = var.values
            return var_data[~self.mask_2D]

        else:
            if depth_index is None:
                raise PyFVCOM2ValueError(
                    "depth_index must be provided for 3D variables"
                )
            var = dataset[var_name].isel(
                {self.time_dim_name: local_time_index, self.depth_dim_name: depth_index}
            )
            var_data = var.values
            return var_data[~self.mask_2D]

    def get_filled_2D_var(self, var_name: str, target_datetime: datetime, tolerance=None) -> np.ndarray:
        """Fill masked values in a 2D variable by interpolation.

        Uses a Delaunay triangulation of unmasked ocean points to linearly
        interpolate over land-masked cells on the regular CMEMS grid.

        Args:
            var_name (str): Variable name.
            target_datetime (datetime): Target datetime to retrieve data for.
            tolerance (timedelta, optional): Maximum allowed time difference. Defaults to None.

        Returns:
            np.ndarray: Filled variable values on the full (lat, lon) grid.
        """
        dataset, local_time_index = self._load_dataset_for_datetime(target_datetime, tolerance)

        if var_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} not found in the dataset"
            )

        var = dataset[var_name].isel({self.time_dim_name: local_time_index})
        var_data = var.values  # shape (lat, lon)

        valid_data = var_data[~self.mask_2D]

        # Build triangulation on first call and cache it for reuse
        if self._surface_triangulation is None:
            source_points = np.column_stack((self._unmasked_lons, self._unmasked_lats))
            self._surface_triangulation = Delaunay(source_points)

        # Interpolate onto the full regular grid
        target_points = np.column_stack((self._lon_grid.ravel(), self._lat_grid.ravel()))
        interpolator = LinearNDInterpolator(self._surface_triangulation, valid_data)
        var_data_filled = interpolator(target_points).reshape(self._lon_grid.shape)

        # Fill remaining NaNs (outside the convex hull) using nearest-neighbor
        nan_mask = np.isnan(var_data_filled)
        if np.any(nan_mask):
            source_points = np.column_stack((self._unmasked_lons, self._unmasked_lats))
            nn_interpolator = NearestNDInterpolator(source_points, valid_data)
            nan_points = np.column_stack((self._lon_grid[nan_mask], self._lat_grid[nan_mask]))
            var_data_filled[nan_mask] = nn_interpolator(nan_points)

        return var_data_filled

    def get_filled_3D_var(self, var_name: str, target_datetime: datetime, tolerance=None) -> np.ndarray:
        """Fill masked values in a 3D variable by interpolation

        First, use griddata to interpolate over all masked surface values.
        Then, for each horizontal point, fill masked depth levels by
        downward extrapolation from the nearest unmasked depth level.

        Args:
            var_name (str): Variable name.
            target_datetime (datetime): Target datetime to retrieve data for.
            tolerance (timedelta, optional): Maximum allowed time difference. Defaults to None.

        Returns:
            np.ndarray: Filled variable values.
        """
        dataset, local_time_index = self._load_dataset_for_datetime(target_datetime, tolerance)
        
        if var_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} was not specified as a 3D variable"
            )

        var = dataset[var_name].isel({self.time_dim_name: local_time_index})
        var_data = var.values  # shape (depth, lat, lon)

        # Create an array to hold filled data
        var_data_filled = np.empty_like(var_data)

        # First, fill the surface layer
        surface = var_data[0, :, :]
        surface_valid = surface[~self.mask_2D]

        # Build triangulation on first call and cache it for reuse
        if self._surface_triangulation is None:
            source_points = np.column_stack((self._unmasked_lons, self._unmasked_lats))
            self._surface_triangulation = Delaunay(source_points)
        
        # Use cached triangulation for optimized interpolation
        target_points = np.column_stack((self._lon_grid.ravel(), self._lat_grid.ravel()))
        interpolator = LinearNDInterpolator(self._surface_triangulation, surface_valid)
        interpolated_surface = interpolator(target_points).reshape(self._lon_grid.shape)

        # Fill remaining NaNs (outside the convex hull) using nearest-neighbor
        nan_mask = np.isnan(interpolated_surface)
        if np.any(nan_mask):
            source_points = np.column_stack((self._unmasked_lons, self._unmasked_lats))
            nn_interpolator = NearestNDInterpolator(source_points, surface_valid)
            nan_points = np.column_stack((self._lon_grid[nan_mask], self._lat_grid[nan_mask]))
            interpolated_surface[nan_mask] = nn_interpolator(nan_points)

        var_data_filled[0, :, :] = interpolated_surface

        # Copy in unmasked values for other depth levels
        var_data_filled[1:, :, :][~self.mask_3D[1:, :, :]] = var_data[1:, :, :][
            ~self.mask_3D[1:, :, :]
        ]

        # Now extrapolate downwards
        bottom_indices = self.get_bottom_indices()
        for j in range(var_data.shape[1]):
            for i in range(var_data.shape[2]):
                k = bottom_indices[j, i]
                var_data_filled[k:, j, i] = var_data_filled[k, j, i]

        return var_data_filled

    def close(self):
        """Close all datasets to free up resources."""
        if self._metadata_dataset is not None:
            self._metadata_dataset.close()
            self._metadata_dataset = None
            
    # Backward compatibility methods that convert time_index to datetime
    def get_var_by_index(self, var_name: str, time_index: int = 0, depth_index: int = None) -> np.ndarray:
        """Backward compatibility method using time index"""
        target_datetime = self._all_dates[time_index]
        return self.get_var(var_name, target_datetime, depth_index)
        
    def get_unmasked_var_by_index(self, var_name: str, time_index: int = 0, depth_index: int = None) -> np.ndarray:
        """Backward compatibility method using time index"""
        target_datetime = self._all_dates[time_index]
        return self.get_unmasked_var(var_name, target_datetime, depth_index)
        
    def get_filled_3D_var_by_index(self, var_name: str, time_index: int = 0) -> np.ndarray:
        """Backward compatibility method using time index"""
        target_datetime = self._all_dates[time_index]
        return self.get_filled_3D_var(var_name, target_datetime)
