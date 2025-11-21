"""Interpolation functions"""

import datetime
import numpy as np
from abc import ABC, abstractmethod
from scipy import interpolate
from typing import NamedTuple, Optional

from .grid import InterpolationCoordinates
from .cmems_reader import CMEMSReader, default_fvcom_to_cmems_var_names
from .fvcom_reader import FVCOMReader
from .exceptions import PyFVCOM2ValueError

__all__ = ["InterpolationCoordinates", "Interpolator", "CMEMSInterpolator", "FVCOMInterpolator"]


class InterpolationCoordinates:
    def __init__(self, dates: np.ndarray, depths: np.ndarray, lats: np.ndarray, lons: np.ndarray):
        """Initialize the InterpolationCoordinates with date, depth, latitude, and longitude arrays.

        Args:
            dates (np.ndarray): 1D array of datetime objects.
            depths (np.ndarray): 1D array of depth values.
            lats (np.ndarray): 1D array of latitude values.
            lons (np.ndarray): 1D array of longitude values.
        """
        self._dates = dates
        self._depths = depths
        self._lats = lats
        self._lons = lons
    
    # Add getters and setters for each attribute if needed
    @property
    def dates(self):
        return self._dates
    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def depths(self):
        return self._depths
    @depths.setter
    def depths(self, value):
        self._depths = value

    @property
    def lats(self):
        return self._lats
    @lats.setter
    def lats(self, value):
        self._lats = value

    @property
    def lons(self): 
        return self._lons
    @lons.setter
    def lons(self, value):
        self._lons = value


class Interpolator(ABC):
    """Abstract base class for interpolation operations."""

    def __init__(self):
        pass

    @abstractmethod
    def interpolate(self, coordinates: InterpolationCoordinates, fvcom_var_name: str) -> np.ndarray:
        """Perform interpolation operation.
        
        This method must be implemented by subclasses to define
        the specific interpolation behavior.

        Args:
            coordinates (InterpolationCoordinates): Coordinates on the FVCOM grid.
            fvcom_var_name (str): Name of the FVCOM variable to interpolate.
        """
        pass


class CMEMSInterpolator(Interpolator):
    """ CMEMS interpolator class
    
    Args:
        cmems_reader (CMEMSReader): An instance of CMEMSReader with loaded data.
        fvcom_name_map (dict): A mapping of variable names between FVCOM and CMEMS.
        The keys are FVCOM variable names and the values are CMEMS variable names.
    
    """

    def __init__(self, cmems_reader: CMEMSReader, fvcom_to_cmems_var_names: Optional[dict] = None):
        super().__init__()

        self.cmems_reader = cmems_reader

        if fvcom_to_cmems_var_names is None:
            self.fvcom_to_cmems_var_names = default_fvcom_to_cmems_var_names
        else:
            self.fvcom_to_cmems_var_names = fvcom_to_cmems_var_names

    def interpolate(self, coordinates: InterpolationCoordinates, fvcom_var_name: str) -> np.ndarray:
        """Perform interpolation operation for CMEMS data.

        Args:
            coordinates (InterpolationCoordinates): Space and time coordinates for the FVCOM grid; i.e., these
            are the times and locations where we want interpolated data.
            fvcom_var_name (str): Name of the FVCOM variable that we want interpolated data for. This
            will be matched to the corresponding CMEMS variable name using the provided mapping.

        Returns:
            np.ndarray: Interpolated variable on the FVCOM grid. For 2D this will be (time, points),
            and for 3D this will be (time, depth, points), where points may be either nodes or elements
            depending on the variable.
        """
        cmems_var_name = self.fvcom_to_cmems_var_names.get(fvcom_var_name)

        print(f"Interpolating CMEMS {cmems_var_name} to FVCOM grid...")

        # Calculate the number of dimensions of the CMEMS variable
        var_ndims = self.cmems_reader.get_var_ndims(cmems_var_name)

        # If a 2D spatial variable (time, lat, lon)
        if var_ndims == 3:
            return self._interpolate_2d(coordinates, cmems_var_name)
        # If a 3D spatial variable (time, depth, lat, lon)
        elif var_ndims == 4:
            return self._interpolate_3d(coordinates, cmems_var_name)

    def _interpolate_2d(self, coordinates: InterpolationCoordinates, cmems_var_name: str) -> np.ndarray:
        """Interpolate a 2D CMEMS variable onto the FVCOM grid.

        Args:
            coordinates (InterpolationCoordinates): Space and time coordinates for the FVCOM grid.
            cmems_var_name (str): Name of the CMEMS variable to interpolate.

        Returns:
            np.ndarray: Interpolated variable on the FVCOM grid.
        """
        # Determine time indices from the coordinates, which provides either a single
        # date/time as a datetime object or a list of datetime objects
        if isinstance(coordinates.dates, datetime):
            dates = [coordinates.dates]
        else:
            dates = coordinates.dates

        # Determine the number of dates and points
        n_dates = len(dates)
        n_points = len(coordinates.lons)

        # Initialise array to hold interpolated data
        interpolated_data = np.empty((n_dates, n_points), dtype=np.float32)

        # Loop over each time index to perform interpolation
        for d_idx, target_date in enumerate(dates):
            # Confirm the date is within the CMEMS data range
            if not self.cmems_reader.contains_date(target_date):
                raise PyFVCOM2ValueError(
                    f"Target date {target_date} is outside the range of CMEMS data dates."
                )

            # Determine the time index closest to target_date
            closest_date_index = self.cmems_reader.get_closest_date_index(target_date)

            # Get CMEMS unmasked lons/lats
            unmasked_data = self.cmems_reader.get_unmasked_variable(
                cmems_var_name, closest_date_index
            )

            interpolated_data[d_idx, :] = interpolate.griddata(
                (self.cmems_reader.unmasked_lons, self.cmems_reader.unmasked_lats),
                unmasked_data,
                (coordinates.lons, coordinates.lats),
                method="linear",
            )
        
        return interpolated_data

    def _interpolate_3d(self, coordinates: InterpolationCoordinates, cmems_var_name: str) -> np.ndarray:
        """Interpolate a 3D CMEMS variable onto the FVCOM grid.

        Args:
            coordinates (InterpolationCoordinates): Space and time coordinates for the FVCOM grid.
            cmems_var_name (str): Name of the CMEMS variable to interpolate.

        Returns:
            np.ndarray: Interpolated variable on the FVCOM grid.
        """
        # Determine time indices from the coordinates, which provides either a single
        # date/time as a datetime object or a list of datetime objects
        if isinstance(coordinates.dates, datetime):
            dates = [coordinates.dates]
        else:
            dates = coordinates.dates

        # Determine the number of dates and points
        n_dates = len(dates)
        n_depths = coordinates.depths.shape[0]
        n_points = coordinates.lons.shape[0]

        # Loop over each time index to perform interpolation
        interpolated_data = np.empty((n_dates, n_depths, n_points), dtype=np.float32)

        for d_idx, target_date in enumerate(dates):
            # Confirm the date is within the CMEMS data range
            if not self.cmems_reader.contains_date(target_date):
                raise PyFVCOM2ValueError(
                    f"Target date {target_date} is outside the range of CMEMS data dates."
                )

            # Determine the time index closest to target_date
            closest_date_index = self.cmems_reader.get_closest_date_index(target_date)

            # Get the filled 3D CMEMS variable data
            var_filled = self.cmems_reader.get_filled_3D_var(cmems_var_name, closest_date_index)

            # First, interpolate onto the horizontal grid for each depth level
            var_on_fvcom_horizontal_grid = np.empty(
                (self.cmems_reader.n_depths, n_points), dtype=var_filled.dtype
            )

            for depth_index in range(self.cmems_reader.n_depths):
                layer_data = var_filled[depth_index, :, :]

                interp = interpolate.RegularGridInterpolator(
                    (self.cmems_reader.lons, self.cmems_reader.lats), layer_data.T
                )
                var_on_fvcom_horizontal_grid[depth_index, :] = interp((coordinates.lons, coordinates.lats))

            # Next, interpolate onto the FVCOM vertical sigma layers for each horizontal point
            var_on_fvcom_grid = np.empty((n_depths, n_points), dtype=var_filled.dtype)

            for i in range(n_points):
                var_profile = var_on_fvcom_horizontal_grid[:, i]
                target_depths = coordinates.depths[:, i]

                interp = interpolate.interp1d(
                    self.cmems_reader.depth_levels,
                    var_profile,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                var_on_fvcom_grid[:, i] = interp(target_depths)
            
            interpolated_data[d_idx, :, :] = var_on_fvcom_grid
        
        return interpolated_data


class FVCOMInterpolator(Interpolator):
    
    def __init__(self, fvcom_reader: FVCOMReader):
        super().__init__()

        self.fvcom_reader = fvcom_reader

    def interpolate(self, coordinates: NamedTuple, fvcom_var_name: str) -> np.ndarray:
        """Perform interpolation operation for FVCOM data.

        Args:
            coordinates (NamedTuple): Coordinates on the FVCOM grid.
            fvcom_var_name (str): Name of the FVCOM variable to interpolate.

        Returns:
            np.ndarray: Interpolated variable on the FVCOM grid.
        """
        # Implement FVCOM-specific interpolation logic here
        pass

