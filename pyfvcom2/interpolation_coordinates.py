"""Interpolation coordinates class"""

import numpy as np

__all__ = ["InterpolationCoordinates"]


class InterpolationCoordinates:
    def __init__(
        self, dates: np.ndarray, depths: np.ndarray, lats: np.ndarray, lons: np.ndarray
    ):
        """Initialize the InterpolationCoordinates with date, depth, latitude, and longitude arrays.

        The date array should be 1D. The latitude and longitude arrays should also be 1D and of the same
        length - representing the horizontal points. For FVCOM 3D variables, the length of the lon and
        lat arrays will be either the number of nodes or elements in the FVCOM grid, depending on whether the
        variable is node-based or element-based. The depth array should be 2D, with shape (n_depths, n_horizontal_points),
        where n_horizontal_points is the length of the lat/lon arrays. For regularly gridded data sources,
        the lat and lon arrays must be converted into 1D arrays representing all horizontal points in the grid.

        Args:
            dates (np.ndarray): 1D array of datetime objects.
            depths (np.ndarray): 2D array of depth values (shape: n_depths x n_horizontal_points).
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
