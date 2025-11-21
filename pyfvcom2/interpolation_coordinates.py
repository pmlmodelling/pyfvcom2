"""Interpolation coordinates class"""

import numpy as np

__all__ = ["InterpolationCoordinates"]


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