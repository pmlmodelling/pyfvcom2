"""Interpolation coordinates class"""

import numpy as np

__all__ = ["InterpolationCoordinates"]


class InterpolationCoordinates:
    def __init__(
        self, dates: np.ndarray, x3: np.ndarray, x2: np.ndarray, x1: np.ndarray,
        coordinate_system: str
    ):
        """Initialize the InterpolationCoordinates with date, x3, x2, and x1 arrays.

        The date array should be 1D. The x2 and x1 arrays should also be 1D and of the same
        length - representing the horizontal points (see args below). For FVCOM 3D variables, the length of the x1 and
        x2 arrays will be either the number of nodes or elements in the FVCOM grid, depending on whether the
        variable is node-based or element-based. The x3 array corresponds to depth, and should be 2D, with shape (n_depths, n_horizontal_points),
        where n_horizontal_points is the length of the x2/x1 arrays. For regularly gridded data sources,
        the x2 and x1 arrays must be converted into 1D arrays representing all horizontal points in the grid.

        Args:
            dates (np.ndarray): 1D array of datetime objects.
            x3 (np.ndarray): 2D array of x3 values (typically depth; shape: n_depths x n_horizontal_points).
            x2 (np.ndarray): 1D array of x2 values (typically y or latitude).
            x1 (np.ndarray): 1D array of x1 values (typically x or longitude).
            coordinate_system (str): Coordinate system of the x1, x2, and x3 arrays ("geographic", "cartesian").
        """
        self._dates = dates
        self._x3 = x3
        self._x2 = x2
        self._x1 = x1
        if coordinate_system not in ["geographic", "cartesian"]:
            raise ValueError("coordinate_system must be either 'geographic' or 'cartesian'")
        self.coordinate_system = coordinate_system

    # Add getters and setters for each attribute if needed
    @property
    def dates(self):
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def x3(self):
        return self._x3

    @x3.setter
    def x3(self, value):
        self._x3 = value

    @property
    def x2(self):
        return self._x2

    @x2.setter
    def x2(self, value):
        self._x2 = value

    @property
    def x1(self):
        return self._x1

    @x1.setter
    def x1(self, value):
        self._x1 = value
    
    @property
    def coordinate_system(self):
        return self._coordinate_system

    @coordinate_system.setter
    def coordinate_system(self, value):
        if value not in ["geographic", "cartesian"]:
            raise ValueError("coordinate_system must be either 'geographic' or 'cartesian'")
        self._coordinate_system = value
