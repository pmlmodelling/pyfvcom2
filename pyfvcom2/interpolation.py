"""Interpolation functions"""

import numpy as np
from abc import ABC, abstractmethod
from scipy import interpolate
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from matplotlib.tri import Triangulation, LinearTriInterpolator
from typing import Optional

from .cmems_reader import CMEMSReader, default_fvcom_to_cmems_var_names
from .fvcom_reader import FVCOMReader
from .exceptions import PyFVCOM2ValueError
from .interpolation_coordinates import InterpolationCoordinates
from .coordinates import sigma_to_z_coords

__all__ = ["InterpolationCoordinates", "Interpolator", "CMEMSInterpolator", "FVCOMInterpolator"]


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

        print(f"Interpolating CMEMS {cmems_var_name} to FVCOM grid.")

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
        try:
            # Try to get length - if it fails, it's a single datetime object
            len(coordinates.dates)
            dates = coordinates.dates
        except TypeError:
            # Single datetime object, wrap in list
            dates = [coordinates.dates]

        # Determine the number of dates and points
        n_dates = len(dates)
        n_points = len(coordinates.x1)

        # Initialise array to hold interpolated data
        interpolated_data = np.empty((n_dates, n_points), dtype=np.float32)

        # Build Delaunay triangulation once for reuse across all time steps
        # This avoids the expensive triangulation rebuild in scipy.interpolate.griddata
        source_points = np.column_stack((
            self.cmems_reader.unmasked_lons, 
            self.cmems_reader.unmasked_lats
        ))
        tri = Delaunay(source_points)
        
        # Target points for interpolation
        target_points = np.column_stack((coordinates.x1, coordinates.x2))

        # Loop over each time index to perform interpolation
        for d_idx, target_date in enumerate(dates):
            print(f"Interpolating CMEMS {cmems_var_name} to FVCOM grid for date: {target_date}.")

            # Get CMEMS unmasked data for this time step
            unmasked_data = self.cmems_reader.get_unmasked_variable(
                cmems_var_name, target_date
            )

            # Create interpolator with pre-built triangulation (reuses triangulation)
            interpolator = LinearNDInterpolator(tri, unmasked_data)
            interpolated_data[d_idx, :] = interpolator(target_points)
            
            # Check for NaN values indicating out-of-bounds points
            # NB - Might there be situations when one wants to fill the NaNs
            # using, e.g., nearest-neighbor interpolation? NB the method is already
            # interpolating over masked internal points, so this would only apply
            # for points that also sit outside the convex hull of the unstructured
            # grid which has been constructed from the regular CMEMS grid. In most
            # cases, I expect one would want to use a different source of data
            # for these points.
            nan_mask = np.isnan(interpolated_data[d_idx, :])
            if np.any(nan_mask):
                nan_indices = np.where(nan_mask)[0]
                nan_coords = target_points[nan_indices]
                error_msg = (f"Out-of-bounds interpolation detected for {len(nan_indices)} points "
                           f"at time {target_date}.\n"
                           f"Points outside CMEMS grid coverage: "
                           f"{[(coord[0], coord[1]) for coord in nan_coords[:5]]}")
                if len(nan_indices) > 5:
                    error_msg += f" ... and {len(nan_indices) - 5} more points"
                raise PyFVCOM2ValueError(error_msg)
        
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
        try:
            # Try to get length - if it fails, it's a single datetime object
            len(coordinates.dates)
            dates = coordinates.dates
        except TypeError:
            # Single datetime object, wrap in list
            dates = [coordinates.dates]

        # Determine the number of dates and points
        n_dates = len(dates)
        n_depths = coordinates.x3.shape[0]
        n_points = coordinates.x1.shape[0]

        # Loop over each time index to perform interpolation
        interpolated_data = np.empty((n_dates, n_depths, n_points), dtype=np.float32)

        for d_idx, target_date in enumerate(dates):
            print(f"Interpolating CMEMS {cmems_var_name} to FVCOM grid for date: {target_date}.")
            
            # Get the filled 3D CMEMS variable data
            var_filled = self.cmems_reader.get_filled_3D_var(cmems_var_name, target_date)

            # First, interpolate onto the horizontal grid for each depth level
            var_on_fvcom_horizontal_grid = np.empty(
                (self.cmems_reader.n_depths, n_points), dtype=var_filled.dtype
            )

            for depth_index in range(self.cmems_reader.n_depths):
                layer_data = var_filled[depth_index, :, :]

                interp = interpolate.RegularGridInterpolator(
                    (self.cmems_reader.lons, self.cmems_reader.lats), layer_data.T
                )
                var_on_fvcom_horizontal_grid[depth_index, :] = interp((coordinates.x1, coordinates.x2))

            # Next, interpolate onto the FVCOM vertical sigma layers for each horizontal point
            var_on_fvcom_grid = np.empty((n_depths, n_points), dtype=var_filled.dtype)

            for i in range(n_points):
                var_profile = var_on_fvcom_horizontal_grid[:, i]
                target_depths = coordinates.x3[:, i]

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
    """ FVCOM interpolator class
    
    There are several methods that could be used for interpolating data:

    1. Nearest-neighbor interpolation
    2. Linear interpolation from a delaunay triangulation
    3. Radial basis function interpolation

    These different methods have different trade-offs in terms of speed and accuracy, with RBFs being
    the most accurate but also the slowest. For now, we only support linear interpolation using
    scipy's LinearTriInterpolator. However, this could be extended in future to support other methods.

    Further notes:
    - In nested FVCOM grids, the elements are identical in the overlap zone, meaning there is no
    need to interpolate from one grid to another. Instead, one can simply extract the data from the
    parent grid at the locations of the child grid nodes/elements.
    - For node-based variables (e.g., zeta, h), interpolation is performed using the grid nodes. We use
    the input grid's connectivity (triangles) to build a triangulation for the nodes (i.e., the nv array).
    - For element-based variables (e.g., u, v), we must construct a new triangulation based on the element
    centroids (xc, yc).
    - We should allow for extrapolation of values outside the convex hull of the triangulation, which may
    be necessary when moving from a coarse grid to a finer grid. This is done by testing for NaN values in
    the interpolated output and filling them using nearest-neighbor interpolation. A warning is issued when
    this occurs.
    - Once data has been interpolated onto all horizontal surfaces of the input grid, we then interpolate
    vertically onto the supplied depth levels.
    - Options to parallelise the interpolation step are included.

    Args:
        fvcom_reader (FVCOMReader): An instance of FVCOMReader with loaded data.
    """
    
    def __init__(self, fvcom_reader: FVCOMReader):
        super().__init__()

        self.fvcom_reader = fvcom_reader

        # These are created through lazy initialisation. Note for
        # element based data, form a new triangulation based on
        # element centroids.
        self._triangulation_nodes_cartesian = None
        self._triangulation_nodes_geographic = None
        self._triangulation_elements_cartesian = None
        self._triangulation_elements_geographic = None

    @property
    def triangulation_nodes_cartesian(self):
        """Triangulation for FVCOM grid nodes in cartesian coordinates."""
        if self._triangulation_nodes_cartesian is not None:
            return self._triangulation_nodes_cartesian

        self._triangulation_nodes_cartesian = Triangulation(
            self.fvcom_reader.grid.x,
            self.fvcom_reader.grid.y,
            triangles=self.fvcom_reader.grid.triangles)
        return self._triangulation_nodes_cartesian

    @property
    def triangulation_nodes_geographic(self):
        """Triangulation for FVCOM grid nodes in geographic coordinates."""
        if self._triangulation_nodes_geographic is not None:
            return self._triangulation_nodes_geographic

        self._triangulation_nodes_geographic = Triangulation(
            self.fvcom_reader.grid.lon,
            self.fvcom_reader.grid.lat,
            triangles=self.fvcom_reader.grid.triangles)
        return self._triangulation_nodes_geographic

    @property
    def triangulation_elements_cartesian(self):
        """Triangulation for FVCOM grid elements in cartesian coordinates."""
        if self._triangulation_elements_cartesian is not None:
            return self._triangulation_elements_cartesian

        self._triangulation_elements_cartesian = Triangulation(
            self.fvcom_reader.grid.xc,
            self.fvcom_reader.grid.yc)
        return self._triangulation_elements_cartesian

    @property
    def triangulation_elements_geographic(self):
        """Triangulation for FVCOM grid elements in geographic coordinates."""
        if self._triangulation_elements_geographic is not None:
            return self._triangulation_elements_geographic

        self._triangulation_elements_geographic = Triangulation(
            self.fvcom_reader.grid.lonc,
            self.fvcom_reader.grid.latc)
        return self._triangulation_elements_geographic

    def interpolate(self, coordinates: InterpolationCoordinates, fvcom_var_name: str,
                    extrapolate_horizontally: Optional[bool] = False,
                    extrapolate_down: Optional[bool] = False,
                    extrapolate_up: Optional[bool] = True,
                    apply_land_sea_mask: Optional[bool] = True,
                    land_sea_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Perform interpolation operation for FVCOM data.

        The land_sea_mask is only applied when extrapolate is False. If not provided and extrapolate is False,
        it is automatically generated from the supplied coordinates object through a call to
        `self.generate_land_sea_mask`. The argument is included to save on computation time in cases where in
        `interpolate` is called multiple times with the same coordinates, e.g., when interpolating multiple
        variables onto the same grid.

        Args:
            coordinates (InterpolationCoordinates): Coordinates on the FVCOM grid.
            fvcom_var_name (str): Name of the FVCOM variable to interpolate.
            extrapolate_horizontally (Optional[bool]): Whether to allow extrapolation outside the horizontal grid.
            extrapolate_down (Optional[bool]): Whether to allow extrapolation below the bottom grid.
            extrapolate_up (Optional[bool]): Whether to allow extrapolation above the surface grid.
            apply_land_sea_mask (Optional[bool]): Whether to apply the land-sea mask during interpolation.
            land_sea_mask (Optional[np.ndarray]): Optional land-sea mask to apply during interpolation.

        Returns:
            np.ndarray: Interpolated variable on the FVCOM grid.
        """
        # Get variable dimensions and their names
        var_dims = self.fvcom_reader.get_var_dimensions(fvcom_var_name)
        
        # Check if time dimension is present
        has_time = 'time' in var_dims
        
        # Set has_depth flag
        has_depth = self.fvcom_reader.get_vertical_position(fvcom_var_name) is not None

        # If not extrapolating, generate land-sea mask is not provided
        if apply_land_sea_mask and land_sea_mask is None:
            land_sea_mask = ~self.generate_land_sea_mask(coordinates)

        # Route to appropriate interpolation method based on dimensions
        if not has_time and not has_depth:
            # Case 1: 2D time independent (e.g., bathymetry: [node] or [nele])
            var = self._interpolate_2d_static(coordinates, fvcom_var_name, extrapolate_horizontally=extrapolate_horizontally)

            if apply_land_sea_mask:
                # Apply land-sea mask to interpolated variable
                var[land_sea_mask] = np.nan
            return var

        elif has_time and not has_depth:
            # Case 2: 2D time dependent (e.g., zeta: [time, node])
            var = self._interpolate_2d_time_dependent(coordinates, fvcom_var_name, extrapolate_horizontally=extrapolate_horizontally)
            
            if apply_land_sea_mask:
                # Apply land-sea mask to interpolated variable
                var[:, land_sea_mask] = np.nan
            return var

        elif has_time and has_depth:
            # Case 3: 3D time dependent (e.g., temp: [time, siglay, node])
            var = self._interpolate_3d_time_dependent(coordinates, fvcom_var_name, extrapolate_horizontally=extrapolate_horizontally,
                                                      extrapolate_down=extrapolate_down, extrapolate_up=extrapolate_up)
            
            if apply_land_sea_mask:
                # Apply land-sea mask to interpolated variable
                var[:, :, land_sea_mask] = np.nan
            return var

        else:
            raise PyFVCOM2ValueError(
                f"Unsupported variable dimensions for {fvcom_var_name}: {var_dims}"
            )
    
    def _interpolate_2d_static(self, coordinates: InterpolationCoordinates,
                               fvcom_var_name: str,
                               extrapolate_horizontally: Optional[bool] = False) -> np.ndarray:
        """Interpolate a 2D time-independent FVCOM variable.
        
        Args:
            coordinates (InterpolationCoordinates): Target coordinates.
            fvcom_var_name (str): Name of the FVCOM variable.
            extrapolate_horizontally (Optional[bool]): Whether to allow extrapolation outside the grid.
            
        Returns:
            np.ndarray: Interpolated variable data with shape (n_points,).
        """
        n_points = len(coordinates.x1)

        # Initialise array to hold interpolated data
        interpolated_data = np.empty((n_points), dtype=np.float32)

        # FVCOM data for interpolation
        data = self.fvcom_reader.get_var(fvcom_var_name)

        # Create interpolator
        var_is_node_based = self.fvcom_reader.var_is_node_based(fvcom_var_name)
        interpolator = self._get_linear_interpolator_for_variable(var_is_node_based, coordinates.horizontal_coordinate_system, data)

        # Interpolate
        interpolated_data[:] = interpolator(coordinates.x1, coordinates.x2)

        # Fill NaNs indicating out-of-bounds points using nearest-neighbor interpolation
        if extrapolate_horizontally:
            target_points = np.column_stack((coordinates.x1, coordinates.x2))
            self._fill_nans_nearest_neighbor(interpolated_data, target_points, var_is_node_based, data)

        return interpolated_data
    
    def _interpolate_2d_time_dependent(self, coordinates: InterpolationCoordinates,
                                       fvcom_var_name: str,
                                       extrapolate_horizontally: Optional[bool] = False) -> np.ndarray:
        """Interpolate a 2D time-dependent FVCOM variable.
        
        Args:
            coordinates (InterpolationCoordinates): Target coordinates.
            fvcom_var_name (str): Name of the FVCOM variable.
            extrapolate_horizontally (Optional[bool]): Whether to allow extrapolation outside the grid.
            
        Returns:
            np.ndarray: Interpolated variable data with shape (n_times, n_points).
        """
        try:
            len(coordinates.dates)
            dates = coordinates.dates
        except TypeError:
            # Single datetime object, wrap in list
            dates = [coordinates.dates]

        # Determine the number of dates and points
        n_dates = len(dates)
        n_points = len(coordinates.x1)

        # Initialise array to hold interpolated data
        interpolated_data = np.empty((n_dates, n_points), dtype=np.float32)

        # Is the variable node or element based?
        var_is_node_based = self.fvcom_reader.var_is_node_based(fvcom_var_name)

        # Loop over each time index to perform interpolation
        for d_idx, target_date in enumerate(dates):
            print(f"Interpolating FVCOM {fvcom_var_name} to FVCOM grid for date: {target_date}.")

            # FVCOM data for this time step
            data = self.fvcom_reader.get_var(fvcom_var_name, target_date)

            # Create interpolator
            interpolator = self._get_linear_interpolator_for_variable(var_is_node_based, coordinates.horizontal_coordinate_system, data)

            interpolated_data[d_idx, :] = interpolator(coordinates.x1, coordinates.x2)

            # Fill NaNs indicating out-of-bounds points using nearest-neighbor interpolation
            if extrapolate_horizontally:
                target_points = np.column_stack((coordinates.x1, coordinates.x2))
                self._fill_nans_nearest_neighbor(interpolated_data[d_idx, :], target_points,
                                                 var_is_node_based, data,
                                                 context_msg=f"at time {target_date}")

        return interpolated_data

    def _interpolate_3d_time_dependent(self, coordinates: InterpolationCoordinates,
                                       fvcom_var_name: str,
                                       extrapolate_horizontally: Optional[bool] = False,
                                       extrapolate_down: Optional[bool] = False,
                                       extrapolate_up: Optional[bool] = True) -> np.ndarray:
        """Interpolate a 3D time-dependent FVCOM variable.
        
        Args:
            coordinates (InterpolationCoordinates): Target coordinates.
            fvcom_var_name (str): Name of the FVCOM variable.
            extrapolate_horizontally (Optional[bool]): Whether to allow horizontal extrapolation outside the grid.
            extrapolate_down (Optional[bool]): Whether to allow extrapolation below the grid.
            extrapolate_up (Optional[bool]): Whether to allow extrapolation above the grid. NB Useful if highest depth point is a layer
            center and thus does not reach all the way to the surface. Defaults to True for this reason.
            
        Returns:
            np.ndarray: Interpolated variable data with shape (n_times, n_depths, n_points).
        """
        try:
            len(coordinates.dates)
            dates = coordinates.dates
        except TypeError:
            dates = [coordinates.dates]

        # Determine the number of dates, depths and points we will interpolate to
        n_dates = len(dates)
        n_depths = coordinates.x3.shape[0]
        n_points = coordinates.x1.shape[0]

        # The number of depth levels in the FVCOM data (variable dependent, as can be
        # defined at layer centers or layer interfaces)
        n_depths_fvcom = self.fvcom_reader.get_n_depth_levels(fvcom_var_name)

        # Is the variable node or element based?
        var_is_node_based = self.fvcom_reader.var_is_node_based(fvcom_var_name)

        # Target points for interpolation
        target_points = np.column_stack((coordinates.x1, coordinates.x2))

        # Initialise array to hold interpolated data
        interpolated_var = np.empty((n_dates, n_depths, n_points), dtype=np.float32)

        # Loop over each time index to perform interpolation
        for d_idx, target_date in enumerate(dates):
            print(f"Interpolating FVCOM {fvcom_var_name} for date: {target_date}.")

            # FVCOM variable for this time point
            fvcom_var = self.fvcom_reader.get_var(fvcom_var_name, target_date)

            # Read the correct FVCOM depth coordinates
            if coordinates.vertical_coordinate_system == 'z':
                # Interpolation to be done in z coordinates. Depths are spatially
                # variable in FVCOM, so we need to interpolate these to the target
                # horizontal grid before performing vertical interpolation.
                fvcom_depths = self.fvcom_reader.get_time_dep_z_levels(
                    fvcom_var_name,
                    target_datetime=target_date,
                    relative_to_free_surface=True
                )

            else: # sigma coordinates
                # Interpolation to be done in sigma coordinates. Typically, sigma coordinates
                # do not vary horizontally, but we allow for it here. They do not vary in time.
                fvcom_depths = self.fvcom_reader.get_sigma_levels(fvcom_var_name)

            # Array holding variable data interpolated onto the new horizontal grid at each
            # FVCOM depth level
            var_on_target_horizontal_grid = np.empty(
                (n_depths_fvcom, n_points), dtype=fvcom_var.dtype
            )

            # Array holding depth data interpolated onto the new horizontal grid at each
            # FVCOM depth level (only used for vertical interpolation)
            depth_on_target_horizontal_grid = np.empty(
                (n_depths_fvcom, n_points), dtype=fvcom_depths.dtype
            )

            # Loop over all FVCOM depth levels 
            for i in range(n_depths_fvcom):
                # Horizontal interpolation of depth levels for this time step and depth level.
                depth_interp = self._get_linear_interpolator_for_variable(var_is_node_based,
                                                                          coordinates.horizontal_coordinate_system, fvcom_depths[i,:])
                depth_on_target_horizontal_grid[i, :] = depth_interp(coordinates.x1, coordinates.x2)
                if extrapolate_horizontally:
                    self._fill_nans_nearest_neighbor(depth_on_target_horizontal_grid[i, :], target_points,
                                                     var_is_node_based, fvcom_depths[i, :])

                # Horizontal interpolation of variable for this time step and depth level.
                var_interp = self._get_linear_interpolator_for_variable(var_is_node_based, coordinates.horizontal_coordinate_system, fvcom_var[i, :])
                var_on_target_horizontal_grid[i, :] = var_interp(coordinates.x1, coordinates.x2)
                if extrapolate_horizontally:
                    self._fill_nans_nearest_neighbor(var_on_target_horizontal_grid[i, :], target_points,
                                                     var_is_node_based, fvcom_var[i, :],
                                                     context_msg=f"at depth index {i} and time {target_date}")

            # Next, interpolate onto depth levels of each vertical point
            var_on_target_grid = np.empty((n_depths, n_points), dtype=fvcom_var.dtype)

            for i in range(n_points):
                # Set fill_value per point based on extrapolate_up and extrapolate_down.
                fill_value = (np.nan if not extrapolate_down else var_on_target_horizontal_grid[0, i],
                              np.nan if not extrapolate_up else var_on_target_horizontal_grid[-1, i])

                interp = interpolate.interp1d(
                    depth_on_target_horizontal_grid[:, i],
                    var_on_target_horizontal_grid[:, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value=fill_value
                )
                target_depths = coordinates.x3[:, i]
                var_on_target_grid[:, i] = interp(target_depths)

            interpolated_var[d_idx, :, :] = var_on_target_grid

        return interpolated_var

    def _get_linear_interpolator_for_variable(self, var_is_node_based: bool,
                                              horizontal_coordinate_system: str,
                                              data: np.ndarray) -> LinearTriInterpolator:
        """Get the appropriate linear interpolator based on whether the variable is node or element based.
        
        Args:
            var_is_node_based (bool): True if the variable is node based, False if element based.
            horizontal_coordinate_system (str): The coordinate system of the data ('geographic' or 'cartesian').
            data (np.ndarray): Data array for the variable.
        Returns:
            LinearTriInterpolator: The interpolator object.
        """
        # Determine if variable is node or element based
        if var_is_node_based:
            # Node based variable
            if horizontal_coordinate_system == "geographic":
                triangulation = self.triangulation_nodes_geographic
            else:
                triangulation = self.triangulation_nodes_cartesian
        else:
            # Element based variable
            if horizontal_coordinate_system == "geographic":
                triangulation = self.triangulation_elements_geographic
            else:
                triangulation = self.triangulation_elements_cartesian

        # Create interpolator
        interpolator = LinearTriInterpolator(
            triangulation,
            data
        )
        
        return interpolator

    def _fill_nans_nearest_neighbor(self, interpolated_data: np.ndarray, target_points: np.ndarray,
                                    var_is_node_based: bool, source_data: np.ndarray,
                                    warn: bool = False, context_msg: str = "") -> None:
        """Check for NaN values in interpolated data and fill using nearest-neighbor interpolation.

        Args:
            interpolated_data: 1D array of interpolated values (modified in-place).
            target_points: 2D array of target point coordinates, shape (n_points, 2).
            var_is_node_based: Whether the variable is node-based.
            source_data: Source data array for building the nearest-neighbor interpolator.
            warn: Whether to print a warning message when NaN values are found.
            context_msg: Additional context appended to the warning (e.g. "at time 2024-01-01").
        """
        nan_mask = np.isnan(interpolated_data)
        if not np.any(nan_mask):
            return

        nan_indices = np.where(nan_mask)[0]

        if warn:
            context = f" {context_msg}" if context_msg else ""
            print(
                f"Warning: Out-of-bounds interpolation detected for {len(nan_indices)} points{context}.\n"
                f"Points outside FVCOM grid coverage: "
                f"{[(coord[0], coord[1]) for coord in target_points[nan_indices][:1]]}"
            )
            if len(nan_indices) > 1:
                print(f" ... and {len(nan_indices) - 1} more points")
            print("These will be filled using nearest-neighbor interpolation.")

        nn_interpolator = self._get_nn_interpolator_for_variable(var_is_node_based, source_data)
        interpolated_data[nan_indices] = nn_interpolator(target_points[nan_indices])

    def _get_nn_interpolator_for_variable(self, var_is_node_based: bool,
                                          data: np.ndarray) -> NearestNDInterpolator:
        """Get the appropriate nearest-neighbor interpolator
        
        Args:
            var_is_node_based (bool): True if the variable is node based, False if element based.
            data (np.ndarray): Data array for the variable.
        Returns:
            NearestNDInterpolator: The nearest-neighbor interpolator object.
        """
        # Determine if variable is node or element based
        if var_is_node_based:
            # Node based variable
            x = self.fvcom_reader.grid.x
            y = self.fvcom_reader.grid.y
        else:
            # Element based variable
            x = self.fvcom_reader.grid.xc
            y = self.fvcom_reader.grid.yc
        
        # Create nearest-neighbor interpolator
        interpolator = interpolate.NearestNDInterpolator(
            np.column_stack((x, y)),
            data
        )
        
        return interpolator

    def generate_land_sea_mask(self, coordinates: InterpolationCoordinates) -> np.ndarray:
        """Generate a land-sea mask for the given interpolation coordinates
        
        The mask indicates which points are within the convex hull of the FVCOM grid and which are not,
        and thus which are sea points and which are land points.

        Args:
            coordinates (InterpolationCoordinates): The interpolation coordinates for which to generate the mask.
        
        Returns:
            np.ndarray: A boolean array of shape (n_points,) where True indicates a sea point and False indicates a land point.
        """
        # Use the triangulation of the grid nodes to determine which points are within the convex hull of the grid
        if coordinates.horizontal_coordinate_system == "geographic":
            triangulation = self.triangulation_nodes_geographic
        else:
            triangulation = self.triangulation_nodes_cartesian
        mask = triangulation.get_trifinder()(coordinates.x1, coordinates.x2) != -1

        return mask

