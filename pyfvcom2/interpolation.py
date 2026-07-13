"""Interpolation functions"""

import numpy as np
from abc import ABC, abstractmethod
from scipy import interpolate
from scipy.spatial import Delaunay, QhullError
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from matplotlib.tri import Triangulation, LinearTriInterpolator
from typing import Optional

from .cmems_reader import CMEMSReader, default_fvcom_to_cmems_var_names
from .coordinates import sigma_to_z_coords, pol2cart, cart2pol
from .exceptions import PyFVCOM2ValueError
from .fvcom_reader import FVCOMReader
from .interpolation_coordinates import InterpolationCoordinates
from .nemo_reader import NEMOReader, default_fvcom_to_nemo_var_names
from .tide_reader import HarmonicsData

__all__ = [
    "InterpolationCoordinates",
    "Interpolator",
    "CMEMSInterpolator",
    "NEMOInterpolator",
    "FVCOMInterpolator",
    "TPXOInterpolator",
]


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
        if cmems_var_name is None:
            raise PyFVCOM2ValueError(
                f"No CMEMS variable mapping found for FVCOM variable '{fvcom_var_name}'. "
                f"Available mappings: {self.fvcom_to_cmems_var_names}"
            )

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

        # Loop over each time index to perform interpolation
        for d_idx, target_date in enumerate(dates):
            print(f"Interpolating CMEMS {cmems_var_name} to FVCOM grid for date: {target_date}.")

            # Get the filled 2D CMEMS variable data on the regular grid, with
            # linear interpolation in time between bracketing CMEMS time steps.
            t0, t1, alpha = self.cmems_reader._get_bracketing_times(target_date)
            var_filled_0 = self.cmems_reader.get_filled_2D_var(cmems_var_name, t0)
            if alpha == 0.0:
                var_filled = var_filled_0
            else:
                var_filled_1 = self.cmems_reader.get_filled_2D_var(cmems_var_name, t1)
                var_filled = (1.0 - alpha) * var_filled_0 + alpha * var_filled_1

            # Interpolate from the regular grid onto the FVCOM points
            interp = interpolate.RegularGridInterpolator(
                (self.cmems_reader.lons, self.cmems_reader.lats), var_filled.T
            )
            interpolated_data[d_idx, :] = interp((coordinates.x1, coordinates.x2))

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

            # Get the filled 3D CMEMS variable data, with linear interpolation
            # in time between bracketing CMEMS time steps.
            t0, t1, alpha = self.cmems_reader._get_bracketing_times(target_date)
            var_filled_0 = self.cmems_reader.get_filled_3D_var(cmems_var_name, t0)
            if alpha == 0.0:
                var_filled = var_filled_0
            else:
                var_filled_1 = self.cmems_reader.get_filled_3D_var(cmems_var_name, t1)
                var_filled = (1.0 - alpha) * var_filled_0 + alpha * var_filled_1

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


class NEMOInterpolator(Interpolator):
    """NEMO interpolator for native-grid fields.

    Supports scalar 2D and 3D variables on curvilinear geographic grids. NEMO
    U/V velocities are interpolated from their native staggered grids and
    rotated onto east/north components before being returned as FVCOM ``u`` and
    ``v``.

    Args:
        nemo_reader: A :class:`NEMOReader` instance with one or more grid file
            groups loaded.
        fvcom_to_nemo_var_names: Mapping from FVCOM names to NEMO variable
            names. Defaults to temperature, salinity, sea-surface height and
            velocity mappings.
    """

    def __init__(
        self,
        nemo_reader: NEMOReader,
        fvcom_to_nemo_var_names: Optional[dict] = None,
    ):
        super().__init__()

        self.nemo_reader = nemo_reader
        self._tri_cache: dict = {}  # cache Delaunay triangulations keyed by (grid id, mask hash)
        self._velocity_cache: dict = {}  # cache rotated (u, v) pair keyed by target coord hash
        self._var_read_cache: dict = {}  # cache raw NEMO field reads keyed by (var, time, grid)
        self._z_read_cache: dict = {}    # cache raw NEMO z-coord reads keyed by (var, time, grid)

        if fvcom_to_nemo_var_names is None:
            self.fvcom_to_nemo_var_names = default_fvcom_to_nemo_var_names
        else:
            self.fvcom_to_nemo_var_names = fvcom_to_nemo_var_names

    def _resolve_nemo_var_name(self, candidates):
        """Return the first available NEMO variable name from a string or tuple of candidates."""
        if isinstance(candidates, str):
            return candidates
        for candidate in candidates:
            try:
                self.nemo_reader.grid_for_variable(candidate)
                return candidate
            except PyFVCOM2ValueError:
                continue
        raise PyFVCOM2ValueError(
            f"None of the candidate NEMO variable names {list(candidates)} "
            "were found in the supplied files"
        )

    @staticmethod
    def _as_date_list(dates):
        try:
            len(dates)
            return dates
        except TypeError:
            return [dates]

    def _filled_curvilinear_interpolation(
        self,
        source_lons: np.ndarray,
        source_lats: np.ndarray,
        source_data: np.ndarray,
        target_lons: np.ndarray,
        target_lats: np.ndarray,
    ) -> np.ndarray:
        """Interpolate one curvilinear source layer onto target points.

        Delaunay triangulations are cached on the first call for each unique
        (source grid, valid-mask) combination and reused on subsequent calls,
        avoiding the dominant O(n log n) triangulation cost across time steps.
        """
        source_lons = np.asarray(source_lons)
        source_lats = np.asarray(source_lats)
        source_data = np.asarray(source_data)

        valid_mask = (
            np.isfinite(source_lons)
            & np.isfinite(source_lats)
            & np.isfinite(source_data)
        )

        if not np.any(valid_mask):
            raise PyFVCOM2ValueError(
                "Cannot interpolate NEMO data because the source layer has no "
                "finite values."
            )

        source_points = np.column_stack(
            (source_lons[valid_mask].ravel(), source_lats[valid_mask].ravel())
        )
        source_values = source_data[valid_mask].ravel()
        target_points = np.column_stack((target_lons, target_lats))

        if len(source_values) >= 3:
            try:
                # Build the Delaunay triangulation once per unique (grid, mask)
                # combination and reuse it for subsequent time steps / depth levels.
                cache_key = (id(source_lons), hash(valid_mask.tobytes()))
                if cache_key not in self._tri_cache:
                    self._tri_cache[cache_key] = Delaunay(source_points)
                tri = self._tri_cache[cache_key]
                interpolator = LinearNDInterpolator(tri, source_values)
                interpolated = interpolator(target_points)
            except QhullError:
                nearest = NearestNDInterpolator(source_points, source_values)
                return nearest(target_points)
        else:
            nearest = NearestNDInterpolator(source_points, source_values)
            return nearest(target_points)

        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest = NearestNDInterpolator(source_points, source_values)
            interpolated[nan_mask] = nearest(target_points[nan_mask])

        return interpolated

    def _get_time_interpolated_var(
        self,
        nemo_var_name: str,
        target_date,
        grid: str,
    ) -> np.ndarray:
        """Read a NEMO variable with linear temporal interpolation.

        Raw reads from the NEMO files are cached so that all hourly output
        steps bracketed by the same pair of daily records only trigger two
        file reads in total rather than two per time step.
        """
        t0, t1, alpha = self.nemo_reader.get_bracketing_times(target_date, grid)

        key_0 = (nemo_var_name, t0, grid)
        if key_0 not in self._var_read_cache:
            self._var_read_cache[key_0] = self.nemo_reader.get_var(nemo_var_name, t0, grid=grid)
        data_0 = self._var_read_cache[key_0]

        if alpha == 0.0:
            return data_0

        key_1 = (nemo_var_name, t1, grid)
        if key_1 not in self._var_read_cache:
            self._var_read_cache[key_1] = self.nemo_reader.get_var(nemo_var_name, t1, grid=grid)
        data_1 = self._var_read_cache[key_1]

        return (1.0 - alpha) * data_0 + alpha * data_1

    def _get_time_interpolated_vertical_coordinates(
        self,
        nemo_var_name: str,
        target_date,
        grid: str,
    ) -> np.ndarray:
        """Read NEMO z coordinates with linear temporal interpolation.

        Raw reads are cached for the same reason as _get_time_interpolated_var.
        """
        t0, t1, alpha = self.nemo_reader.get_bracketing_times(target_date, grid)

        key_0 = (nemo_var_name, t0, grid)
        if key_0 not in self._z_read_cache:
            self._z_read_cache[key_0] = self.nemo_reader.get_vertical_coordinates(
                nemo_var_name,
                target_datetime=t0,
                grid=grid,
                positive_down=False,
            )
        z_0 = self._z_read_cache[key_0]

        if alpha == 0.0:
            return z_0

        key_1 = (nemo_var_name, t1, grid)
        if key_1 not in self._z_read_cache:
            self._z_read_cache[key_1] = self.nemo_reader.get_vertical_coordinates(
                nemo_var_name,
                target_datetime=t1,
                grid=grid,
                positive_down=False,
            )
        z_1 = self._z_read_cache[key_1]

        return (1.0 - alpha) * z_0 + alpha * z_1

    def interpolate(
        self,
        coordinates: InterpolationCoordinates,
        fvcom_var_name: str,
    ) -> np.ndarray:
        """Interpolate a NEMO variable onto supplied FVCOM coordinates."""
        if coordinates.horizontal_coordinate_system != "geographic":
            raise PyFVCOM2ValueError(
                "NEMOInterpolator currently requires geographic target "
                "coordinates."
            )

        candidates = self.fvcom_to_nemo_var_names.get(fvcom_var_name)
        if candidates is None:
            raise PyFVCOM2ValueError(
                f"No NEMO variable mapping found for FVCOM variable "
                f"'{fvcom_var_name}'. Available mappings: "
                f"{self.fvcom_to_nemo_var_names}"
            )

        if fvcom_var_name in ("u", "v"):
            return self._interpolate_velocity(coordinates, fvcom_var_name)

        nemo_var_name = self._resolve_nemo_var_name(candidates)
        grid = self.nemo_reader.grid_for_variable(nemo_var_name)
        var_ndims = self.nemo_reader.get_var_ndims(nemo_var_name, grid=grid)

        print(f"Interpolating NEMO {nemo_var_name} to FVCOM grid.")

        if var_ndims == 2:
            return self._interpolate_2d_static(coordinates, nemo_var_name, grid)
        elif var_ndims == 3:
            return self._interpolate_2d(coordinates, nemo_var_name, grid)
        elif var_ndims == 4:
            return self._interpolate_3d(coordinates, nemo_var_name, grid)

        raise PyFVCOM2ValueError(
            f"Unsupported NEMO variable dimensions for {nemo_var_name}: "
            f"{var_ndims}"
        )

    def _interpolate_2d_static(
        self,
        coordinates: InterpolationCoordinates,
        nemo_var_name: str,
        grid: str,
    ) -> np.ndarray:
        n_points = len(coordinates.x1)
        interpolated_data = np.empty((n_points), dtype=np.float32)

        var_data = self.nemo_reader.get_var(nemo_var_name, grid=grid)
        interpolated_data[:] = self._filled_curvilinear_interpolation(
            self.nemo_reader.lons_for_variable(nemo_var_name, grid),
            self.nemo_reader.lats_for_variable(nemo_var_name, grid),
            var_data,
            coordinates.x1,
            coordinates.x2,
        )

        return interpolated_data

    def _interpolate_velocity(
        self,
        coordinates: InterpolationCoordinates,
        fvcom_var_name: str,
    ) -> np.ndarray:
        """Interpolate native NEMO U/V and rotate to east/north components.

        Both rotated components are computed together on the first call and
        cached so that the second call (for the other component) is free.
        """
        u_candidates = self.fvcom_to_nemo_var_names.get("u")
        v_candidates = self.fvcom_to_nemo_var_names.get("v")
        if u_candidates is None or v_candidates is None:
            raise PyFVCOM2ValueError(
                "NEMO velocity interpolation requires mappings for both "
                "'u' and 'v'."
            )
        u_var_name = self._resolve_nemo_var_name(u_candidates)
        v_var_name = self._resolve_nemo_var_name(v_candidates)

        u_grid = self.nemo_reader.grid_for_variable(u_var_name)
        v_grid = self.nemo_reader.grid_for_variable(v_var_name)

        # Cache key based on target horizontal coordinates so both u and v
        # share the same cache entry when called with identical nest positions.
        coord_key = (
            hash(np.asarray(coordinates.x1).tobytes()),
            hash(np.asarray(coordinates.x2).tobytes()),
        )

        if coord_key not in self._velocity_cache:
            print(f"Interpolating and rotating NEMO {u_var_name}/{v_var_name}.")

            native_u = self._interpolate_3d(coordinates, u_var_name, u_grid)
            native_v = self._interpolate_3d(coordinates, v_var_name, v_grid)

            angle_grid = "T" if "T" in self.nemo_reader.grid_names else u_grid
            grid_angle = self.nemo_reader.grid_angle(angle_grid)
            cos_angle = self._filled_curvilinear_interpolation(
                self.nemo_reader.lons(angle_grid),
                self.nemo_reader.lats(angle_grid),
                np.cos(grid_angle),
                coordinates.x1,
                coordinates.x2,
            )
            sin_angle = self._filled_curvilinear_interpolation(
                self.nemo_reader.lons(angle_grid),
                self.nemo_reader.lats(angle_grid),
                np.sin(grid_angle),
                coordinates.x1,
                coordinates.x2,
            )
            vector_length = np.hypot(cos_angle, sin_angle)
            valid_angle = vector_length > 0.0
            cos_angle[valid_angle] = cos_angle[valid_angle] / vector_length[valid_angle]
            sin_angle[valid_angle] = sin_angle[valid_angle] / vector_length[valid_angle]

            rotated_u = (
                native_u * cos_angle[np.newaxis, np.newaxis, :]
                - native_v * sin_angle[np.newaxis, np.newaxis, :]
            )
            rotated_v = (
                native_u * sin_angle[np.newaxis, np.newaxis, :]
                + native_v * cos_angle[np.newaxis, np.newaxis, :]
            )
            self._velocity_cache[coord_key] = (rotated_u, rotated_v)

        rotated_u, rotated_v = self._velocity_cache[coord_key]
        return rotated_u if fvcom_var_name == "u" else rotated_v

    def _interpolate_2d(
        self,
        coordinates: InterpolationCoordinates,
        nemo_var_name: str,
        grid: str,
    ) -> np.ndarray:
        dates = self._as_date_list(coordinates.dates)
        n_dates = len(dates)
        n_points = len(coordinates.x1)

        interpolated_data = np.empty((n_dates, n_points), dtype=np.float32)

        for d_idx, target_date in enumerate(dates):
            print(
                f"Interpolating NEMO {nemo_var_name} to FVCOM grid for "
                f"date: {target_date}."
            )

            var_data = self._get_time_interpolated_var(
                nemo_var_name, target_date, grid
            )
            interpolated_data[d_idx, :] = self._filled_curvilinear_interpolation(
                self.nemo_reader.lons_for_variable(nemo_var_name, grid),
                self.nemo_reader.lats_for_variable(nemo_var_name, grid),
                var_data,
                coordinates.x1,
                coordinates.x2,
            )

        return interpolated_data

    def _interpolate_3d(
        self,
        coordinates: InterpolationCoordinates,
        nemo_var_name: str,
        grid: str,
    ) -> np.ndarray:
        """Interpolate a 3-D NEMO variable onto FVCOM sigma coordinates.

        Performance notes
        -----------------
        * Source-grid lons/lats are extracted once outside the time loop.
        * ``depth_on_horizontal_grid`` (the horizontal projection of NEMO z
          coordinates onto the nest points) is computed only once per unique
          (t0, t1) bracket.  For fixed z-level grids (e.g. AMM7) all time
          steps share a single bracket, so the 45-depth × n_points horizontal
          interpolation of depths is done exactly once rather than 25 times.
        * Per-point vertical-interpolation geometry (valid-depth mask, sort
          order, unique-depth indices, clipped target depths) is built once
          from the first depth computation and reused for every subsequent
          time step; only the interpolated values change.
        * ``scipy.interpolate.interp1d`` (Python-object overhead per call) is
          replaced by ``np.interp`` (pure-C, no construction cost).
        """
        if coordinates.vertical_coordinate_system != "z":
            raise PyFVCOM2ValueError(
                "NEMOInterpolator currently requires z target coordinates for "
                "3D variables."
            )

        dates = self._as_date_list(coordinates.dates)
        n_dates = len(dates)
        n_depths = coordinates.x3.shape[0]
        n_points = coordinates.x1.shape[0]

        interpolated_data = np.empty((n_dates, n_depths, n_points), dtype=np.float32)

        # Source grid is constant across time steps — extract once.
        src_lons = self.nemo_reader.lons_for_variable(nemo_var_name, grid)
        src_lats = self.nemo_reader.lats_for_variable(nemo_var_name, grid)

        # depth_on_horizontal_grid cache: keyed by (t0, t1) bracket.
        # For fixed z-level grids the NEMO depth coordinate is the same for
        # every time step, so we only compute this once per bracket.
        _depth_hgrid: dict = {}

        # Per-point vertical interpolation parameters built from the depth
        # profile the first time depth_on_horizontal_grid is available.
        # Each entry is None (no valid data) or a tuple
        #   ('const', value)  — single valid depth, return constant
        #   ('interp', valid_mask, sort_idx, uniq_idx, sd_unique, td)
        _vert_params: list | None = None
        _vert_params_bracket: tuple | None = None  # (t0, t1) that built _vert_params

        for d_idx, target_date in enumerate(dates):
            print(
                f"Interpolating NEMO {nemo_var_name} to FVCOM grid for "
                f"date: {target_date}."
            )

            var_data = self._get_time_interpolated_var(
                nemo_var_name, target_date, grid
            )
            z_data = self._get_time_interpolated_vertical_coordinates(
                nemo_var_name, target_date, grid
            )
            n_nemo_depths = z_data.shape[0]

            # ---- Horizontal interpolation of the variable (always fresh) ----
            var_on_horizontal_grid = np.empty(
                (n_nemo_depths, n_points), dtype=np.float32
            )
            for depth_index in range(n_nemo_depths):
                layer_data = var_data[depth_index, :, :]
                if not np.any(np.isfinite(layer_data)):
                    var_on_horizontal_grid[depth_index, :] = np.nan
                    continue
                var_on_horizontal_grid[depth_index, :] = (
                    self._filled_curvilinear_interpolation(
                        src_lons, src_lats, layer_data,
                        coordinates.x1, coordinates.x2,
                    )
                )

            # ---- depth_on_horizontal_grid (cached per bracket) ----
            t0, t1, _ = self.nemo_reader.get_bracketing_times(target_date, grid)
            bracket = (t0, t1)

            if bracket not in _depth_hgrid:
                depth_on_horizontal_grid = np.empty(
                    (n_nemo_depths, n_points), dtype=np.float32
                )
                for depth_index in range(n_nemo_depths):
                    layer_data = var_data[depth_index, :, :]
                    if not np.any(np.isfinite(layer_data)):
                        depth_on_horizontal_grid[depth_index, :] = np.nan
                        continue
                    layer_depths = np.where(
                        np.isfinite(layer_data),
                        z_data[depth_index, :, :],
                        np.nan,
                    )
                    depth_on_horizontal_grid[depth_index, :] = (
                        self._filled_curvilinear_interpolation(
                            src_lons, src_lats, layer_depths,
                            coordinates.x1, coordinates.x2,
                        )
                    )

                # Cache when z-levels are fixed: z_0 == z_1 (e.g. AMM7).
                z0_key = (nemo_var_name, t0, grid)
                z1_key = (nemo_var_name, t1, grid)
                z_fixed = t0 == t1 or (
                    z0_key in self._z_read_cache
                    and z1_key in self._z_read_cache
                    and np.array_equal(
                        self._z_read_cache[z0_key],
                        self._z_read_cache[z1_key],
                    )
                )
                if z_fixed:
                    _depth_hgrid[bracket] = depth_on_horizontal_grid

                # Invalidate per-point params when the depth profile changes.
                if bracket != _vert_params_bracket:
                    _vert_params = None
                    _vert_params_bracket = bracket
            else:
                depth_on_horizontal_grid = _depth_hgrid[bracket]

            # ---- Vertical interpolation ----
            var_on_target_grid = np.empty((n_depths, n_points), dtype=np.float32)

            if _vert_params is None:
                # Build per-point geometry from current depth profile.
                _vert_params = []
                for point_index in range(n_points):
                    sd = depth_on_horizontal_grid[:, point_index]
                    sv = var_on_horizontal_grid[:, point_index]
                    valid = np.isfinite(sd) & np.isfinite(sv)

                    if not np.any(valid):
                        _vert_params.append(None)
                        var_on_target_grid[:, point_index] = np.nan
                        continue

                    sd_v = sd[valid]
                    sv_v = sv[valid]
                    sort_idx = np.argsort(sd_v)
                    sd_sorted = sd_v[sort_idx]
                    sd_unique, uniq_idx = np.unique(sd_sorted, return_index=True)

                    if len(sd_unique) == 1:
                        _vert_params.append(('const', valid, sort_idx, uniq_idx))
                        var_on_target_grid[:, point_index] = sv_v[sort_idx][uniq_idx][0]
                        continue

                    td = np.clip(
                        coordinates.x3[:, point_index],
                        sd_unique[0],
                        sd_unique[-1],
                    )
                    _vert_params.append(('interp', valid, sort_idx, uniq_idx, sd_unique, td))
                    var_on_target_grid[:, point_index] = np.interp(
                        td, sd_unique, sv_v[sort_idx][uniq_idx]
                    )
            else:
                # Reuse cached geometry — only values change between time steps.
                for point_index in range(n_points):
                    p = _vert_params[point_index]
                    if p is None:
                        var_on_target_grid[:, point_index] = np.nan
                        continue
                    kind = p[0]
                    valid, sort_idx, uniq_idx = p[1], p[2], p[3]
                    sv_reindexed = (
                        var_on_horizontal_grid[:, point_index][valid][sort_idx][uniq_idx]
                    )
                    if kind == 'const':
                        var_on_target_grid[:, point_index] = sv_reindexed[0]
                    else:
                        sd_unique, td = p[4], p[5]
                        var_on_target_grid[:, point_index] = np.interp(
                            td, sd_unique, sv_reindexed
                        )

            interpolated_data[d_idx, :, :] = var_on_target_grid

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
            self.fvcom_reader.x_nodes,
            self.fvcom_reader.y_nodes,
            triangles=self.fvcom_reader.grid.triangles)
        return self._triangulation_nodes_cartesian

    @property
    def triangulation_nodes_geographic(self):
        """Triangulation for FVCOM grid nodes in geographic coordinates."""
        if self._triangulation_nodes_geographic is not None:
            return self._triangulation_nodes_geographic

        self._triangulation_nodes_geographic = Triangulation(
            self.fvcom_reader.lon_nodes,
            self.fvcom_reader.lat_nodes,
            triangles=self.fvcom_reader.grid.triangles)
        return self._triangulation_nodes_geographic

    @property
    def triangulation_elements_cartesian(self):
        """Triangulation for FVCOM grid elements in cartesian coordinates."""
        if self._triangulation_elements_cartesian is not None:
            return self._triangulation_elements_cartesian

        self._triangulation_elements_cartesian = Triangulation(
            self.fvcom_reader.x_elements,
            self.fvcom_reader.y_elements)
        return self._triangulation_elements_cartesian

    @property
    def triangulation_elements_geographic(self):
        """Triangulation for FVCOM grid elements in geographic coordinates."""
        if self._triangulation_elements_geographic is not None:
            return self._triangulation_elements_geographic

        self._triangulation_elements_geographic = Triangulation(
            self.fvcom_reader.lon_elements,
            self.fvcom_reader.lat_elements)
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
            x = self.fvcom_reader.x_nodes
            y = self.fvcom_reader.y_nodes
        else:
            # Element based variable
            x = self.fvcom_reader.x_elements
            y = self.fvcom_reader.y_elements
        
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


class TPXOInterpolator(Interpolator):
    """TPXO tidal harmonics interpolator.

    Interpolates TPXO tidal harmonic amplitudes and phases from the regular
    TPXO grid onto target positions (e.g., FVCOM open boundary nodes or elements).

    Phase interpolation is handled by decomposing amplitude/phase into two
    cartesian components before interpolation, then converting back. This
    avoids artifacts from the 360-to-0 degree phase wrapping discontinuity.

    Longitude convention mismatches between the TPXO grid (often 0-360) and the
    target positions (which may use -180 to 180) are detected and corrected
    automatically.

    Args:
        harmonics_data: Pre-loaded harmonics data from a TPXOHarmonicsReader
            or TPXOComplexHarmonicsReader.
        interp_method: Interpolation method passed to scipy's
            RegularGridInterpolator. Defaults to 'linear'.
    """

    def __init__(self, harmonics_data: HarmonicsData, interp_method: str = 'linear'):
        super().__init__()
        self.harmonics_data = harmonics_data
        self.interp_method = interp_method

    def interpolate(self, coordinates: InterpolationCoordinates, fvcom_var_name: str = None) -> HarmonicsData:
        """Interpolate TPXO harmonics onto target positions.

        Args:
            coordinates: Target positions for interpolation. Uses x1 (longitude)
                and x2 (latitude).
            fvcom_var_name: Not used for TPXO interpolation. Included for
                compatibility with the Interpolator interface.

        Returns:
            HarmonicsData with amplitudes and phases interpolated to the target
            positions, each shaped (n_points, n_constituents).
        """
        target_lon = coordinates.x1
        target_lat = coordinates.x2

        harmonics_lon = np.array(self.harmonics_data.longitude, copy=True)
        harmonics_lat = np.array(self.harmonics_data.latitude, copy=True)
        amplitudes = np.asarray(self.harmonics_data.amplitudes)
        phases = np.asarray(self.harmonics_data.phases)

        # RegularGridInterpolator requires 1D coordinate arrays
        if harmonics_lon.ndim != 1 or harmonics_lat.ndim != 1:
            harmonics_lon = np.unique(harmonics_lon)
            harmonics_lat = np.unique(harmonics_lat)

        # Convert amplitude/phase to cartesian components to avoid
        # phase wrapping artifacts during interpolation. The components
        # are given the names "harmonics_component_1" and "harmonics_component_2".
        harmonics_component_1, harmonics_component_2 = pol2cart(amplitudes, phases, degrees=True)

        # Align longitude conventions between source and target grids
        harmonics_lon, harmonics_component_1, harmonics_component_2 = self._align_longitudes(
            target_lon, harmonics_lon, harmonics_component_1, harmonics_component_2
        )

        # Interpolate each constituent onto the target positions
        n_constituents = amplitudes.shape[0]
        n_points = len(target_lon)
        interp_component_1 = np.empty((n_constituents, n_points))
        interp_component_2 = np.empty((n_constituents, n_points))

        for i in range(n_constituents):
            comp_1_interpolator = interpolate.RegularGridInterpolator(
                (harmonics_lon, harmonics_lat), harmonics_component_1[i],
                method=self.interp_method, bounds_error=False, fill_value=None
            )
            comp_2_interpolator = interpolate.RegularGridInterpolator(
                (harmonics_lon, harmonics_lat), harmonics_component_2[i],
                method=self.interp_method, bounds_error=False, fill_value=None
            )
            interp_component_1[i] = comp_1_interpolator((target_lon, target_lat))
            interp_component_2[i] = comp_2_interpolator((target_lon, target_lat))

        # Convert back to amplitude and phase
        interp_amplitudes, interp_phases = cart2pol(interp_component_1, interp_component_2, degrees=True)

        # Transpose to (n_points, n_constituents) to match predict_tide expectations
        return HarmonicsData(
            longitude=target_lon,
            latitude=target_lat,
            amplitudes=interp_amplitudes.T,
            phases=interp_phases.T,
            constituents=self.harmonics_data.constituents,
        )

    @staticmethod
    def _align_longitudes(target_lon, harmonics_lon, harmonics_component_1, harmonics_component_2):
        """Align harmonics longitude convention with target positions.

        Detects whether the harmonics and target use different longitude
        conventions (0-360 vs -180 to 180) and shifts the harmonics data
        accordingly. The longitude array is then sorted to ensure it is
        monotonically increasing, as required by RegularGridInterpolator.

        Args:
            target_lon: Target longitude positions.
            harmonics_lon: Harmonics grid longitudes (1D).
            harmonics_component_1: First cartesian component, shape (n_const, n_lon, n_lat).
            harmonics_component_2: Second cartesian component, shape (n_const, n_lon, n_lat).

        Returns:
            Tuple of (harmonics_lon, harmonics_component_1, harmonics_component_2)
            with aligned and sorted longitudes.
        """
        if np.any(harmonics_lon > 180) and np.any(target_lon < 0):
            harmonics_lon = np.where(
                harmonics_lon > 180, harmonics_lon - 360, harmonics_lon
            )
        elif np.any(harmonics_lon < 0) and np.any(target_lon > 180):
            harmonics_lon = np.where(
                harmonics_lon < 0, harmonics_lon + 360, harmonics_lon
            )

        # Sort so longitudes are monotonically increasing
        sort_idx = np.argsort(harmonics_lon)
        harmonics_lon = harmonics_lon[sort_idx]
        harmonics_component_1 = harmonics_component_1[:, sort_idx, :]
        harmonics_component_2 = harmonics_component_2[:, sort_idx, :]

        return harmonics_lon, harmonics_component_1, harmonics_component_2
