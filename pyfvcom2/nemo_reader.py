"""Read native NEMO model output."""

__all__ = [
    "NEMOReader",
    "default_nemo_grid_var_names",
    "default_nemo_mask_var_names",
    "default_nemo_thickness_var_names",
    "default_fvcom_to_nemo_var_names",
    "default_nemo_zero_profile_mask_var_names",
    "default_nemo_zero_value_mask_var_names",
]

import bisect
from datetime import datetime, timedelta
from os import PathLike
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from pyfvcom2.exceptions import PyFVCOM2ValueError


PathInput = Union[str, PathLike]
FilePathInput = Union[PathInput, Sequence[PathInput]]
GridFilePathInput = Union[FilePathInput, Mapping[str, FilePathInput]]
MaskFilePathInput = Union[PathInput, Mapping[str, PathInput]]


default_nemo_grid_var_names = {
    "T": {
        "time": "time_counter",
        "lon": "nav_lon_grid_T",
        "lat": "nav_lat_grid_T",
        "depth": "deptht",
        "x": "x_grid_T",
        "y": "y_grid_T",
    },
    "U": {
        "time": "time_counter",
        "lon": "nav_lon",
        "lat": "nav_lat",
        "depth": "depthu",
        "x": "x",
        "y": "y",
    },
    "V": {
        "time": "time_counter",
        "lon": "nav_lon",
        "lat": "nav_lat",
        "depth": "depthv",
        "x": "x",
        "y": "y",
    },
    "W": {
        "time": "time_counter",
        "lon": "nav_lon",
        "lat": "nav_lat",
        "depth": "depthw",
        "x": "x",
        "y": "y",
    },
}


default_nemo_mask_var_names = {
    "T": "tmask",
    "U": "umask",
    "V": "vmask",
    "W": "wmask",
}


default_nemo_thickness_var_names = {
    "T": "e3t",
    "U": "e3u",
    "V": "e3v",
    "W": "e3w",
}


default_fvcom_to_nemo_var_names = {
    "temp": "votemper",
    "salinity": "vosaline",
    "zeta": "sossheig",
    "u": ("vozocrtx", "uo"),
    "v": ("vomecrty", "vo"),
}


default_nemo_zero_profile_mask_var_names = ("votemper", "vosaline")
default_nemo_zero_value_mask_var_names = ("votemper", "vosaline")


_GRID_COORD_FALLBACKS = {
    "T": {
        "lon": ("nav_lon_grid_T", "nav_lon"),
        "lat": ("nav_lat_grid_T", "nav_lat"),
        "depth": ("deptht", "depth"),
        "x": ("x_grid_T", "x"),
        "y": ("y_grid_T", "y"),
    },
    "U": {
        "lon": ("nav_lon",),
        "lat": ("nav_lat",),
        "depth": ("depthu", "depth"),
        "x": ("x",),
        "y": ("y",),
    },
    "V": {
        "lon": ("nav_lon",),
        "lat": ("nav_lat",),
        "depth": ("depthv", "depth"),
        "x": ("x",),
        "y": ("y",),
    },
    "W": {
        "lon": ("nav_lon", "nav_lon_grid_W"),
        "lat": ("nav_lat", "nav_lat_grid_W"),
        "depth": ("depthw", "depth"),
        "x": ("x", "x_grid_W"),
        "y": ("y", "y_grid_W"),
    },
}


class NEMOReader:
    """Read native NEMO files split by computational grid.

    Native NEMO outputs are commonly written as separate files for variables
    defined on T, U, V and W grids. Unlike CMEMS products, these files often use
    two-dimensional curvilinear longitude and latitude coordinates. This reader
    keeps those per-grid coordinates intact and builds a time-to-file index for
    each supplied grid.

    Args:
        file_paths: A path/list of paths for T-grid data, or a mapping of grid
            name (``"T"``, ``"U"``, ``"V"``, ``"W"``) to path/list of paths.
        grid_var_names: Optional per-grid variable-name overrides. Each grid
            entry can override ``time``, ``lon``, ``lat``, ``depth``, ``x`` or
            ``y`` names.
        mask_file_paths: Optional path to a NEMO mesh-mask file, or a mapping
            of grid name to mask file. A single path is applied to every
            supplied grid.
        mask_var_names: Optional per-grid mask variable names. Defaults are
            ``tmask``, ``umask``, ``vmask`` and ``wmask``.
        thickness_var_names: Optional per-grid cell-thickness variable names.
            Defaults are ``e3t``, ``e3u``, ``e3v`` and ``e3w``.
        zero_profile_mask_var_names: Optional variable names for which all-depth
            zero profiles should be treated as invalid when no explicit mask is
            available. Defaults to AMM-style temperature and salinity names.
        zero_value_mask_var_names: Optional variable names for which exact zero
            values should be treated as invalid when no explicit mask is
            available. This handles AMM-style below-bottom padding in
            temperature and salinity fields.
    """

    def __init__(
        self,
        file_paths: GridFilePathInput,
        grid_var_names: Optional[Mapping[str, Mapping[str, str]]] = None,
        mask_file_paths: Optional[MaskFilePathInput] = None,
        mask_var_names: Optional[Mapping[str, str]] = None,
        thickness_var_names: Optional[Mapping[str, str]] = None,
        zero_profile_mask_var_names: Optional[Sequence[str]] = (
            default_nemo_zero_profile_mask_var_names
        ),
        zero_value_mask_var_names: Optional[Sequence[str]] = (
            default_nemo_zero_value_mask_var_names
        ),
    ):
        self.file_paths = self._normalise_file_paths(file_paths)
        self.grid_var_names = self._normalise_grid_var_names(grid_var_names)
        self.mask_file_paths = self._normalise_mask_file_paths(mask_file_paths)
        self.mask_var_names = self._normalise_mask_var_names(mask_var_names)
        self.thickness_var_names = self._normalise_thickness_var_names(
            thickness_var_names
        )
        if zero_profile_mask_var_names is None:
            self.zero_profile_mask_var_names = set()
        else:
            self.zero_profile_mask_var_names = set(zero_profile_mask_var_names)
        if zero_value_mask_var_names is None:
            self.zero_value_mask_var_names = set()
        else:
            self.zero_value_mask_var_names = set(zero_value_mask_var_names)

        self._metadata_datasets = {}
        self._mask_datasets = {}
        self._time_to_file_map = {}
        self._all_dates = {}

        for grid, paths in self.file_paths.items():
            if len(paths) == 0:
                raise PyFVCOM2ValueError(f"No NEMO files supplied for {grid} grid")

            self._metadata_datasets[grid] = xr.open_dataset(paths[0])
            self._resolve_grid_var_names(grid)
            self._validate_grid(grid)
            self._build_time_index_mapping(grid)

        for grid, mask_file_path in self.mask_file_paths.items():
            if grid not in self.file_paths:
                raise PyFVCOM2ValueError(
                    f"NEMO mask supplied for {grid} grid, but no {grid} data "
                    "files were supplied."
                )
            self._mask_datasets[grid] = xr.open_dataset(mask_file_path)
            self._resolve_mask_var_name(grid)
            self._validate_mask(grid)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        """Close metadata and mask datasets held by the reader."""
        for dataset in self._metadata_datasets.values():
            dataset.close()
        for dataset in self._mask_datasets.values():
            dataset.close()

    @staticmethod
    def _normalise_file_list(file_paths: FilePathInput) -> list:
        if isinstance(file_paths, (str, PathLike)):
            return [str(file_paths)]

        return [str(path) for path in file_paths]

    @classmethod
    def _normalise_file_paths(cls, file_paths: GridFilePathInput) -> dict:
        if isinstance(file_paths, Mapping):
            return {
                str(grid).upper(): cls._normalise_file_list(paths)
                for grid, paths in file_paths.items()
            }

        return {"T": cls._normalise_file_list(file_paths)}

    def _normalise_grid_var_names(
        self, grid_var_names: Optional[Mapping[str, Mapping[str, str]]]
    ) -> dict:
        names = {
            grid: values.copy()
            for grid, values in default_nemo_grid_var_names.items()
        }

        if grid_var_names is not None:
            for grid, values in grid_var_names.items():
                grid_name = str(grid).upper()
                names.setdefault(grid_name, {})
                names[grid_name].update(values)

        for grid in self.file_paths:
            names.setdefault(grid, {"time": "time_counter"})

        return names

    def _normalise_mask_file_paths(
        self, mask_file_paths: Optional[MaskFilePathInput]
    ) -> dict:
        if mask_file_paths is None:
            return {}

        if isinstance(mask_file_paths, Mapping):
            return {
                str(grid).upper(): str(path)
                for grid, path in mask_file_paths.items()
            }

        mask_file_path = str(mask_file_paths)
        return {grid: mask_file_path for grid in self.file_paths}

    def _normalise_mask_var_names(
        self, mask_var_names: Optional[Mapping[str, str]]
    ) -> dict:
        names = default_nemo_mask_var_names.copy()
        if mask_var_names is not None:
            for grid, var_name in mask_var_names.items():
                names[str(grid).upper()] = var_name
        return names

    def _normalise_thickness_var_names(
        self, thickness_var_names: Optional[Mapping[str, str]]
    ) -> dict:
        names = default_nemo_thickness_var_names.copy()
        if thickness_var_names is not None:
            for grid, var_name in thickness_var_names.items():
                names[str(grid).upper()] = var_name
        return names

    def _resolve_grid_var_names(self, grid: str) -> None:
        """Resolve default coordinate names against variables in a dataset."""
        dataset = self._metadata_datasets[grid]
        names = self.grid_var_names[grid]
        fallbacks = _GRID_COORD_FALLBACKS.get(grid, {})

        for logical_name, candidates in fallbacks.items():
            configured_name = names.get(logical_name)
            if configured_name in dataset.variables or configured_name in dataset.dims:
                continue

            for candidate in candidates:
                if candidate in dataset.variables or candidate in dataset.dims:
                    names[logical_name] = candidate
                    break

    def _validate_grid(self, grid: str) -> None:
        dataset = self._metadata_datasets[grid]
        names = self.grid_var_names[grid]

        required_names = ("time", "lon", "lat")
        for logical_name in required_names:
            var_name = names.get(logical_name)
            if var_name not in dataset.variables and var_name not in dataset.dims:
                raise PyFVCOM2ValueError(
                    f"NEMO {grid} grid is missing required {logical_name} "
                    f"variable/dimension '{var_name}' in {self.file_paths[grid][0]}"
                )

    def _resolve_mask_var_name(self, grid: str) -> None:
        mask_dataset = self._mask_datasets[grid]
        configured_name = self.mask_var_names.get(grid)
        if configured_name in mask_dataset.variables:
            return

        candidates = [
            default_nemo_mask_var_names.get(grid),
            "tmask",
            "umask",
            "vmask",
            "wmask",
            "mask",
        ]
        for candidate in candidates:
            if candidate in mask_dataset.variables:
                self.mask_var_names[grid] = candidate
                return

    def _validate_mask(self, grid: str) -> None:
        mask_var_name = self.mask_var_names.get(grid)
        if mask_var_name not in self._mask_datasets[grid].variables:
            raise PyFVCOM2ValueError(
                f"NEMO {grid} mask variable '{mask_var_name}' not found in "
                f"{self.mask_file_paths[grid]}"
            )

    def _build_time_index_mapping(self, grid: str) -> None:
        """Build a mapping from datetime to ``(file_path, local_time_index)``."""
        time_name = self.time_var_name(grid)

        self._time_to_file_map[grid] = {}
        all_dates = []

        for file_path in self.file_paths[grid]:
            with xr.open_dataset(file_path) as dataset:
                if time_name not in dataset.variables and time_name not in dataset.dims:
                    raise PyFVCOM2ValueError(
                        f"NEMO {grid} file {file_path} is missing time "
                        f"variable/dimension '{time_name}'"
                    )

                times = dataset[time_name].data
                for local_idx, time_val in enumerate(times):
                    if time_val not in self._time_to_file_map[grid]:
                        self._time_to_file_map[grid][time_val] = (
                            file_path,
                            local_idx,
                        )
                    all_dates.append(time_val)

        self._all_dates[grid] = sorted(set(all_dates))

    @staticmethod
    def _normalise_datetime(target_datetime, dates):
        if isinstance(target_datetime, datetime):
            first_date = dates[0] if len(dates) > 0 else None
            if isinstance(first_date, np.datetime64):
                return np.datetime64(target_datetime)

        return target_datetime

    @staticmethod
    def _normalise_tolerance(tolerance):
        if tolerance is None:
            return None

        if isinstance(tolerance, timedelta):
            nanoseconds = int(tolerance.total_seconds() * 1_000_000_000)
            return np.timedelta64(nanoseconds, "ns")

        return np.timedelta64(tolerance)

    def _load_dataset_for_datetime(self, target_datetime, grid: str, tolerance=None):
        """Load the dataset containing the closest available time."""
        grid = self._normalise_grid_name(grid)
        dates = self._all_dates[grid]
        target_datetime = self._normalise_datetime(target_datetime, dates)

        if len(dates) == 0:
            raise PyFVCOM2ValueError(f"No dates available for NEMO {grid} grid")

        start_date = dates[0]
        end_date = dates[-1]

        if target_datetime in self._time_to_file_map[grid]:
            file_path, local_time_index = self._time_to_file_map[grid][
                target_datetime
            ]
        else:
            exact_matches = [date for date in dates if date == target_datetime]
            if exact_matches:
                file_path, local_time_index = self._time_to_file_map[grid][
                    exact_matches[0]
                ]
                return xr.open_dataset(file_path), local_time_index

            if target_datetime < start_date or target_datetime > end_date:
                raise PyFVCOM2ValueError(
                    f"Target datetime {target_datetime} is outside the available "
                    f"NEMO {grid} range [{start_date} to {end_date}]"
                )

            time_diffs = [abs(date - target_datetime) for date in dates]
            closest_idx = time_diffs.index(min(time_diffs))
            closest_time = dates[closest_idx]

            tolerance = self._normalise_tolerance(tolerance)
            if tolerance is not None and min(time_diffs) > tolerance:
                raise PyFVCOM2ValueError(
                    f"Closest available NEMO {grid} time ({closest_time}) is "
                    f"{min(time_diffs)} away from target ({target_datetime}), "
                    f"which exceeds tolerance ({tolerance})"
                )

            file_path, local_time_index = self._time_to_file_map[grid][closest_time]

        return xr.open_dataset(file_path), local_time_index

    @staticmethod
    def _normalise_grid_name(grid: str) -> str:
        return str(grid).upper()

    def _find_grid_for_variable(self, var_name: str, grid: Optional[str] = None) -> str:
        if grid is not None:
            grid_name = self._normalise_grid_name(grid)
            if grid_name not in self._metadata_datasets:
                raise PyFVCOM2ValueError(f"NEMO {grid_name} grid has not been loaded")
            if var_name not in self._metadata_datasets[grid_name].variables:
                raise PyFVCOM2ValueError(
                    f"Variable {var_name} not found in NEMO {grid_name} grid"
                )
            return grid_name

        matching_grids = [
            grid_name
            for grid_name, dataset in self._metadata_datasets.items()
            if var_name in dataset.variables
        ]

        if len(matching_grids) == 0:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} not found in supplied NEMO files"
            )

        if len(matching_grids) > 1:
            raise PyFVCOM2ValueError(
                f"Variable {var_name} exists on multiple NEMO grids "
                f"{matching_grids}; pass grid=... explicitly"
            )

        return matching_grids[0]

    def grid_for_variable(self, var_name: str, grid: Optional[str] = None) -> str:
        """Return the NEMO grid containing ``var_name``."""
        return self._find_grid_for_variable(var_name, grid)

    def _get_depth_dim_for_var(self, dataset: xr.Dataset, var_name: str):
        var_dims = dataset[var_name].dims
        depth_candidates = {
            names.get("depth")
            for names in self.grid_var_names.values()
            if names.get("depth") is not None
        }
        depth_candidates.update(("depth", "deptht", "depthu", "depthv", "depthw"))

        for dim in var_dims:
            if dim in depth_candidates:
                return dim

        return None

    @staticmethod
    def _coord_is_longitude(coord_name: str, coord) -> bool:
        standard_name = str(coord.attrs.get("standard_name", "")).lower()
        return standard_name == "longitude" or "lon" in coord_name.lower()

    @staticmethod
    def _coord_is_latitude(coord_name: str, coord) -> bool:
        standard_name = str(coord.attrs.get("standard_name", "")).lower()
        return standard_name == "latitude" or "lat" in coord_name.lower()

    def _get_horizontal_coord_names_for_var(
        self,
        dataset: xr.Dataset,
        var_name: str,
        grid: str,
    ) -> Tuple[str, str]:
        """Return longitude/latitude coordinate names for a NEMO variable."""
        var = dataset[var_name]
        horizontal_dims = tuple(var.dims[-2:]) if len(var.dims) >= 2 else ()
        lon_name = None
        lat_name = None

        coord_candidates = str(var.attrs.get("coordinates", "")).split()
        for coord_name in coord_candidates:
            if coord_name not in dataset.variables:
                continue

            coord = dataset[coord_name]
            if horizontal_dims and tuple(coord.dims) != horizontal_dims:
                continue

            if lon_name is None and self._coord_is_longitude(coord_name, coord):
                lon_name = coord_name
            if lat_name is None and self._coord_is_latitude(coord_name, coord):
                lat_name = coord_name

        if lon_name is None or lat_name is None:
            for coord_name, coord in dataset.variables.items():
                if horizontal_dims and tuple(coord.dims) != horizontal_dims:
                    continue

                if lon_name is None and self._coord_is_longitude(coord_name, coord):
                    lon_name = coord_name
                if lat_name is None and self._coord_is_latitude(coord_name, coord):
                    lat_name = coord_name

                if lon_name is not None and lat_name is not None:
                    break

        if lon_name is None:
            lon_name = self.lon_var_name(grid)
        if lat_name is None:
            lat_name = self.lat_var_name(grid)

        return lon_name, lat_name

    def _logical_grid_for_variable(
        self,
        dataset: xr.Dataset,
        var_name: str,
        grid: str,
    ) -> str:
        """Infer the NEMO staggered grid used by a variable."""
        var = dataset[var_name]
        names = set(var.dims)
        names.update(str(var.attrs.get("coordinates", "")).split())

        grid_hints = {
            "W": (
                "depthw",
                "x_grid_W",
                "y_grid_W",
                "nav_lon_grid_W",
                "nav_lat_grid_W",
            ),
            "U": (
                "depthu",
                "x_grid_U",
                "y_grid_U",
                "nav_lon_grid_U",
                "nav_lat_grid_U",
            ),
            "V": (
                "depthv",
                "x_grid_V",
                "y_grid_V",
                "nav_lon_grid_V",
                "nav_lat_grid_V",
            ),
            "T": (
                "deptht",
                "x_grid_T",
                "y_grid_T",
                "nav_lon_grid_T",
                "nav_lat_grid_T",
            ),
        }
        for logical_grid, hints in grid_hints.items():
            if any(hint in names for hint in hints):
                return logical_grid

        return grid

    def _get_depth_values_for_var(
        self,
        dataset: xr.Dataset,
        var_name: str,
        grid: str,
    ) -> np.ndarray:
        depth_dim = self._get_depth_dim_for_var(dataset, var_name)
        if depth_dim is not None and depth_dim in dataset.variables:
            return dataset[depth_dim].values

        depth_name = self.depth_var_name(grid)
        if depth_name not in dataset.variables:
            raise PyFVCOM2ValueError(
                f"NEMO {grid} variable {var_name} has no depth coordinate"
            )

        return dataset[depth_name].values

    def _prepare_var_selection(
        self,
        var_name: str,
        target_datetime=None,
        grid: Optional[str] = None,
        depth_index: Optional[int] = None,
        tolerance=None,
    ):
        grid = self._find_grid_for_variable(var_name, grid)

        if target_datetime is None:
            dataset = self._metadata_datasets[grid]
            close_dataset = False
            local_time_index = None
            selector = {}
        else:
            dataset, local_time_index = self._load_dataset_for_datetime(
                target_datetime, grid, tolerance
            )
            close_dataset = True
            selector = {}

        if var_name not in dataset.variables:
            if close_dataset:
                dataset.close()
            raise PyFVCOM2ValueError(
                f"Variable {var_name} not found in NEMO {grid} grid"
            )

        time_name = self.time_var_name(grid)
        if local_time_index is not None and time_name in dataset[var_name].dims:
            selector[time_name] = local_time_index

        depth_dim = self._get_depth_dim_for_var(dataset, var_name)
        if depth_index is not None:
            if depth_dim is None:
                if close_dataset:
                    dataset.close()
                raise PyFVCOM2ValueError(
                    f"Variable {var_name} has no recognised depth dimension"
                )
            selector[depth_dim] = depth_index

        return grid, dataset, selector, close_dataset

    @staticmethod
    def _mask_values_to_invalid(mask_values: np.ndarray) -> np.ndarray:
        """Convert NEMO ocean masks to boolean invalid-point masks."""
        mask_values = np.asarray(mask_values)
        if mask_values.dtype == bool:
            return ~mask_values

        return (~np.isfinite(mask_values)) | (mask_values <= 0)

    def _get_external_mask(
        self,
        grid: str,
        dataset: xr.Dataset,
        var_name: str,
        selector: Mapping[str, int],
        target_shape,
    ):
        if grid not in self._mask_datasets:
            return None

        mask_dataset = self._mask_datasets[grid]
        logical_grid = self._logical_grid_for_variable(dataset, var_name, grid)
        mask_var_name = None
        for candidate in (
            self.mask_var_names.get(logical_grid),
            self.mask_var_names.get(grid),
        ):
            if candidate in mask_dataset.variables:
                mask_var_name = candidate
                break
        if mask_var_name is None:
            raise PyFVCOM2ValueError(
                f"No mask variable found for NEMO {grid} variable {var_name}"
            )

        mask = mask_dataset[mask_var_name]
        mask_selector = {}

        var_dims = dataset[var_name].dims
        depth_dim = self._get_depth_dim_for_var(dataset, var_name)
        selected_time_index = selector.get(self.time_var_name(grid))
        selected_depth_index = selector.get(depth_dim)
        time_names = {"time", "time_counter", "t"}
        depth_names = {
            "z",
            "nav_lev",
            "depth",
            "deptht",
            "depthu",
            "depthv",
            "depthw",
        }

        for dim in mask.dims:
            if dim in selector:
                mask_selector[dim] = selector[dim]
            elif dim == self.time_var_name(grid) and dim in var_dims:
                continue
            elif dim == depth_dim and dim in var_dims:
                continue
            elif dim in time_names and selected_time_index is not None:
                if selected_time_index < mask.sizes[dim]:
                    mask_selector[dim] = selected_time_index
                elif mask.sizes[dim] == 1:
                    mask_selector[dim] = 0
            elif dim in depth_names and selected_depth_index is not None:
                if selected_depth_index < mask.sizes[dim]:
                    mask_selector[dim] = selected_depth_index
                elif mask.sizes[dim] == 1:
                    mask_selector[dim] = 0
            elif dim in time_names and mask.sizes[dim] == 1:
                mask_selector[dim] = 0
            elif dim in depth_names and dim not in var_dims and mask.sizes[dim] == 1:
                mask_selector[dim] = 0
            elif dim not in var_dims and mask.sizes[dim] == 1:
                mask_selector[dim] = 0

        invalid_mask = self._mask_values_to_invalid(mask.isel(mask_selector).values)
        invalid_mask = np.squeeze(invalid_mask)

        # A 3D tmask is often used for 2D sea-surface fields. Use the surface
        # layer in that case, after scalar time dimensions have been removed.
        while invalid_mask.ndim > len(target_shape):
            invalid_mask = invalid_mask[0, ...]

        try:
            return np.broadcast_to(invalid_mask, target_shape)
        except ValueError as exc:
            raise PyFVCOM2ValueError(
                f"NEMO {grid} mask shape {invalid_mask.shape} cannot be "
                f"broadcast to {var_name} selection shape {target_shape}"
            ) from exc

    def _get_zero_profile_mask(
        self,
        dataset: xr.Dataset,
        var_name: str,
        selector: Mapping[str, int],
        target_shape,
    ):
        if var_name not in self.zero_profile_mask_var_names:
            return None

        depth_dim = self._get_depth_dim_for_var(dataset, var_name)
        if depth_dim is None:
            return None

        profile_selector = {
            dim: index for dim, index in selector.items() if dim != depth_dim
        }
        profile = dataset[var_name].isel(profile_selector).values
        profile = np.asanyarray(profile)
        if np.ma.is_masked(profile):
            profile = np.ma.filled(profile, np.nan)

        profile_dims = [
            dim for dim in dataset[var_name].dims if dim not in profile_selector
        ]
        depth_axis = profile_dims.index(depth_dim)

        finite_profile = np.isfinite(profile)
        zero_or_missing_profile = (~finite_profile) | np.isclose(profile, 0.0)
        invalid_horizontal = np.all(zero_or_missing_profile, axis=depth_axis)

        selected_dims = [
            dim for dim in dataset[var_name].dims if dim not in selector
        ]
        if depth_dim in selected_dims:
            depth_axis = selected_dims.index(depth_dim)
            invalid_horizontal = np.expand_dims(invalid_horizontal, axis=depth_axis)

        try:
            return np.broadcast_to(invalid_horizontal, target_shape)
        except ValueError as exc:
            raise PyFVCOM2ValueError(
                f"NEMO inferred zero-profile mask shape "
                f"{invalid_horizontal.shape} cannot be broadcast to {var_name} "
                f"selection shape {target_shape}"
            ) from exc

    def _get_zero_value_mask(
        self,
        data: np.ndarray,
        var_name: str,
    ):
        if var_name not in self.zero_value_mask_var_names:
            return None

        if not np.issubdtype(np.asarray(data).dtype, np.number):
            return None

        return np.isclose(data, 0.0)

    @property
    def grid_names(self):
        """Return the NEMO grid names supplied to the reader."""
        return tuple(self.file_paths.keys())

    def time_var_name(self, grid: str = "T") -> str:
        grid = self._normalise_grid_name(grid)
        return self.grid_var_names[grid]["time"]

    def lon_var_name(self, grid: str = "T") -> str:
        grid = self._normalise_grid_name(grid)
        return self.grid_var_names[grid]["lon"]

    def lat_var_name(self, grid: str = "T") -> str:
        grid = self._normalise_grid_name(grid)
        return self.grid_var_names[grid]["lat"]

    def depth_var_name(self, grid: str = "T") -> str:
        grid = self._normalise_grid_name(grid)
        return self.grid_var_names[grid]["depth"]

    def dates(self, grid: str = "T") -> np.ndarray:
        """Return available dates for a NEMO grid."""
        grid = self._normalise_grid_name(grid)
        return np.array(self._all_dates[grid])

    def time_span(self, grid: str = "T"):
        """Return start, end and count for a NEMO grid's available times."""
        dates = self.dates(grid)
        if len(dates) == 0:
            return None

        return {"start": dates[0], "end": dates[-1], "count": len(dates)}

    def lons(self, grid: str = "T") -> np.ndarray:
        """Return native two-dimensional longitudes for a NEMO grid."""
        grid = self._normalise_grid_name(grid)
        return self._metadata_datasets[grid][self.lon_var_name(grid)].values

    def lats(self, grid: str = "T") -> np.ndarray:
        """Return native two-dimensional latitudes for a NEMO grid."""
        grid = self._normalise_grid_name(grid)
        return self._metadata_datasets[grid][self.lat_var_name(grid)].values

    def depth_levels(self, grid: str = "T", positive_down: bool = True) -> np.ndarray:
        """Return depth levels for a NEMO grid.

        NEMO depth coordinates are conventionally positive down. Set
        ``positive_down=False`` to return pyfvcom2-style z coordinates.
        """
        grid = self._normalise_grid_name(grid)
        depth_name = self.depth_var_name(grid)
        if depth_name not in self._metadata_datasets[grid].variables:
            raise PyFVCOM2ValueError(f"NEMO {grid} grid has no depth variable")

        depths = self._metadata_datasets[grid][depth_name].values
        if positive_down:
            return depths

        return -depths

    def lons_for_variable(
        self,
        var_name: str,
        grid: Optional[str] = None,
    ) -> np.ndarray:
        """Return native longitudes for a NEMO variable."""
        grid = self._find_grid_for_variable(var_name, grid)
        dataset = self._metadata_datasets[grid]
        lon_name, _ = self._get_horizontal_coord_names_for_var(
            dataset,
            var_name,
            grid,
        )
        return dataset[lon_name].values

    def lats_for_variable(
        self,
        var_name: str,
        grid: Optional[str] = None,
    ) -> np.ndarray:
        """Return native latitudes for a NEMO variable."""
        grid = self._find_grid_for_variable(var_name, grid)
        dataset = self._metadata_datasets[grid]
        _, lat_name = self._get_horizontal_coord_names_for_var(
            dataset,
            var_name,
            grid,
        )
        return dataset[lat_name].values

    def depth_levels_for_variable(
        self,
        var_name: str,
        grid: Optional[str] = None,
        positive_down: bool = True,
    ) -> np.ndarray:
        """Return depth levels for a NEMO variable."""
        grid = self._find_grid_for_variable(var_name, grid)
        depths = self._get_depth_values_for_var(
            self._metadata_datasets[grid],
            var_name,
            grid,
        )
        if positive_down:
            return depths

        return -depths

    def get_vertical_coordinates(
        self,
        var_name: str,
        target_datetime=None,
        grid: Optional[str] = None,
        positive_down: bool = False,
    ) -> np.ndarray:
        """Return vertical coordinates for a NEMO variable selection.

        If the relevant NEMO cell-thickness variable (e.g. ``e3t``) is present,
        cell-centre depths are reconstructed from cumulative layer thicknesses.
        This captures time-varying layer thicknesses caused by the free surface.
        If no thickness variable is available, the static depth coordinate is
        broadcast over the horizontal grid.

        Args:
            var_name: NEMO variable name whose grid/depth convention to use.
            target_datetime: Optional time at which to evaluate time-varying
                cell thicknesses.
            grid: Optional explicit NEMO grid name.
            positive_down: Return positive-down depths when True, otherwise
                return z coordinates with negative values below the surface.
        """
        grid = self._find_grid_for_variable(var_name, grid)
        dataset = self._metadata_datasets[grid]
        logical_grid = self._logical_grid_for_variable(dataset, var_name, grid)
        thickness_var_name = self.thickness_var_names.get(logical_grid)

        if (
            thickness_var_name is not None
            and thickness_var_name in dataset.variables
        ):
            thickness = self.get_var(
                thickness_var_name,
                target_datetime=target_datetime,
                grid=grid,
                apply_mask=False,
            )
            if thickness.ndim == 4:
                thickness = thickness[0, :, :, :]
            if thickness.ndim != 3:
                raise PyFVCOM2ValueError(
                    f"NEMO {grid} thickness variable {thickness_var_name} "
                    f"has unsupported shape {thickness.shape}"
                )
            depths = np.cumsum(thickness, axis=0) - 0.5 * thickness
        else:
            depths_1d = self.depth_levels_for_variable(
                var_name,
                grid,
                positive_down=True,
            )
            horizontal_shape = self.lons_for_variable(var_name, grid).shape
            depths = np.broadcast_to(
                depths_1d[:, np.newaxis, np.newaxis],
                (len(depths_1d), *horizontal_shape),
            )

        if positive_down:
            return depths

        return -depths

    @staticmethod
    def _wrapped_lon_delta(lon_1, lon_0):
        """Return longitude difference wrapped onto [-180, 180] degrees."""
        return (lon_1 - lon_0 + 180.0) % 360.0 - 180.0

    def grid_angle(self, grid: str = "T") -> np.ndarray:
        """Return the local NEMO i-axis angle relative to east in radians.

        The angle is estimated from neighbouring longitude/latitude points on
        the selected grid. It is intended for rotating native NEMO i/j velocity
        components onto east/north components on regional curvilinear grids.
        """
        grid = self._normalise_grid_name(grid)
        lons = np.asarray(self.lons(grid), dtype=np.float64)
        lats = np.asarray(self.lats(grid), dtype=np.float64)

        if lons.ndim != 2 or lats.ndim != 2:
            raise PyFVCOM2ValueError(
                f"NEMO {grid} grid angle calculation requires 2D lon/lat arrays"
            )
        if lons.shape[1] < 2:
            raise PyFVCOM2ValueError(
                f"NEMO {grid} grid angle calculation requires at least two "
                "points in the x direction"
            )

        dlon = np.empty_like(lons, dtype=np.float64)
        dlat = np.empty_like(lats, dtype=np.float64)

        dlon[:, 1:-1] = self._wrapped_lon_delta(lons[:, 2:], lons[:, :-2])
        dlat[:, 1:-1] = lats[:, 2:] - lats[:, :-2]
        dlon[:, 0] = self._wrapped_lon_delta(lons[:, 1], lons[:, 0])
        dlat[:, 0] = lats[:, 1] - lats[:, 0]
        dlon[:, -1] = self._wrapped_lon_delta(lons[:, -1], lons[:, -2])
        dlat[:, -1] = lats[:, -1] - lats[:, -2]

        dx = dlon * np.cos(np.deg2rad(lats))
        dy = dlat

        return np.arctan2(dy, dx)

    def contains_date(self, date_time, grid: str = "T") -> bool:
        """Return whether a grid contains ``date_time`` within its date range."""
        dates = self.dates(grid)
        if len(dates) == 0:
            return False

        date_time = self._normalise_datetime(date_time, dates)
        return dates[0] <= date_time <= dates[-1]

    def get_closest_date_index(self, date_time, grid: str = "T") -> int:
        """Return the index of the closest available date for a grid."""
        dates = self.dates(grid)
        date_time = self._normalise_datetime(date_time, dates)
        time_diffs = [abs(date - date_time) for date in dates]
        return time_diffs.index(min(time_diffs))

    def get_bracketing_times(self, target_datetime, grid: str = "T"):
        """Find the two NEMO time steps bracketing ``target_datetime``.

        Returns:
            tuple: ``(t0, t1, alpha)`` where ``alpha`` is the fractional weight
            for ``t1``. Exact matches return ``(t, t, 0.0)``.
        """
        grid = self._normalise_grid_name(grid)
        dates = list(self._all_dates[grid])
        target_datetime = self._normalise_datetime(target_datetime, dates)

        if len(dates) == 0:
            raise PyFVCOM2ValueError(f"No dates available for NEMO {grid} grid")

        if target_datetime < dates[0] or target_datetime > dates[-1]:
            raise PyFVCOM2ValueError(
                f"Target datetime {target_datetime} is outside the available "
                f"NEMO {grid} range [{dates[0]} to {dates[-1]}]"
            )

        if target_datetime in self._time_to_file_map[grid]:
            return target_datetime, target_datetime, 0.0

        exact_matches = [date for date in dates if date == target_datetime]
        if exact_matches:
            exact_time = exact_matches[0]
            return exact_time, exact_time, 0.0

        idx = bisect.bisect_right(dates, target_datetime)
        t0 = dates[idx - 1]
        t1 = dates[idx]

        dt_total = (t1 - t0) / np.timedelta64(1, "s")
        dt_target = (target_datetime - t0) / np.timedelta64(1, "s")
        alpha = float(dt_target / dt_total)

        return t0, t1, alpha

    def get_var_ndims(self, var_name: str, grid: Optional[str] = None) -> int:
        """Return the number of dimensions for a NEMO variable."""
        grid = self._find_grid_for_variable(var_name, grid)
        return len(self._metadata_datasets[grid][var_name].dims)

    def get_var(
        self,
        var_name: str,
        target_datetime=None,
        grid: Optional[str] = None,
        depth_index: Optional[int] = None,
        tolerance=None,
        apply_mask: bool = True,
    ) -> np.ndarray:
        """Return values for a NEMO variable.

        If ``target_datetime`` is omitted, the variable is read from the first
        metadata file without time selection. If ``depth_index`` is supplied for
        a 3D variable, only that vertical level is returned.
        """
        grid, dataset, selector, close_dataset = self._prepare_var_selection(
            var_name,
            target_datetime=target_datetime,
            grid=grid,
            depth_index=depth_index,
            tolerance=tolerance,
        )

        try:
            data = dataset[var_name].isel(selector).values
            if apply_mask:
                external_mask = self._get_external_mask(
                    grid,
                    dataset,
                    var_name,
                    selector,
                    np.shape(data),
                )
                if external_mask is not None:
                    if not np.issubdtype(np.asarray(data).dtype, np.floating):
                        data = data.astype(np.float64)
                    else:
                        data = data.copy()
                    data[external_mask] = np.nan
                zero_profile_mask = self._get_zero_profile_mask(
                    dataset,
                    var_name,
                    selector,
                    np.shape(data),
                )
                if zero_profile_mask is not None:
                    if not np.issubdtype(np.asarray(data).dtype, np.floating):
                        data = data.astype(np.float64)
                    else:
                        data = data.copy()
                    data[zero_profile_mask] = np.nan
                zero_value_mask = self._get_zero_value_mask(data, var_name)
                if zero_value_mask is not None:
                    if not np.issubdtype(np.asarray(data).dtype, np.floating):
                        data = data.astype(np.float64)
                    else:
                        data = data.copy()
                    data[zero_value_mask] = np.nan

            return data
        finally:
            if close_dataset:
                dataset.close()

    def get_mask(
        self,
        var_name: str,
        target_datetime=None,
        grid: Optional[str] = None,
        depth_index: Optional[int] = None,
        tolerance=None,
    ) -> np.ndarray:
        """Return a boolean mask for a NEMO variable selection."""
        grid, dataset, selector, close_dataset = self._prepare_var_selection(
            var_name,
            target_datetime=target_datetime,
            grid=grid,
            depth_index=depth_index,
            tolerance=tolerance,
        )

        try:
            data = dataset[var_name].isel(selector).values

            if np.ma.is_masked(data):
                data_mask = np.ma.getmaskarray(data)
            elif np.issubdtype(np.asarray(data).dtype, np.number):
                data_mask = np.isnan(data)
            else:
                data_mask = np.zeros(np.shape(data), dtype=bool)

            external_mask = self._get_external_mask(
                grid,
                dataset,
                var_name,
                selector,
                np.shape(data),
            )
            if external_mask is not None:
                data_mask = data_mask | external_mask

            zero_profile_mask = self._get_zero_profile_mask(
                dataset,
                var_name,
                selector,
                np.shape(data),
            )
            if zero_profile_mask is not None:
                data_mask = data_mask | zero_profile_mask

            zero_value_mask = self._get_zero_value_mask(data, var_name)
            if zero_value_mask is not None:
                data_mask = data_mask | zero_value_mask

            return data_mask
        finally:
            if close_dataset:
                dataset.close()
