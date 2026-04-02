import xarray as xr
import numpy as np
from typing import NamedTuple, Optional, List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from .exceptions import PyFVCOM2ValueError

__all__ = [
    "FVCOMHarmonicsNames", "TPXOHarmonicsNames", "TPXOComplexHarmonicsNames", 
    "HarmonicsData", "HarmonicsReader", "FVCOMHarmonicsReader", 
    "TPXOHarmonicsReader", "TPXOComplexHarmonicsReader"
]

# Named tuples to hold variable names for different harmonics data formats
FVCOMHarmonicsNames = NamedTuple(
    "FVCOMHarmonicsNames",
    [
        ("amplitude_var_name", str),
        ("phase_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
        ("apply_transport_unit_conversion", bool),
        ("bathy_var_name", Optional[str]),
    ],
)


TPXOHarmonicsNames = NamedTuple(
    "TPXOHarmonicsNames",
    [
        ("amplitude_var_name", str),
        ("phase_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
        ("apply_transport_unit_conversion", bool),
        ("bathy_var_name", Optional[str]),
    ],
)


TPXOComplexHarmonicsNames = NamedTuple(
    "TPXOComplexHarmonicsNames",
    [
        ("part1_var_name", str),
        ("part2_var_name", str),
        ("lon_var_name", str),
        ("lat_var_name", str),
        ("constituents_var_name", str),
        ("apply_transport_unit_conversion", bool),
        ("bathy_var_name", Optional[str]),
    ],
)


# Named tuple to hold returned harmonics data
HarmonicsData = NamedTuple(
    "HarmonicsData",
    [
        ("longitude", np.ndarray),
        ("latitude", np.ndarray),
        ("amplitudes", np.ndarray),
        ("phases", np.ndarray),
        ("constituents", List[str]),
    ],
)


class HarmonicsReader(ABC):
    """Abstract base class for reading tidal harmonics data from netCDF files."""

    def __init__(self, file_path: Union[str, Dict[str, str]]):
        """Initialise the reader.

        Args:
            file_path: Path to the harmonics data file. Either a single
                path (str) when all constituents are in one file, or a
                dict mapping constituent names (uppercase) to file paths
                when each constituent is stored in a separate file.
        """
        self.file_path = file_path

    @abstractmethod
    def read_harmonics(
        self, requested_constituents: List[str], var_names: NamedTuple
    ) -> HarmonicsData:
        """Read harmonics data for the specified constituents and variable.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for the specific data format (FVCOM, TPXO, etc.).
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        pass

    def _parse_constituents(self, constituents_var):
        """Parse the constituents variable from the netCDF file to extract available constituent names.

        Args:
            constituents_var: The xarray DataArray containing constituent names from the netCDF file.
        Returns:
            List of available constituent names in the file, formatted as uppercase strings with whitespace stripped.
        """
        try:
            n_const = constituents_var.sizes['nc']
        except KeyError:
            n_const = None

        if n_const is None:
            # Single constituent case - return a list with one name
            return [constituents_var.data.item().decode().upper().strip()]
        else:
            names = []
            for i in range(n_const):
                name = constituents_var.data[i].decode().upper().strip()
                names.append(name)

        return names

    def _check_constituents(
        self, requested_constituents: List[str], constituents_var_name
    ):
        """Check if the requested constituents are available in the file(s).

        Args:
            requested_constituents: List of tidal constituent names to read.
            constituents_var_name: Name of the variable in the netCDF file that contains constituent names.
        Raises:
            PyFVCOM2ValueError: If any requested constituent is not available in the file(s).
        """
        if isinstance(self.file_path, dict):
            # Per-constituent files: each key is a constituent name
            missing = [c for c in requested_constituents if c not in self.file_path]
            if missing:
                raise PyFVCOM2ValueError(
                    f"The following requested constituents have no file path entry: {missing}"
                )
            # Also verify each file actually contains the named constituent
            for constituent in requested_constituents:
                with xr.open_dataset(str(self.file_path[constituent])) as tides:
                    available = self._parse_constituents(tides.variables[constituents_var_name])
                if constituent not in available:
                    raise PyFVCOM2ValueError(
                        f"Constituent '{constituent}' not found in file "
                        f"'{self.file_path[constituent]}'. Available: {available}"
                    )
        else:
            with xr.open_dataset(str(self.file_path)) as tides:
                available_constituents = self._parse_constituents(tides.variables[constituents_var_name])

            missing = [c for c in requested_constituents if c not in available_constituents]
            if missing:
                raise PyFVCOM2ValueError(
                    f"The following requested constituents are not available in the file: {missing}"
                )

    @staticmethod
    def _fill_land_points(lons, lats, amplitudes, phases):
        """Fill land points (where amplitude is zero) using nearest ocean neighbour.

        TPXO uses 0.0 in amplitude arrays to indicate land. These zeros
        contaminate the interpolation if left in place. This method identifies
        land points using a mask where the amplitude is zero for all
        constituents, then fills them with values from the nearest ocean point
        using a KD-tree lookup.

        Args:
            lons: 2D longitude array (nx, ny).
            lats: 2D latitude array (nx, ny).
            amplitudes: Amplitude array, shape (n_constituents, nx, ny).
            phases: Phase array, shape (n_constituents, nx, ny).

        Returns:
            Tuple of (amplitudes, phases) with land points filled.
        """
        amplitudes = np.array(amplitudes, dtype=float)
        phases = np.array(phases, dtype=float)

        # Build 2D coordinate grids if needed
        if lons.ndim == 1 and lats.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lons, lats, indexing='ij')
        else:
            lon_2d, lat_2d = np.asarray(lons), np.asarray(lats)

        # Land mask: True where amplitude is zero or NaN for ALL constituents
        land_mask = np.all((amplitudes == 0.0) | np.isnan(amplitudes), axis=0)

        if not np.any(land_mask):
            return amplitudes, phases

        ocean_mask = ~land_mask
        ocean_points = np.column_stack((lon_2d[ocean_mask], lat_2d[ocean_mask]))
        land_points = np.column_stack((lon_2d[land_mask], lat_2d[land_mask]))

        # Find nearest ocean point for each land point
        tree = cKDTree(ocean_points)
        _, nearest_idx = tree.query(land_points)

        n_constituents = amplitudes.shape[0]
        for i in range(n_constituents):
            amplitudes[i][land_mask] = amplitudes[i][ocean_mask][nearest_idx]
            phases[i][land_mask] = phases[i][ocean_mask][nearest_idx]

        return amplitudes, phases

    @staticmethod
    def _reduce_coords_to_1d(lons, lats):
        """Reduce 2-D coordinate arrays to 1-D if they form a regular grid.

        Some TPXO files store lon/lat as 2-D arrays (one value per grid
        cell) while others store them as 1-D vectors.  When the arrays
        are 2-D, the values along one axis should be constant (i.e. a
        meshgrid).  This method extracts the unique values along each
        axis and checks that they faithfully reconstruct the original
        2-D arrays.

        Args:
            lons: Longitude array (1-D or 2-D).
            lats: Latitude array (1-D or 2-D).

        Returns:
            Tuple of (lons_1d, lats_1d).

        Raises:
            PyFVCOM2ValueError: If the 2-D arrays cannot be reduced to
                1-D without loss of information.
        """
        if lons.ndim == 1 and lats.ndim == 1:
            return lons, lats

        if lons.ndim != 2 or lats.ndim != 2:
            raise PyFVCOM2ValueError(
                f"Expected lon/lat arrays of 1 or 2 dimensions, "
                f"got lon ndim={lons.ndim}, lat ndim={lats.ndim}."
            )

        lons_1d = np.unique(lons)
        lats_1d = np.unique(lats)

        # Verify the 1-D vectors reconstruct the original 2-D arrays
        lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d, indexing='ij')
        if lon_2d.shape != lons.shape or not np.allclose(lon_2d, lons):
            raise PyFVCOM2ValueError(
                "Cannot reduce 2-D longitude array to 1-D: the values "
                "do not form a regular meshgrid."
            )
        if lat_2d.shape != lats.shape or not np.allclose(lat_2d, lats):
            raise PyFVCOM2ValueError(
                "Cannot reduce 2-D latitude array to 1-D: the values "
                "do not form a regular meshgrid."
            )

        return lons_1d, lats_1d

    @staticmethod
    def _shift_longitudes(lons, *arrays):
        """Shift longitudes from 0-360 to -180-180 and reorder arrays accordingly.

        If longitudes are already in -180-180, this is a no-op.

        Args:
            lons: 1-D array of longitudes.
            *arrays: Data arrays whose longitude axis (axis=1 for 3-D,
                axis=0 for 2-D) will be reordered to match the sorted
                longitudes.

        Returns:
            Tuple of (lons, *arrays) with longitudes shifted and arrays
            reordered.
        """
        if lons.max() <= 180.0:
            return (lons, *arrays)

        lons = np.where(lons > 180, lons - 360, lons)
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]

        reordered = []
        for arr in arrays:
            if arr.ndim == 3:
                reordered.append(arr[:, sort_idx, :])
            elif arr.ndim == 2:
                reordered.append(arr[sort_idx, :])
            else:
                reordered.append(arr[sort_idx])
        return (lons, *reordered)

    @staticmethod
    def _convert_bathy_to_metres(bathy, units):
        """Convert a bathymetry array to metres based on its units string.

        Args:
            bathy: Bathymetry array.
            units: Units string from the netCDF variable (e.g. 'cm', 'km').

        Returns:
            Bathymetry array in metres.

        Raises:
            PyFVCOM2ValueError: If the units string is not recognised.
        """
        units_lower = units.strip().lower()
        if units_lower in ('m', 'meter', 'meters', 'metre', 'metres'):
            return bathy
        elif units_lower in ('cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres'):
            return bathy / 100.0
        elif units_lower in ('km', 'kilometer', 'kilometers', 'kilometre', 'kilometres'):
            return bathy * 1000.0
        else:
            raise PyFVCOM2ValueError(
                f"Unrecognised bathymetry units '{units}'. "
                f"Expected one of: m, cm, km (or their long forms)."
            )

    @staticmethod
    def _get_length_scale_to_metres(units):
        """Return the multiplicative factor that converts a length unit to metres.

        The *units* string may be a pure length (e.g. ``'cm'``), a velocity
        (e.g. ``'cm/s'``), or a transport term (e.g. ``'cm^2/s'``).  Only
        the length component varies across TPXO files; time is always in
        seconds.  This method extracts the length prefix and returns the
        appropriate scale factor so that the caller can multiply amplitudes
        to obtain metres (or m/s, m^2/s, etc.).

        Recognised length prefixes: mm, cm, m, km.

        Args:
            units: Units string from the netCDF variable attribute.

        Returns:
            Scale factor (float) to multiply amplitudes by.

        Raises:
            PyFVCOM2ValueError: If the length prefix is not recognised.
        """
        units_stripped = units.strip().lower()

        # Split off the time part (e.g. '/s') if present
        parts = units_stripped.split('/')
        length_with_exp = parts[0].strip()

        # Check for an exponent (e.g. 'cm^2' for transport)
        if '^' in length_with_exp:
            length_part, exp_str = length_with_exp.split('^', 1)
            length_part = length_part.strip()
            exponent = int(exp_str.strip())
        else:
            length_part = length_with_exp
            exponent = 1

        scale_map = {
            'mm': 0.001,
            'millimeter': 0.001, 'millimeters': 0.001,
            'millimetre': 0.001, 'millimetres': 0.001,
            'cm': 0.01,
            'centimeter': 0.01, 'centimeters': 0.01,
            'centimetre': 0.01, 'centimetres': 0.01,
            'm': 1.0,
            'meter': 1.0, 'meters': 1.0,
            'metre': 1.0, 'metres': 1.0,
            'km': 1000.0,
            'kilometer': 1000.0, 'kilometers': 1000.0,
            'kilometre': 1000.0, 'kilometres': 1000.0,
        }

        if length_part not in scale_map:
            raise PyFVCOM2ValueError(
                f"Unrecognised length unit '{length_part}' in units string '{units}'. "
                f"Expected one of: mm, cm, m, km (or their long forms)."
            )

        return scale_map[length_part] ** exponent

    @staticmethod
    def _compute_bbox_indices(
        lons_1d: np.ndarray,
        lats_1d: np.ndarray,
        bbox: Tuple[float, float, float, float],
        margin: float = 1.0,
    ) -> Tuple[slice, slice, np.ndarray, np.ndarray]:
        """Compute index slices that subset 1-D lon/lat arrays to a bounding box.

        Longitudes are expected to already be in -180–180 (see
        ``_shift_longitudes``).  The bbox should use the same convention.

        Args:
            lons_1d: 1-D array of longitudes (must be in -180 to 180).
            lats_1d: 1-D array of latitudes from the file.
            bbox: (lon_min, lon_max, lat_min, lat_max) in -180 to 180.
            margin: Extra padding in degrees added around the bbox.

        Returns:
            (lon_slice, lat_slice, lons_subset, lats_subset) where the
            slices index into the supplied arrays and the subset arrays
            are the coordinate values for the selected region.
        """
        lon_min, lon_max, lat_min, lat_max = bbox

        # Expand by margin
        lon_min -= margin
        lon_max += margin
        lat_min -= margin
        lat_max += margin

        # Longitude indices
        lon_idx = np.where((lons_1d >= lon_min) & (lons_1d <= lon_max))[0]
        if lon_idx.size == 0:
            raise PyFVCOM2ValueError(
                f"No longitude values found within bbox "
                f"[{lon_min}, {lon_max}] (file range "
                f"[{lons_1d.min()}, {lons_1d.max()}])."
            )
        lon_sl = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)

        # Latitude indices
        lat_idx = np.where((lats_1d >= lat_min) & (lats_1d <= lat_max))[0]
        if lat_idx.size == 0:
            raise PyFVCOM2ValueError(
                f"No latitude values found within bbox "
                f"[{lat_min}, {lat_max}] (file range "
                f"[{lats_1d.min()}, {lats_1d.max()}])."
            )
        lat_sl = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)

        return lon_sl, lat_sl, lons_1d[lon_sl], lats_1d[lat_sl]


class FVCOMHarmonicsReader(HarmonicsReader):
    """A class to read FVCOM harmonics data from a netCDF file."""

    def __init__(self, file_path: str):
        super().__init__(file_path)

    def read_harmonics(
        self, requested_constituents: List[str], var_names: FVCOMHarmonicsNames
    ) -> HarmonicsData:
        """Read FVCOM harmonics data for the specified constituents and variable.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for FVCOM data.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        with xr.open_dataset(str(self.file_path)) as tides:
            # Read available constituents from file
            constituent_names = self._parse_constituents(tides[var_names.constituents_var_name])

            # Determine the indices of the requested constituents
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            # Read coordinate data
            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            # Read amplitude and phase data
            amplitudes = tides[var_names.amplitude_var_name].isel(nconsts=const_indices)
            phases = tides[var_names.phase_var_name].isel(nconsts=const_indices)

            # If necessary, reorder the array so that constiuents are the first dimension
            amplitudes = amplitudes.transpose("nconsts", ...)
            phases = phases.transpose("nconsts", ...)

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


class TPXOHarmonicsReader(HarmonicsReader):
    """A class to read TPXO harmonics data stored as amplitude and phase from a netCDF file."""

    def __init__(self, file_path: Union[str, Dict[str, str]]):
        super().__init__(file_path)

    def _read_single_file(self, file_path, requested_constituents, var_names):
        """Read amplitude and phase data from a single file.

        Returns:
            Tuple of (lons, lats, amplitudes, phases) where amplitudes
            and phases have shape (n_constituents, nx, ny).
        """
        with xr.open_dataset(str(file_path)) as tides:
            constituent_names = self._parse_constituents(tides[var_names.constituents_var_name])
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            if "nc" in tides.dims:
                amplitudes = tides[var_names.amplitude_var_name].isel(nc=const_indices)
                phases = tides[var_names.phase_var_name].isel(nc=const_indices)
                amplitudes = amplitudes.transpose("nc", ...).values
                phases = phases.transpose("nc", ...).values
            else:
                amplitudes = tides[var_names.amplitude_var_name].values[np.newaxis, ...]
                phases = tides[var_names.phase_var_name].values[np.newaxis, ...]

            amp_units = tides[var_names.amplitude_var_name].attrs.get('units', 'm')
            scale = self._get_length_scale_to_metres(amp_units)
            if scale != 1.0:
                amplitudes = amplitudes * scale

        return lons, lats, amplitudes, phases

    def read_harmonics(
        self, requested_constituents: List[str], var_names: TPXOHarmonicsNames,
        fill_land: bool = True,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_margin: float = 1.0,
    ) -> HarmonicsData:
        """Read TPXO harmonics data for the specified constituents.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for TPXO data.
            fill_land: Whether to fill land points (where amplitude == 0) by interpolation from ocean points. Defaults to True.
            bbox: Optional (lon_min, lon_max, lat_min, lat_max) bounding box
                to spatially subset the data before loading. Uses the target
                coordinate system (e.g. -180 to 180). Dramatically reduces
                memory usage and processing time for high-resolution files.
            bbox_margin: Extra margin in degrees around the bbox. Defaults to 1.0.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        if isinstance(self.file_path, dict):
            # Per-constituent files: read each one individually and stack
            amp_list, phase_list = [], []
            lons = lats = None
            for constituent in requested_constituents:
                fp = self.file_path[constituent]
                c_lons, c_lats, c_amp, c_phase = self._read_single_file(
                    fp, [constituent], var_names
                )
                if lons is None:
                    lons, lats = c_lons, c_lats
                amp_list.append(c_amp[0])
                phase_list.append(c_phase[0])
            amplitudes = np.stack(amp_list, axis=0)
            phases = np.stack(phase_list, axis=0)
        else:
            lons, lats, amplitudes, phases = self._read_single_file(
                self.file_path, requested_constituents, var_names
            )

        # Reduce 2-D coordinate arrays to 1-D if applicable
        lons, lats = self._reduce_coords_to_1d(lons, lats)

        # Shift longitudes from 0-360 to -180-180 and reorder data
        lons, amplitudes, phases = self._shift_longitudes(lons, amplitudes, phases)

        # Spatial subsetting (lons are now in -180 to 180)
        if bbox is not None:
            lon_sl, lat_sl, lons, lats = self._compute_bbox_indices(
                lons, lats, bbox, margin=bbox_margin
            )
            amplitudes = amplitudes[:, lon_sl, lat_sl]
            phases = phases[:, lon_sl, lat_sl]

        # Fill land points (where amplitude == 0) by interpolation from ocean points
        if fill_land:
            amplitudes, phases = self._fill_land_points(lons, lats, amplitudes, phases)
        else:
            land_mask = np.all(amplitudes == 0.0, axis=0)
            amplitudes[:, land_mask] = np.nan
            phases[:, land_mask] = np.nan

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


class TPXOComplexHarmonicsReader(HarmonicsReader):
    """A class to read TPXO harmonics data stored as complex (real/imaginary) from a netCDF file."""

    def __init__(self, file_path: Union[str, Dict[str, str]]):
        super().__init__(file_path)

    def _read_single_file(self, file_path, requested_constituents, var_names):
        """Read real/imaginary data and coordinates from a single file.

        Returns:
            Tuple of (lons, lats, real, imag, real_units) where real and
            imag have shape (n_constituents, nx, ny).
        """
        with xr.open_dataset(str(file_path)) as tides:
            constituent_names = self._parse_constituents(tides[var_names.constituents_var_name])
            const_indices = [constituent_names.index(i) for i in requested_constituents]

            lons = tides[var_names.lon_var_name].data[:]
            lats = tides[var_names.lat_var_name].data[:]

            if "nc" in tides.dims:
                real = tides[var_names.part1_var_name].isel(nc=const_indices)
                imag = tides[var_names.part2_var_name].isel(nc=const_indices)
                real = real.transpose("nc", ...).values
                imag = imag.transpose("nc", ...).values
            else:
                real = tides[var_names.part1_var_name].values[np.newaxis, ...]
                imag = tides[var_names.part2_var_name].values[np.newaxis, ...]

            real_units = tides[var_names.part1_var_name].attrs.get('units', '')
            imag_units = tides[var_names.part2_var_name].attrs.get('units', '')

            if real_units != imag_units:
                raise PyFVCOM2ValueError(
                    f"Real and imaginary parts have different units: "
                    f"real units='{real_units}', imag units='{imag_units}'."
                )

        return lons, lats, real, imag, real_units

    def read_harmonics(
        self, requested_constituents: List[str], var_names: TPXOComplexHarmonicsNames,
        fill_land: bool = True,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_margin: float = 1.0,
        bathy_file: Optional[np.ndarray] = None,
    ) -> HarmonicsData:
        """Read TPXO complex harmonics data for the specified constituents.

        Args:
            requested_constituents: List of tidal constituent names to read.
            var_names: NamedTuple containing variable names for TPXO data.
            fill_land: Whether to fill land points (where amplitude == 0) by interpolation from ocean points. Defaults to True.
            bbox: Optional (lon_min, lon_max, lat_min, lat_max) bounding box
                to spatially subset the data before loading. Uses the target
                coordinate system (e.g. -180 to 180). Dramatically reduces
                memory usage and processing time for high-resolution files.
            bbox_margin: Extra margin in degrees around the bbox. Defaults to 1.0.
            bathy_file: Optional path to a netCDF file containing bathymetry data on the same grid, required if apply_transport_unit_conversion is True.
        Returns:
            HarmonicsData: NamedTuple containing longitude, latitude, amplitudes, phases, and constituents
        """
        self._check_constituents(
            requested_constituents, var_names.constituents_var_name
        )

        # If transport unit conversion is needed, read bathymetry data
        if var_names.apply_transport_unit_conversion:
            if var_names.bathy_var_name is None:
                raise PyFVCOM2ValueError(
                    "apply_transport_unit_conversion is True but bathy_var_name is not provided in var_names."
                )
            if bathy_file is None:
                raise PyFVCOM2ValueError(
                    "apply_transport_unit_conversion is True but no bathy_file path provided."
                )
            with xr.open_dataset(str(bathy_file)) as bathy_ds:
                bathy = bathy_ds[var_names.bathy_var_name].data[:]
                bathy_units = bathy_ds[var_names.bathy_var_name].attrs.get('units', 'm')
                bathy = self._convert_bathy_to_metres(bathy, bathy_units)
                bathy = np.where(bathy == 0, np.nan, bathy)

        if isinstance(self.file_path, dict):
            # Per-constituent files: read each one individually and stack
            real_list, imag_list = [], []
            lons = lats = None
            real_units = None
            for constituent in requested_constituents:
                fp = self.file_path[constituent]
                c_lons, c_lats, c_real, c_imag, c_units = self._read_single_file(
                    fp, [constituent], var_names
                )
                if lons is None:
                    lons, lats = c_lons, c_lats
                    real_units = c_units
                real_list.append(c_real[0])
                imag_list.append(c_imag[0])
            real = np.stack(real_list, axis=0)
            imag = np.stack(imag_list, axis=0)
        else:
            lons, lats, real, imag, real_units = self._read_single_file(
                self.file_path, requested_constituents, var_names
            )

        # Reduce 2-D coordinate arrays to 1-D if applicable
        lons, lats = self._reduce_coords_to_1d(lons, lats)

        # Shift longitudes from 0-360 to -180-180 and reorder data
        if var_names.apply_transport_unit_conversion:
            lons, real, imag, bathy = self._shift_longitudes(lons, real, imag, bathy)
        else:
            lons, real, imag = self._shift_longitudes(lons, real, imag)

        # Spatial subsetting (lons are now in -180 to 180)
        if bbox is not None:
            lon_sl, lat_sl, lons, lats = self._compute_bbox_indices(
                lons, lats, bbox, margin=bbox_margin
            )
            real = real[:, lon_sl, lat_sl]
            imag = imag[:, lon_sl, lat_sl]
            if var_names.apply_transport_unit_conversion:
                bathy = bathy[lon_sl, lat_sl]

        # Convert complex to amplitude and phase
        amplitudes = np.array(np.abs(real + 1j * imag), dtype=float)
        phases = np.array((np.arctan2(-imag, real) / np.pi) * 180, dtype=float)

        # Convert amplitude length units to metres
        scale = self._get_length_scale_to_metres(real_units)
        if scale != 1.0:
            amplitudes = amplitudes * scale

        # Convert transport to velocity by dividing through bathymetry
        if var_names.apply_transport_unit_conversion:
            amplitudes = amplitudes / bathy[np.newaxis, :, :]

        # Fill land points (where amplitude == 0) by interpolation from ocean points
        if fill_land:
            amplitudes, phases = self._fill_land_points(lons, lats, amplitudes, phases)
        else:
            amp_land_mask = np.all(amplitudes == 0.0, axis=0)
            phase_land_mask = np.all(phases == 0.0, axis=0)
            amplitudes[:, amp_land_mask] = np.nan
            phases[:, phase_land_mask] = np.nan

        return HarmonicsData(
            longitude=lons,
            latitude=lats,
            amplitudes=amplitudes,
            phases=phases,
            constituents=requested_constituents,
        )


def create_harmonics_reader(reader_type: str, file_path: Union[str, Dict[str, str]]) -> HarmonicsReader:
    """Factory function to create the appropriate harmonics reader.

    Args:
        reader_type: Type of reader to create. Options:
            - 'fvcom': FVCOMHarmonicsReader
            - 'tpxo': TPXOHarmonicsReader (amplitude/phase format)
            - 'tpxo_complex': TPXOComplexHarmonicsReader (real/imaginary format)
        file_path: Path to the harmonics data file(s). Either a single
            path (str) when all constituents are in one file, or a dict
            mapping constituent names to file paths.

    Returns:
        Instance of the appropriate reader subclass.

    Raises:
        ValueError: If reader_type is not recognized.
    """
    if reader_type.lower() == "fvcom":
        return FVCOMHarmonicsReader(file_path)
    elif reader_type.lower() == "tpxo":
        return TPXOHarmonicsReader(file_path)
    elif reader_type.lower() == "tpxo_complex":
        return TPXOComplexHarmonicsReader(file_path)
    else:
        valid_types = ["fvcom", "tpxo", "tpxo_complex"]
        raise ValueError(
            f"Unknown reader_type '{reader_type}'. Valid options: {valid_types}"
        )


def get_fvcom_harmonics_names(var_name: str) -> FVCOMHarmonicsNames:
    """Get the standard variable names for FVCOM harmonics data

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        FVCOMHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        amplitude_name, phase_name = "z_amp", "z_phase"
        lon_name, lat_name = "lon", "lat"
    elif var_name == "u":
        amplitude_name, phase_name = "u_amp", "u_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "v":
        amplitude_name, phase_name = "v_amp", "v_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "ua":
        amplitude_name, phase_name = "ua_amp", "ua_phase"
        lon_name, lat_name = "lonc", "latc"
    elif var_name == "va":
        amplitude_name, phase_name = "va_amp", "va_phase"
        lon_name, lat_name = "lonc", "latc"
    else:
        raise PyFVCOM2ValueError(f"Unknown FVCOM variable name '{var_name}'")

    return FVCOMHarmonicsNames(
        amplitude_var_name=amplitude_name,
        phase_var_name=phase_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="z_const_names",
        apply_transport_unit_conversion=False,
        bathy_var_name=None,
    )


def get_tpxo_harmonics_names(var_name: str) -> TPXOHarmonicsNames:
    """Get the standard variable names for TPXO harmonics data stored as amplitude and phase.

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        TPXOHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        amplitude_name, phase_name = "ha", "hp"
        lon_name, lat_name = "lon_z", "lat_z"
    elif var_name == "u":
        amplitude_name, phase_name = "ua", "up"
        lon_name, lat_name = "lon_u", "lat_u"
    elif var_name == "v":
        amplitude_name, phase_name = "va", "vp"
        lon_name, lat_name = "lon_v", "lat_v"
    else:
        raise PyFVCOM2ValueError(f"Unknown TPXO variable name '{var_name}'")

    return TPXOHarmonicsNames(
        amplitude_var_name=amplitude_name,
        phase_var_name=phase_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="con",
        apply_transport_unit_conversion=False,
        bathy_var_name=None,
    )


def get_tpxo_complex_harmonics_names(var_name: str) -> TPXOComplexHarmonicsNames:
    """Get the standard variable names for TPXO harmonics data stored as complex (real/imaginary).

    Args:
        var_name: Base variable name for the harmonics data.
    Returns:
        TPXOComplexHarmonicsNames: NamedTuple containing standard variable names.
    Raises:
        PyFVCOM2ValueError: If var_name is not recognized.
    """
    if var_name == "zeta":
        part1_name, part2_name = "hRe", "hIm"
        lon_name, lat_name = "lon_z", "lat_z"
        bathy_var_name = "hz"
        apply_transport_unit_conversion = False
    elif var_name == "u":
        part1_name, part2_name = "uRe", "uIm"
        lon_name, lat_name = "lon_u", "lat_u"
        bathy_var_name = "hu"
        apply_transport_unit_conversion = True
    elif var_name == "v":
        part1_name, part2_name = "vRe", "vIm"
        lon_name, lat_name = "lon_v", "lat_v"
        bathy_var_name = "hv"
        apply_transport_unit_conversion = True
    else:
        raise PyFVCOM2ValueError(f"Unknown TPXO variable name '{var_name}'")

    return TPXOComplexHarmonicsNames(
        part1_var_name=part1_name,
        part2_var_name=part2_name,
        lon_var_name=lon_name,
        lat_var_name=lat_name,
        constituents_var_name="con",
        apply_transport_unit_conversion=apply_transport_unit_conversion,
        bathy_var_name=bathy_var_name,
    )
