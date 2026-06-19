from __future__ import annotations

import os
import warnings
import numpy as np
from datetime import datetime, timedelta
from math import ceil
from typing import Optional

from scipy.interpolate import interp1d

from .fvcom_writer import FVCOMWriter
from .version import full_version
from .grid import Grid, connectivity
from .exceptions import PyFVCOM2ValueError


__all__ = ["RiverManager"]


_ERSEM_ATTRIBUTES: dict[str, dict] = {
    'N1_p':     {'long_name': 'phosphate phosphorus',                          'units': 'mmol P/m^3'},
    'N3_n':     {'long_name': 'nitrate nitrogen',                              'units': 'mmol N/m^3'},
    'N4_n':     {'long_name': 'ammonium nitrogen',                             'units': 'mmol N/m^3'},
    'N5_s':     {'long_name': 'silicate silicate',                             'units': 'mmol Si/m^3'},
    'O2_o':     {'long_name': 'dissolved Oxygen',                              'units': 'mmol O_2/m^3'},
    'O3_TA':    {'long_name': 'carbonate total alkalinity',                    'units': 'mmol C/m^3'},
    'O3_c':     {'long_name': 'carbonate total dissolved inorganic carbon',    'units': 'mmol C/m^3'},
    'O3_bioalk':{'long_name': 'carbonate bioalkalinity',                       'units': 'umol/kg'},
    'Z4_c':     {'long_name': 'mesozooplankton carbon',                        'units': 'mg C/m^3'},
    'Z5_n':     {'long_name': 'microzooplankton nitrogen',                     'units': 'mmol N/m^3'},
    'Z5_c':     {'long_name': 'microzooplankton carbon',                       'units': 'mg C/m^3'},
    'Z5_p':     {'long_name': 'microzooplankton phosphorus',                   'units': 'mmol P/m^3'},
    'Z6_n':     {'long_name': 'nanoflagellates nitrogen',                      'units': 'mmol N/m^3'},
    'Z6_c':     {'long_name': 'nanoflagellates carbon',                        'units': 'mg C/m^3'},
    'Z6_p':     {'long_name': 'nanoflagellates phosphorus',                    'units': 'mmol P/m^3'},
}

_ERSEM_ZOOPLANKTON_DEFAULTS: dict[str, float] = {
    'Z4_c': 1.2e-6,
    'Z5_c': 7.2e-6,   'Z5_n': 0.12e-6,   'Z5_p': 0.0113e-6,
    'Z6_c': 2.4e-6,   'Z6_n': 0.0505e-6, 'Z6_p': 0.0047e-6,
}

_EPOCH = datetime(1970, 1, 1)


def _haversine_km(
    lon1: float,
    lat1: float,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    """Compute haversine distances in km from one point to an array of points.

    Args:
        lon1: Longitude of the source point in decimal degrees.
        lat1: Latitude of the source point in decimal degrees.
        lon2: Target longitudes in decimal degrees.
        lat2: Target latitudes in decimal degrees.

    Returns:
        Great-circle distances in km, same shape as lon2/lat2.
    """
    R = 6371.0
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    dlon = np.radians(lon2 - lon1)
    dlat = np.radians(lat2 - lat1)
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _datetimes_to_seconds(times: list[datetime]) -> np.ndarray:
    """Convert a list of datetimes to seconds since _EPOCH as float64."""
    return np.array([(t - _EPOCH).total_seconds() for t in times], dtype=np.float64)


class RiverManager:
    """Manager for FVCOM river freshwater forcing.

    Accepts raw river time series data at geographic positions, assigns each
    river to the nearest coastline node or element, merges rivers that share a
    grid location, splits rivers whose peak flux exceeds a threshold, then
    writes the FVCOM-format NetCDF forcing file and river namelist.

    Typical usage::

        mgr = RiverManager(grid, placement='node', flux_threshold=50.0)
        mgr.add_river_data(lon, lat, names, times, flux, temperature)
        mgr.create_forcing_file('river.nc', start, end, timedelta(days=1))
        mgr.create_namelist_file('river.nml', 'river.nc')

    Attributes:
        grid: The Grid instance used for coastal geometry.
    """

    def __init__(
        self,
        grid: Grid,
        placement: str = 'node',
        distance_threshold: float = 1.0,
        flux_threshold: float = 50.0,
    ) -> None:
        """Initialise the RiverManager.

        Args:
            grid: Grid instance providing mesh geometry and open boundary
                definitions.
            placement: Whether to place rivers on mesh nodes (``'node'``) or
                element centroids (``'element'``). Defaults to ``'node'``.
            distance_threshold: Maximum distance in km from a river position to
                the nearest coastline location. Rivers farther than this are
                dropped with a warning. Defaults to 1.0 km.
            flux_threshold: Maximum permitted flux (m³ s⁻¹) at any single grid
                location at any time step. Rivers that exceed this are split
                across adjacent coastline locations. Defaults to 50.0 m³ s⁻¹.

        Raises:
            PyFVCOM2ValueError: If ``placement`` is not ``'node'`` or
                ``'element'``.
        """
        if placement not in ('node', 'element'):
            raise PyFVCOM2ValueError(
                f"placement must be 'node' or 'element', got '{placement}'."
            )
        self._grid = grid
        self._placement = placement
        self._distance_threshold = distance_threshold
        self._flux_threshold = flux_threshold

        # Cached coastline geometry (computed lazily on first add_river_data call).
        self._coastline_indices: Optional[np.ndarray] = None  # node or element indices
        self._coastline_lons: Optional[np.ndarray] = None
        self._coastline_lats: Optional[np.ndarray] = None

        # Processed river data (populated by add_river_data).
        self._names: list[str] = []
        self._grid_locations: list[int] = []  # 0-indexed node or element indices
        self._times: list[datetime] = []
        self._flux: Optional[np.ndarray] = None         # (n_times, n_rivers)
        self._temperature: Optional[np.ndarray] = None  # (n_times, n_rivers)
        self._salinity: Optional[np.ndarray] = None     # (n_times, n_rivers)
        self._ersem: Optional[dict[str, np.ndarray]] = None
        self._other_variables: Optional[dict] = None

    @property
    def n_rivers(self) -> int:
        """Number of rivers after processing.

        Returns:
            Count of rivers, or 0 if add_river_data has not been called.
        """
        return len(self._names)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_river_data(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        names: list[str],
        times: list[datetime],
        flux: np.ndarray,
        temperature: np.ndarray,
        salinity: Optional[np.ndarray] = None,
        ersem: Optional[dict[str, np.ndarray]] = None,
        other_variables: Optional[dict] = None,
    ) -> None:
        """Process and store river forcing data.

        Assigns each river to the nearest coastline grid location, merges any
        rivers that land on the same location, and splits rivers whose peak
        flux exceeds ``flux_threshold``.  Negative flux values remaining after
        all processing are clamped to zero with a warning.

        Calling this method a second time replaces all previously stored data.

        Args:
            lon: Longitude of each river mouth in decimal degrees,
                shape ``(n_rivers,)``.
            lat: Latitude of each river mouth in decimal degrees,
                shape ``(n_rivers,)``.
            names: Name for each river, length ``n_rivers``.
            times: Datetime objects for the input time axis, length
                ``n_times``.
            flux: River discharge in m³ s⁻¹, shape ``(n_times, n_rivers)``.
            temperature: River temperature in °C, shape
                ``(n_times, n_rivers)``.
            salinity: River salinity in PSU, shape ``(n_times, n_rivers)``.
                Defaults to zero for all rivers and times if not supplied.
            ersem: ERSEM biogeochemical variables.  Each key is a variable
                name (e.g. ``'N1_p'``) and each value is an array of shape
                ``(n_times, n_rivers)``.  Zooplankton defaults from WCO L4
                initial conditions are auto-populated for missing Z4/Z5/Z6
                keys.
            other_variables: Additional variables to write to the forcing
                file.  Each key is a variable name and each value is a dict
                with keys ``'data'`` (array of shape ``(n_times, n_rivers)``)
                and ``'attributes'`` (dict of netCDF attributes).

        Raises:
            PyFVCOM2ValueError: If array shapes are inconsistent.
        """
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        names = list(names)
        flux = np.asarray(flux, dtype=float)
        temperature = np.asarray(temperature, dtype=float)

        n_rivers_in = len(lon)
        n_times = len(times)

        # --- 1. Validate shapes ---
        if len(lat) != n_rivers_in:
            raise PyFVCOM2ValueError(
                f"lat length ({len(lat)}) does not match lon length ({n_rivers_in})."
            )
        if len(names) != n_rivers_in:
            raise PyFVCOM2ValueError(
                f"names length ({len(names)}) does not match lon length ({n_rivers_in})."
            )
        if flux.shape != (n_times, n_rivers_in):
            raise PyFVCOM2ValueError(
                f"flux shape {flux.shape} expected ({n_times}, {n_rivers_in})."
            )
        if temperature.shape != (n_times, n_rivers_in):
            raise PyFVCOM2ValueError(
                f"temperature shape {temperature.shape} expected ({n_times}, {n_rivers_in})."
            )

        # --- 2. Default salinity ---
        if salinity is None:
            salinity = np.zeros((n_times, n_rivers_in), dtype=float)
        else:
            salinity = np.asarray(salinity, dtype=float)
            if salinity.shape != (n_times, n_rivers_in):
                raise PyFVCOM2ValueError(
                    f"salinity shape {salinity.shape} expected ({n_times}, {n_rivers_in})."
                )

        if ersem is not None:
            for var, arr in ersem.items():
                arr = np.asarray(arr, dtype=float)
                if arr.shape != (n_times, n_rivers_in):
                    raise PyFVCOM2ValueError(
                        f"ersem['{var}'] shape {arr.shape} expected ({n_times}, {n_rivers_in})."
                    )
                ersem[var] = arr

        if other_variables is not None:
            for var, spec in other_variables.items():
                if 'data' not in spec or 'attributes' not in spec:
                    raise PyFVCOM2ValueError(
                        f"other_variables['{var}'] must have 'data' and 'attributes' keys."
                    )
                arr = np.asarray(spec['data'], dtype=float)
                if arr.shape != (n_times, n_rivers_in):
                    raise PyFVCOM2ValueError(
                        f"other_variables['{var}']['data'] shape {arr.shape} "
                        f"expected ({n_times}, {n_rivers_in})."
                    )
                other_variables[var]['data'] = arr

        # --- 3. Ensure coastline is computed ---
        self._compute_coastline()

        # --- 4. Distance filter and nearest-location assignment ---
        keep = []
        nearest_coast_idx = []  # index into self._coastline_indices

        for i in range(n_rivers_in):
            dists = _haversine_km(lon[i], lat[i], self._coastline_lons, self._coastline_lats)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= self._distance_threshold:
                keep.append(i)
                nearest_coast_idx.append(min_idx)

        n_removed = n_rivers_in - len(keep)
        if n_removed > 0:
            warnings.warn(
                f"{n_removed} river(s) removed: nearest coastline location "
                f"exceeds {self._distance_threshold} km threshold.",
                stacklevel=2,
            )
        if not keep:
            warnings.warn("No rivers remain after distance filtering.", stacklevel=2)
            self._names = []
            self._grid_locations = []
            self._times = list(times)
            self._flux = np.empty((n_times, 0))
            self._temperature = np.empty((n_times, 0))
            self._salinity = np.empty((n_times, 0))
            self._ersem = {} if ersem is not None else None
            self._other_variables = other_variables
            return

        keep = np.asarray(keep, dtype=int)
        nearest_coast_idx = np.asarray(nearest_coast_idx, dtype=int)
        grid_locs = self._coastline_indices[nearest_coast_idx]  # 0-indexed node/elem indices

        flux = flux[:, keep]
        temperature = temperature[:, keep]
        salinity = salinity[:, keep]
        names_keep = [names[i] for i in keep]
        if ersem is not None:
            ersem = {k: v[:, keep] for k, v in ersem.items()}
        if other_variables is not None:
            other_variables = {k: {'data': v['data'][:, keep], 'attributes': v['attributes']}
                               for k, v in other_variables.items()}

        # --- 5. Merge coincident rivers ---
        unique_locs, inv = np.unique(grid_locs, return_inverse=True)
        n_unique = len(unique_locs)

        merged_flux = np.zeros((n_times, n_unique), dtype=float)
        merged_temp = np.zeros((n_times, n_unique), dtype=float)
        merged_salt = np.zeros((n_times, n_unique), dtype=float)
        merged_names: list[str] = []
        merged_ersem: dict[str, np.ndarray] = {}
        merged_other: dict = {}

        if ersem is not None:
            for var in ersem:
                merged_ersem[var] = np.zeros((n_times, n_unique), dtype=float)
        if other_variables is not None:
            for var in other_variables:
                merged_other[var] = {
                    'data': np.zeros((n_times, n_unique), dtype=float),
                    'attributes': other_variables[var]['attributes'],
                }

        for ui in range(n_unique):
            same = np.where(inv == ui)[0]
            total_flux = flux[:, same].sum(axis=1)          # (n_times,)
            merged_flux[:, ui] = total_flux
            safe_total = np.where(total_flux == 0.0, 1.0, total_flux)
            w = flux[:, same] / safe_total[:, np.newaxis]   # (n_times, n_in_group)
            merged_temp[:, ui] = (temperature[:, same] * w).sum(axis=1)
            merged_salt[:, ui] = (salinity[:, same] * w).sum(axis=1)
            merged_names.append('_'.join(names_keep[j] for j in same))
            if ersem is not None:
                for var in ersem:
                    merged_ersem[var][:, ui] = (ersem[var][:, same] * w).sum(axis=1)
            if other_variables is not None:
                for var in other_variables:
                    merged_other[var]['data'][:, ui] = (
                        other_variables[var]['data'][:, same] * w
                    ).sum(axis=1)

        # Auto-populate missing ERSEM zooplankton defaults.
        if ersem is not None:
            for var, default_val in _ERSEM_ZOOPLANKTON_DEFAULTS.items():
                if var not in merged_ersem:
                    merged_ersem[var] = np.full((n_times, n_unique), default_val, dtype=float)

        # --- 6. Split large-flux rivers ---
        # Work with lists of columns so we can append new rivers.
        split_names: list[str] = list(merged_names)
        split_locs: list[int] = list(unique_locs.tolist())
        flux_cols: list[np.ndarray] = [merged_flux[:, i].copy() for i in range(n_unique)]
        temp_cols: list[np.ndarray] = [merged_temp[:, i].copy() for i in range(n_unique)]
        salt_cols: list[np.ndarray] = [merged_salt[:, i].copy() for i in range(n_unique)]
        ersem_cols: dict[str, list[np.ndarray]] = {}
        other_cols: dict[str, list[np.ndarray]] = {}

        if ersem is not None:
            for var in merged_ersem:
                ersem_cols[var] = [merged_ersem[var][:, i].copy() for i in range(n_unique)]
        other_attrs: dict[str, dict] = {}
        if other_variables is not None:
            for var in merged_other:
                other_cols[var] = [merged_other[var]['data'][:, i].copy() for i in range(n_unique)]
                other_attrs[var] = merged_other[var]['attributes']

        occupied: set[int] = set(split_locs)

        i = 0
        while i < len(flux_cols):
            col = flux_cols[i]
            max_flux = float(col.max())
            if max_flux > self._flux_threshold:
                n_splits = int(ceil(max_flux / self._flux_threshold))
                orig_name = split_names[i]
                orig_loc = split_locs[i]
                each_col = col / n_splits

                split_names[i] = f'{orig_name}_1'
                flux_cols[i] = each_col

                new_locs = self._find_free_locations(orig_loc, n_splits - 1, occupied)

                for j, new_loc in enumerate(new_locs, 2):
                    occupied.add(new_loc)
                    split_names.append(f'{orig_name}_{j}')
                    split_locs.append(new_loc)
                    flux_cols.append(each_col.copy())
                    temp_cols.append(temp_cols[i].copy())
                    salt_cols.append(salt_cols[i].copy())
                    for var in ersem_cols:
                        ersem_cols[var].append(ersem_cols[var][i].copy())
                    for var in other_cols:
                        other_cols[var].append(other_cols[var][i].copy())
            i += 1

        # --- 7. Negative flux check ---
        all_flux = np.column_stack(flux_cols) if flux_cols else np.empty((n_times, 0))
        if np.any(all_flux < 0.0):
            warnings.warn(
                "Negative river flux values detected. Setting negative values to zero.",
                stacklevel=2,
            )
            all_flux = np.maximum(all_flux, 0.0)

        # --- 8. Store processed results ---
        self._names = split_names
        self._grid_locations = split_locs
        self._times = list(times)
        self._flux = all_flux
        self._temperature = (
            np.column_stack(temp_cols) if temp_cols else np.empty((n_times, 0))
        )
        self._salinity = (
            np.column_stack(salt_cols) if salt_cols else np.empty((n_times, 0))
        )

        if ersem is not None:
            self._ersem = {
                var: (
                    np.column_stack(ersem_cols[var])
                    if ersem_cols.get(var)
                    else np.empty((n_times, 0))
                )
                for var in ersem_cols
            }
        else:
            self._ersem = None

        if other_variables is not None:
            self._other_variables = {
                var: {
                    'data': (
                        np.column_stack(other_cols[var])
                        if other_cols.get(var)
                        else np.empty((n_times, 0))
                    ),
                    'attributes': other_attrs[var],
                }
                for var in other_cols
            }
        else:
            self._other_variables = None

    def create_forcing_file(
        self,
        output_path: str,
        start_date: datetime,
        end_date: datetime,
        interval: timedelta,
        format: str = 'NETCDF4',
        **kwargs,
    ) -> None:
        """Write the FVCOM river forcing NetCDF file.

        River time series are linearly interpolated from the input times
        (supplied via :meth:`add_river_data`) to the requested output grid.
        The output grid must be covered by the input data range; requesting
        dates outside that range raises an error.

        Args:
            output_path: Path to write the NetCDF file.
            start_date: First output datetime.
            end_date: Last output datetime (inclusive).
            interval: Time step between output records.
            format: NetCDF format string passed to FVCOMWriter. Defaults to
                ``'NETCDF4'``.
            **kwargs: Additional keyword arguments forwarded to FVCOMWriter.
                Pass ``ncopts`` (dict) to control per-variable compression.

        Raises:
            PyFVCOM2ValueError: If :meth:`add_river_data` has not been called,
                or if the output time range is not fully covered by the input
                data.
        """
        if self._flux is None:
            raise PyFVCOM2ValueError(
                "No river data available. Call add_river_data first."
            )

        n_steps = int((end_date - start_date) / interval) + 1
        out_times = [start_date + i * interval for i in range(n_steps)]

        t_in_min = min(self._times)
        t_in_max = max(self._times)
        if out_times[0] < t_in_min or out_times[-1] > t_in_max:
            raise PyFVCOM2ValueError(
                f"Requested output range [{out_times[0]}, {out_times[-1]}] is not "
                f"fully covered by input data range [{t_in_min}, {t_in_max}]."
            )

        ncopts = kwargs.pop('ncopts', {})

        t_in_sec = _datetimes_to_seconds(self._times)
        t_out_sec = _datetimes_to_seconds(out_times)

        n_rivers = self.n_rivers

        def _interp(data: np.ndarray) -> np.ndarray:
            if n_rivers == 0:
                return np.empty((len(out_times), 0))
            f = interp1d(t_in_sec, data, kind='linear', axis=0, bounds_error=True)
            return f(t_out_sec)

        flux_out = _interp(self._flux)
        temp_out = _interp(self._temperature)
        salt_out = _interp(self._salinity)

        dims = {
            'namelen': 80,
            'rivers': n_rivers,
            'time': 0,
            'DateStrLen': 26,
        }
        global_attributes = {
            'type': 'FVCOM RIVER FORCING FILE',
            'title': '',
            'history': f'File created using PyFVCOM2 version {full_version}',
            'filename': os.path.basename(output_path),
            'Conventions': 'CF-1.0',
        }

        with FVCOMWriter(
            str(output_path), dims,
            global_attributes=global_attributes,
            clobber=True, format=format, **kwargs
        ) as ncfile:

            name_chars = np.zeros((n_rivers, 80), dtype='S1')
            for i, name in enumerate(self._names):
                b = name[:80].encode('ascii', errors='replace').ljust(80)[:80]
                name_chars[i] = np.frombuffer(b, dtype='S1')
            ncfile.add_variable(
                'river_names', name_chars, ['rivers', 'namelen'],
                attributes={'long_name': 'river names'}, format='c',
            )

            ncfile.write_fvcom_time(out_times, ncopts=ncopts)

            atts = {'long_name': 'river runoff flux', 'units': 'm^3 s^{-1}'}
            ncfile.add_variable('river_flux', flux_out, ['time', 'rivers'],
                                attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'river runoff temperature', 'units': 'Celsius'}
            ncfile.add_variable('river_temp', temp_out, ['time', 'rivers'],
                                attributes=atts, ncopts=ncopts)

            atts = {'long_name': 'river runoff salinity', 'units': 'PSU'}
            ncfile.add_variable('river_salt', salt_out, ['time', 'rivers'],
                                attributes=atts, ncopts=ncopts)

            if self._ersem is not None:
                for var, data in self._ersem.items():
                    ersem_out = _interp(data)
                    atts = _ERSEM_ATTRIBUTES.get(var, {'long_name': var})
                    ncfile.add_variable(var, ersem_out, ['time', 'rivers'],
                                        attributes=atts, ncopts=ncopts)

            if self._other_variables is not None:
                for var, spec in self._other_variables.items():
                    other_out = _interp(spec['data'])
                    ncfile.add_variable(var, other_out, ['time', 'rivers'],
                                        attributes=spec['attributes'], ncopts=ncopts)

    def create_namelist_file(
        self,
        output_path: str,
        forcing_file: str,
        vertical_distribution: str | np.ndarray = 'uniform',
    ) -> None:
        """Write the FVCOM river namelist file.

        Writes one ``&NML_RIVER`` block per river.  Grid location indices are
        written as 1-based integers as required by FVCOM.

        Args:
            output_path: Path to write the namelist file.
            forcing_file: Path to the river forcing NetCDF file as it will
                appear inside the namelist (typically a relative path used by
                FVCOM at run time).
            vertical_distribution: Vertical distribution of river inflow.
                Pass ``'uniform'`` (default) to compute equal weights from the
                grid sigma layers, an ``np.ndarray`` of weights (one per sigma
                layer, must sum to 1), or a pre-formatted space-separated
                string.

        Raises:
            PyFVCOM2ValueError: If :meth:`add_river_data` has not been called.
        """
        if self._flux is None:
            raise PyFVCOM2ValueError(
                "No river data available. Call add_river_data first."
            )

        if isinstance(vertical_distribution, np.ndarray):
            vdist_str = ' '.join(f'{v:.6f}' for v in vertical_distribution)
        elif vertical_distribution == 'uniform':
            n_layers = self._grid.n_sigma_layers
            frac = 1.0 / n_layers
            vdist_str = ' '.join(f'{frac:.6f}' for _ in range(n_layers))
        else:
            vdist_str = str(vertical_distribution)

        with open(output_path, 'w') as f:
            for name, loc in zip(self._names, self._grid_locations):
                f.write('&NML_RIVER\n')
                f.write(f" RIVER_NAME = '{name}',\n")
                f.write(f" RIVER_FILE = '{forcing_file}',\n")
                f.write(f" RIVER_GRID_LOCATION = {loc + 1},\n")
                f.write(f" RIVER_VERTICAL_DISTRIBUTION = '{vdist_str}'\n")
                f.write('/\n')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_coastline(self) -> None:
        """Compute and cache coastline node/element indices.

        Called lazily by :meth:`add_river_data` on first invocation.
        """
        if self._coastline_indices is not None:
            return

        p = np.column_stack([self._grid.lon_nodes, self._grid.lat_nodes])
        _, _, e2t, bnd = connectivity(p, self._grid.triangles)

        obc_node_set: set[int] = set()
        for ob in self._grid.open_boundaries:
            obc_node_set.update(ob.node_indices.tolist())

        # Coastline nodes: boundary nodes that are not OBC nodes.
        all_bnd_nodes = np.where(bnd)[0]
        is_obc = np.isin(all_bnd_nodes, list(obc_node_set)) if obc_node_set else np.zeros(len(all_bnd_nodes), dtype=bool)
        coastline_nodes = all_bnd_nodes[~is_obc]

        if self._placement == 'node':
            self._coastline_indices = coastline_nodes
            self._coastline_lons = self._grid.lon_nodes[coastline_nodes]
            self._coastline_lats = self._grid.lat_nodes[coastline_nodes]
        else:
            # Coastline elements: boundary elements with no OBC nodes.
            bnd_edge_mask = e2t[:, 1] == -1
            all_bnd_elems = np.unique(e2t[bnd_edge_mask, 0])
            if obc_node_set:
                bnd_elem_nodes = self._grid.triangles[all_bnd_elems]  # (n, 3)
                has_obc = np.isin(bnd_elem_nodes, list(obc_node_set)).any(axis=1)
                self._coastline_indices = all_bnd_elems[~has_obc]
            else:
                self._coastline_indices = all_bnd_elems
            self._coastline_lons = self._grid.lon_elements[self._coastline_indices]
            self._coastline_lats = self._grid.lat_elements[self._coastline_indices]

    def _find_free_locations(
        self,
        orig_loc: int,
        n_needed: int,
        occupied: set[int],
    ) -> list[int]:
        """Find ``n_needed`` free coastline locations near ``orig_loc``.

        Locations are returned in order of increasing distance from the
        original river location.

        Args:
            orig_loc: 0-indexed node or element index of the original river.
            n_needed: Number of additional free locations required.
            occupied: Set of already-occupied location indices; updated as
                new locations are selected so callers see the latest state.

        Returns:
            List of n_needed free coastline location indices (0-indexed).

        Raises:
            PyFVCOM2ValueError: If fewer than ``n_needed`` free locations can
                be found on the coastline.
        """
        if self._placement == 'node':
            orig_lon = self._grid.lon_nodes[orig_loc]
            orig_lat = self._grid.lat_nodes[orig_loc]
        else:
            orig_lon = self._grid.lon_elements[orig_loc]
            orig_lat = self._grid.lat_elements[orig_loc]

        dists = _haversine_km(orig_lon, orig_lat, self._coastline_lons, self._coastline_lats)
        sorted_idx = np.argsort(dists)

        result: list[int] = []
        for idx in sorted_idx:
            candidate = int(self._coastline_indices[idx])
            if candidate not in occupied:
                result.append(candidate)
                occupied.add(candidate)
                if len(result) == n_needed:
                    break

        if len(result) < n_needed:
            raise PyFVCOM2ValueError(
                f"Cannot find {n_needed} free coastline locations near location "
                f"{orig_loc}; only {len(result)} available on the coastline."
            )
        return result

def read_river_config(file_name,  zeroindex=True):
    """
    Parse an FVCOM river namelist file to extract the parameters and their values. Returns a dict of the parameters
    with the associated values for all the rivers defined in the namelist.

    Parameters
    ----------
    file_name : str
        Full path to an FVCOM Rivers name list.
    zeroindex : bool, optional
        Set to False to keep indices as 1-based rather than converting to 0-based. Defaults to True (i.e. return
        zero-indexed indices).

    Returns
    -------
    rivers : dict
        Dict of the parameters for each river defined in the name list.
        Dictionary keys are the name list parameter names (e.g. RIVER_NAME).

    """

    rivers = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            if line and not line.startswith('&') and not line.startswith('/'):
                param, value = [i.strip(",' ") for i in line.split('=')]
                if param in rivers:
                    rivers[param].append(value)
                else:
                    rivers[param] = [value]

        print('Found {} rivers.'.format(len(rivers['RIVER_NAME'])))

    if zeroindex and 'RIVER_GRID_LOCATION' in rivers:
        rivers['RIVER_GRID_LOCATION'] = [int(i) - 1 for i in rivers['RIVER_GRID_LOCATION']]

    return rivers

