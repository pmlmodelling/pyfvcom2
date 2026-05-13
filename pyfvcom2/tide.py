from pyfvcom2.exceptions import PyFVCOM2ValueError
import pandas
import numpy as np
from typing import Optional
from netCDF4 import date2num
from datetime import datetime, timedelta
from utide import solve as utide_solve, reconstruct, ut_constants
from utide.utilities import Bunch
import multiprocessing

from .interpolation_coordinates import InterpolationCoordinates
from .tide_reader import HarmonicsData
from .fvcom_writer import FVCOMWriter
from .fvcom_reader import FVCOMReader


# Modified Julian Day zero point
MJD_ZERO_POINT = "1858-11-17"

# Default tidal constituents: eight primaries + three shallow-water.
DEFAULT_CONSTIT = ('M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'M4', 'MS4', 'MN4')


class TideManager:
    """Manages tidal harmonics interpolation and prediction.

    Orchestrates the pipeline from TPXO tidal harmonic data through
    interpolation onto target positions to tidal time series prediction
    via UTide.

    Typical usage:
        1. Create a TideManager with the desired constituents.
        2. Register TPXOInterpolator instances for each variable (zeta, u, v)
           via add_interpolator.
        3. Pass the TideManager to NestManager.add_tidal_data(), which calls
           predict() for each variable with the appropriate target positions.

    Args:
        constituents: List of tidal constituent names (e.g. ['M2', 'S2']).
        parallel: Whether to run UTide predictions in parallel. Default True.
        pool_size: Number of parallel processes. Default None (use all CPUs).
    """

    def __init__(self, constituents: list[str], parallel: bool = True,
                 pool_size: Optional[int] = None):
        self._constituents = constituents
        self._parallel = parallel
        self._pool_size = pool_size
        self._interpolators = {}

    @property
    def constituents(self) -> list[str]:
        """List of tidal constituent names."""
        return self._constituents

    def add_interpolator(self, variable: str, interpolator) -> None:
        """Register a TPXOInterpolator for a tidal variable.

        Args:
            variable: Tidal variable name ('zeta', 'u', or 'v').
            interpolator: TPXOInterpolator loaded with harmonics data
                for this variable.
        """
        self._interpolators[variable] = interpolator

    def predict(self, variable: str, datetimes: np.ndarray,
                longitudes: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
        """Interpolate harmonics onto target positions and predict tides.

        Args:
            variable: Variable name ('zeta', 'u', or 'v').
            datetimes: Array of datetime objects for prediction times.
            longitudes: Target longitude positions.
            latitudes: Target latitude positions.

        Returns:
            Predicted tidal time series, shape (n_times, n_points).
        """
        if variable not in self._interpolators:
            raise PyFVCOM2ValueError(
                f"No interpolator registered for '{variable}'. "
                f"Call add_interpolator first."
            )

        interpolated = self._interpolate_harmonics(variable, longitudes, latitudes)

        results = predict_tide(
            datetimes, self._constituents,
            interpolated.amplitudes, interpolated.phases,
            latitudes, self._parallel, self._pool_size
        )

        return np.asarray(results).T  # (n_times, n_points)

    def _interpolate_harmonics(self, variable: str, longitudes: np.ndarray,
                               latitudes: np.ndarray) -> HarmonicsData:
        """Interpolate harmonics onto target positions.

        Args:
            variable: Variable name.
            longitudes: Target longitudes.
            latitudes: Target latitudes.

        Returns:
            HarmonicsData with amplitudes and phases interpolated to targets.
        """
        interpolator = self._interpolators[variable]

        # TPXOInterpolator.interpolate only uses x1 (lon) and x2 (lat)
        coords = InterpolationCoordinates(
            dates=np.array([]),
            x3=np.empty((0, len(longitudes))),
            x2=latitudes,
            x1=longitudes,
            horizontal_coordinate_system='geographic',
            vertical_coordinate_system='z',
        )
        return interpolator.interpolate(coords)


def predict_tide(
    datetimes: np.ndarray,
    constituents: list[str],
    amplitudes: np.ndarray,
    phases: np.ndarray,
    latitudes: np.ndarray,
    parallel: bool = True,
    pool_size: Optional[int] = None,
):
    """
    Reconstruct tidal variations in zeta, u, v etc at the provided datetimes and latitudes
    using the provided tidal constituent amplitudes and phases.

    Args:
    datetimes : np.ndarray
        Array of datetime objects for prediction times.
    interval : float
        Time interval between datetimes in days.
    constituents : list[str]
        List of tidal constituent names to read.
    amplitudes : np.ndarray
        Amplitude of the relevant constituents shaped [nlocs, nconst].
    phases : np.ndarray
        Array of the phase of the relevant constituents shaped [nlocs, nconst].
    latitudes : np.ndarray
        Latitudes of the positions to predict.
    parallel : bool, optional
        Whether to run the predictions in parallel using multiprocessing. Default is True.
    pool_size : int, optional
        Number of parallel processes to use. If 1, runs serially. Default is
        1.
    Returns:
    results : list[np.ndarray]
        List of predicted zeta time series arrays for each location.
    """
    const_indices = np.asarray(
        [ut_constants["const"]["name"].tolist().index(i) for i in constituents]
    )
    frq = ut_constants["const"]["freq"][const_indices]

    # Extend the provided datetimes by +/- 1 day to avoid edge effects in UTide (TBC). Maintain an
    # appropriate time interval for the predictions.
    extended_datetimes = extend_datetime_array(datetimes, extension_length_in_days=1.0)

    # Convert extended datetimes to datenums
    extended_times = date2num(extended_datetimes, units=f"days since {MJD_ZERO_POINT} 00:00:00")

    # Take reference time to be the median time in the input times
    ref_time = np.median(extended_times)

    coef = Bunch(name=constituents, mean=0, slope=0)
    coef["aux"] = Bunch(reftime=ref_time, lind=const_indices, frq=frq)
    coef["aux"]["opt"] = Bunch(
        twodim=False,
        nodsatlint=False,
        nodsatnone=False,
        gwchlint=False,
        gwchnone=False,
        notrend=True,
        prefilt=[],
        nodiagn=True,
    )

    # Prepare the time data for predicting the time series.
    # UTide needs netCDF date2num times.
    args = [
        (latitudes[i], extended_times, coef, amplitudes[i], phases[i])
        for i in range(len(latitudes))
    ]
    if parallel is False or pool_size == 1:
        results = []
        for arg in args:
            results.append(reconstruct_wrapper(arg))
    else:
        pool = multiprocessing.Pool(pool_size)
        results = pool.map(reconstruct_wrapper, args)
        pool.close()

    # Remove the extended time predictions to return only those for the original datetimes
    mask = (extended_datetimes >= datetimes[0]) & (extended_datetimes <= datetimes[-1])
    results = [r[mask] for r in results]

    return results


def reconstruct_wrapper(args: tuple) -> np.ndarray:
    """
    For the given time and coefficients (in coef) reconstruct the tidal elevation or current component time
    series at the given latitude.

    Args:
    args : tuple
        Tuple of (lats, times, coef, amplitudes, phases) where:
        - lats: Latitude of the position to predict.
        - times: Array of datenums (days since MJD zero point).
        - coef: UTide coefficients Bunch.
        - amplitudes: Amplitude of the relevant constituents shaped [nconst].
        - phases: Phase of the relevant constituents shaped [nconst].

    Returns:
    zeta : np.ndarray
        Time series of surface elevations.

    Notes
    -----
    Uses utide.reconstruct() for the predicted tide. Accepts a single tuple
    argument for compatibility with multiprocessing.Pool.map.

    """
    lats, times, coef, amplitudes, phases = args
    coef["aux"]["lat"] = lats
    coef["A"] = amplitudes
    coef["g"] = phases
    coef["A_ci"] = np.zeros(amplitudes.shape)
    coef["g_ci"] = np.zeros(phases.shape)
    pred = reconstruct(times, coef, epoch=f"{MJD_ZERO_POINT}")
    zeta = pred["h"]

    return zeta


def extend_datetime_array(datetimes: np.ndarray, extension_length_in_days: float=1.0) -> np.ndarray:
    """ Extend a datetime array by a given number of days at both ends.

    TODO This was in the original PyFVCOM, although in a different form. Unsure whether it is actually
    needed for UTide. A simpler approach would be to just add one day at each end, but this might
    create unevenly spaced time arrays. Would this matter? Need to test it.

    Args:
        datetimes (np.ndarray): Original array of datetime objects.
        extension_length_in_days (float): Number of days to extend at both ends.

    Returns:
        np.ndarray: Extended array of datetime objects.
    """
    # Calculate the interval between original datetimes. Check the interval is consistent across the array.
    if len(datetimes) < 2:
        raise ValueError("datetimes array must contain at least two elements to determine the interval.")

    # Does the time interval need to be consistent across the array? Copying what was done in PyFVCOM.
    intervals = np.diff(datetimes)
    interval = intervals[0]
    if not np.all(intervals == interval):
        raise PyFVCOM2ValueError("Inconsistent time intervals found.")
    
    # Use the extension length to create new start and end datetimes
    extended_datetimes_start = datetimes[0] - timedelta(days=extension_length_in_days)
    extended_datetimes_end = datetimes[-1] + timedelta(days=extension_length_in_days)

    # Create the extended datetime array with the same interval
    extended_datetimes = pandas.date_range(extended_datetimes_start, extended_datetimes_end,
                                           freq=timedelta(days=interval.days, seconds=interval.seconds))

    return extended_datetimes.to_pydatetime()


class HarmonicAnalysisWriter(FVCOMWriter):
    """Write tidal harmonic analysis results to a NetCDF file.

    Subclasses FVCOMWriter to provide methods for writing the grid, sigma
    coordinates, constituent names, harmonic amplitudes/phases, and optionally
    raw or predicted time series.

    Typical usage::

        dims = {
            'node': nx, 'nele': ne, 'siglay': nz, 'siglev': nzlev,
            'three': 3, 'nconsts': len(consts),
            'NameStrLen': 4, 'DateStrLen': 26,
        }
        global_atts = {'title': 'FVCOM harmonic analysis', ...}
        with HarmonicAnalysisWriter('harmonics.nc', dims, global_atts) as nc:
            nc.add_grid(lon, lat, lonc, latc, h, h_center, nv, siglay, siglev, consts)
            nc.write_fvcom_time(times)   # only needed when writing time series
            nc.add_harmonics('zeta', z_amp, z_phase)
            nc.add_harmonics('ua', ua_amp, ua_phase)
            nc.add_harmonics('u', u_amp, u_phase)

    Notes
    -----
    Dimensions must be passed to __init__ before calling add_grid() or
    add_harmonics(). The 'time' dimension should be set to 0 (unlimited) when
    writing raw or predicted time series, and omitted otherwise.

    The ncopts compression settings default to zlib level 7, matching the
    original PyFVCOM HarmonicOutput behaviour.
    """

    # Depth-resolved variables that require a siglay dimension.
    _SIGLAY_VARS = {'u', 'v'}
    # Variables located at nodes (vs. elements).
    _NODE_VARS = {'zeta'}
    # Default compression options.
    _NCOPTS = {'zlib': True, 'complevel': 7}

    def add_grid(self, lon, lat, lonc, latc, h, h_center, nv, siglay, siglev, consts):
        """Write grid coordinates, bathymetry, connectivity and constituent names.

        Parameters
        ----------
        lon, lat : array-like
            Node longitudes and latitudes.
        lonc, latc : array-like
            Element centre longitudes and latitudes.
        h : array-like
            Node bathymetry (m, positive down).
        h_center : array-like
            Element centre bathymetry (m, positive down).
        nv : array-like
            Node-to-element connectivity table, shape (3, nele).
        siglay : array-like
            Sigma layer coordinates, shape (siglay, node).
        siglev : array-like
            Sigma level coordinates, shape (siglev, node).
        consts : list of str
            Tidal constituent names, padded to 4 characters (e.g. 'M2  ').
        """
        ncopts = self._NCOPTS

        self.add_variable('lon', lon, ['node'],
                          attributes={'units': 'degrees_east',
                                      'long_name': 'nodal longitude',
                                      'standard_name': 'longitude'}, ncopts=ncopts)
        self.add_variable('lat', lat, ['node'],
                          attributes={'units': 'degrees_north',
                                      'long_name': 'nodal latitude',
                                      'standard_name': 'latitude'}, ncopts=ncopts)
        self.add_variable('lonc', lonc, ['nele'],
                          attributes={'units': 'degrees_east',
                                      'long_name': 'element centre longitude',
                                      'standard_name': 'longitude'}, ncopts=ncopts)
        self.add_variable('latc', latc, ['nele'],
                          attributes={'units': 'degrees_north',
                                      'long_name': 'element centre latitude',
                                      'standard_name': 'latitude'}, ncopts=ncopts)
        self.add_variable('h', h, ['node'],
                          attributes={'long_name': 'Bathymetry',
                                      'standard_name': 'sea_floor_depth_below_geoid',
                                      'units': 'm',
                                      'positive': 'down'}, ncopts=ncopts)
        self.add_variable('h_center', h_center, ['nele'],
                          attributes={'long_name': 'Bathymetry',
                                      'standard_name': 'sea_floor_depth_below_geoid',
                                      'units': 'm',
                                      'positive': 'down',
                                      'grid_location': 'center'}, ncopts=ncopts)
        self.add_variable('nv', nv, ['three', 'nele'],
                          attributes={'long_name': 'nodes surrounding element'},
                          format='i', ncopts=ncopts)
        self.add_variable('siglay', siglay, ['siglay', 'node'],
                          attributes={'long_name': 'Sigma Layers',
                                      'standard_name': 'ocean_sigma/general_coordinate',
                                      'positive': 'up',
                                      'valid_min': -1.0,
                                      'valid_max': 0.0,
                                      'formula_terms': 'sigma: siglay eta: zeta depth: h'}, ncopts=ncopts)
        self.add_variable('siglev', siglev, ['siglev', 'node'],
                          attributes={'long_name': 'Sigma Levels',
                                      'standard_name': 'ocean_sigma/general_coordinate',
                                      'positive': 'up',
                                      'valid_min': -1.0,
                                      'valid_max': 0.0,
                                      'formula_terms': 'sigma: siglev eta: zeta depth: h'}, ncopts=ncopts)

        # Constituent names: one variable per field type (z, u, v).
        for prefix in ('z', 'u', 'v'):
            self.add_variable(f'{prefix}_const_names', consts,
                              ['nconsts', 'NameStrLen'],
                              attributes={'long_name': f'Tidal constituent names for {prefix}'},
                              format='c', ncopts=ncopts)

    def add_harmonics(self, variable, amplitude, phase):
        """Write harmonic amplitude and phase for a given variable.

        Parameters
        ----------
        variable : str
            One of 'zeta', 'ua', 'va', 'u', 'v'.
        amplitude : array-like
            Harmonic amplitudes (m or m/s). Shape is (nconsts, node) for
            'zeta', (nconsts, nele) for depth-averaged velocity, or
            (nconsts, siglay, nele) for depth-resolved velocity.
        phase : array-like
            Harmonic phases (degrees), same shape as amplitude.
        """
        ncopts = self._NCOPTS

        if variable == 'zeta':
            dims = ['nconsts', 'node']
            coords = 'lon lat nconsts'
        elif variable in ('ua', 'va'):
            dims = ['nconsts', 'nele']
            coords = 'nconsts lonc latc'
        elif variable in ('u', 'v'):
            dims = ['nconsts', 'siglay', 'nele']
            coords = 'nconsts lonc latc'
        else:
            raise PyFVCOM2ValueError(
                f"Unsupported variable '{variable}'. "
                "Expected one of 'zeta', 'ua', 'va', 'u', 'v'."
            )

        short = variable
        self.add_variable(f'{short}_amp', amplitude, dims,
                          attributes={'long_name': f'Tidal harmonic amplitude of {short}',
                                      'standard_name': f'{short}_amplitude',
                                      'units': 'm' if variable == 'zeta' else 'm s-1',
                                      'coordinates': coords}, ncopts=ncopts)
        self.add_variable(f'{short}_phase', phase, dims,
                          attributes={'long_name': f'Tidal harmonic phase of {short}',
                                      'standard_name': f'{short}_phase',
                                      'units': 'degrees',
                                      'coordinates': coords}, ncopts=ncopts)

    def add_raw(self, variable, data):
        """Write the raw (input) time series used in the harmonic analysis.

        Requires a 'time' dimension to have been created and write_fvcom_time()
        to have been called first.

        Parameters
        ----------
        variable : str
            One of 'zeta', 'ua', 'va', 'u', 'v'.
        data : array-like
            Time series data. Shape is (time, node) for 'zeta', (time, nele)
            for depth-averaged, or (time, siglay, nele) for depth-resolved.
        """
        self._write_timeseries(variable, data, suffix='raw')

    def add_predicted(self, variable, data):
        """Write a predicted time series reconstructed from the harmonics.

        Requires a 'time' dimension to have been created and write_fvcom_time()
        to have been called first.

        Parameters
        ----------
        variable : str
            One of 'zeta', 'ua', 'va', 'u', 'v'.
        data : array-like
            Predicted time series. Same shape conventions as add_raw().
        """
        self._write_timeseries(variable, data, suffix='pred')

    def _write_timeseries(self, variable, data, suffix):
        """Common implementation for add_raw() and add_predicted()."""
        ncopts = self._NCOPTS
        label = 'Modelled' if suffix == 'raw' else 'Predicted'

        if variable == 'zeta':
            dims = ['time', 'node']
            coords = 'time lat lon'
            long_name = f'{label} surface elevation'
        elif variable in ('ua', 'va'):
            dims = ['time', 'nele']
            coords = 'time latc lonc'
            direction = 'Eastward' if variable == 'ua' else 'Northward'
            long_name = f'{label} {direction} depth-averaged velocity'
        elif variable in ('u', 'v'):
            dims = ['time', 'siglay', 'nele']
            coords = 'time latc lonc'
            direction = 'Eastward' if variable == 'u' else 'Northward'
            long_name = f'{label} {direction} velocity'
        else:
            raise PyFVCOM2ValueError(
                f"Unsupported variable '{variable}'. "
                "Expected one of 'zeta', 'ua', 'va', 'u', 'v'."
            )

        units = 'm' if variable == 'zeta' else 'm s-1'
        self.add_variable(f'{variable}_{suffix}', data, dims,
                          attributes={'long_name': long_name,
                                      'units': units,
                                      'coordinates': coords}, ncopts=ncopts)


def analyse_harmonics(
    times: np.ndarray,
    elevations: np.ndarray,
    latitudes: np.ndarray,
    predict: bool = False,
    constit: tuple = DEFAULT_CONSTIT,
    **kwargs,
) -> np.ndarray:
    """Run UTide harmonic analysis on a set of time series.

    Parameters
    ----------
    times : np.ndarray
        Modified Julian Day times, shape (ntimes,).
    elevations : np.ndarray
        Time series data, shape (npositions, ntimes).
    latitudes : np.ndarray
        Latitude for each position, shape (npositions,). Positions with a
        NaN latitude are skipped and left as NaN in the output.
    predict : bool, optional
        If True, also return a reconstructed predicted time series alongside
        the harmonics. Defaults to False.
    constit : tuple, optional
        Tidal constituent names to include in the analysis. Defaults to
        DEFAULT_CONSTIT.
    **kwargs
        Additional keyword arguments forwarded to :func:`utide.solve`. The
        ``verbose`` key is also forwarded to :func:`utide.reconstruct` when
        *predict* is True.

    Returns
    -------
    harmonics : np.ndarray
        Shape ``(npositions, 2, len(constit))``. Index 0 along the second
        axis is phase (degrees), index 1 is amplitude (m or m/s). Positions
        with a NaN latitude are left as NaN.
    predicted : np.ndarray
        Only returned when *predict* is True. Shape ``(npositions, ntimes)``.
        Positions with a NaN latitude are left as NaN.

    Notes
    -----
    UTide returns constituents in an order that may differ from *constit*.
    The output arrays are always re-sorted to match the order given in
    *constit*.
    """
    npositions = len(latitudes)
    harmonics = np.full((npositions, 2, len(constit)), np.nan)
    if predict:
        predicted = np.full((npositions, len(times)), np.nan)

    # Extract verbose for reconstruct; utide.solve also accepts it via **kwargs.
    verbose = kwargs.get('verbose', False)

    for i, (timeseries, lat) in enumerate(zip(elevations, latitudes)):
        if np.isnan(lat):
            continue

        res = utide_solve(t=times, u=timeseries, lat=lat, method='ols',
                          constit=constit, epoch=MJD_ZERO_POINT, **kwargs)

        # UTide returns constituents in a different order; re-sort to match constit.
        c_order = [res['name'].tolist().index(cc) for cc in constit]
        harmonics[i, 0, :] = res['g'][c_order]   # phase
        harmonics[i, 1, :] = res['A'][c_order]   # amplitude

        if predict:
            reconstructed = reconstruct(t=times, coef=res, epoch=MJD_ZERO_POINT,
                                        verbose=verbose)
            predicted[i, :] = reconstructed['h']

    if predict:
        return harmonics, predicted
    return harmonics


class HarmonicsAnalyser:
    """Orchestrate tidal harmonic analysis from FVCOM output.

    Loads data from a :class:`~pyfvcom2.fvcom_reader.FVCOMReader`, runs UTide
    harmonic analysis for each requested variable, and writes results to a
    :class:`HarmonicAnalysisWriter` NetCDF file.

    Parameters
    ----------
    reader : FVCOMReader
        Open reader for the FVCOM model output to analyse.
    output_file : str
        Path to the NetCDF output file to create.
    constit : tuple, optional
        Tidal constituent names. Defaults to :data:`DEFAULT_CONSTIT`.
    predict : bool, optional
        If True, also write a predicted (reconstructed) time series for each
        variable. Defaults to False.
    dump_raw : bool, optional
        If True, also write the raw (input) time series for each variable.
        Defaults to False.

    Examples
    --------
    ::

        reader = FVCOMReader(['output_0001.nc', 'output_0002.nc'])
        analyser = HarmonicsAnalyser(reader, 'harmonics.nc', predict=True)
        analyser.run(['zeta', 'u', 'v'])
    """

    def __init__(
        self,
        reader: FVCOMReader,
        output_file: str,
        constit: tuple = DEFAULT_CONSTIT,
        predict: bool = False,
        dump_raw: bool = False,
    ):
        self._reader = reader
        self._output_file = output_file
        self._constit = constit
        self._predict = predict
        self._dump_raw = dump_raw

        # Pre-compute MJD times once — used by analyse_harmonics and write_fvcom_time.
        self._times = date2num(
            reader.dates, units=f'days since {MJD_ZERO_POINT} 00:00:00'
        )

        # Fixed-width (4-char) constituent name lists for netCDF character array.
        self._cnames = [list(f'{c:4s}') for c in constit]

    def run(self, variables: list = None):
        """Run harmonic analysis for the specified variables.

        Opens the output file, writes the grid and (when applicable) time,
        then runs the analysis variable by variable.

        Parameters
        ----------
        variables : list of str, optional
            FVCOM variable names to analyse. Valid values are ``'zeta'``,
            ``'u'``, ``'v'``, ``'ua'``, ``'va'``. Defaults to all five.
        """
        if variables is None:
            variables = ['zeta', 'u', 'v', 'ua', 'va']

        reader = self._reader

        dims = {
            'node': reader.n_nodes,
            'nele': reader.n_elements,
            'siglay': reader.n_sigma_layers,
            'siglev': reader.n_sigma_levels,
            'three': 3,
            'nconsts': len(self._constit),
            'NameStrLen': 4,
        }
        if self._predict or self._dump_raw:
            dims['time'] = 0          # unlimited
            dims['DateStrLen'] = 26   # ISO-8601 string length for write_fvcom_time

        global_atts = {
            'title': 'FVCOM tidal harmonic analysis',
            'history': 'Created by pyfvcom2 HarmonicsAnalyser',
        }

        # FVCOMReader stores triangles as (nele, 3) 0-based; FVCOM convention
        # for nv is (3, nele) 1-based.
        nv = (reader.grid.triangles + 1).T

        with HarmonicAnalysisWriter(self._output_file, dims, global_atts) as ncout:
            ncout.add_grid(
                lon=reader.lon_nodes,
                lat=reader.lat_nodes,
                lonc=reader.lon_elements,
                latc=reader.lat_elements,
                h=reader.bathy_nodes,
                h_center=reader.bathy_elements,
                nv=nv,
                siglay=reader.sigma_layers_nodes,
                siglev=reader.sigma_levels_nodes,
                consts=self._cnames,
            )

            if self._predict or self._dump_raw:
                ncout.write_fvcom_time(reader.dates)

            for var in variables:
                self._process_variable(var, ncout)

    def _process_variable(self, var: str, ncout: HarmonicAnalysisWriter):
        """Load, analyse, and write harmonics for one variable."""
        reader = self._reader
        is_node = reader.var_is_node_based(var)
        has_depth = reader.get_vertical_position(var) == 'layer_centre'
        latitudes = reader.lat_nodes if is_node else reader.lat_elements
        npositions = reader.n_nodes if is_node else reader.n_elements
        ntimes = len(reader.dates)
        nconsts = len(self._constit)

        if has_depth:
            nz = reader.n_sigma_layers
            amp = np.full((nconsts, nz, npositions), np.nan)
            phase = np.full((nconsts, nz, npositions), np.nan)
            if self._predict:
                pred = np.full((ntimes, nz, npositions), np.nan)
            if self._dump_raw:
                raw = np.full((ntimes, nz, npositions), np.nan)

            for zlev in range(nz):
                # data shape: (ntimes, npositions)
                data = self._load_full_timeseries(var, zlev=zlev)
                if self._dump_raw:
                    raw[:, zlev, :] = data

                # analyse_harmonics expects elevations as (npositions, ntimes)
                result = analyse_harmonics(
                    self._times, data.T, latitudes,
                    predict=self._predict,
                    constit=self._constit,
                )
                if self._predict:
                    harmonics, predicted = result
                    pred[:, zlev, :] = predicted.T   # (ntimes, npositions)
                else:
                    harmonics = result

                amp[:, zlev, :] = harmonics[:, 1, :].T    # (nconsts, npositions)
                phase[:, zlev, :] = harmonics[:, 0, :].T
        else:
            data = self._load_full_timeseries(var)   # (ntimes, npositions)
            if self._dump_raw:
                raw = data.copy()

            result = analyse_harmonics(
                self._times, data.T, latitudes,
                predict=self._predict,
                constit=self._constit,
            )
            if self._predict:
                harmonics, predicted = result
                pred = predicted.T   # (ntimes, npositions)
            else:
                harmonics = result

            amp = harmonics[:, 1, :].T    # (nconsts, npositions)
            phase = harmonics[:, 0, :].T

        ncout.add_harmonics(var, amp, phase)
        if self._dump_raw:
            ncout.add_raw(var, raw)
        if self._predict:
            ncout.add_predicted(var, pred)

    def _load_full_timeseries(self, var: str, zlev: int = None) -> np.ndarray:
        """Load the complete time series for one variable.

        Parameters
        ----------
        var : str
            FVCOM variable name.
        zlev : int, optional
            Sigma layer index (0-based). Required for depth-resolved variables;
            omit for 2-D variables such as ``zeta``, ``ua``, ``va``.

        Returns
        -------
        data : np.ndarray
            Shape ``(ntimes, npositions)``.
        """
        reader = self._reader
        npositions = reader.n_nodes if reader.var_is_node_based(var) else reader.n_elements
        ntimes = len(reader.dates)

        data = np.full((ntimes, npositions), np.nan)
        for t, dt in enumerate(reader.dates):
            # For 2-D vars: get_var returns (npositions,).
            # For 3-D vars: get_var returns (nz, npositions); select zlev.
            raw = reader.get_var(var, target_datetime=dt)
            data[t, :] = raw[zlev, :] if zlev is not None else raw
        return data

