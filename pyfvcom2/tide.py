from pyfvcom2.exceptions import PyFVCOM2ValueError
import pandas
import numpy as np
import sys
import time
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


# ── Private progress-reporting helpers ────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    s = int(seconds)
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m {s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h {m:02d}m'


def _print_progress(label: str, done: int, total: int, t_start: float,
                    _state: list) -> None:
    """Write a rate-limited progress line to stdout.

    Parameters
    ----------
    label : str
        Short description shown on the left (e.g. ``'zeta'`` or ``'u [σ 3/20]'``).
    done : int
        Number of positions completed so far.
    total : int
        Total number of positions to process.
    t_start : float
        Timestamp (from :func:`time.perf_counter`) when processing started.
    _state : list
        Single-element mutable list ``[last_print_time]`` used to rate-limit
        output to at most once per second.  Pass ``[0.0]`` on first call.
    """
    now = time.perf_counter()
    finished = done >= total
    if not finished and (now - _state[0]) < 1.0:
        return
    _state[0] = now

    elapsed = now - t_start
    pct = 100.0 * done / total
    w = len(str(total))   # field width for counts
    if done > 0:
        eta_str = _fmt_time(elapsed * (total - done) / done)
    else:
        eta_str = '?'

    line = (f'  {label:<20s}: {done:{w}d}/{total}  ({pct:5.1f}%)'
            f'  elapsed {_fmt_time(elapsed):>7s}  ETA {eta_str:>7s}')
    if finished:
        print(f'\r{line}', flush=True)
    else:
        print(f'\r{line}', end='', flush=True)


# ─────────────────────────────────────────────────────────────────────────────

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


def _solve_one(args: tuple):
    """Solve UTide harmonics for a single spatial position.

    Defined at module level for :mod:`multiprocessing` pickle compatibility.
    Each argument is packed into a single tuple by :func:`analyse_harmonics`.

    Returns
    -------
    tuple or None
        ``(phase, amplitude, predicted_or_None)`` on success; ``None`` when
        the position has a NaN latitude (output row is left as NaN).
    """
    import warnings
    times, u, lat, constit, epoch, verbose, predict = args
    if np.isnan(lat):
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='utide')
        res = utide_solve(t=times, u=u, lat=lat, method='ols',
                          constit=constit, epoch=epoch, verbose=verbose)
    c_order = [res['name'].tolist().index(cc) for cc in constit]
    g = res['g'][c_order]   # phase
    A = res['A'][c_order]   # amplitude
    h = None
    if predict:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='utide')
            rec = reconstruct(t=times, coef=res, epoch=epoch, verbose=verbose)
        h = rec['h']
    return (g, A, h)


def analyse_harmonics(
    times: np.ndarray,
    elevations: np.ndarray,
    latitudes: np.ndarray,
    predict: bool = False,
    constit: tuple = DEFAULT_CONSTIT,
    pool_size: int = 1,
    label: str = '',
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
    pool_size : int, optional
        Number of worker processes to use. ``1`` (default) runs a simple
        serial loop. Values greater than 1 dispatch positions across a
        :class:`multiprocessing.Pool` via ``imap``, producing near-linear
        speedup on multi-core machines. Use ``multiprocessing.cpu_count()``
        to utilise all available cores. Tasks are yielded lazily so that
        memory use scales with *pool_size*, not with *npositions* — important
        for long records (hourly or sub-hourly over months).
    label : str, optional
        If non-empty, a live progress line is printed to stdout showing
        completed positions, elapsed time, and estimated time remaining.
        :class:`HarmonicsAnalyser` sets this automatically when
        ``show_progress=True``; it can also be set directly when calling
        :func:`analyse_harmonics` standalone. Defaults to ``''`` (no output).
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
    from multiprocessing import Pool
    import warnings

    npositions = len(latitudes)
    harmonics = np.full((npositions, 2, len(constit)), np.nan)
    if predict:
        predicted = np.full((npositions, len(times)), np.nan)

    # Extract verbose for reconstruct; utide.solve also accepts it via **kwargs.
    verbose = kwargs.get('verbose', False)

    # Progress tracking: _state[0] holds the timestamp of the last print.
    _state = [0.0]
    t_start = time.perf_counter()

    if pool_size == 1:
        # ── Serial path ───────────────────────────────────────────────────────
        for i, (timeseries, lat) in enumerate(zip(elevations, latitudes)):
            if label:
                _print_progress(label, i, npositions, t_start, _state)

            if np.isnan(lat):
                continue

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='utide')
                res = utide_solve(t=times, u=timeseries, lat=lat, method='ols',
                                  constit=constit, epoch=MJD_ZERO_POINT, **kwargs)

            # UTide returns constituents in a different order; re-sort to match constit.
            c_order = [res['name'].tolist().index(cc) for cc in constit]
            harmonics[i, 0, :] = res['g'][c_order]   # phase
            harmonics[i, 1, :] = res['A'][c_order]   # amplitude

            if predict:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning, module='utide')
                    reconstructed = reconstruct(t=times, coef=res,
                                                epoch=MJD_ZERO_POINT,
                                                verbose=verbose)
                predicted[i, :] = reconstructed['h']

        if label:
            _print_progress(label, npositions, npositions, t_start, _state)
    else:
        # ── Parallel path ─────────────────────────────────────────────────────
        # Tasks are yielded lazily so only pool_size * chunksize time-series
        # arrays are held in memory at any one time — important for long
        # records (hourly/sub-hourly over months).
        def _task_gen():
            for u, lat in zip(elevations, latitudes):
                yield (times, u, lat, constit, MJD_ZERO_POINT, verbose, predict)

        # chunksize batches tasks per IPC round-trip: small enough to keep
        # memory bounded, large enough to amortise round-trip overhead.
        chunksize = max(1, pool_size * 4)
        with Pool(processes=pool_size) as pool:
            for i, result in enumerate(
                pool.imap(_solve_one, _task_gen(), chunksize=chunksize)
            ):
                if label:
                    _print_progress(label, i, npositions, t_start, _state)
                if result is None:
                    continue   # NaN latitude — leave output row as NaN
                g, A, h = result
                harmonics[i, 0, :] = g
                harmonics[i, 1, :] = A
                if predict and h is not None:
                    predicted[i, :] = h

        if label:
            _print_progress(label, npositions, npositions, t_start, _state)

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
    node_indices : array-like of int, optional
        Integer indices into the full node array selecting the subset of nodes
        to analyse. If None (default), all nodes are analysed. Useful for
        restricting the analysis to a spatial region of interest and avoiding
        the cost of analysing the full mesh. Typically constructed as::

            mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
            node_indices = np.where(mask)[0]

    element_indices : array-like of int, optional
        Integer indices into the full element array selecting the subset of
        elements to analyse. If None (default), all elements are analysed.
        Constructed analogously to *node_indices* using element-centred
        coordinates (``reader.lon_elements``, ``reader.lat_elements``).
    verbose : bool, optional
        If True, print UTide's per-solve progress messages (``"solve: matrix
        prep … solution … done."``). Defaults to False.
    pool_size : int, optional
        Number of worker processes for parallel analysis. ``1`` (default)
        runs serially. Set to the number of physical cores (e.g.
        ``multiprocessing.cpu_count()``) for maximum throughput. Passed
        directly to :func:`analyse_harmonics`.
    show_progress : bool, optional
        If True (default), print a live progress line to stdout for each
        variable (and each sigma layer for 3-D variables) showing the count
        of completed positions, elapsed time, and estimated time remaining.
        Set to False to suppress all progress output.

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
        node_indices: np.ndarray = None,
        element_indices: np.ndarray = None,
        verbose: bool = False,
        pool_size: int = 1,
        show_progress: bool = True,
    ):
        self._reader = reader
        self._output_file = output_file
        self._constit = constit
        self._predict = predict
        self._dump_raw = dump_raw
        self._node_indices = node_indices
        self._element_indices = element_indices
        self._verbose = verbose
        self._pool_size = pool_size
        self._show_progress = show_progress

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
        node_idx = self._node_indices
        elem_idx = self._element_indices

        n_nodes = reader.n_nodes if node_idx is None else len(node_idx)
        n_ele   = reader.n_elements if elem_idx is None else len(elem_idx)

        lon    = reader.lon_nodes    if node_idx is None else reader.lon_nodes[node_idx]
        lat    = reader.lat_nodes    if node_idx is None else reader.lat_nodes[node_idx]
        lonc   = reader.lon_elements if elem_idx is None else reader.lon_elements[elem_idx]
        latc   = reader.lat_elements if elem_idx is None else reader.lat_elements[elem_idx]
        h      = reader.bathy_nodes    if node_idx is None else reader.bathy_nodes[node_idx]
        h_ctr  = reader.bathy_elements if elem_idx is None else reader.bathy_elements[elem_idx]
        siglay = reader.sigma_layers_nodes if node_idx is None else reader.sigma_layers_nodes[:, node_idx]
        siglev = reader.sigma_levels_nodes if node_idx is None else reader.sigma_levels_nodes[:, node_idx]

        # nv is only meaningful for a full-mesh run (subset connectivity
        # would require re-indexing which is not attempted here).
        if node_idx is None:
            nv = (reader.grid.triangles + 1).T
        else:
            # Placeholder: store a zero array; plotting via scatter not tripcolor.
            nv = np.zeros((3, n_ele), dtype=int)

        dims = {
            'node': n_nodes,
            'nele': n_ele,
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

        with HarmonicAnalysisWriter(self._output_file, dims, global_atts) as ncout:
            ncout.add_grid(
                lon=lon, lat=lat, lonc=lonc, latc=latc,
                h=h, h_center=h_ctr, nv=nv,
                siglay=siglay, siglev=siglev,
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
        indices = self._node_indices if is_node else self._element_indices
        lat_all = reader.lat_nodes if is_node else reader.lat_elements
        latitudes = lat_all if indices is None else lat_all[indices]
        npositions = len(latitudes)
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

                label = f'{var} [\u03c3 {zlev + 1}/{nz}]' if self._show_progress else ''
                # analyse_harmonics expects elevations as (npositions, ntimes)
                result = analyse_harmonics(
                    self._times, data.T, latitudes,
                    predict=self._predict,
                    constit=self._constit,
                    pool_size=self._pool_size,
                    label=label,
                    verbose=self._verbose,
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

            label = var if self._show_progress else ''
            result = analyse_harmonics(
                self._times, data.T, latitudes,
                predict=self._predict,
                constit=self._constit,
                pool_size=self._pool_size,
                label=label,
                verbose=self._verbose,
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

        Reads each underlying NetCDF file in a single bulk slice rather than
        one timestep at a time, which is substantially faster for variables
        with many time steps.

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
            Shape ``(ntimes, npositions)``, where npositions reflects any
            spatial subset set on the analyser.
        """
        from netCDF4 import Dataset as _Dataset

        reader = self._reader
        is_node = reader.var_is_node_based(var)
        spatial_idx = self._node_indices if is_node else self._element_indices
        npositions = (reader.n_nodes if is_node else reader.n_elements) \
                     if spatial_idx is None else len(spatial_idx)
        ntimes = len(reader.dates)
        data = np.full((ntimes, npositions), np.nan)

        # Build a per-file list of (global_time_indices, local_time_indices)
        # using the reader's internal time→file mapping.
        file_slices: dict[str, list] = {}
        for global_t, dt in enumerate(reader.dates):
            fp, local_t = reader._time_to_file_map[dt]
            file_slices.setdefault(fp, []).append((global_t, local_t))

        for fp, index_pairs in file_slices.items():
            global_idxs = [p[0] for p in index_pairs]
            local_idxs  = [p[1] for p in index_pairs]
            with _Dataset(fp) as ds:
                raw = ds.variables[var][local_idxs, ...]   # (n_t_local, ...)
                if np.ma.is_masked(raw):
                    raw = np.ma.getdata(raw)
                if zlev is not None:
                    # 3-D variable: (n_t_local, nz, npositions)
                    slice_ = raw[:, zlev, :] if spatial_idx is None else raw[:, zlev, :][:, spatial_idx]
                else:
                    # 2-D variable: (n_t_local, npositions)
                    slice_ = raw if spatial_idx is None else raw[:, spatial_idx]
                data[global_idxs, :] = slice_

        return data

