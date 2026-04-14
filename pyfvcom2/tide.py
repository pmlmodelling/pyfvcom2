from pyfvcom2.exceptions import PyFVCOM2ValueError
import pandas
import numpy as np
from typing import Optional
from netCDF4 import date2num
from datetime import datetime, timedelta
from utide import reconstruct, ut_constants
from utide.utilities import Bunch
import multiprocessing

from .interpolation_coordinates import InterpolationCoordinates
from .tide_reader import HarmonicsData


# Modified Julian Day zero point
MJD_ZERO_POINT = "1858-11-17"


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