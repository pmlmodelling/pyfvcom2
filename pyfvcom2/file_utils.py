"""Tools to assist with file handling."""

import os
from glob import glob
from typing import Optional
from datetime import datetime
from datetime import timedelta
from netCDF4 import Dataset
from cftime import num2pydate
from pyfvcom2.exceptions import PyFVCOM2FileNotFoundError

__all__ = ["find_file", "find_files"]

_TIME_VAR_CANDIDATES = ("time", "time_counter", "time_instant", "t")


def _read_time_as_datetimes(file_path: str):
    """Read the time coordinate from a NetCDF file as a list of datetimes.

    Tries a set of common time variable names in order.
    """
    with Dataset(file_path) as ds:
        time_var_name = next(
            (name for name in _TIME_VAR_CANDIDATES if name in ds.variables),
            None,
        )
        if time_var_name is None:
            raise KeyError(
                f"No recognised time variable found in {file_path}. "
                f"Tried: {_TIME_VAR_CANDIDATES}"
            )
        time_var = ds.variables[time_var_name]
        return num2pydate(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
        )


def find_file(
    dir_name: str,
    file_stem: str,
    date_time: datetime,
    tolerance_hours: Optional[int] = 0,
) -> str:
    """Find a file in a directory matching a given stem and containing date_time within a given tolerance.

    Args:
        dir_name (str): Directory to search for files.
        file_stem (str): Stem of the file name to match.
        date_time (datetime): Datetime to match in the file name.
        tolerance_hours (int, optional): Tolerance in hours for datetime matching. Defaults to 1.

    Returns:
        str: Path to the matched file.
        int: Index of the time variable corresponding to the matched datetime.

    Raises:
        PyFVCOM2FileNotFoundError: If no matching file is found within the tolerance.
    """
    # List all files in the directory with the given stem
    search_pattern = os.path.join(dir_name, f"{file_stem}*")
    candidate_files = glob(search_pattern)

    # Define the time window for matching
    start_time = date_time - timedelta(hours=tolerance_hours)
    end_time = date_time + timedelta(hours=tolerance_hours)

    for file_path in candidate_files:
        datetimes = _read_time_as_datetimes(file_path)

        if datetimes[0] <= start_time and datetimes[-1] >= end_time:
            file_path_found = file_path

            # Determine the time index closest to date_time
            time_diffs = [abs((dt - date_time).total_seconds()) for dt in datetimes]
            closest_time_index = time_diffs.index(min(time_diffs))

            return file_path_found, closest_time_index

    raise PyFVCOM2FileNotFoundError(
        f"No file found in {dir_name} matching stem {file_stem} within {tolerance_hours} hours of {date_time}."
    )


def find_files(
    dir_name: str,
    file_stem: str,
    start_date_time: datetime,
    end_date_time: datetime,
    tolerance_hours: Optional[int] = 0
) -> list[str]:
    """Find all files in a directory matching a given stem and containing datetimes within a given range.

    Args:
        dir_name (str): Directory to search for files.
        file_stem (str): Stem of the file name to match.
        start_date_time (datetime): Start datetime to match in the file name.
        end_date_time (datetime): End datetime to match in the file name.
        tolerance_hours (int, optional): Tolerance in hours for datetime matching. Defaults to 1.

    Returns:
        list[str]: List of paths to the matched files.
    """
    # List all files in the directory with the given stem
    search_pattern = os.path.join(dir_name, f"{file_stem}*")
    candidate_files = glob(search_pattern)

    matched_files = []

    for file_path in candidate_files:
        datetimes = _read_time_as_datetimes(file_path)

        if datetimes[0] <= end_date_time and datetimes[-1] >= start_date_time:
            matched_files.append(file_path)

    if not matched_files:
        raise PyFVCOM2FileNotFoundError(
            f"No files found in {dir_name} matching stem {file_stem} within {tolerance_hours} hours of {start_date_time} - {end_date_time}."
        )

    # Sort the matched files
    matched_files.sort()

    return matched_files
