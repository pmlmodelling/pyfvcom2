"""Tools to assist with file handling."""

import os
from glob import glob
from typing import Optional
from datetime import datetime
from datetime import timedelta
from netCDF4 import Dataset
from cftime import num2pydate
from pyfvcom2.exceptions import PyFVCOM2FileNotFoundError


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
        with Dataset(file_path) as ds:
            # Read time variable and convert to datetime
            datetimes = num2pydate(
                ds.variables["time"][:],
                units=ds.variables["time"].units,
                calendar=(
                    ds.variables["time"].calendar
                    if hasattr(ds.variables["time"], "calendar")
                    else "standard"
                ),
            )

        if datetimes[0] <= start_time and datetimes[-1] >= end_time:
            file_path_found = file_path

            # Determine the time index closest to date_time
            time_diffs = [abs((dt - date_time).total_seconds()) for dt in datetimes]
            closest_time_index = time_diffs.index(min(time_diffs))

            return file_path_found, closest_time_index

    raise PyFVCOM2FileNotFoundError(
        f"No file found in {dir_name} matching stem {file_stem} within {tolerance_hours} hours of {date_time}."
    )
