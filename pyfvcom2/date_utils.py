from datetime import datetime, timedelta


def create_datetime_array(
    start_datetime: datetime,
    end_datetime: datetime,
    delta: timedelta
) -> list[datetime]:
    """Create an array of datetime objects from start to end with a given time delta.

    Args:
        start_datetime (datetime): Start datetime.
        end_datetime (datetime): End datetime.
        delta (timedelta): Time delta between consecutive datetimes.

    Returns:
        list[datetime]: List of datetime objects.
    """
    date_times = []
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        date_times.append(current_datetime)
        current_datetime += delta
    return date_times


def round_time(datetime_raw, rounding_interval=60):
    """Apply rounding to datetime objects

    Rounding is sometimes required when simulation times are written to file
    with limited precision.

    Parameters
    ----------
    datetime_raw: List, Datetime
        List of datetime objects to which rounding should be applied

    rounding_interval: int, optional
        No. of seconds to round to (default 60, or one minute)

    Returns
    -------
    datetime_rounded: List, Datetime
        List of rounded datetime objects
    """
    datetime_rounded = []
    for dt in datetime_raw:
        seconds = (dt - dt.min).seconds
        rounding = (seconds + rounding_interval/2) // rounding_interval * rounding_interval
        datetime_rounded.append(dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond))
    return datetime_rounded