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