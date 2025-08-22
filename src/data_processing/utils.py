"""
Data utility functions for time-based calculations and constants.

This module provides helper functions for calculating time differences
and defines constants used across the data processing pipeline.
"""

import pandas as pd

# Time window constant for analysis (in hours)
WINDOW_HOURS = 48  # 48-hour observation window


def get_hour_difference(end: pd.Series, start: pd.Series) -> pd.Series:
    """
    Calculate the difference between two datetime series in hours.

    Args:
        end (pd.Series): Later datetime series (minuend)
        start (pd.Series): Earlier datetime series (subtrahend)

    Returns:
        pd.Series: Time difference in hours as float values

    Example:
        >>> import pandas as pd
        >>> time1 = pd.Series([pd.Timestamp('2023-01-02 12:30:00')])
        >>> time2 = pd.Series([pd.Timestamp('2023-01-01 12:00:00')])
        >>> get_hour_difference(time1, time2)
        0    24.5
        dtype: float64
    """
    return (end - start) / pd.Timedelta(hours=1)


def get_year_difference(end: pd.Series, start: pd.Series) -> pd.Series:
    """
    Calculate the difference between two datetime series in years.

    This function extracts the year from each datetime and calculates
    the simple year difference (not accounting for months/days).

    Args:
        end (pd.Series): Later datetime series (minuend)
        start (pd.Series): Earlier datetime series (subtrahend)

    Returns:
        pd.Series: Year difference as integer values

    Example:
        >>> import pandas as pd
        >>> time1 = pd.Series([pd.Timestamp('2023-06-15')])
        >>> time2 = pd.Series([pd.Timestamp('1990-03-20')])
        >>> get_year_difference(time1, time2)
        0    33
        dtype: int64
    """
    return end.dt.year - start.dt.year
