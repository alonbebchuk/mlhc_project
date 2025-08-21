import pandas as pd

WINDOW_HOURS: int = 48

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_YEAR = 365 * 24 * SECONDS_PER_HOUR


def get_time_difference(time1: pd.Timestamp, time2: pd.Timestamp, seconds_per_unit: int) -> int:
    return (time1 - time2).total_seconds() / seconds_per_unit
