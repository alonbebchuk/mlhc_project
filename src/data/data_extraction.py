import duckdb
import pandas as pd
from typing import List, Tuple
from data.cohort_and_target_data import get_cohort_and_target_data
from data.static_data import get_static_features
from data.timeseries_data import get_timeseries_features


def extract_data(subject_ids: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drive_path = r"H:\My Drive\MIMIC-III"
    con = duckdb.connect(f"{drive_path}/mimiciii.duckdb")

    hadm_ids, targets = get_cohort_and_target_data(con, subject_ids)
    static_features = get_static_features(con, hadm_ids)
    timeseries_features = get_timeseries_features(con, hadm_ids)

    con.close()

    return hadm_ids, static_features, timeseries_features, targets
