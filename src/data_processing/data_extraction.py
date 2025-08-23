import duckdb
import numpy as np
from typing import List, Tuple
from .cohort_data import get_cohort_hadm_ids_and_targets
from .static_data import get_static_data
from .timeseries_data import get_timeseries_data
from .logging_utils import logger

DUCKDB_PATH = r"H:\My Drive\MIMIC-III\mimiciii.duckdb"


def extract_data(subject_ids: List[int]) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.log_start("extract_data")
    con = duckdb.connect(DUCKDB_PATH)
    hadm_ids, targets = get_cohort_hadm_ids_and_targets(con, subject_ids)
    static_data = get_static_data(con, hadm_ids)
    timeseries_data, timeseries_missingness = get_timeseries_data(con, hadm_ids)
    con.close()
    logger.log_end("extract_data")
    return hadm_ids, static_data, timeseries_data, timeseries_missingness, targets
