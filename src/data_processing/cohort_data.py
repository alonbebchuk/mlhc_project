import duckdb
import numpy as np
import pandas as pd
from typing import List, Tuple

from .logging_utils import logger

MIN_AGE = 18
MAX_AGE = 89
MIN_LOS_HOURS = 54
MORTALITY_EVENT_HOURS = 720
LOS_EVENT_HOURS = 168
READMISSION_EVENT_HOURS = 720
SECONDS_PER_HOUR = 3600.0

COHORT_SQL = f"""
    WITH patient_admissions AS (
        SELECT 
            a.hadm_id::INTEGER AS hadm_id,
            a.has_chartevents_data::INTEGER AS has_chartevents_data,
            EXTRACT(year FROM AGE(a.admittime::TIMESTAMP, p.dob::TIMESTAMP))::INTEGER AS age,
            EXTRACT(epoch FROM (a.dischtime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS los_hours,
            EXTRACT(epoch FROM (p.dod::TIMESTAMP - a.dischtime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS discharge_to_death_hours,
            EXTRACT(epoch FROM (LEAD(a.admittime::TIMESTAMP) OVER (PARTITION BY a.subject_id ORDER BY a.admittime) - a.dischtime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS discharge_to_readmission_hours,
            ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS admission_rank
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.subject_id::INTEGER IN (SELECT subject_id FROM tmp_subject_ids)
    ),
    filtered_cohort AS (
        SELECT *
        FROM patient_admissions
        WHERE admission_rank = 1
          AND age BETWEEN {MIN_AGE} AND {MAX_AGE}
          AND los_hours >= {MIN_LOS_HOURS}
          AND has_chartevents_data = 1
    )
    SELECT 
        hadm_id,
        CASE WHEN discharge_to_death_hours <= {MORTALITY_EVENT_HOURS} THEN 1 ELSE 0 END AS mortality_event,
        CASE WHEN los_hours > {LOS_EVENT_HOURS} THEN 1 ELSE 0 END AS los_event,
        CASE WHEN discharge_to_readmission_hours <= {READMISSION_EVENT_HOURS} THEN 1 ELSE 0 END AS readmission_event
    FROM filtered_cohort
    ORDER BY hadm_id
    """


def get_cohort_hadm_ids_and_targets(con: duckdb.DuckDBPyConnection, subject_ids: List[int]) -> Tuple[List[int], np.ndarray]:
    logger.log_start("get_cohort_hadm_ids_and_targets")
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))
    df = con.execute(COHORT_SQL).fetchdf()
    hadm_ids = df["hadm_id"].tolist()
    targets = df[["mortality_event", "los_event", "readmission_event"]].reset_index(drop=True).values
    logger.log_end("get_cohort_hadm_ids_and_targets")
    return hadm_ids, targets
