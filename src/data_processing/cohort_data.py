"""
Cohort Definition and Target Label Generation for ICU Patient Outcome Prediction

This module defines the patient cohort selection criteria and creates target labels
for three clinical outcomes: mortality, length of stay, and readmission.

The cohort selection follows these inclusion criteria:
- First admission only for each patient
- Age between 18-89 years 
- Length of stay >= 54 hours
- Must have charted events data available

Target outcomes:
- Mortality: Death within 30 days (720 hours) of discharge
- Length of stay: ICU stay > 7 days (168 hours)
- Readmission: Readmission within 30 days (720 hours) of discharge
"""
import duckdb
import numpy as np
import pandas as pd
from typing import List, Tuple

from .logging_utils import logger

# Cohort inclusion criteria constants
MIN_AGE = 18                    # Minimum patient age in years
MAX_AGE = 89                    # Maximum patient age in years
MIN_LOS_HOURS = 54              # Minimum length of stay in hours (2.25 days)

# Target outcome time windows (in hours)
MORTALITY_EVENT_HOURS = 720     # 30 days - time window for mortality after discharge
LOS_EVENT_HOURS = 168           # 7 days - threshold for prolonged length of stay
READMISSION_EVENT_HOURS = 720   # 30 days - time window for readmission after discharge

# Time conversion constant
SECONDS_PER_HOUR = 3600.0

# SQL query to define patient cohort and extract target outcomes
# This complex query performs the following steps:
# 1. Calculate patient characteristics and temporal relationships
# 2. Apply inclusion/exclusion criteria to create final cohort
# 3. Generate binary target labels for three clinical outcomes
COHORT_SQL = f"""
    WITH patient_admissions AS (
        -- Calculate patient-level features and temporal relationships
        SELECT 
            a.hadm_id::INTEGER AS hadm_id,
            a.has_chartevents_data::INTEGER AS has_chartevents_data,
            -- Calculate age at admission
            EXTRACT(year FROM AGE(a.admittime::TIMESTAMP, p.dob::TIMESTAMP))::INTEGER AS age,
            -- Calculate length of stay in hours
            EXTRACT(epoch FROM (a.dischtime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS los_hours,
            -- Calculate time from discharge to death (for mortality outcome)
            EXTRACT(epoch FROM (p.dod::TIMESTAMP - a.dischtime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS discharge_to_death_hours,
            -- Calculate time from discharge to next admission (for readmission outcome)
            EXTRACT(epoch FROM (LEAD(a.admittime::TIMESTAMP) OVER (PARTITION BY a.subject_id ORDER BY a.admittime) - a.dischtime::TIMESTAMP)) / {SECONDS_PER_HOUR} AS discharge_to_readmission_hours,
            -- Rank admissions chronologically for each patient
            ROW_NUMBER() OVER (PARTITION BY a.subject_id ORDER BY a.admittime) AS admission_rank
        FROM admissions a
        JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.subject_id::INTEGER IN (SELECT subject_id FROM tmp_subject_ids)
    ),
    filtered_cohort AS (
        -- Apply inclusion criteria to create final cohort
        SELECT *
        FROM patient_admissions
        WHERE admission_rank = 1                    -- First admission only
          AND age BETWEEN {MIN_AGE} AND {MAX_AGE}   -- Age 18-89 years
          AND los_hours >= {MIN_LOS_HOURS}          -- Minimum 54 hours LOS
          AND has_chartevents_data = 1              -- Must have chart events data
    )
    -- Generate binary target labels for three outcomes
    SELECT 
        hadm_id,
        -- Mortality outcome: death within 30 days of discharge
        CASE WHEN discharge_to_death_hours <= {MORTALITY_EVENT_HOURS} THEN 1 ELSE 0 END AS mortality_event,
        -- Length of stay outcome: ICU stay longer than 7 days
        CASE WHEN los_hours > {LOS_EVENT_HOURS} THEN 1 ELSE 0 END AS los_event,
        -- Readmission outcome: readmission within 30 days of discharge
        CASE WHEN discharge_to_readmission_hours <= {READMISSION_EVENT_HOURS} THEN 1 ELSE 0 END AS readmission_event
    FROM filtered_cohort
    ORDER BY hadm_id
    """


def get_cohort_hadm_ids_and_targets(con: duckdb.DuckDBPyConnection, subject_ids: List[int]) -> Tuple[List[int], np.ndarray]:
    """
    Extract hospital admission IDs and target outcomes for a given set of patient subject IDs.
    
    This function applies cohort selection criteria to filter patients and generates
    binary target labels for three clinical outcomes: mortality, length of stay, and readmission.
    
    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection to MIMIC-III database
        subject_ids (List[int]): List of patient subject IDs to include in cohort
        
    Returns:
        Tuple[List[int], np.ndarray]: 
            - List of hospital admission IDs (hadm_ids) that meet inclusion criteria
            - 2D numpy array of shape (n_admissions, 3) containing binary target labels:
              - Column 0: mortality_event (1 if death within 30 days of discharge, 0 otherwise)
              - Column 1: los_event (1 if length of stay > 7 days, 0 otherwise)  
              - Column 2: readmission_event (1 if readmission within 30 days, 0 otherwise)
              
    Note:
        The function registers the subject_ids as a temporary table in DuckDB for efficient
        SQL execution. Only first admissions for each patient are included in the cohort.
    """
    logger.log_start("get_cohort_hadm_ids_and_targets")
    
    # Register subject IDs as temporary table for SQL query
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))
    
    # Execute cohort SQL to get filtered admissions and target labels
    df = con.execute(COHORT_SQL).fetchdf()
    
    # Extract admission IDs and target matrix
    hadm_ids = df["hadm_id"].tolist()
    targets = df[["mortality_event", "los_event", "readmission_event"]].reset_index(drop=True).values
    
    logger.log_end("get_cohort_hadm_ids_and_targets")
    return hadm_ids, targets
