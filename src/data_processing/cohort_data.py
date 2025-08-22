"""
Cohort data extraction and target variable generation.

This module handles the extraction of patient cohort data from the database
and generates target variables for machine learning models including mortality,
length of stay, and readmission events.
"""

import duckdb
import pandas as pd
from typing import List, Tuple
from data_processing.utils import get_year_difference, get_hour_difference

# Patient eligibility criteria
MIN_AGE = 18  # Minimum age for inclusion (years)
MAX_AGE = 89  # Maximum age for inclusion (years)
MIN_LOS_HOURS = 54  # Minimum length of stay for inclusion (hours)

# Event prediction time windows (hours)
MORTALITY_EVENT_HOURS = 720  # 30 days - predict mortality within 30 days of discharge
LOS_EVENT_HOURS = 168  # 7 days - predict extended length of stay (>7 days)
READMISSION_EVENT_HOURS = 720  # 30 days - predict readmission within 30 days of discharge

# SQL query to extract admission and patient data for the cohort
ADMISSION_SQL = f"""
    SELECT a.subject_id::INTEGER AS subject_id,                      -- Patient identifier
           a.hadm_id::INTEGER AS hadm_id,                            -- Hospital admission identifier
           a.admittime::TIMESTAMP AS admittime,                      -- Admission timestamp
           a.dischtime::TIMESTAMP AS dischtime,                      -- Discharge timestamp
           a.has_chartevents_data::INTEGER AS has_chartevents_data,  -- Whether the patient has chart events data
           p.dob::TIMESTAMP AS dob,                                  -- Patient date of birth
           p.dod::TIMESTAMP AS dod                                   -- Patient date of death (if applicable)
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id                           -- Join with patient demographics
    WHERE a.subject_id::INTEGER IN (SELECT subject_id FROM tmp_subject_ids)  -- Filter to cohort subjects
    ORDER BY a.subject_id, a.admittime                                       -- Sort by patient and admission time
    """

# Columns that will be included in the final target dataset
TARGET_COLUMNS = [
    "hadm_id",              # Hospital admission ID (primary key)
    "mortality_event",      # Binary: death within prediction window
    "los_event",            # Binary: extended length of stay
    "readmission_event"     # Binary: readmission within prediction window
]


def get_cohort_hadm_ids_and_targets(con: duckdb.DuckDBPyConnection, subject_ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
    """
    Extract hospital admission IDs and target variables for a given cohort of patients.

    This function processes patient admissions data to create a clean cohort for machine learning
    by applying inclusion/exclusion criteria and generating binary target variables for prediction tasks.

    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB database connection
        subject_ids (List[int]): List of patient subject IDs to include in the cohort

    Returns:
        Tuple[List[int], pd.DataFrame]: 
            - List of hospital admission IDs (hadm_ids) that meet all criteria
            - DataFrame with target variables (mortality_event, los_event, readmission_event)

    Inclusion Criteria:
        - Age between MIN_AGE and MAX_AGE years at admission
        - Length of stay >= MIN_LOS_HOURS hours
        - If patient died, death must occur >= MIN_LOS_HOURS after admission
        - Must have chart events data available

    Target Variables:
        - mortality_event: Death within MORTALITY_EVENT_HOURS of discharge
        - los_event: Length of stay > LOS_EVENT_HOURS
        - readmission_event: Readmission within READMISSION_EVENT_HOURS of discharge
    """
    # Register subject IDs as a temporary table for SQL query
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))

    # Execute SQL query to get admission and patient data
    df = con.execute(ADMISSION_SQL).fetchdf()

    # Calculate next admission time for readmission events
    # Shift admittime by -1 within each patient group to get the next admission
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)

    # Keep only the first admission per patient for this analysis
    df = df.groupby("subject_id", as_index=False).head(1)

    # Age filter: patient must be between MIN_AGE and MAX_AGE at admission
    keep_age = get_year_difference(df["admittime"], df["dob"]).between(MIN_AGE, MAX_AGE)

    # Length of stay filter: admission must be at least MIN_LOS_HOURS long
    keep_los = get_hour_difference(df["dischtime"], df["admittime"]) >= MIN_LOS_HOURS

    # Death filter: if patient died, death must be at least MIN_LOS_HOURS after admission
    keep_dod = df["dod"].isna() | (get_hour_difference(df["dod"], df["admittime"]) >= MIN_LOS_HOURS)

    # Chart events filter: patient must have chart events data available
    keep_chartevents = df["has_chartevents_data"] == 1

    # Combine all inclusion criteria with logical AND
    keep = keep_age & keep_los & keep_dod & keep_chartevents

    # Filter the dataframe to include only patients meeting all criteria
    df = df[keep].reset_index(drop=True)

    # Mortality event: death within MORTALITY_EVENT_HOURS of discharge
    df["mortality_event"] = df["dod"].notna() & (get_hour_difference(df["dod"], df["dischtime"]) <= MORTALITY_EVENT_HOURS)

    # Length of stay event: admission longer than LOS_EVENT_HOURS
    df["los_event"] = get_hour_difference(df["dischtime"], df["admittime"]) > LOS_EVENT_HOURS

    # Readmission event: next admission within READMISSION_EVENT_HOURS of discharge
    df["readmission_event"] = df["next_admittime"].notna() & (get_hour_difference(df["next_admittime"], df["dischtime"]) <= READMISSION_EVENT_HOURS)

    # Extract final outputs
    hadm_ids = df["hadm_id"].tolist()  # List of admission IDs meeting criteria
    targets = df[TARGET_COLUMNS].reset_index(drop=True)  # Target variables dataframe

    return hadm_ids, targets
