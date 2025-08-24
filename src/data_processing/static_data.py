"""
Static Patient Features Extraction for ICU Outcome Prediction

This module handles extraction of static (time-invariant) patient features from
the MIMIC-III database. Static features include demographics, anthropometric
measurements, and clinical interventions received during the first 48 hours.

The module extracts the following feature categories:
1. Demographics: age, gender, admission details, insurance, ethnicity
2. Anthropometric: height and weight (with unit conversions)
3. Clinical interventions: vasopressors, mechanical ventilation, RRT, sedation, antibiotics
4. ICU admission status within the observation window

All measurements and interventions are captured within the first 48 hours
of hospital admission to avoid future information leakage.
"""
import duckdb
import numpy as np
import pandas as pd
from typing import List

from .logging_utils import logger

# Temporal window configuration
WINDOW_HOURS = 48           # Observation window in hours from admission

# Unit conversion factors for anthropometric measurements
IN_TO_CM_FACTOR = 2.54      # Inches to centimeters conversion
LB_TO_KG_FACTOR = 0.45359237  # Pounds to kilograms conversion

# MIMIC-III item IDs for anthropometric measurements
HEIGHT_IN_ITEMIDS = [920, 1394]        # Height measurements in inches
HEIGHT_CM_ITEMIDS = [226730]           # Height measurements in centimeters  
WEIGHT_KG_ITEMIDS = [763, 3580, 226512, 224639]  # Weight measurements in kilograms
WEIGHT_LB_ITEMIDS = [3581, 226531]     # Weight measurements in pounds

# MIMIC-III item IDs for clinical interventions
# Vasopressor administration (cardiovascular and metavision systems)
VASOPRESSOR_CV_ITEMIDS = [30047, 30120, 30044, 30119, 30309, 30127, 30312, 30051, 30043, 30307, 30042, 30306]
VASOPRESSOR_MV_ITEMIDS = [221906, 221289, 221749, 222315, 221662, 221653]

# Mechanical ventilation (procedure events and chart events)
VENTILATION_PROCEDURE_ITEMIDS = [225468, 224385, 224391]
VENTILATION_CHART_ITEMIDS = [224684, 224685, 224686, 220339, 505, 506, 60, 444, 224695, 218, 224738, 223834, 467]

# Renal replacement therapy (RRT) - procedure and chart events
RRT_PROCEDURE_ITEMIDS = [225802, 225803, 225805, 224270]
RRT_CHART_ITEMIDS = [226499, 227357, 152, 224149, 582]

# Sedation administration (cardiovascular and metavision systems)
SEDATION_CV_ITEMIDS = [30131, 30124, 30166, 30121]
SEDATION_MV_ITEMIDS = [222168, 225150, 221385, 221668]

# Regular expression pattern to identify antibiotic medications
# Covers major antibiotic classes: beta-lactams, aminoglycosides, fluoroquinolones, etc.
ANTIBIOTIC_REGEX = r'(amoxicillin|ampicillin|oxacillin|penicillin|piperacillin|tazobactam|zosyn|cefazolin|cefepime|ceftazidime|ceftriaxone|cefuroxime|meropenem|imipenem|ertapenem|vancomycin|amikacin|gentamicin|tobramycin|azithromycin|ciprofloxacin|levofloxacin|clindamycin|doxycycline|metronidazole|rifampin|daptomycin|linezolid)'

# SQL query to extract static patient features within the observation window
# This complex query extracts demographics, anthropometric data, and clinical interventions
# All data is captured within the first 48 hours to avoid future information leakage
STATIC_SQL = f"""
    WITH height_data AS (
        -- Extract height measurements with unit conversion (inches -> cm)
        SELECT DISTINCT ON (c.hadm_id) 
            c.hadm_id,
            CASE 
                WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_in_itemids) THEN c.valuenum::DOUBLE * {IN_TO_CM_FACTOR}
                ELSE c.valuenum::DOUBLE 
            END AS height
        FROM chartevents c
        JOIN admissions a ON c.hadm_id = a.hadm_id
            WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
            AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_in_itemids) OR c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_cm_itemids))
            AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
            AND c.valuenum IS NOT NULL
            AND c.error = 0
        ORDER BY c.hadm_id, c.charttime
    ),
    weight_data AS (
        -- Extract weight measurements with unit conversion (pounds -> kg)
        SELECT DISTINCT ON (c.hadm_id)
            c.hadm_id,
            CASE 
                WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_lb_itemids) THEN c.valuenum::DOUBLE * {LB_TO_KG_FACTOR}
                ELSE c.valuenum::DOUBLE 
            END AS weight
        FROM chartevents c
        JOIN admissions a ON c.hadm_id = a.hadm_id
            WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
            AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_kg_itemids) OR c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_lb_itemids))
            AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
            AND c.valuenum IS NOT NULL
            AND c.error = 0
        ORDER BY c.hadm_id, c.charttime
    )
    -- Main query: extract comprehensive static patient features
    SELECT 
        a.hadm_id::INTEGER AS hadm_id,
        -- Demographic and administrative features
        a.admission_type,                    -- Type of admission (emergency, elective, etc.)
        a.admission_location,                -- Location patient was admitted from
        a.insurance,                         -- Insurance type
        a.language,                          -- Primary language
        a.religion,                          -- Religious affiliation
        a.marital_status,                    -- Marital status
        a.ethnicity,                         -- Ethnicity/race
        CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,  -- Binary gender (1=Male, 0=Female)
        EXTRACT(year FROM AGE(a.admittime::TIMESTAMP, p.dob::TIMESTAMP))::INTEGER AS age,  -- Age at admission
        -- Anthropometric measurements (may be NULL if not recorded)
        COALESCE(h.height) AS height,        -- Height in centimeters
        COALESCE(w.weight) AS weight,        -- Weight in kilograms
        -- Clinical intervention indicators (binary features within observation window)
        -- Vasopressor administration: Check both CareVue and MetaVision systems
        CASE WHEN EXISTS (
            SELECT 1 FROM inputevents_cv ie 
            WHERE ie.hadm_id = a.hadm_id 
              AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_cv_itemids)
              AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) OR EXISTS (
            SELECT 1 FROM inputevents_mv ie 
            WHERE ie.hadm_id = a.hadm_id 
              AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_mv_itemids)
              AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) THEN 1 ELSE 0 END AS received_vasopressor,
        -- Mechanical ventilation: Check both procedure events and chart events
        CASE WHEN EXISTS (
            SELECT 1 FROM procedureevents_mv pe 
            WHERE pe.hadm_id = a.hadm_id 
              AND pe.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_proc_itemids)
              AND pe.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) OR EXISTS (
            SELECT 1 FROM chartevents c 
            WHERE c.hadm_id = a.hadm_id 
              AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_chart_itemids)
              AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) THEN 1 ELSE 0 END AS recieved_mechanical_ventilation,
        -- Renal replacement therapy (RRT): Check both procedure and chart events
        CASE WHEN EXISTS (
            SELECT 1 FROM procedureevents_mv p 
            WHERE p.hadm_id = a.hadm_id 
              AND p.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_proc_itemids)
              AND p.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) OR EXISTS (
            SELECT 1 FROM chartevents c 
            WHERE c.hadm_id = a.hadm_id 
              AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_chart_itemids)
              AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) THEN 1 ELSE 0 END AS received_rrt,
        -- Sedation administration: Check both CareVue and MetaVision systems
        CASE WHEN EXISTS (
            SELECT 1 FROM inputevents_cv ie 
            WHERE ie.hadm_id = a.hadm_id 
              AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_cv_itemids)
              AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) OR EXISTS (
            SELECT 1 FROM inputevents_mv ie 
            WHERE ie.hadm_id = a.hadm_id 
              AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_mv_itemids)
              AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) THEN 1 ELSE 0 END AS received_sedation,
        -- Antibiotic administration: Check prescriptions using regex pattern matching
        CASE WHEN EXISTS (
            SELECT 1 FROM prescriptions p 
            WHERE p.hadm_id = a.hadm_id 
              AND LOWER(COALESCE(p.drug, '')) ~ '{ANTIBIOTIC_REGEX}'
              AND p.startdate::DATE BETWEEN a.admittime::DATE AND (a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS)::DATE
        ) THEN 1 ELSE 0 END AS received_antibiotic,
        -- ICU admission: Check if patient was admitted to ICU within observation window
        CASE WHEN EXISTS (
            SELECT 1 FROM icustays i 
            WHERE i.hadm_id = a.hadm_id 
              AND i.intime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        ) THEN 1 ELSE 0 END AS reached_icu
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id
    LEFT JOIN height_data h ON a.hadm_id = h.hadm_id
    LEFT JOIN weight_data w ON a.hadm_id = w.hadm_id
    WHERE a.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
    ORDER BY a.hadm_id
    """

# Column definitions for static features organized by data type
# These lists define the expected structure of extracted static data

# Categorical features requiring encoding (demographics and administrative)
CATEGORICAL_COLUMNS = [
    "admission_type",      # Type of admission (emergency, elective, urgent)
    "admission_location",  # Location admitted from (emergency dept, clinic, transfer)
    "insurance",          # Insurance type (Medicare, Medicaid, private, etc.)
    "language",           # Primary language
    "religion",           # Religious affiliation  
    "marital_status",     # Marital status (single, married, divorced, etc.)
    "ethnicity",          # Ethnicity/race categories
    "gender"              # Gender (already binary encoded in SQL)
]

# Numeric features that are always present (no missing values expected)
NUMERIC_COLUMNS_WITHOUT_MISSING = [
    "age"                 # Age at admission (always calculable from dates)
]

# Numeric features that may have missing values requiring imputation
NUMERIC_COLUMNS_WITH_MISSING = [
    "height",             # Height in centimeters (may not be recorded)
    "weight"              # Weight in kilograms (may not be recorded)
]

# All numeric columns (with and without missing values)
NUMERIC_COLUMNS = NUMERIC_COLUMNS_WITHOUT_MISSING + NUMERIC_COLUMNS_WITH_MISSING

# Binary indicator features for clinical interventions (0/1 values)
BINARY_COLUMNS = [
    "received_vasopressor",           # Received vasopressor medications
    "recieved_mechanical_ventilation", # Received mechanical ventilation support
    "received_rrt",                   # Received renal replacement therapy
    "received_sedation",              # Received sedation medications
    "received_antibiotic",            # Received antibiotic medications
    "reached_icu"                     # Was admitted to ICU
]

# Complete list of all static feature columns in expected order
STATIC_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS


def get_static_data(con: duckdb.DuckDBPyConnection, hadm_ids: List[int]) -> np.ndarray:
    """
    Extract static patient features for given hospital admission IDs.
    
    Executes a complex SQL query to extract comprehensive static features including
    demographics, anthropometric measurements, and clinical interventions within
    the first 48 hours of admission.
    
    Args:
        con (duckdb.DuckDBPyConnection): Active DuckDB connection to MIMIC-III database
        hadm_ids (List[int]): List of hospital admission IDs to extract features for
        
    Returns:
        np.ndarray: 2D array of shape (n_admissions, n_static_features) containing
                   extracted static features in the order defined by STATIC_COLUMNS
                   
    Features extracted:
        - Demographics: admission type/location, insurance, language, religion, marital status, ethnicity
        - Patient characteristics: age, gender, height, weight
        - Clinical interventions: vasopressors, mechanical ventilation, RRT, sedation, antibiotics
        - Care location: ICU admission status
        
    Note:
        The function registers multiple temporary tables in DuckDB for efficient
        SQL execution with item ID lookups. Missing categorical values are filled
        with 'missing' string for consistent downstream processing.
    """
    logger.log_start("get_static_data")
    
    # Register admission IDs as temporary table for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    
    # Register item ID lists as temporary tables for efficient SQL joins
    con.register("tmp_height_in_itemids", pd.DataFrame({"itemid": HEIGHT_IN_ITEMIDS}))
    con.register("tmp_height_cm_itemids", pd.DataFrame({"itemid": HEIGHT_CM_ITEMIDS}))
    con.register("tmp_weight_kg_itemids", pd.DataFrame({"itemid": WEIGHT_KG_ITEMIDS}))
    con.register("tmp_weight_lb_itemids", pd.DataFrame({"itemid": WEIGHT_LB_ITEMIDS}))
    con.register("tmp_vaso_cv_itemids", pd.DataFrame({"itemid": VASOPRESSOR_CV_ITEMIDS}))
    con.register("tmp_vaso_mv_itemids", pd.DataFrame({"itemid": VASOPRESSOR_MV_ITEMIDS}))
    con.register("tmp_vent_proc_itemids", pd.DataFrame({"itemid": VENTILATION_PROCEDURE_ITEMIDS}))
    con.register("tmp_vent_chart_itemids", pd.DataFrame({"itemid": VENTILATION_CHART_ITEMIDS}))
    con.register("tmp_rrt_proc_itemids", pd.DataFrame({"itemid": RRT_PROCEDURE_ITEMIDS}))
    con.register("tmp_rrt_chart_itemids", pd.DataFrame({"itemid": RRT_CHART_ITEMIDS}))
    con.register("tmp_sed_cv_itemids", pd.DataFrame({"itemid": SEDATION_CV_ITEMIDS}))
    con.register("tmp_sed_mv_itemids", pd.DataFrame({"itemid": SEDATION_MV_ITEMIDS}))
    
    # Execute static features SQL query
    df = con.execute(STATIC_SQL).fetchdf()
    
    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Fill missing categorical values with 'missing' for consistent encoding
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna('missing')
    
    # Extract features in expected column order
    static_data = df[STATIC_COLUMNS].values
    
    logger.log_end("get_static_data")
    return static_data
