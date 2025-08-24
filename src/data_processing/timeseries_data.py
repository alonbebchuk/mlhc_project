"""
Time-Series Data Extraction for ICU Patient Monitoring

This module handles extraction and aggregation of time-series patient monitoring data
from the MIMIC-III database, including laboratory values and vital signs collected
during the first 48 hours of hospital admission.

The module processes two main categories of time-series data:
1. Laboratory values: Blood chemistry, hematology, coagulation studies
2. Vital signs: Heart rate, blood pressure, respiratory rate, temperature, oxygen saturation

Features are aggregated by hour using statistical summaries (min, max, mean, median,
standard deviation, count, quartiles) to create a dense representation suitable
for machine learning models.

Key processing steps:
- Quality filtering using clinically reasonable value ranges
- Unit standardization (temperature conversion from Fahrenheit to Celsius)
- Temporal aggregation by admission hour (1-48 hours)
- Statistical summary computation for each hour-feature combination
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

from .logging_utils import logger

# Temporal window configuration
WINDOW_HOURS = 48           # Observation window in hours from admission

# Laboratory test metadata with MIMIC-III item IDs and clinically reasonable ranges
# Each entry contains: itemid (MIMIC-III identifier), standardized name, and valid range
# Multiple item IDs may map to the same lab test due to different measurement systems
LAB_METADATA = [
    {'itemid': 50862, 'name': 'albumin', 'min': 0.1, 'max': 10},           # Serum albumin (g/dL)
    {'itemid': 50868, 'name': 'anion_gap', 'min': 1, 'max': 40},           # Anion gap (mEq/L)
    {'itemid': 50882, 'name': 'bicarbonate', 'min': 1, 'max': 100},        # Bicarbonate (mEq/L)
    {'itemid': 50885, 'name': 'bilirubin_total', 'min': 0, 'max': 20},     # Total bilirubin (mg/dL)
    {'itemid': 51006, 'name': 'bun', 'min': 2, 'max': 200},                # Blood urea nitrogen (mg/dL)
    {'itemid': 50902, 'name': 'chloride', 'min': 80, 'max': 130},          # Chloride (mEq/L) - CareVue
    {'itemid': 50806, 'name': 'chloride', 'min': 80, 'max': 130},          # Chloride (mEq/L) - MetaVision
    {'itemid': 50912, 'name': 'creatinine', 'min': 0.1, 'max': 15},        # Serum creatinine (mg/dL)
    {'itemid': 50931, 'name': 'glucose', 'min': 15, 'max': 2000},          # Blood glucose (mg/dL) - CareVue
    {'itemid': 50809, 'name': 'glucose', 'min': 15, 'max': 2000},          # Blood glucose (mg/dL) - MetaVision
    {'itemid': 51221, 'name': 'hematocrit', 'min': 10, 'max': 80},         # Hematocrit (%) - CareVue
    {'itemid': 50810, 'name': 'hematocrit', 'min': 10, 'max': 80},         # Hematocrit (%) - MetaVision
    {'itemid': 51222, 'name': 'hemoglobin', 'min': 2, 'max': 25},          # Hemoglobin (g/dL) - CareVue
    {'itemid': 50811, 'name': 'hemoglobin', 'min': 2, 'max': 25},          # Hemoglobin (g/dL) - MetaVision
    {'itemid': 51237, 'name': 'inr', 'min': 0.5, 'max': 10},               # International normalized ratio
    {'itemid': 50813, 'name': 'lactate', 'min': 0.2, 'max': 15},           # Serum lactate (mmol/L)
    {'itemid': 50960, 'name': 'magnesium', 'min': 0.5, 'max': 4},          # Serum magnesium (mg/dL)
    {'itemid': 50970, 'name': 'phosphate', 'min': 0.1, 'max': 20},         # Serum phosphate (mg/dL)
    {'itemid': 51265, 'name': 'platelet_count', 'min': 0.1, 'max': 1000},  # Platelet count (K/uL)
    {'itemid': 50971, 'name': 'potassium', 'min': 1, 'max': 10},           # Serum potassium (mEq/L) - CareVue
    {'itemid': 50822, 'name': 'potassium', 'min': 1, 'max': 10},           # Serum potassium (mEq/L) - MetaVision
    {'itemid': 51274, 'name': 'pt', 'min': 5, 'max': 50},                  # Prothrombin time (seconds)
    {'itemid': 51275, 'name': 'ptt', 'min': 5, 'max': 200},                # Partial thromboplastin time (seconds)
    {'itemid': 50983, 'name': 'sodium', 'min': 100, 'max': 200},           # Serum sodium (mEq/L) - CareVue
    {'itemid': 50824, 'name': 'sodium', 'min': 100, 'max': 200},           # Serum sodium (mEq/L) - MetaVision
    {'itemid': 51301, 'name': 'wbc_count', 'min': 0.2, 'max': 150},        # White blood cell count (K/uL) - CareVue
    {'itemid': 51300, 'name': 'wbc_count', 'min': 0.2, 'max': 150},        # White blood cell count (K/uL) - MetaVision
]

# Vital signs metadata with MIMIC-III item IDs and clinically reasonable ranges
# Multiple item IDs per vital sign accommodate different monitoring systems and locations
VITAL_METADATA = [
    # Diastolic blood pressure (mmHg) - multiple monitoring systems
    {'itemid': 8368, 'name': 'diastolic_bp', 'min': 20, 'max': 240},       # CareVue - invasive
    {'itemid': 8440, 'name': 'diastolic_bp', 'min': 20, 'max': 240},       # CareVue - non-invasive
    {'itemid': 8441, 'name': 'diastolic_bp', 'min': 20, 'max': 240},       # CareVue - non-invasive
    {'itemid': 8555, 'name': 'diastolic_bp', 'min': 20, 'max': 240},       # CareVue - invasive
    {'itemid': 220180, 'name': 'diastolic_bp', 'min': 20, 'max': 240},     # MetaVision - invasive
    {'itemid': 220051, 'name': 'diastolic_bp', 'min': 20, 'max': 240},     # MetaVision - non-invasive
    # Blood glucose (mg/dL) - chart events (point-of-care testing)
    {'itemid': 807, 'name': 'glucose', 'min': 15, 'max': 2000},            # CareVue - bedside glucose
    {'itemid': 811, 'name': 'glucose', 'min': 15, 'max': 2000},            # CareVue - bedside glucose
    {'itemid': 1529, 'name': 'glucose', 'min': 15, 'max': 2000},           # CareVue - bedside glucose
    {'itemid': 3744, 'name': 'glucose', 'min': 15, 'max': 2000},           # CareVue - bedside glucose
    {'itemid': 3745, 'name': 'glucose', 'min': 15, 'max': 2000},           # CareVue - bedside glucose
    {'itemid': 220621, 'name': 'glucose', 'min': 15, 'max': 2000},         # MetaVision - bedside glucose
    {'itemid': 225664, 'name': 'glucose', 'min': 15, 'max': 2000},         # MetaVision - bedside glucose
    {'itemid': 226537, 'name': 'glucose', 'min': 15, 'max': 2000},         # MetaVision - bedside glucose
    # Heart rate (beats per minute)
    {'itemid': 211, 'name': 'heart_rate', 'min': 10, 'max': 300},          # CareVue
    {'itemid': 220045, 'name': 'heart_rate', 'min': 10, 'max': 300},       # MetaVision
    # Mean arterial pressure (mmHg)
    {'itemid': 52, 'name': 'mean_bp', 'min': 20, 'max': 245},              # CareVue - invasive
    {'itemid': 443, 'name': 'mean_bp', 'min': 20, 'max': 245},             # CareVue - invasive
    {'itemid': 456, 'name': 'mean_bp', 'min': 20, 'max': 245},             # CareVue - non-invasive
    {'itemid': 6702, 'name': 'mean_bp', 'min': 20, 'max': 245},            # CareVue - invasive
    {'itemid': 220052, 'name': 'mean_bp', 'min': 20, 'max': 245},          # MetaVision - invasive
    {'itemid': 220181, 'name': 'mean_bp', 'min': 20, 'max': 245},          # MetaVision - invasive
    {'itemid': 225312, 'name': 'mean_bp', 'min': 20, 'max': 245},          # MetaVision - invasive
    # Respiratory rate (breaths per minute)
    {'itemid': 615, 'name': 'respiratory_rate', 'min': 5, 'max': 100},     # CareVue
    {'itemid': 618, 'name': 'respiratory_rate', 'min': 5, 'max': 100},     # CareVue
    {'itemid': 220210, 'name': 'respiratory_rate', 'min': 5, 'max': 100},  # MetaVision
    {'itemid': 224690, 'name': 'respiratory_rate', 'min': 5, 'max': 100},  # MetaVision
    # Oxygen saturation (%)
    {'itemid': 646, 'name': 'spo2', 'min': 5, 'max': 100},                 # CareVue
    {'itemid': 220277, 'name': 'spo2', 'min': 5, 'max': 100},              # MetaVision
    # Systolic blood pressure (mmHg)
    {'itemid': 51, 'name': 'systolic_bp', 'min': 20, 'max': 250},          # CareVue - invasive
    {'itemid': 442, 'name': 'systolic_bp', 'min': 20, 'max': 250},         # CareVue - invasive
    {'itemid': 455, 'name': 'systolic_bp', 'min': 20, 'max': 250},         # CareVue - non-invasive
    {'itemid': 6701, 'name': 'systolic_bp', 'min': 20, 'max': 250},        # CareVue - invasive
    {'itemid': 220050, 'name': 'systolic_bp', 'min': 20, 'max': 250},      # MetaVision - invasive
    {'itemid': 220179, 'name': 'systolic_bp', 'min': 20, 'max': 250},      # MetaVision - invasive
    # Body temperature (Celsius after conversion)
    {'itemid': 676, 'name': 'temperature', 'min': 20, 'max': 45},          # CareVue - Celsius
    {'itemid': 223762, 'name': 'temperature', 'min': 20, 'max': 45},       # MetaVision - Celsius
    {'itemid': 223761, 'name': 'temperature', 'min': 20, 'max': 45},       # MetaVision - Fahrenheit (converted)
    {'itemid': 678, 'name': 'temperature', 'min': 20, 'max': 45},          # CareVue - Fahrenheit (converted)
]

# Time conversion and temperature unit handling
SECONDS_PER_HOUR = 3600.0                           # Conversion factor for time calculations
FAHRENHEIT_ITEMIDS = [223761, 678]                  # Item IDs that record temperature in Fahrenheit
FAHRENHEIT_TO_CELSIUS = "(c.valuenum::DOUBLE - 32) * 5/9"  # SQL expression for F to C conversion

# SQL query for extracting and aggregating laboratory values by hour
# Computes statistical summaries for each lab test within each admission hour
LAB_SQL = f"""
    SELECT 
        l.hadm_id::INTEGER AS hadm_id,
        m.name AS name,
        -- Calculate admission hour (starting from hour 1, not 0)
        GREATEST(1, CAST(CEIL(EXTRACT(epoch FROM (l.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR}) AS INTEGER)) AS admission_hour,
        -- Statistical aggregation functions for each hour-feature combination
        MIN(l.valuenum::DOUBLE) AS min,        -- Minimum value in the hour
        MAX(l.valuenum::DOUBLE) AS max,        -- Maximum value in the hour
        AVG(l.valuenum::DOUBLE) AS mean,       -- Average value in the hour
        MEDIAN(l.valuenum::DOUBLE) AS median,  -- Median value in the hour
        STDDEV_SAMP(l.valuenum::DOUBLE) AS std, -- Standard deviation in the hour
        COUNT(l.valuenum) AS count,            -- Number of measurements in the hour
        QUANTILE_DISC(l.valuenum::DOUBLE, 0.25) AS q25,  -- 25th percentile
        QUANTILE_DISC(l.valuenum::DOUBLE, 0.75) AS q75   -- 75th percentile
    FROM labevents l
    JOIN admissions a ON l.hadm_id = a.hadm_id
    JOIN tmp_lab_meta m ON l.itemid = m.itemid
    WHERE l.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND l.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND l.valuenum IS NOT NULL
      AND l.valuenum::DOUBLE BETWEEN m.min AND m.max  -- Quality filtering using clinical ranges
    GROUP BY l.hadm_id, m.name, admission_hour
    ORDER BY l.hadm_id, m.name, admission_hour
    """

# SQL query for extracting and aggregating vital signs by hour
# Includes temperature unit conversion (Fahrenheit to Celsius) and statistical summaries
VITAL_SQL = f"""
    SELECT 
        c.hadm_id::INTEGER AS hadm_id,
        m.name AS name,
        -- Calculate admission hour (starting from hour 1, not 0)
        GREATEST(1, CAST(CEIL(EXTRACT(epoch FROM (c.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR}) AS INTEGER)) AS admission_hour,
        -- Statistical aggregation with temperature unit conversion
        MIN(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) AS min,
        MAX(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) AS max,
        AVG(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) AS mean,
        MEDIAN(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) AS median,
        STDDEV_SAMP(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) AS std,
        COUNT(c.valuenum) AS count,
        QUANTILE_DISC(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END, 0.25) AS q25,
        QUANTILE_DISC(CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END, 0.75) AS q75
    FROM chartevents c
    JOIN admissions a ON c.hadm_id = a.hadm_id
    JOIN tmp_vital_meta m ON c.itemid = m.itemid
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER = 0                        -- Exclude error measurements
      AND (CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) BETWEEN m.min AND m.max  -- Quality filtering
    GROUP BY c.hadm_id, m.name, admission_hour
    ORDER BY c.hadm_id, m.name, admission_hour
    """

# Data processing configuration for time-series aggregation
GROUP_COLUMNS = ['hadm_id', 'name', 'admission_hour']    # Columns used for grouping in SQL queries
STAT_COLUMNS = ['min', 'max', 'mean', 'median', 'std', 'count', 'q25', 'q75']  # Statistical summary columns
STAT_FILLNA_VALUES = {'std': 0.0}                       # Fill values for missing statistics (std=0 when only 1 measurement)

# Generate complete list of time-series feature names
# Creates features like: albumin_min, albumin_max, albumin_mean, heart_rate_min, etc.
TIMESERIES_COLUMNS = [
    f"{name}_{stat}"
    for name in sorted(list(set(meta['name'] for meta in LAB_METADATA + VITAL_METADATA)))
    for stat in STAT_COLUMNS
]


def get_timeseries_data(con, hadm_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract and aggregate time-series laboratory and vital sign data for given admissions.
    
    Processes both laboratory values and vital signs from the MIMIC-III database,
    aggregating measurements by hour using multiple statistical summaries to create
    dense time-series representations suitable for machine learning models.
    
    Args:
        con: Active DuckDB connection to MIMIC-III database
        hadm_ids (List[int]): List of hospital admission IDs to extract data for
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - timeseries_data: 3D array of shape (n_admissions, 48_hours, n_features)
              containing statistical summaries of vital signs and labs for each hour
            - timeseries_missingness: 3D binary array of same shape indicating 
              missing measurements (1=missing, 0=present)
              
    Processing steps:
        1. Execute SQL queries to extract and aggregate lab/vital data by hour
        2. Apply quality filtering using clinically reasonable value ranges
        3. Convert temperature units (Fahrenheit to Celsius) where needed
        4. Reshape data into dense 3D tensor format with missingness indicators
        5. Fill missing statistical values (e.g., std=0 for single measurements)
        
    Note:
        The function creates a dense tensor representation where missing hours
        are filled with NaN values. Hours are numbered 1-48 from admission time.
    """
    logger.log_start("get_timeseries_data")
    
    # Prepare metadata for SQL queries
    lab_meta = pd.DataFrame(LAB_METADATA)
    vital_meta = pd.DataFrame(VITAL_METADATA)
    
    # Register temporary tables for efficient SQL execution
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_lab_meta", lab_meta[['itemid', 'name', 'min', 'max']])
    con.register("tmp_vital_meta", vital_meta[['itemid', 'name', 'min', 'max']])
    con.register("tmp_fahrenheit_itemids", pd.DataFrame({"itemid": FAHRENHEIT_ITEMIDS}))
    
    # Execute SQL queries to get aggregated time-series data
    lab_df = con.execute(LAB_SQL).fetchdf()
    vital_df = con.execute(VITAL_SQL).fetchdf()
    
    # Combine lab and vital signs data
    df = pd.concat([lab_df, vital_df], ignore_index=True)
    
    # Fill missing statistical values with appropriate defaults
    for stat_col, fill_value in STAT_FILLNA_VALUES.items():
        df[stat_col] = df[stat_col].fillna(fill_value)
    
    # Reshape from wide format to long format for tensor creation
    melted = df.melt(id_vars=GROUP_COLUMNS, value_vars=STAT_COLUMNS, var_name='stat', value_name='value')
    melted['feature'] = melted['name'] + '_' + melted['stat']
    
    # Create dense 3D tensor: (n_admissions, n_hours, n_features)
    shape = (len(hadm_ids), WINDOW_HOURS, len(TIMESERIES_COLUMNS))
    timeseries_data = np.full(shape, np.nan)                    # Initialize with NaN (missing)
    timeseries_missingness = np.ones(shape, dtype=np.uint8)     # Initialize with 1 (missing)
    
    # Create index mappings for efficient tensor assignment
    hadm_indices = pd.Categorical(melted['hadm_id'], categories=hadm_ids).codes
    hour_indices = melted['admission_hour'].values - 1  # Convert to 0-based indexing
    feature_indices = pd.Categorical(melted['feature'], categories=TIMESERIES_COLUMNS).codes
    
    # Assign observed values to tensor and mark as present
    idx = (hadm_indices, hour_indices, feature_indices)
    timeseries_data[idx] = melted['value']
    timeseries_missingness[idx] = 0  # Mark as present (not missing)
    
    logger.log_end("get_timeseries_data")
    return timeseries_data, timeseries_missingness
