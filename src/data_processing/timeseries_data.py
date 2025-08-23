import pandas as pd
import numpy as np
from typing import List, Tuple

from .logging_utils import logger

WINDOW_HOURS = 48

LAB_METADATA = [
    {'itemid': 50862, 'name': 'albumin', 'min': 0.1, 'max': 10},
    {'itemid': 50868, 'name': 'anion_gap', 'min': 1, 'max': 40},
    {'itemid': 50882, 'name': 'bicarbonate', 'min': 1, 'max': 100},
    {'itemid': 50885, 'name': 'bilirubin_total', 'min': 0, 'max': 20},
    {'itemid': 51006, 'name': 'bun', 'min': 2, 'max': 200},
    {'itemid': 50902, 'name': 'chloride', 'min': 80, 'max': 130},
    {'itemid': 50806, 'name': 'chloride', 'min': 80, 'max': 130},
    {'itemid': 50912, 'name': 'creatinine', 'min': 0.1, 'max': 15},
    {'itemid': 50931, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 50809, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 51221, 'name': 'hematocrit', 'min': 10, 'max': 80},
    {'itemid': 50810, 'name': 'hematocrit', 'min': 10, 'max': 80},
    {'itemid': 51222, 'name': 'hemoglobin', 'min': 2, 'max': 25},
    {'itemid': 50811, 'name': 'hemoglobin', 'min': 2, 'max': 25},
    {'itemid': 51237, 'name': 'inr', 'min': 0.5, 'max': 10},
    {'itemid': 50813, 'name': 'lactate', 'min': 0.2, 'max': 15},
    {'itemid': 50960, 'name': 'magnesium', 'min': 0.5, 'max': 4},
    {'itemid': 50970, 'name': 'phosphate', 'min': 0.1, 'max': 20},
    {'itemid': 51265, 'name': 'platelet_count', 'min': 0.1, 'max': 1000},
    {'itemid': 50971, 'name': 'potassium', 'min': 1, 'max': 10},
    {'itemid': 50822, 'name': 'potassium', 'min': 1, 'max': 10},
    {'itemid': 51274, 'name': 'pt', 'min': 5, 'max': 50},
    {'itemid': 51275, 'name': 'ptt', 'min': 5, 'max': 200},
    {'itemid': 50983, 'name': 'sodium', 'min': 100, 'max': 200},
    {'itemid': 50824, 'name': 'sodium', 'min': 100, 'max': 200},
    {'itemid': 51301, 'name': 'wbc_count', 'min': 0.2, 'max': 150},
    {'itemid': 51300, 'name': 'wbc_count', 'min': 0.2, 'max': 150},
]

VITAL_METADATA = [
    {'itemid': 8368, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 8440, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 8441, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 8555, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 220180, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 220051, 'name': 'diastolic_bp', 'min': 20, 'max': 240},
    {'itemid': 807, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 811, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 1529, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 3744, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 3745, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 220621, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 225664, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 226537, 'name': 'glucose', 'min': 15, 'max': 2000},
    {'itemid': 211, 'name': 'heart_rate', 'min': 10, 'max': 300},
    {'itemid': 220045, 'name': 'heart_rate', 'min': 10, 'max': 300},
    {'itemid': 52, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 443, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 456, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 6702, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 220052, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 220181, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 225312, 'name': 'mean_bp', 'min': 20, 'max': 245},
    {'itemid': 615, 'name': 'respiratory_rate', 'min': 5, 'max': 100},
    {'itemid': 618, 'name': 'respiratory_rate', 'min': 5, 'max': 100},
    {'itemid': 220210, 'name': 'respiratory_rate', 'min': 5, 'max': 100},
    {'itemid': 224690, 'name': 'respiratory_rate', 'min': 5, 'max': 100},
    {'itemid': 646, 'name': 'spo2', 'min': 5, 'max': 100},
    {'itemid': 220277, 'name': 'spo2', 'min': 5, 'max': 100},
    {'itemid': 51, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 442, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 455, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 6701, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 220050, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 220179, 'name': 'systolic_bp', 'min': 20, 'max': 250},
    {'itemid': 676, 'name': 'temperature', 'min': 20, 'max': 45},
    {'itemid': 223762, 'name': 'temperature', 'min': 20, 'max': 45},
    {'itemid': 223761, 'name': 'temperature', 'min': 20, 'max': 45},
    {'itemid': 678, 'name': 'temperature', 'min': 20, 'max': 45},
]

SECONDS_PER_HOUR = 3600.0
FAHRENHEIT_ITEMIDS = [223761, 678]
FAHRENHEIT_TO_CELSIUS = "(c.valuenum::DOUBLE - 32) * 5/9"

LAB_SQL = f"""
    SELECT 
        l.hadm_id::INTEGER AS hadm_id,
        m.name AS name,
        GREATEST(1, CAST(CEIL(EXTRACT(epoch FROM (l.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR}) AS INTEGER)) AS admission_hour,
        MIN(l.valuenum::DOUBLE) AS min,
        MAX(l.valuenum::DOUBLE) AS max,
        AVG(l.valuenum::DOUBLE) AS mean,
        MEDIAN(l.valuenum::DOUBLE) AS median,
        STDDEV_SAMP(l.valuenum::DOUBLE) AS std,
        COUNT(l.valuenum) AS count,
        QUANTILE_DISC(l.valuenum::DOUBLE, 0.25) AS q25,
        QUANTILE_DISC(l.valuenum::DOUBLE, 0.75) AS q75
    FROM labevents l
    JOIN admissions a ON l.hadm_id = a.hadm_id
    JOIN tmp_lab_meta m ON l.itemid = m.itemid
    WHERE l.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND l.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND l.valuenum IS NOT NULL
      AND l.valuenum::DOUBLE BETWEEN m.min AND m.max
    GROUP BY l.hadm_id, m.name, admission_hour
    ORDER BY l.hadm_id, m.name, admission_hour
    """

VITAL_SQL = f"""
    SELECT 
        c.hadm_id::INTEGER AS hadm_id,
        m.name AS name,
        GREATEST(1, CAST(CEIL(EXTRACT(epoch FROM (c.charttime::TIMESTAMP - a.admittime::TIMESTAMP)) / {SECONDS_PER_HOUR}) AS INTEGER)) AS admission_hour,
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
      AND c.error::INTEGER = 0
      AND (CASE WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_fahrenheit_itemids) THEN {FAHRENHEIT_TO_CELSIUS} ELSE c.valuenum::DOUBLE END) BETWEEN m.min AND m.max
    GROUP BY c.hadm_id, m.name, admission_hour
    ORDER BY c.hadm_id, m.name, admission_hour
    """

GROUP_COLUMNS = ['hadm_id', 'name', 'admission_hour']
STAT_COLUMNS = ['min', 'max', 'mean', 'median', 'std', 'count', 'q25', 'q75']
STAT_FILLNA_VALUES = {'std': 0.0}

TIMESERIES_COLUMNS = [
    f"{name}_{stat}"
    for name in sorted(list(set(meta['name'] for meta in LAB_METADATA + VITAL_METADATA)))
    for stat in STAT_COLUMNS
]


def get_timeseries_data(con, hadm_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    logger.log_start("get_timeseries_data")
    lab_meta = pd.DataFrame(LAB_METADATA)
    vital_meta = pd.DataFrame(VITAL_METADATA)
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_lab_meta", lab_meta[['itemid', 'name', 'min', 'max']])
    con.register("tmp_vital_meta", vital_meta[['itemid', 'name', 'min', 'max']])
    con.register("tmp_fahrenheit_itemids", pd.DataFrame({"itemid": FAHRENHEIT_ITEMIDS}))
    lab_df = con.execute(LAB_SQL).fetchdf()
    vital_df = con.execute(VITAL_SQL).fetchdf()
    df = pd.concat([lab_df, vital_df], ignore_index=True)
    for stat_col, fill_value in STAT_FILLNA_VALUES.items():
        df[stat_col] = df[stat_col].fillna(fill_value)
    melted = df.melt(id_vars=GROUP_COLUMNS, value_vars=STAT_COLUMNS, var_name='stat', value_name='value')
    melted['feature'] = melted['name'] + '_' + melted['stat']
    shape = (len(hadm_ids), WINDOW_HOURS, len(TIMESERIES_COLUMNS))
    timeseries_data = np.full(shape, np.nan)
    timeseries_missingness = np.ones(shape, dtype=np.uint8)
    hadm_indices = pd.Categorical(melted['hadm_id'], categories=hadm_ids).codes
    hour_indices = melted['admission_hour'].values - 1
    feature_indices = pd.Categorical(melted['feature'], categories=TIMESERIES_COLUMNS).codes
    idx = (hadm_indices, hour_indices, feature_indices)
    timeseries_data[idx] = melted['value']
    timeseries_missingness[idx] = 0
    logger.log_end("get_timeseries_data")
    return timeseries_data, timeseries_missingness
