import pandas as pd
import numpy as np
from typing import List

WINDOW_HOURS = 48
SECONDS_PER_HOUR = 3600

VITAL_COLUMNS = ["subject_id", "hadm_id", "name", "valuenum", "hour_from_admission"]
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
    {'itemid': 223761, 'name': 'temperature', 'min': 68, 'max': 113, 'apply_func': lambda x: (x - 32) * 5/9},
    {'itemid': 678, 'name': 'temperature', 'min': 68, 'max': 113, 'apply_func': lambda x: (x - 32) * 5/9},
]


def query_vitals_48h(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Query vital sign events within the first 48 hours and filter by metadata bounds."""
    meta = pd.DataFrame(VITAL_METADATA)
    itemids = meta["itemid"].tolist()

    con.register("tmp_vital_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql = f"""
    SELECT c.subject_id::INTEGER AS subject_id,
           c.hadm_id::INTEGER AS hadm_id,
           c.itemid::INTEGER AS itemid,
           c.charttime::TIMESTAMP AS charttime,
           c.valuenum::DOUBLE AS valuenum,
           a.admittime::TIMESTAMP AS admittime
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vital_itemids)
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER = 0
    """
    df = con.execute(sql).fetchdf()
    df = df.merge(meta, on="itemid")[(df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])].reset_index(drop=True)

    for _, row in meta.iterrows():
        if 'apply_func' in row:
            mask = df['itemid'] == row['itemid']
            df.loc[mask, 'valuenum'] = df.loc[mask, 'valuenum'].apply(row['apply_func'])

    time_diff_hours = np.ceil((df["charttime"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR)
    time_diff_hours = np.maximum(time_diff_hours, 1)
    df["hour_from_admission"] = time_diff_hours.astype(int)

    df = df[VITAL_COLUMNS]
    return df
