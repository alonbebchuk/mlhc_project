import pandas as pd
import numpy as np
from typing import List

WINDOW_HOURS = 48
SECONDS_PER_HOUR = 3600

LAB_COLUMNS = ["subject_id", "hadm_id", "name", "valuenum", "hour_from_admission"]
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


def query_labs_48h(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Query lab events within the first 48 hours and filter by metadata bounds."""
    meta = pd.DataFrame(LAB_METADATA)
    itemids = meta["itemid"].tolist()

    con.register("tmp_lab_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql = f"""
    SELECT l.subject_id::INTEGER AS subject_id,
           l.hadm_id::INTEGER AS hadm_id,
           l.itemid::INTEGER AS itemid,
           l.charttime::TIMESTAMP AS charttime,
           l.valuenum::DOUBLE AS valuenum,
           a.admittime::TIMESTAMP AS admittime
    FROM labevents l
    JOIN admissions a ON l.subject_id = a.subject_id AND l.hadm_id = a.hadm_id
    WHERE l.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND l.itemid::INTEGER IN (SELECT itemid FROM tmp_lab_itemids)
      AND l.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND l.valuenum IS NOT NULL
    """
    df = con.execute(sql).fetchdf()
    df = df.merge(meta, on="itemid")[(df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])].reset_index(drop=True)

    time_diff_hours = np.ceil((df["charttime"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR)
    time_diff_hours = np.maximum(time_diff_hours, 1)
    df["hour_from_admission"] = time_diff_hours.astype(int)

    df = df[LAB_COLUMNS]
    return df
