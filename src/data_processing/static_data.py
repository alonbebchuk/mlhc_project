import duckdb
import numpy as np
import pandas as pd
from typing import List

from .logging_utils import logger

WINDOW_HOURS = 48
IN_TO_CM_FACTOR = 2.54
LB_TO_KG_FACTOR = 0.45359237

HEIGHT_IN_ITEMIDS = [920, 1394]
HEIGHT_CM_ITEMIDS = [226730]
WEIGHT_KG_ITEMIDS = [763, 3580, 226512, 224639]
WEIGHT_LB_ITEMIDS = [3581, 226531]
VASOPRESSOR_CV_ITEMIDS = [30047, 30120, 30044, 30119, 30309, 30127, 30312, 30051, 30043, 30307, 30042, 30306]
VASOPRESSOR_MV_ITEMIDS = [221906, 221289, 221749, 222315, 221662, 221653]
VENTILATION_PROCEDURE_ITEMIDS = [225468, 224385, 224391]
VENTILATION_CHART_ITEMIDS = [224684, 224685, 224686, 220339, 505, 506, 60, 444, 224695, 218, 224738, 223834, 467]
RRT_PROCEDURE_ITEMIDS = [225802, 225803, 225805, 224270]
RRT_CHART_ITEMIDS = [226499, 227357, 152, 224149, 582]
SEDATION_CV_ITEMIDS = [30131, 30124, 30166, 30121]
SEDATION_MV_ITEMIDS = [222168, 225150, 221385, 221668]
ANTIBIOTIC_REGEX = r'(amoxicillin|ampicillin|oxacillin|penicillin|piperacillin|tazobactam|zosyn|cefazolin|cefepime|ceftazidime|ceftriaxone|cefuroxime|meropenem|imipenem|ertapenem|vancomycin|amikacin|gentamicin|tobramycin|azithromycin|ciprofloxacin|levofloxacin|clindamycin|doxycycline|metronidazole|rifampin|daptomycin|linezolid)'

STATIC_SQL = f"""
    WITH height_data AS (
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
    SELECT 
        a.hadm_id::INTEGER AS hadm_id,
        a.admission_type,
        a.admission_location,
        a.insurance,
        a.language,
        a.religion,
        a.marital_status,
        a.ethnicity,
        CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender,
        EXTRACT(year FROM AGE(a.admittime::TIMESTAMP, p.dob::TIMESTAMP))::INTEGER AS age,
        COALESCE(h.height) AS height,
        COALESCE(w.weight) AS weight,
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
        CASE WHEN EXISTS (
            SELECT 1 FROM prescriptions p 
            WHERE p.hadm_id = a.hadm_id 
              AND LOWER(COALESCE(p.drug, '')) ~ '{ANTIBIOTIC_REGEX}'
              AND p.startdate::DATE BETWEEN a.admittime::DATE AND (a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS)::DATE
        ) THEN 1 ELSE 0 END AS received_antibiotic,
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

CATEGORICAL_COLUMNS = [
    "admission_type",
    "admission_location",
    "insurance",
    "language",
    "religion",
    "marital_status",
    "ethnicity",
    "gender"
]

NUMERIC_COLUMNS_WITHOUT_MISSING = [
    "age"
]

NUMERIC_COLUMNS_WITH_MISSING = [
    "height",
    "weight"
]

NUMERIC_COLUMNS = NUMERIC_COLUMNS_WITHOUT_MISSING + NUMERIC_COLUMNS_WITH_MISSING

BINARY_COLUMNS = [
    "received_vasopressor",
    "recieved_mechanical_ventilation",
    "received_rrt",
    "received_sedation",
    "received_antibiotic",
    "reached_icu"
]

STATIC_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS


def get_static_data(con: duckdb.DuckDBPyConnection, hadm_ids: List[int]) -> np.ndarray:
    logger.log_start("get_static_data")
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
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
    df = con.execute(STATIC_SQL).fetchdf()
    df.columns = df.columns.str.lower()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna('missing')
    static_data = df[STATIC_COLUMNS].values
    logger.log_end("get_static_data")
    return static_data
