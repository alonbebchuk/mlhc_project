# from config import *
from config import WINDOW_HOURS


# ---- base pulls ----
ICUQ = f"""
    SELECT a.subject_id::INTEGER AS subject_id,
           a.hadm_id::INTEGER AS hadm_id,
           a.admittime::TIMESTAMP AS admittime,
           a.dischtime::TIMESTAMP AS dischtime,
           a.admission_type AS admission_type,
           a.admission_location AS admission_location,
           a.insurance AS insurance,
           a.language AS language,
           a.religion AS religion,
           a.marital_status AS marital_status,
           a.ethnicity AS ethnicity,
           a.edregtime::TIMESTAMP AS edregtime,
            --    a.diagnosis AS diagnosis,
           a.has_chartevents_data::INTEGER AS has_chartevents_data,
           p.gender AS gender,
           p.dob::TIMESTAMP AS dob,
           p.dod::TIMESTAMP AS dod
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id
    WHERE a.subject_id::INTEGER IN (SELECT subject_id FROM tmp_subject_ids)
    """

LABQUERY = f"""
    SELECT l.subject_id::INTEGER AS subject_id,
           l.hadm_id::INTEGER AS hadm_id,
           l.charttime::TIMESTAMP AS charttime,
           l.itemid::INTEGER AS itemid,
           l.valuenum::DOUBLE AS valuenum
    FROM labevents l
    JOIN admissions a ON l.subject_id = a.subject_id AND l.hadm_id = a.hadm_id
    WHERE l.itemid::INTEGER IN (SELECT itemid FROM tmp_lab_itemids)
      AND l.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND l.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND l.valuenum IS NOT NULL
    """

VITQUER = f"""
    SELECT c.subject_id::INTEGER AS subject_id,
           c.hadm_id::INTEGER AS hadm_id,
           c.charttime::TIMESTAMP AS charttime,
           c.itemid::INTEGER AS itemid,
           c.valuenum::DOUBLE AS valuenum
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.itemid::INTEGER IN (SELECT itemid FROM tmp_vital_itemids)
      AND c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER = 0
    """

ICU_INTIME = f"""
    SELECT i.hadm_id::INTEGER AS hadm_id,
           MIN(i.intime)::TIMESTAMP AS first_icu_intime
    FROM icustays i
    JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE i.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND i.intime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    GROUP BY i.hadm_id
    """

# ---- height / weight (first 48h) ----
HEIGHT_48H = f"""
SELECT c.hadm_id::INTEGER AS hadm_id,
       c.charttime::TIMESTAMP AS charttime,
       c.itemid::INTEGER AS itemid,
       c.valuenum::DOUBLE AS valuenum
FROM chartevents c
JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_itemids)
  AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
  AND c.valuenum IS NOT NULL
  AND c.error::INTEGER = 0
"""

WEIGHT_48H = f"""
SELECT c.hadm_id::INTEGER AS hadm_id,
       c.charttime::TIMESTAMP AS charttime,
       c.itemid::INTEGER AS itemid,
       c.valuenum::DOUBLE AS valuenum
FROM chartevents c
JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_itemids)
  AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
  AND c.valuenum IS NOT NULL
  AND c.error::INTEGER = 0
"""

# ---- treatments / exposures (first 48h) ----
# vasopressors
VASO_CV_48H = f"""
SELECT ie.hadm_id::INTEGER AS hadm_id
FROM inputevents_cv ie
JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
  AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""
VASO_MV_48H = f"""
SELECT ie.hadm_id::INTEGER AS hadm_id
FROM inputevents_mv ie
JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
  AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""

# sedation
SED_CV_48H = f"""
SELECT ie.hadm_id::INTEGER AS hadm_id
FROM inputevents_cv ie
JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
  AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""
SED_MV_48H = f"""
SELECT ie.hadm_id::INTEGER AS hadm_id
FROM inputevents_mv ie
JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
  AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""

# ventilation signal
VENT_48H = f"""
SELECT c.hadm_id::INTEGER AS hadm_id,
       c.itemid::INTEGER AS itemid,
       LOWER(COALESCE(c.value, '')) AS value
FROM chartevents c
JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_itemids)
  AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
  AND c.error::INTEGER = 0
"""

# RRT
RRT_PROC_48H = f"""
SELECT p.hadm_id::INTEGER AS hadm_id
FROM procedureevents_mv p
JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND p.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_proc_itemids)
  AND p.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""
RRT_CHART_48H = f"""
SELECT c.hadm_id::INTEGER AS hadm_id
FROM chartevents c
JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_chart_itemids)
  AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
  AND c.error::INTEGER = 0
"""

# antibiotics (drug text filtered in Python)
RX_ABX_48H = f"""
SELECT p.hadm_id::INTEGER AS hadm_id,
       LOWER(COALESCE(p.drug, '')) AS drug
FROM prescriptions p
JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND p.startdate::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""

# microbiology – positive blood cultures
MICRO_BLOOD_48H = f"""
SELECT m.hadm_id::INTEGER AS hadm_id
FROM microbiologyevents m
JOIN admissions a ON m.hadm_id = a.hadm_id AND m.subject_id = a.subject_id
WHERE m.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
  AND LOWER(COALESCE(m.spec_type_desc, '')) LIKE '%blood culture%'
  AND m.org_name IS NOT NULL
  AND m.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP
                                 AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
"""
