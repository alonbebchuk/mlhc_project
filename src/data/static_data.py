import pandas as pd
from typing import List
from data.common import SECONDS_PER_YEAR, WINDOW_HOURS, get_time_difference

HEIGHT_IN_ITEMIDS = [920, 1394, 4187, 3486, 226707]
HEIGHT_CM_ITEMIDS = [3485, 4188]
WEIGHT_LB_ITEMIDS = [3581, 226531]
WEIGHT_KG_ITEMIDS = [762, 763, 3723, 3580, 226512]
VASOPRESSOR_ITEMIDS = [221906, 30047, 30120, 221289, 30044, 30119, 30309,
                       221749, 30127, 30128, 221662, 30043, 30307, 222315, 30051, 30042, 30306]
SEDATION_ITEMIDS = [222168, 30131, 221668, 30124, 221744,
                    225972, 225942, 30150, 30308, 30118, 30149, 225150]
VENTILATION_ITEMIDS = [223849, 60, 437, 505, 506, 686, 220339, 224700,
                       639, 654, 681, 682, 683, 684, 224684, 224685, 224686, 467, 223848]
RRT_PROCEDURE_ITEMIDS = [225802, 225803, 225441]
RRT_CHART_ITEMIDS = [152]
ANTIBIOTIC_REGEX = r"(vancomycin|zosyn|piperacillin|tazobactam|cefepime|meropenem|levofloxacin|azithromycin|ceftriaxone|metronidazole)"

IN_TO_CM_FACTOR = 2.54
LB_TO_KG_FACTOR = 0.45359237


BASE_SQL = """
    SELECT a.hadm_id::INTEGER AS hadm_id,
           a.admittime::TIMESTAMP AS admittime,
           a.admission_type AS admission_type,
           a.admission_location AS admission_location,
           a.insurance AS insurance,
           a.language AS language,
           a.religion AS religion,
           a.marital_status AS marital_status,
           a.ethnicity AS ethnicity,
           p.gender AS gender,
           p.dob::TIMESTAMP AS dob
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id
    WHERE a.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
    """
HEIGHT_SQL = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id,
           CASE 
               WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_in_itemids) THEN c.valuenum * {IN_TO_CM_FACTOR}
               ELSE c.valuenum 
           END AS height,
           c.charttime::TIMESTAMP AS charttime
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_in_itemids) OR 
           c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_cm_itemids))
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER == 0
    ORDER BY c.hadm_id, c.charttime
    """
WEIGHT_SQL = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id,
           CASE 
               WHEN c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_lb_itemids) THEN c.valuenum * {LB_TO_KG_FACTOR}
               ELSE c.valuenum 
           END AS weight,
           c.charttime::TIMESTAMP AS charttime
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_kg_itemids) OR 
           c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_lb_itemids))
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER == 0
    ORDER BY c.hadm_id, c.charttime
    """
VASOPRESSOR_SQL = f"""
    SELECT DISTINCT hadm_id, 1 AS received_vasopressor
    FROM (
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_cv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
          AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_mv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
          AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) combined
    """
VENTILATOR_SQL = f"""
    SELECT DISTINCT c.hadm_id::INTEGER AS hadm_id, 1 AS was_mechanically_ventilated
    FROM chartevents c
    JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_itemids)
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.error::INTEGER == 0
      AND (c.itemid::INTEGER NOT IN (467, 223848) OR
           (c.itemid::INTEGER = 467 AND LOWER(COALESCE(c.value, '')) LIKE '%ventilator%') OR
           (c.itemid::INTEGER = 223848 AND LOWER(COALESCE(c.value, '')) NOT LIKE '%other%'))
    """
RRT_SQL = f"""
    SELECT DISTINCT hadm_id, 1 AS received_rrt
    FROM (
        SELECT p.hadm_id::INTEGER AS hadm_id
        FROM procedureevents_mv p
        JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
        WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND p.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_proc_itemids)
          AND p.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT c.hadm_id::INTEGER AS hadm_id
        FROM chartevents c
        JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
        WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_chart_itemids)
          AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
          AND c.error::INTEGER == 0
    ) combined
    """
SEDATION_SQL = f"""
    SELECT DISTINCT hadm_id, 1 AS received_sedation
    FROM (
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_cv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
          AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_mv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
          AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) combined
    """
ANTIBIOTIC_SQL = f"""
    SELECT DISTINCT p.hadm_id::INTEGER AS hadm_id, 1 AS received_antibiotic
    FROM prescriptions p
    JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
    WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND p.startdate::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND LOWER(COALESCE(p.drug, '')) ~ '{ANTIBIOTIC_REGEX}'
    """
ICU_SQL = f"""
    SELECT i.hadm_id::INTEGER AS hadm_id,
           MIN(i.intime)::TIMESTAMP AS first_icu_intime
    FROM icustays i
    JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE i.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND i.intime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    GROUP BY i.hadm_id
    """


def get_base_data(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Query basic admission and patient data."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    df = con.execute(BASE_SQL).fetchdf()

    df["age"] = (get_time_difference(df["admittime"],
                 df["dob"], SECONDS_PER_YEAR)).astype(int)

    df["gender"] = (df["gender"] == "M").astype(int)

    df = df.drop(columns=["dob"])

    return df


def add_height_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Add height feature (inches converted to cm)."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_height_in_itemids", pd.DataFrame(
        {"itemid": HEIGHT_IN_ITEMIDS}))
    con.register("tmp_height_cm_itemids", pd.DataFrame(
        {"itemid": HEIGHT_CM_ITEMIDS}))

    height_df = con.execute(HEIGHT_SQL).fetchdf()

    height_df = height_df.groupby("hadm_id").first().reset_index()[
        ["hadm_id", "height"]]

    return height_df


def add_weight_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Add weight feature (pounds converted to kg)."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_weight_lb_itemids", pd.DataFrame(
        {"itemid": WEIGHT_LB_ITEMIDS}))
    con.register("tmp_weight_kg_itemids", pd.DataFrame(
        {"itemid": WEIGHT_KG_ITEMIDS}))

    weight_df = con.execute(WEIGHT_SQL).fetchdf()

    weight_df = weight_df.groupby("hadm_id").first().reset_index()[
        ["hadm_id", "weight"]]

    return weight_df


def add_vasopressor_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Add received_vasopressor binary feature."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vaso_itemids", pd.DataFrame(
        {"itemid": VASOPRESSOR_ITEMIDS}))

    vaso_df = con.execute(VASOPRESSOR_SQL).fetchdf()

    return vaso_df


def add_ventilation_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Add was_mechanically_ventilated binary feature."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vent_itemids", pd.DataFrame(
        {"itemid": VENTILATION_ITEMIDS}))

    vent_df = con.execute(VENTILATOR_SQL).fetchdf()

    return vent_df


def add_rrt_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    """Add received_rrt binary feature."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_rrt_proc_itemids", pd.DataFrame(
        {"itemid": RRT_PROCEDURE_ITEMIDS}))
    con.register("tmp_rrt_chart_itemids", pd.DataFrame(
        {"itemid": RRT_CHART_ITEMIDS}))

    rrt_df = con.execute(RRT_SQL).fetchdf()

    return rrt_df


def add_sedation_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_sed_itemids", pd.DataFrame({"itemid": SEDATION_ITEMIDS}))

    sed_df = con.execute(SEDATION_SQL).fetchdf()

    return sed_df


def add_antibiotic_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    ab_df = con.execute(ANTIBIOTIC_SQL).fetchdf()

    return ab_df


def add_icu_feature(con, hadm_ids: List[int]) -> pd.DataFrame:
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    icu_df = con.execute(ICU_SQL).fetchdf()

    return icu_df


def get_static_features(con, hadm_ids: List[int]) -> pd.DataFrame:
    df = get_base_data(con, hadm_ids)

    df = df.merge(add_height_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_weight_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_vasopressor_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_ventilation_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_rrt_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_sedation_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_antibiotic_feature(con, hadm_ids), on="hadm_id", how="left")
    df = df.merge(add_icu_feature(con, hadm_ids), on="hadm_id", how="left")

    return df
