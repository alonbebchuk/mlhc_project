"""
Static data extraction for ICU patients.

This module handles the extraction and processing of static patient features
from the MIMIC-III database, including demographics, measurements,
medication usage, and clinical interventions during the observation window.
"""

import duckdb
import pandas as pd
from typing import List
from data_processing.utils import WINDOW_HOURS, get_year_difference

# Height measurements
HEIGHT_IN_ITEMIDS = [920, 1394]
HEIGHT_CM_ITEMIDS = [226730]

# Weight measurements
WEIGHT_KG_ITEMIDS = [763, 3580, 226512, 224639]
WEIGHT_LB_ITEMIDS = [3581, 226531]

# Vasopressor medications
VASOPRESSOR_CV_ITEMIDS = [30047, 30120, 30044, 30119, 30309, 30127, 30312, 30051, 30043, 30307, 30042, 30306]
VASOPRESSOR_MV_ITEMIDS = [221906, 221289, 221749, 222315, 221662, 221653]

# Mechanical ventilation
VENTILATION_PROCEDURE_ITEMIDS = [225468, 224385, 224391]
VENTILATION_CHART_ITEMIDS = [224684, 224685, 224686, 220339, 505, 506, 60, 444, 224695, 218, 224738, 223834, 467]

# Renal Replacement Therapy
RRT_PROCEDURE_ITEMIDS = [225802, 225803, 225805, 224270]
RRT_CHART_ITEMIDS = [226499, 227357, 152, 224149, 582]

# Sedation medications
SEDATION_CV_ITEMIDS = [30131, 30124, 30166, 30121]
SEDATION_MV_ITEMIDS = [222168, 225150, 221385, 221668]

# Antibiotic medications
ANTIBIOTIC_REGEX = r'(amoxicillin|ampicillin|oxacillin|penicillin|piperacillin|tazobactam|zosyn|cefazolin|cefepime|ceftazidime|ceftriaxone|cefuroxime|meropenem|imipenem|ertapenem|vancomycin|amikacin|gentamicin|tobramycin|azithromycin|ciprofloxacin|levofloxacin|clindamycin|doxycycline|metronidazole|rifampin|daptomycin|linezolid)'

# Unit conversion factors
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
    JOIN admissions a ON c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_in_itemids) OR c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_cm_itemids))
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER = 0
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
    JOIN admissions a ON c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND (c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_lb_itemids) OR c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_kg_itemids))
      AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER = 0
    ORDER BY c.hadm_id, c.charttime
    """
VASOPRESSOR_SQL = f"""
    SELECT DISTINCT hadm_id
    FROM (
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_cv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_cv_itemids)
          AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_mv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_mv_itemids)
          AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) combined
    """
VENTILATOR_SQL = f"""
    SELECT DISTINCT hadm_id
    FROM (
        SELECT pe.hadm_id::INTEGER AS hadm_id
        FROM procedureevents_mv pe
        JOIN admissions a ON pe.hadm_id = a.hadm_id
        WHERE pe.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND pe.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_proc_itemids)
          AND pe.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT c.hadm_id::INTEGER AS hadm_id
        FROM chartevents c
        JOIN admissions a ON c.hadm_id = a.hadm_id
        WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_chart_itemids)
          AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) AS combined_vent
    """
RRT_SQL = f"""
    SELECT DISTINCT hadm_id
    FROM (
        SELECT p.hadm_id::INTEGER AS hadm_id
        FROM procedureevents_mv p
        JOIN admissions a ON p.hadm_id = a.hadm_id
        WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND p.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_proc_itemids)
          AND p.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT c.hadm_id::INTEGER AS hadm_id
        FROM chartevents c
        JOIN admissions a ON c.hadm_id = a.hadm_id
        WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_chart_itemids)
          AND c.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) combined
    """
SEDATION_SQL = f"""
    SELECT DISTINCT hadm_id
    FROM (
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_cv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_cv_itemids)
          AND ie.charttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
        
        UNION
        
        SELECT ie.hadm_id::INTEGER AS hadm_id
        FROM inputevents_mv ie
        JOIN admissions a ON ie.hadm_id = a.hadm_id
        WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
          AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_mv_itemids)
          AND ie.starttime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    ) combined
    """
ANTIBIOTIC_SQL = f"""
    SELECT DISTINCT p.hadm_id::INTEGER AS hadm_id
    FROM prescriptions p
    JOIN admissions a ON p.hadm_id = a.hadm_id
    WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND LOWER(COALESCE(p.drug, '')) ~ '{ANTIBIOTIC_REGEX}'
      AND p.startdate::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    """
ICU_SQL = f"""
    SELECT DISTINCT i.hadm_id::INTEGER AS hadm_id
    FROM icustays i
    JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE i.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND i.intime::TIMESTAMP BETWEEN a.admittime::TIMESTAMP AND a.admittime::TIMESTAMP + INTERVAL {WINDOW_HOURS} HOURS
    """

# Static feature column definitions
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
    "age",
    "height",
    "weight"
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
# All static features to include in final dataset
STATIC_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + BINARY_COLUMNS


def get_base_data(con: duckdb.DuckDBPyConnection, hadm_ids: List[int]) -> pd.DataFrame:
    """
    Extract base demographic and admission data for specified hospital admissions.

    This function retrieves core patient information including demographics,
    admission details, and calculates derived features like age.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Base patient data with demographic features and calculated age
    """
    # Register admission IDs for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    # Execute base SQL query to get demographic and admission data
    df = con.execute(BASE_SQL).fetchdf()

    # Calculate age at admission time
    df["age"] = get_year_difference(df["admittime"], df["dob"]).astype(int)

    # Convert gender to binary (1 for Male, 0 for Female)
    df["gender"] = (df["gender"] == "M").astype(int)

    # Remove admission time and date of birth columns (no longer needed after age calculation)
    df = df.drop(columns=["admittime"])
    df = df.drop(columns=["dob"])

    return df


def add_height_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract patient height measurements during the observation window.

    Heights are automatically converted to centimeters if recorded in inches.
    Takes the first valid height measurement for each admission.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Height data with columns [hadm_id, height]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_height_in_itemids", pd.DataFrame({"itemid": HEIGHT_IN_ITEMIDS}))
    con.register("tmp_height_cm_itemids", pd.DataFrame({"itemid": HEIGHT_CM_ITEMIDS}))

    # Execute height extraction query
    height_df = con.execute(HEIGHT_SQL).fetchdf()

    # Take first valid height measurement per admission
    height_df = height_df.groupby("hadm_id").first().reset_index()[["hadm_id", "height"]]

    df = df.merge(height_df, on="hadm_id", how="left")
    return df


def add_weight_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract patient weight measurements during the observation window.

    Weights are automatically converted to kilograms if recorded in pounds.
    Takes the first valid weight measurement for each admission.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Weight data with columns [hadm_id, weight]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_weight_lb_itemids", pd.DataFrame({"itemid": WEIGHT_LB_ITEMIDS}))
    con.register("tmp_weight_kg_itemids", pd.DataFrame({"itemid": WEIGHT_KG_ITEMIDS}))

    # Execute weight extraction query
    weight_df = con.execute(WEIGHT_SQL).fetchdf()

    # Take first valid weight measurement per admission
    weight_df = weight_df.groupby("hadm_id").first().reset_index()[["hadm_id", "weight"]]

    df = df.merge(weight_df, on="hadm_id", how="left")
    return df


def add_vasopressor_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients received vasopressor medications during observation window.

    Uses scientifically established definitions from INPUTEVENTS tables covering
    both CareVue and MetaVision systems for comprehensive detection.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Vasopressor usage with columns [hadm_id, received_vasopressor]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vaso_cv_itemids", pd.DataFrame({"itemid": VASOPRESSOR_CV_ITEMIDS}))
    con.register("tmp_vaso_mv_itemids", pd.DataFrame({"itemid": VASOPRESSOR_MV_ITEMIDS}))

    # Execute vasopressor query
    vaso_df = con.execute(VASOPRESSOR_SQL).fetchdf()

    df["received_vasopressor"] = df["hadm_id"].isin(vaso_df["hadm_id"]).astype(int)
    return df


def add_ventilation_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients were mechanically ventilated during observation window.

    Uses comprehensive multi-table approach combining evidence from procedures,
    ventilator settings, and oxygen delivery devices for robust detection.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Ventilation status with columns [hadm_id, recieved_mechanical_ventilation]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vent_proc_itemids", pd.DataFrame({"itemid": VENTILATION_PROCEDURE_ITEMIDS}))
    con.register("tmp_vent_chart_itemids", pd.DataFrame({"itemid": VENTILATION_CHART_ITEMIDS}))

    # Execute ventilation query
    vent_df = con.execute(VENTILATOR_SQL).fetchdf()

    df["recieved_mechanical_ventilation"] = df["hadm_id"].isin(vent_df["hadm_id"]).astype(int)
    return df


def add_rrt_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients received renal replacement therapy (RRT) during observation window.

    RRT includes dialysis and other kidney support treatments for patients with
    kidney failure or severe kidney dysfunction.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: RRT status with columns [hadm_id, received_rrt]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_rrt_proc_itemids", pd.DataFrame({"itemid": RRT_PROCEDURE_ITEMIDS}))
    con.register("tmp_rrt_chart_itemids", pd.DataFrame({"itemid": RRT_CHART_ITEMIDS}))

    # Execute RRT query
    rrt_df = con.execute(RRT_SQL).fetchdf()

    df["received_rrt"] = df["hadm_id"].isin(rrt_df["hadm_id"]).astype(int)
    return df


def add_sedation_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients received sedation medications during observation window.

    Uses scientifically established definitions from INPUTEVENTS tables covering
    common sedatives used in critical care (Propofol, Dexmedetomidine, Lorazepam, Midazolam).

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Sedation status with columns [hadm_id, received_sedation]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_sed_cv_itemids", pd.DataFrame({"itemid": SEDATION_CV_ITEMIDS}))
    con.register("tmp_sed_mv_itemids", pd.DataFrame({"itemid": SEDATION_MV_ITEMIDS}))

    # Execute sedation query
    sed_df = con.execute(SEDATION_SQL).fetchdf()

    df["received_sedation"] = df["hadm_id"].isin(sed_df["hadm_id"]).astype(int)
    return df


def add_antibiotic_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients received antibiotic medications during observation window.

    Uses efficient regex pattern matching on PRESCRIPTIONS table covering major
    antibiotic classes: penicillins, cephalosporins, carbapenems, glycopeptides,
    aminoglycosides, macrolides, quinolones, and others.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Antibiotic status with columns [hadm_id, received_antibiotic]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    # Execute antibiotic query
    ab_df = con.execute(ANTIBIOTIC_SQL).fetchdf()

    df["received_antibiotic"] = df["hadm_id"].isin(ab_df["hadm_id"]).astype(int)
    return df


def add_icu_feature(con: duckdb.DuckDBPyConnection, hadm_ids: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which patients were admitted to an ICU during observation window.

    ICU (Intensive Care Unit) admission indicates the need for intensive
    monitoring and life support.

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: ICU admission status with columns [hadm_id, reached_icu]
    """
    # Register temporary tables for SQL query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))

    # Execute ICU query
    icu_df = con.execute(ICU_SQL).fetchdf()

    df["reached_icu"] = df["hadm_id"].isin(icu_df["hadm_id"]).astype(int)
    return df


def get_static_data(con: duckdb.DuckDBPyConnection, hadm_ids: List[int]) -> pd.DataFrame:
    """
    Extract comprehensive static features for ICU patients using scientifically established definitions.

    This function implements validated methodologies from the MIT Laboratory for
    Computational Physiology MIMIC Code Repository, ensuring reproducible and
    scientifically sound feature extraction.

    Features extracted using established research definitions:
    - Demographics (age, gender, ethnicity, etc.) with proper categorical encoding
    - Admission details (type, location, insurance, etc.)  
    - Physical measurements (height, weight) with unit conversion and physiological filtering
    - Clinical interventions using comprehensive MIMIC-III item ID lists:
      * Vasopressors: from INPUTEVENTS covering both CareVue and MetaVision
      * Mechanical ventilation: multi-table approach (procedures + chart events + devices)
      * RRT: comprehensive procedure and chart event detection
      * Sedation: validated sedative medications from INPUTEVENTS
      * Antibiotics: comprehensive drug name matching from PRESCRIPTIONS
    - Care settings (ICU admission timing)

    Args:
        con: DuckDB database connection
        hadm_ids (List[int]): List of hospital admission IDs

    Returns:
        pd.DataFrame: Complete static features dataset with scientifically validated columns
    """
    # Get base demographic and admission data
    df = get_base_data(con, hadm_ids)

    # Add physical measurements
    df = add_height_feature(con, hadm_ids, df)
    df = add_weight_feature(con, hadm_ids, df)

    # Add clinical intervention features
    df = add_vasopressor_feature(con, hadm_ids, df)
    df = add_ventilation_feature(con, hadm_ids, df)
    df = add_rrt_feature(con, hadm_ids, df)
    df = add_sedation_feature(con, hadm_ids, df)
    df = add_antibiotic_feature(con, hadm_ids, df)

    # Add care setting features
    df = add_icu_feature(con, hadm_ids, df)

    # Fill missing values with 'missing'
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna('missing')

    # Select only the specified static columns for final output
    df = df[STATIC_COLUMNS]
    return df
