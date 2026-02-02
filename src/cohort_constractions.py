"""
cohort.py
Main file for building cohorts from MIMIC-III.
Provides utilities to extract cohorts, apply inclusion/exclusion criteria,
and compute early (first 48h) features and targets.
"""

# import duckdb
# import io
# import os
# import pickle
import pandas as pd
# from pathlib import Path
# from google.auth.transport.requests import Request
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
from typing import Dict, List, Tuple
from queries import *   # SQL queries stored in external file
from config import *    # Config constants (thresholds, enums, etc.)

# from queries import (
#     ICUQ, LABQUERY, VITQUER, ICU_INTIME,
#     HEIGHT_48H, WEIGHT_48H,
#     VASO_CV_48H, VASO_MV_48H,
#     SED_CV_48H, SED_MV_48H,
#     VENT_48H,
#     RRT_PROC_48H, RRT_CHART_48H,
#     RX_ABX_48H, MICRO_BLOOD_48H,
# )
# from config import (
#     WINDOW_HOURS, GAP_HOURS, MIN_LOS_HOURS, MIN_AGE, MAX_AGE,
#     PROLONGED_LOS_THRESHOLD_DAYS, READMISSION_WINDOW_DAYS, MORTALITY_WINDOW_DAYS,
#     SECONDS_PER_HOUR, SECONDS_PER_YEAR,
#     ADMISSION_TYPE, ADMISSION_LOCATION, INSURANCE, LANGUAGE, RELIGION, MARTIAL_STATUS, ETHNICITY
# )

# WINDOW_HOURS: int = 48
# GAP_HOURS: int = 6
# MIN_LOS_HOURS: int = 54
# MIN_AGE: int = 18
# MAX_AGE: int = 89

# PROLONGED_LOS_THRESHOLD_DAYS: int = 7
# READMISSION_WINDOW_DAYS: int = 30
# MORTALITY_WINDOW_DAYS: int = 30

# SECONDS_PER_HOUR = 60 * 60
# SECONDS_PER_YEAR = 365 * 24 * SECONDS_PER_HOUR


# SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# ---------- small helpers ----------

def _assert_required_columns(df: pd.DataFrame, cols: list[str], name: str):
    """
    Utility to assert that a DataFrame has required columns.
    Raises ValueError if any column is missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

# def parse_enum(series: pd.Series, values, other: str = "OTHER") -> pd.Series:
#     # Keep NA as <NA> using pandas StringDtype, then normalize
#     s = series.astype('string').str.upper()
#     s = s.fillna("")
#     return s.where(s.isin(values), other)

# def normalize_categorical_enums(df: pd.DataFrame) -> pd.DataFrame:
#     for col, vocab in [
#         ("admission_type", ADMISSION_TYPE),
#         ("admission_location", ADMISSION_LOCATION),
#         ("insurance", INSURANCE),
#         ("language", LANGUAGE),
#         ("religion", RELIGION),
#         ("marital_status", MARTIAL_STATUS),
#         ("ethnicity", ETHNICITY),
#     ]:
#         if col in df.columns:
#             df[col] = parse_enum(df[col], vocab)
#     return df


def parse_enum(series: pd.Series, values: List[str], other: str = "OTHER") -> pd.Series:
    """
    Normalize categorical values into a fixed vocabulary.
    Args:
        series: pandas Series with raw string values
        values: allowed list of strings
        other: label to assign to out-of-vocabulary entries
    Returns:
        Series of uppercase values, mapped to `values` or `other`
    """
    s = series.astype(str).str.upper().fillna("")
    return s.where(s.isin(values), other=other)


def normalize_categorical_enums(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalization to categorical fields (admission_type, insurance, etc.)
    using vocabularies defined in `config.py`.
    Args:
        df: admissions DataFrame with raw categorical fields
    Returns:
        DataFrame with normalized categorical columns
    """
    df["admission_type"] = parse_enum(df["admission_type"], ADMISSION_TYPE)
    df["admission_location"] = parse_enum(df["admission_location"], ADMISSION_LOCATION)
    df["insurance"] = parse_enum(df["insurance"], INSURANCE)
    df["language"] = parse_enum(df["language"], LANGUAGE)
    df["religion"] = parse_enum(df["religion"], RELIGION)
    df["marital_status"] = parse_enum(df["marital_status"], MARTIAL_STATUS)
    df["ethnicity"] = parse_enum(df["ethnicity"], ETHNICITY)
    return df


# ---------- IO loaders ----------

def load_initial_subjects(path: str) -> List[int]:
    """
    Load initial cohort subject IDs from CSV file.
    Args:
        path: Path to CSV containing a `subject_id` column.
    Returns:
        List of subject_id integers.
    """
    df = pd.read_csv(path, dtype={"subject_id": "int64"})
    # _assert_required_columns(df, ["subject_id"], "initial_cohort_csv")
    subject_ids = df["subject_id"].tolist()
    return subject_ids


def load_metadata(meta_path: str) -> pd.DataFrame:
    """
    Load itemid metadata (min/max valid value ranges) from CSV.
    Args:
        meta_path: Path to metadata CSV with columns [itemid, min, max].
    Returns:
        DataFrame with itemid (int), min (float), max (float).
    """
    # df = pd.read_csv(meta_path, usecols=["itemid", "min", "max"])
    # _assert_required_columns(df, ["itemid","min","max"], "metadata_csv")
    # return df.astype({"itemid": "int64", "min": "float64", "max": "float64"})
    df = pd.read_csv(
        meta_path,
        usecols=["itemid", "min", "max"],
        dtype={"itemid": "int64", "min": "float64", "max": "float64"},
    )
    return df


# ---------- Base queries ----------

def query_base_admissions(con, subject_ids: List[int]) -> pd.DataFrame:
    """
    Query base admission data for given subject_ids. (minimal base without ICU join)
    Args:
        con: duckdb connection
        subject_ids: list of subject IDs
    Returns:
        DataFrame of admissions and patient-level info.
    """
    # Register subject IDs in DuckDB memory table
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))
    df = con.execute(ICUQ).fetchdf()
    return df


def query_labs_48h(con, hadm_ids: List[int], labs_meta_csv: str) -> pd.DataFrame:
    """
    Query lab events in first 48h and filter out-of-range values.
    Args:
        con: duckdb connection
        hadm_ids: list of admission IDs
        labs_meta_csv: metadata CSV with valid ranges
    Returns:
        Filtered DataFrame with lab values within valid ranges.
    """
    meta = load_metadata(labs_meta_csv)
    itemids = meta["itemid"].tolist()

    # Register allowed itemids and admissions
    con.register("tmp_lab_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    df = con.execute(LABQUERY).fetchdf()

    # Join with metadata and filter valid ranges
    df = df.merge(meta, on="itemid")
    mask = (df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])
    df = df.loc[mask, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]].reset_index(drop=True)
    return df


def query_vitals_48h(con, hadm_ids: List[int], vitals_meta_csv: str) -> pd.DataFrame:
    """
    Query vital signs in first 48h and filter out-of-range values.
    Args:
        con: duckdb connection
        hadm_ids: list of admission IDs
        vitals_meta_csv: metadata CSV with valid ranges
    Returns:
        Filtered DataFrame with vital sign values within valid ranges.
    """
    meta = load_metadata(vitals_meta_csv)
    itemids = meta["itemid"].tolist()

    con.register("tmp_vital_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    df = con.execute(VITQUER).fetchdf()

    # Join with metadata and keep only values within min/max range
    df = df.merge(meta, on="itemid")
    mask = (df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])
    df = df.loc[mask, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]].reset_index(drop=True)
    return df


# ---------- Cohort construction & targets ----------

def create_cohort_and_targets(adm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply cohort inclusion/exclusion criteria, keep first admission,
    and derive prediction targets (mortality, prolonged LOS, readmission).
    Args:
        adm_df: admissions DataFrame from query_base_admissions()
    Returns:
        cohort: DataFrame of cleaned admissions with demographic/categorical features
        targets: DataFrame with labels (mortality, prolonged_los, readmission_30d)
    """
    # Track size of dataset
    print(f"  Starting with {len(adm_df)} total admissions")

    # Sort by subject, then admission time; add "next admission" for readmission label
    df_all = adm_df.sort_values(["subject_id", "admittime"]).copy()
    df_all["next_admittime"] = df_all.groupby("subject_id")["admittime"].shift(-1)

    # Keep only first admission per subject
    df = df_all.groupby("subject_id", as_index=False).head(1).copy()
    print(f"  After keeping first admission per subject: {len(df)} admissions")

    # Compute derived features
    df["admission_age"] = (df["admittime"] - df["dob"]).dt.total_seconds() / SECONDS_PER_YEAR
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR
    df["died_before_min_window"] = df["dod"].notna() & (
        (df["dod"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR < MIN_LOS_HOURS)

    # Apply inclusion/exclusion rules
    age_ok = df["admission_age"].between(MIN_AGE, MAX_AGE, inclusive="both")
    los_ok = df["los_hours"] >= MIN_LOS_HOURS
    chartevents_ok = df["has_chartevents_data"] == 1
    not_died_early = ~df["died_before_min_window"]

    print(f"  Age criteria ({MIN_AGE}-{MAX_AGE}): {age_ok.sum()}/{len(df)} patients")
    print(f"  LOS criteria (>={MIN_LOS_HOURS}h): {los_ok.sum()}/{len(df)} patients")
    print(f"  Has chartevents data: {chartevents_ok.sum()}/{len(df)} patients")
    print(f"  Did not die before {MIN_LOS_HOURS}h: {not_died_early.sum()}/{len(df)} patients")

    # Final keep mask
    keep = age_ok & los_ok & chartevents_ok & not_died_early
    print(f"  Final cohort after all criteria: {keep.sum()}/{len(df)} patients")

    excluded = df.loc[~keep]
    if len(excluded) > 0:
        print(f"  Excluded {len(excluded)} patients:")

        age_excluded = ~(excluded['admission_age'].between(MIN_AGE, MAX_AGE, inclusive="both"))
        los_excluded = ~(excluded['los_hours'] >= MIN_LOS_HOURS)
        chartevents_excluded = ~(excluded['has_chartevents_data'] == 1)
        died_early_excluded = excluded['died_before_min_window']

        print(f"    - Age not {MIN_AGE}-{MAX_AGE}: {age_excluded.sum()}")
        print(f"    - LOS < {MIN_LOS_HOURS}h: {los_excluded.sum()}")
        print(f"    - No chartevents data: {chartevents_excluded.sum()}")
        print(f"    - Died before {MIN_LOS_HOURS}h: {died_early_excluded.sum()}")

    df = df.loc[keep].reset_index(drop=True)  # Apply mask

    # Define prediction targets
    df["mortality"] = (
        df["dod"].notna() & 
        (df["dod"] <= (df["dischtime"] + pd.Timedelta(days=MORTALITY_WINDOW_DAYS)))
        ).astype(int)
    df["prolonged_los"] = ((df["dischtime"] - df["admittime"]) > pd.Timedelta(days=PROLONGED_LOS_THRESHOLD_DAYS)).astype(int)
    df["readmission_30d"] = (
        df["next_admittime"].notna() & 
        (df["next_admittime"] <= (df["dischtime"] + pd.Timedelta(days=READMISSION_WINDOW_DAYS)))
        ).astype(int)

    # Normalize categorical fields (insurance, ethnicity, etc.)
    # Normalize categorical enums after filtering/labeling
    df = normalize_categorical_enums(df)

    # Split into output tables
    cohort = df[[
        "subject_id","hadm_id","admittime","admission_type","admission_location",
        "insurance","language","religion","marital_status","ethnicity","edregtime",
        "gender","admission_age"
    ]]
    targets = df[["subject_id","hadm_id","mortality","prolonged_los","readmission_30d"]]
    # cohort = df[[
    #     "subject_id",
    #     "hadm_id",
    #     "admittime",
    #     "admission_type",
    #     "admission_location",
    #     "insurance",
    #     "language",
    #     "religion",
    #     "marital_status",
    #     "ethnicity",
    #     "edregtime",
    #     # "diagnosis",
    #     "gender",
    #     "admission_age"
    # ]]

    return cohort, targets


# ---------- Feature add-ons (first 48h) ----------

def add_first_icu_intime(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add earliest ICU intime within WINDOW_HOURS (48h) of admission for each hadm_id.
    Returns:
        cohort_df with an added column: first_icu_intime (nullable).
    """
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    icu = con.execute(ICU_INTIME).fetchdf()
    cohort_df = cohort_df.merge(icu, on="hadm_id", how="left")
    return cohort_df


def add_first_height(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first recorded height (converted to cm) within WINDOW_HOURS (48h).
    Uses HEIGHT_48H to fetch events; handles both inch and cm itemids.
    Keeps the earliest charttime per hadm_id.
    """
    # Chartevents itemids for height (inches vs centimeters)
    height_in_itemids = [920, 1394, 4187, 3486, 226707]
    height_cm_itemids = [3485, 4188]
    height_itemids = height_in_itemids + height_cm_itemids

    # Register limits and execute query
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_height_itemids", pd.DataFrame({"itemid": height_itemids}))
    h = con.execute(HEIGHT_48H).fetchdf()

    # Convert to centimeters where needed and take the first available value per admission
    h = h.copy()
    h["height_cm"] = h["valuenum"].astype(float)
    inch_mask = h["itemid"].isin(height_in_itemids)
    h.loc[inch_mask, "height_cm"] = h.loc[inch_mask, "height_cm"] * 2.54
    h = h.sort_values(["hadm_id", "charttime"]).groupby("hadm_id", as_index=False).head(1)[["hadm_id", "height_cm"]]

    cohort_df = cohort_df.merge(h, on="hadm_id", how="left")
    return cohort_df


def add_first_weight(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first recorded weight (converted to kg) within WINDOW_HOURS (48h).
    Uses WEIGHT_48H; supports kg and lb itemids; keeps the earliest value.
    """
    weight_kg_itemids = [762, 763, 3723, 3580, 226512]
    weight_lb_itemids = [3581, 226531]
    weight_itemids = weight_kg_itemids + weight_lb_itemids

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_weight_itemids", pd.DataFrame({"itemid": weight_itemids}))
    
    w = con.execute(WEIGHT_48H).fetchdf()

    w = w.copy()
    w["weight_kg"] = w["valuenum"].astype(float)
    lb_mask = w["itemid"].isin(weight_lb_itemids)
    w.loc[lb_mask, "weight_kg"] = w.loc[lb_mask, "weight_kg"] * 0.45359237
    w = w.sort_values(["hadm_id", "charttime"]).groupby("hadm_id", as_index=False).head(1)[["hadm_id", "weight_kg"]]

    cohort_df = cohort_df.merge(w, on="hadm_id", how="left")
    return cohort_df


def add_received_vasopressor_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions that received any vasopressor in first WINDOW_HOURS (48h).
    Pulls from both inputevents_cv and inputevents_mv using VASO_CV_48H / VASO_MV_48H.
    """
    vaso_itemids = [221906, 30047, 30120, 221289, 30044, 30119, 30309, 221749,
                    30127, 30128, 221662, 30043, 30307, 222315, 30051, 30042, 30306]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vaso_itemids", pd.DataFrame({"itemid": vaso_itemids}))
    cv = con.execute(VASO_CV_48H).fetchdf()
    mv = con.execute(VASO_MV_48H).fetchdf()
    both = pd.concat([cv, mv], ignore_index=True)

    # If any row exists for a hadm_id -> flag = 1
    flag = both.groupby("hadm_id").size().gt(0).astype(int).rename("received_vasopressor").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_vasopressor"] = cohort_df["received_vasopressor"].fillna(0).astype(int)
    return cohort_df


def add_received_sedation_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions that received any sedative in first WINDOW_HOURS (48h).
    Looks in inputevents_cv and inputevents_mv using SED_CV_48H / SED_MV_48H.
    """
    sed_itemids = [222168, 30131, 221668, 30124, 221744, 225972, 225942, 30150,
                   30308, 30118, 30149, 225150]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_sed_itemids", pd.DataFrame({"itemid": sed_itemids}))
    cv = con.execute(SED_CV_48H).fetchdf()
    mv = con.execute(SED_MV_48H).fetchdf()
    both = pd.concat([cv, mv], ignore_index=True)

    flag = both.groupby("hadm_id").size().gt(0).astype(int).rename("received_sedation").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_sedation"] = cohort_df["received_sedation"].fillna(0).astype(int)
    return cohort_df


def add_was_mechanically_ventilated_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions with evidence of mechanical ventilation in first WINDOW_HOURS (48h).
    Looks in CHARTEVENTS.
    Heuristics:
      - Presence of ventilator settings (e.g., PEEP, tidal volume, 223849),
      - OR oxygen device charted as a ventilator (exclude vague "other").
    """
    vent_setting_itemids = [223849]
    peep_itemids = [60, 437, 505, 506, 686, 220339, 224700]
    tv_itemids = [639, 654, 681, 682, 683, 684, 224684, 224685, 224686]
    oxygen_device_itemids = [467, 223848]
    vent_itemids = vent_setting_itemids + peep_itemids + tv_itemids + oxygen_device_itemids

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vent_itemids", pd.DataFrame({"itemid": vent_itemids}))
    vent = con.execute(VENT_48H).fetchdf()

    # Any chart of settings implies ventilation
    has_setting = vent[vent["itemid"].isin(vent_setting_itemids + peep_itemids + tv_itemids)]
    # oxygen device codes: look for explicit ventilator indications
    ox = vent[vent["itemid"].isin(oxygen_device_itemids)]
    ox_flag = (
        (ox["itemid"] == 467) & (ox["value"].str.contains("ventilator", na=False))
    ) | (
        (ox["itemid"] == 223848) & (~ox["value"].str.contains("other", na=False))
    )
    ox = ox.loc[ox_flag]
    # Union both signals, then flag per hadm_id
    vent_agg = pd.concat([has_setting[["hadm_id"]], ox[["hadm_id"]]], ignore_index=True)
    flag = vent_agg.groupby("hadm_id").size().gt(0).astype(int).rename("was_mechanically_ventilated").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["was_mechanically_ventilated"] = cohort_df["was_mechanically_ventilated"].fillna(0).astype(int)
    return cohort_df


def add_received_rrt_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions that received renal replacement therapy (RRT) in first WINDOW_HOURS (48h).
    Sources:
      - procedureevents_mv with RRT itemids
      - chartevents for an RRT indicator itemid
    """
    rrt_proc_itemids = [225802, 225803, 225441]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_rrt_proc_itemids", pd.DataFrame({"itemid": rrt_proc_itemids}))
    con.register("tmp_rrt_chart_itemids", pd.DataFrame({"itemid": [152]}))
    rrt_proc = con.execute(RRT_PROC_48H).fetchdf()
    rrt_chart = con.execute(RRT_CHART_48H).fetchdf()
    rrt = pd.concat([rrt_proc, rrt_chart], ignore_index=True)

    flag = rrt.groupby("hadm_id").size().gt(0).astype(int).rename("received_rrt").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_rrt"] = cohort_df["received_rrt"].fillna(0).astype(int)
    return cohort_df


def add_received_antibiotic_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions that received a broad antibiotic in first WINDOW_HOURS (48h).
    Looks in PRESCRIPTIONS.
    - Executes RX_ABX_48H to get prescription text.
    - Uses a simple keyword regex for common broad agents.
    """
    ab_keywords = [
        "vancomycin", "zosyn", "piperacillin", "tazobactam",
        "cefepime", "meropenem", "levofloxacin", "azithromycin",
        "ceftriaxone", "metronidazole"
    ]
    pattern = "|".join(ab_keywords)

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    rx = con.execute(RX_ABX_48H).fetchdf()

    # Case-insensitive search in drug names
    drugs = rx["drug"].astype(str)
    rx = rx.loc[drugs.str.contains(pattern, case=False, regex=True)]
    flag = rx.groupby("hadm_id").size().gt(0).astype(int).rename("received_antibiotic").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_antibiotic"] = cohort_df["received_antibiotic"].fillna(0).astype(int)
    return cohort_df


def add_positive_blood_culture_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag (0/1) admissions with a positive blood culture in first WINDOW_HOURS (48h).
    MICRO_BLOOD_48H already looks in MICROBIOLOGYEVENTS and:
      - filters to blood-culture-like specimens,
      - requires a non-null organism name,
      - applies the time window relative to admission.
    """
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    micro = con.execute(MICRO_BLOOD_48H).fetchdf()

    flag = micro.groupby("hadm_id").size().gt(0).astype(int).rename("positive_blood_culture").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["positive_blood_culture"] = cohort_df["positive_blood_culture"].fillna(0).astype(int)
    return cohort_df


# ---------- Orchestrator ----------

def extract_raw(con, initial_cohort_csv: str, labs_csv: str, vitals_csv: str) -> Dict[str, pd.DataFrame]:
    """
    End-to-end extraction pipeline. Steps:
        1) Load initial subjects from CSV.
        2) Query base admissions/patients for these subjects.
        3) Build first-admission cohort + label targets.
        4) Add first-48h features (ICU intime, height/weight, treatments, ventilation, RRT, micro).
        5) Pull 48h labs and vitals (filtered by metadata bounds).
        6) Return all four DataFrames in a dict.
    Args:
        con: duckdb connection
        initial_cohort_csv: CSV path with subject_id list
        labs_csv: path to lab metadata CSV
        vitals_csv: path to vital metadata CSV
    Returns:
        Dict with:
          - "cohort": admissions and features
          - "labs": lab events (48h)
          - "vitals": vital signs (48h)
          - "targets": prediction labels
    """
    # Load subject IDs
    subject_ids = load_initial_subjects(initial_cohort_csv)
    # print(f"Loaded {len(subject_ids)} subject IDs from initial cohort: {subject_ids}")
    print(f"Loaded {len(subject_ids)} subject IDs from initial cohort")

    # Query admissions for these subjects. Admissions/patient base table
    base = query_base_admissions(con, subject_ids)
    print(f"Found {len(base)} admissions for these subjects")

    # Apply inclusion/exclusion and build labels. First-admission cohort + targets
    cohort, targets = create_cohort_and_targets(base)
    print(f"After inclusion/exclusion criteria: {len(cohort)} patients remaining")
    hadm_ids = cohort["hadm_id"].tolist()

    # Enrich with 48h clinical features. Early feature augmentation (first 48h window)
    cohort = add_first_icu_intime(con, hadm_ids, cohort)
    cohort = add_first_height(con, hadm_ids, cohort)
    cohort = add_first_weight(con, hadm_ids, cohort)
    cohort = add_received_vasopressor_flag(con, hadm_ids, cohort)
    cohort = add_received_sedation_flag(con, hadm_ids, cohort)
    cohort = add_received_antibiotic_flag(con, hadm_ids, cohort)
    cohort = add_was_mechanically_ventilated_flag(con, hadm_ids, cohort)
    cohort = add_received_rrt_flag(con, hadm_ids, cohort)
    cohort = add_positive_blood_culture_flag(con, hadm_ids, cohort)
    cohort = normalize_categorical_enums(cohort) # Normalize categorical fields again in case any were added/merged

    # Query event-level data for labs and vitals
    labs = query_labs_48h(con, hadm_ids, labs_csv)
    vitals = query_vitals_48h(con, hadm_ids, vitals_csv)

    return {"cohort": cohort, "labs": labs, "vitals": vitals, "targets": targets}