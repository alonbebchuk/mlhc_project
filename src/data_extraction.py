import pandas as pd
from typing import Dict, List, Tuple


WINDOW_HOURS: int = 48
GAP_HOURS: int = 6
MIN_LOS_HOURS: int = 54
MIN_AGE: int = 18
MAX_AGE: int = 89

PROLONGED_LOS_THRESHOLD_DAYS: int = 7
READMISSION_WINDOW_DAYS: int = 30
MORTALITY_WINDOW_DAYS: int = 30

SECONDS_PER_HOUR = 60 * 60
SECONDS_PER_YEAR = 365 * 24 * SECONDS_PER_HOUR


def load_initial_subjects(path: str) -> List[int]:
    """Load the initial cohort subject IDs from CSV."""
    df = pd.read_csv(path, dtype={"subject_id": "int64"})
    subject_ids = df["subject_id"].tolist()
    return subject_ids


def load_metadata(meta_path: str) -> pd.DataFrame:
    """Load itemids with min/max bounds from a metadata CSV."""
    df = pd.read_csv(
        meta_path,
        usecols=["itemid", "min", "max"],
        dtype={"itemid": "int64", "min": "float64", "max": "float64"},
    )
    return df


def query_base_admissions(con, subject_ids: List[int]) -> pd.DataFrame:
    """Query admissions and patients (minimal base without ICU join)."""
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))
    sql = f"""
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
    df = con.execute(sql).fetchdf()
    return df


def parse_enum(series: pd.Series, values: List[str], other: str = "OTHER") -> pd.Series:
    s = series.astype(str).str.upper().fillna("")
    return s.where(s.isin(values), other=other)


def normalize_categorical_enums(df: pd.DataFrame) -> pd.DataFrame:
    admission_type_values = {
        "EMERGENCY",
        "URGENT",
        "ELECTIVE",
    }
    admission_location_values = {
        "EMERGENCY ROOM ADMIT",
        "TRANSFER FROM HOSP/EXTRAM",
        "TRANSFER FROM OTHER HEALT",
        "CLINIC REFERRAL/PREMATURE",
        "** INFO NOT AVAILABLE **",
        "TRANSFER FROM SKILLED NUR",
        "TRSF WITHIN THIS FACILITY",
        "HMO REFERRAL/SICK",
        "PHYS REFERRAL/NORMAL DELI",
    }
    insurance_values = {
        "MEDICARE",
        "PRIVATE",
        "MEDICAID",
        "GOVERNMENT",
        "SELF_PAY",
    }
    language_values = {
        "ENGL",
        "SPAN",
        "RUSS",
        "CANT",
        "PORT",
        "MAND",
        "HAIT",
        "FREN",
        "GREE",
        "ITAL",
        "CAPE",
        "VIET",
        "ARAB",
    }
    religion_values = {
        "CATHOLIC",
        "NOT_SPECIFIED",
        "UNOBTAINABLE",
        "PROTESTANT_QUAKER",
        "JEWISH",
        "OTHER",
        "CHRISTIAN_SCIENTIST",
        "BUDDHIST",
        "MUSLIM",
        "JEHOVAHS_WITNESS",
        "GREEK_ORTHODOX",
        "HINDU",
        "UNITARIAN_UNIVERSALIST",
        "SEVENTH_DAY_ADVENTIST",
        "ROMANIAN_EAST_ORTH",
        "BAPTIST",
        "EPISCOPALIAN",
        "LUTHERAN",
        "METHODIST",
        "HEBREW",
    }
    marital_status_values = {
        "MARRIED",
        "SINGLE",
        "WIDOWED",
        "DIVORCED",
        "SEPARATED",
        "UNKNOWN_NOT_SPECIFIED",
        "LIVING_WITH_PARTNER",
    }
    ethnicity_values = {
        "WHITE",
        "BLACK_AFRICAN_AMERICAN",
        "UNKNOWN_NOT_SPECIFIED",
        "HISPANIC_OR_LATINO",
        "ASIAN",
        "UNABLE_TO_OBTAIN",
        "PATIENT_DECLINED_TO_ANSWER",
        "OTHER",
        "ASIAN_CHINESE",
        "HISPANIC_LATINO_PUERTO_RICAN",
        "BLACK_CAPE_VERDEAN",
        "WHITE_RUSSIAN",
        "MULTI_RACE_ETHNICITY",
        "BLACK_HAITIAN",
        "ASIAN_VIETNAMESE",
        "ASIAN_CAMBODIAN",
        "WHITE_EASTERN_EUROPEAN",
        "ASIAN_FILIPINO",
        "HISPANIC_LATINO_DOMINICAN",
        "WHITE_OTHER_EUROPEAN",
        "PORTUGUESE",
        "BLACK_AFRICAN",
        "MIDDLE_EASTERN",
        "ASIAN_INDIAN",
        "HISPANIC_LATINO_GUATEMALAN",
        "ASIAN_ASIAN_INDIAN",
        "HISPANIC_LATINO_CUBAN",
        "AMERICAN_INDIAN_ALASKA_NATIVE",
        "HISPANIC_LATINO_SALVADORAN",
        "NATIVE_HAWAIIAN_OR_OTHER_PACIFIC_ISLANDER",
        "ASIAN_KOREAN",
        "CARIBBEAN_ISLAND",
        "SOUTH_AMERICAN",
    }

    df["admission_type"] = parse_enum(df["admission_type"], admission_type_values)
    df["admission_location"] = parse_enum(df["admission_location"], admission_location_values)
    df["insurance"] = parse_enum(df["insurance"], insurance_values)
    df["language"] = parse_enum(df["language"], language_values)
    df["religion"] = parse_enum(df["religion"], religion_values)
    df["marital_status"] = parse_enum(df["marital_status"], marital_status_values)
    df["ethnicity"] = parse_enum(df["ethnicity"], ethnicity_values)
    return df


def create_cohort_and_targets(adm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply inclusion/exclusion criteria, keep first admission per subject, and label targets."""
    df_all = adm_df.sort_values(["subject_id", "admittime"]).copy()
    df_all["next_admittime"] = df_all.groupby("subject_id")["admittime"].shift(-1)

    df = df_all.groupby("subject_id", as_index=False).head(1).copy()
    df["admission_age"] = (df["admittime"] - df["dob"]).dt.total_seconds() / SECONDS_PER_YEAR
    df["los_hours"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR
    df["died_before_min_window"] = df["dod"].notna() & ((df["dod"] - df["admittime"]).dt.total_seconds() / SECONDS_PER_HOUR < MIN_LOS_HOURS)

    keep = (
        (df["admission_age"].between(MIN_AGE, MAX_AGE, inclusive="both"))
        & (df["los_hours"] >= MIN_LOS_HOURS)
        & (df["has_chartevents_data"] == 1)
        & (~df["died_before_min_window"])
    )
    df = df.loc[keep].reset_index(drop=True)

    df["mortality"] = (df["dod"].notna() & (df["dod"] <= (df["dischtime"] + pd.Timedelta(days=MORTALITY_WINDOW_DAYS)))).astype(int)
    df["prolonged_los"] = ((df["dischtime"] - df["admittime"]) > pd.Timedelta(days=PROLONGED_LOS_THRESHOLD_DAYS)).astype(int)
    df["readmission_30d"] = (df["next_admittime"].notna() & (df["next_admittime"] <= (df["dischtime"] + pd.Timedelta(days=READMISSION_WINDOW_DAYS)))).astype(int)

    df = normalize_categorical_enums(df)

    cohort = df[[
        "subject_id",
        "hadm_id",
        "admittime",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "religion",
        "marital_status",
        "ethnicity",
        "edregtime",
        # "diagnosis",
        "gender",
        "admission_age"
    ]]
    targets = df[[
        "subject_id",
        "hadm_id",
        "mortality",
        "prolonged_los",
        "readmission_30d"
    ]]

    return cohort, targets


def add_first_icu_intime(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add first ICU intime within 48h to cohort."""
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql = f"""
    SELECT i.hadm_id::INTEGER AS hadm_id,
           MIN(i.intime)::TIMESTAMP AS first_icu_intime
    FROM icustays i
    JOIN admissions a ON i.hadm_id = a.hadm_id
    WHERE i.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND i.intime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    GROUP BY i.hadm_id
    """
    icu = con.execute(sql).fetchdf()

    cohort_df = cohort_df.merge(icu, on="hadm_id", how="left")
    return cohort_df


def add_first_height(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add first recorded height (cm) within 48h to cohort."""
    height_in_itemids = [920, 1394, 4187, 3486, 226707]
    height_cm_itemids = [3485, 4188]
    height_itemids = height_in_itemids + height_cm_itemids

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_height_itemids", pd.DataFrame({"itemid": height_itemids}))
    sql_h = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id,
           c.charttime::TIMESTAMP AS charttime,
           c.itemid::INTEGER AS itemid,
           c.valuenum::DOUBLE AS valuenum
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_height_itemids)
      AND c.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER == 0
    """
    h = con.execute(sql_h).fetchdf()

    h["height_cm"] = h.apply(
        lambda r: float(r["valuenum"]) * 2.54 if int(r["itemid"]) in height_in_itemids else float(r["valuenum"]),
        axis=1,
    )
    h = h.sort_values(["hadm_id", "charttime"]).groupby("hadm_id", as_index=False).head(1)[["hadm_id", "height_cm"]]

    cohort_df = cohort_df.merge(h, on="hadm_id", how="left")
    return cohort_df


def add_first_weight(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add first recorded weight (kg) within 48h to cohort."""
    weight_kg_itemids = [762, 763, 3723, 3580, 226512]
    weight_lb_itemids = [3581, 226531]
    weight_itemids = weight_kg_itemids + weight_lb_itemids

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_weight_itemids", pd.DataFrame({"itemid": weight_itemids}))
    sql_w = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id,
           c.charttime::TIMESTAMP AS charttime,
           c.itemid::INTEGER AS itemid,
           c.valuenum::DOUBLE AS valuenum
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_weight_itemids)
      AND c.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER == 0
    """
    w = con.execute(sql_w).fetchdf()

    w["weight_kg"] = w.apply(
        lambda r: float(r["valuenum"]) * 0.45359237 if int(r["itemid"]) in weight_lb_itemids else float(r["valuenum"]),
        axis=1,
    )
    w = w.sort_values(["hadm_id", "charttime"]).groupby("hadm_id", as_index=False).head(1)[["hadm_id", "weight_kg"]]

    cohort_df = cohort_df.merge(w, on="hadm_id", how="left")
    return cohort_df


def add_received_vasopressor_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'received_vasopressor' based on INPUTEVENTS within 48h of admission."""
    vaso_itemids = [221906, 30047, 30120, 221289, 30044, 30119, 30309, 221749, 30127, 30128, 221662, 30043, 30307, 222315, 30051, 30042, 30306]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vaso_itemids", pd.DataFrame({"itemid": vaso_itemids}))
    sql_cv = f"""
    SELECT ie.hadm_id::INTEGER AS hadm_id
    FROM inputevents_cv ie
    JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
    WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
      AND ie.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    sql_mv = f"""
    SELECT ie.hadm_id::INTEGER AS hadm_id
    FROM inputevents_mv ie
    JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
    WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_vaso_itemids)
      AND ie.starttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    cv = con.execute(sql_cv).fetchdf()
    mv = con.execute(sql_mv).fetchdf()
    both = pd.concat([cv, mv], ignore_index=True)

    flag = both.groupby("hadm_id").size().gt(0).astype(int).rename("received_vasopressor").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_vasopressor"] = cohort_df["received_vasopressor"].fillna(0).astype(int)
    return cohort_df


def add_received_sedation_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'received_sedation' based on INPUTEVENTS within 48h of admission."""
    sed_itemids = [222168, 30131, 221668, 30124, 221744, 225972, 225942, 30150, 30308, 30118, 30149, 225150]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_sed_itemids", pd.DataFrame({"itemid": sed_itemids}))
    sql_cv = f"""
    SELECT ie.hadm_id::INTEGER AS hadm_id
    FROM inputevents_cv ie
    JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
    WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
      AND ie.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    sql_mv = f"""
    SELECT ie.hadm_id::INTEGER AS hadm_id
    FROM inputevents_mv ie
    JOIN admissions a ON ie.hadm_id = a.hadm_id AND ie.subject_id = a.subject_id
    WHERE ie.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND ie.itemid::INTEGER IN (SELECT itemid FROM tmp_sed_itemids)
      AND ie.starttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    cv = con.execute(sql_cv).fetchdf()
    mv = con.execute(sql_mv).fetchdf()
    both = pd.concat([cv, mv], ignore_index=True)

    flag = both.groupby("hadm_id").size().gt(0).astype(int).rename("received_sedation").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_sedation"] = cohort_df["received_sedation"].fillna(0).astype(int)
    return cohort_df


def add_received_antibiotic_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'received_antibiotic' based on PRESCRIPTIONS within 48h of admission."""
    ab_keywords = ["vancomycin", "zosyn", "piperacillin", "tazobactam", "cefepime", "meropenem", "levofloxacin", "azithromycin", "ceftriaxone", "metronidazole"]
    pattern = "|".join(ab_keywords)

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql_rx = f"""
    SELECT p.hadm_id::INTEGER AS hadm_id, LOWER(COALESCE(p.drug, '')) AS drug
    FROM prescriptions p
    JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
    WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND p.startdate BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    rx = con.execute(sql_rx).fetchdf()

    drugs = rx["drug"].astype(str)
    rx = rx.loc[drugs.str.contains(pattern, case=False, regex=True)]
    flag = rx.groupby("hadm_id").size().gt(0).astype(int).rename("received_antibiotic").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_antibiotic"] = cohort_df["received_antibiotic"].fillna(0).astype(int)
    return cohort_df


def add_was_mechanically_ventilated_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'was_mechanically_ventilated' based on CHARTEVENTS within 48h of admission."""
    vent_setting_itemids = [223849]
    peep_itemids = [60, 437, 505, 506, 686, 220339, 224700]
    tv_itemids = [639, 654, 681, 682, 683, 684, 224684, 224685, 224686]
    oxygen_device_itemids = [467, 223848]
    vent_itemids = vent_setting_itemids + peep_itemids + tv_itemids + oxygen_device_itemids

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_vent_itemids", pd.DataFrame({"itemid": vent_itemids}))
    sql_vent = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id, c.itemid::INTEGER AS itemid, LOWER(COALESCE(c.value, '')) AS value
    FROM chartevents c
    JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_vent_itemids)
      AND c.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND c.error::INTEGER == 0
    """
    vent = con.execute(sql_vent).fetchdf()

    has_setting = vent[vent["itemid"].isin(vent_setting_itemids + peep_itemids + tv_itemids)]
    ox = vent[vent["itemid"].isin(oxygen_device_itemids)]
    ox_flag = (
        (ox["itemid"] == 467) & (ox["value"].str.contains("ventilator", na=False))
    ) | (
        (ox["itemid"] == 223848) & (~ox["value"].str.contains("other", na=False))
    )
    ox = ox.loc[ox_flag]
    vent_agg = pd.concat([has_setting[["hadm_id"]], ox[["hadm_id"]]], ignore_index=True)
    flag = vent_agg.groupby("hadm_id").size().gt(0).astype(int).rename("was_mechanically_ventilated").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["was_mechanically_ventilated"] = cohort_df["was_mechanically_ventilated"].fillna(0).astype(int)
    return cohort_df


def add_received_rrt_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'received_rrt' based on PROCEDUREEVENTS_MV/CHARTEVENTS within 48h of admission."""
    rrt_proc_itemids = [225802, 225803, 225441]

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    con.register("tmp_rrt_proc_itemids", pd.DataFrame({"itemid": rrt_proc_itemids}))
    sql_rrt_proc = f"""
    SELECT p.hadm_id::INTEGER AS hadm_id
    FROM procedureevents_mv p
    JOIN admissions a ON p.hadm_id = a.hadm_id AND p.subject_id = a.subject_id
    WHERE p.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND p.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_proc_itemids)
      AND p.starttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    con.register("tmp_rrt_chart_itemids", pd.DataFrame({"itemid": [152]}))
    sql_rrt_chart = f"""
    SELECT c.hadm_id::INTEGER AS hadm_id
    FROM chartevents c
    JOIN admissions a ON c.hadm_id = a.hadm_id AND c.subject_id = a.subject_id
    WHERE c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.itemid::INTEGER IN (SELECT itemid FROM tmp_rrt_chart_itemids)
      AND c.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND c.error::INTEGER == 0
    """
    rrt_proc = con.execute(sql_rrt_proc).fetchdf()
    rrt_chart = con.execute(sql_rrt_chart).fetchdf()
    rrt = pd.concat([rrt_proc, rrt_chart], ignore_index=True)

    flag = rrt.groupby("hadm_id").size().gt(0).astype(int).rename("received_rrt").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["received_rrt"] = cohort_df["received_rrt"].fillna(0).astype(int)
    return cohort_df


def add_positive_blood_culture_flag(con, hadm_ids: List[int], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 flag 'positive_blood_culture' from MICROBIOLOGYEVENTS within 48h of admission."""

    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql_micro = f"""
    SELECT m.hadm_id::INTEGER AS hadm_id
    FROM microbiologyevents m
    JOIN admissions a ON m.hadm_id = a.hadm_id AND m.subject_id = a.subject_id
    WHERE m.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND LOWER(COALESCE(m.spec_type_desc, '')) LIKE '%blood culture%'
      AND m.org_name IS NOT NULL
      AND m.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
    """
    micro = con.execute(sql_micro).fetchdf()

    flag = micro.groupby("hadm_id").size().gt(0).astype(int).rename("positive_blood_culture").reset_index()

    cohort_df = cohort_df.merge(flag, on="hadm_id", how="left")
    cohort_df["positive_blood_culture"] = cohort_df["positive_blood_culture"].fillna(0).astype(int)
    return cohort_df


def query_labs_48h(con, hadm_ids: List[int], labs_meta_csv: str) -> pd.DataFrame:
    """Query lab events within the first 48 hours and filter by metadata bounds."""
    meta = load_metadata(labs_meta_csv)
    itemids = meta["itemid"].tolist()

    con.register("tmp_lab_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql = f"""
    SELECT l.subject_id::INTEGER AS subject_id,
           l.hadm_id::INTEGER AS hadm_id,
           l.charttime::TIMESTAMP AS charttime,
           l.itemid::INTEGER AS itemid,
           l.valuenum::DOUBLE AS valuenum
    FROM labevents l
    JOIN admissions a ON l.subject_id = a.subject_id AND l.hadm_id = a.hadm_id
    WHERE l.itemid::INTEGER IN (SELECT itemid FROM tmp_lab_itemids)
      AND l.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND l.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND l.valuenum IS NOT NULL
    """
    df = con.execute(sql).fetchdf()

    df = df.merge(meta, on="itemid")
    mask = (df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])

    df = df.loc[mask, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]].reset_index(drop=True)
    return df


def query_vitals_48h(con, hadm_ids: List[int], vitals_meta_csv: str) -> pd.DataFrame:
    """Query vital sign events within the first 48 hours and filter by metadata bounds."""
    meta = load_metadata(vitals_meta_csv)
    itemids = meta["itemid"].tolist()

    con.register("tmp_vital_itemids", pd.DataFrame({"itemid": itemids}))
    con.register("tmp_hadm_ids", pd.DataFrame({"hadm_id": hadm_ids}))
    sql = f"""
    SELECT c.subject_id::INTEGER AS subject_id,
           c.hadm_id::INTEGER AS hadm_id,
           c.charttime::TIMESTAMP AS charttime,
           c.itemid::INTEGER AS itemid,
           c.valuenum::DOUBLE AS valuenum
    FROM chartevents c
    JOIN admissions a ON c.subject_id = a.subject_id AND c.hadm_id = a.hadm_id
    WHERE c.itemid::INTEGER IN (SELECT itemid FROM tmp_vital_itemids)
      AND c.hadm_id::INTEGER IN (SELECT hadm_id FROM tmp_hadm_ids)
      AND c.charttime BETWEEN a.admittime AND a.admittime + INTERVAL {WINDOW_HOURS} HOUR
      AND c.valuenum IS NOT NULL
      AND c.error::INTEGER == 0
    """
    df = con.execute(sql).fetchdf()

    df = df.merge(meta, on="itemid")
    mask = (df["valuenum"] >= df["min"]) & (df["valuenum"] <= df["max"])

    df = df.loc[mask, ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]].reset_index(drop=True)
    return df


def extract_raw(con, initial_cohort_csv: str, labs_csv: str, vitals_csv: str) -> Dict[str, pd.DataFrame]:
    """Orchestrate raw extraction for the first-admission cohort."""
    subject_ids = load_initial_subjects(initial_cohort_csv)
    base = query_base_admissions(con, subject_ids)

    cohort, targets = create_cohort_and_targets(base)
    hadm_ids = cohort["hadm_id"].tolist()

    cohort = add_first_icu_intime(con, hadm_ids, cohort)
    cohort = add_first_height(con, hadm_ids, cohort)
    cohort = add_first_weight(con, hadm_ids, cohort)
    cohort = add_received_vasopressor_flag(con, hadm_ids, cohort)
    cohort = add_received_sedation_flag(con, hadm_ids, cohort)
    cohort = add_received_antibiotic_flag(con, hadm_ids, cohort)
    cohort = add_was_mechanically_ventilated_flag(con, hadm_ids, cohort)
    cohort = add_received_rrt_flag(con, hadm_ids, cohort)
    cohort = add_positive_blood_culture_flag(con, hadm_ids, cohort)
    cohort = normalize_categorical_enums(cohort)

    labs = query_labs_48h(con, hadm_ids, labs_csv)
    vitals = query_vitals_48h(con, hadm_ids, vitals_csv)

    return {
        "cohort": cohort,
        "labs": labs,
        "vitals": vitals,
        "targets": targets,
    }
