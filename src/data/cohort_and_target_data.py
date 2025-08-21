import pandas as pd
from typing import List, Tuple
from data.common import SECONDS_PER_HOUR, SECONDS_PER_YEAR, get_time_difference

MIN_AGE = 18
MAX_AGE = 89
MIN_LOS_HOURS = 54

MORTALITY_DAYS = 30
PROLONGED_LOS_DAYS = 7
READMISSION_DAYS = 30

ADMISSION_SQL = f"""
    SELECT a.subject_id::INTEGER AS subject_id,
           a.hadm_id::INTEGER AS hadm_id,
           a.admittime::TIMESTAMP AS admittime,
           a.dischtime::TIMESTAMP AS dischtime,
           p.dob::TIMESTAMP AS dob,
           p.dod::TIMESTAMP AS dod
    FROM admissions a
    JOIN patients p ON a.subject_id = p.subject_id
    WHERE a.subject_id::INTEGER IN (SELECT subject_id FROM tmp_subject_ids)
    ORDER BY a.subject_id, a.admittime
    """

TARGET_COLUMNS = ["hadm_id", "mortality", "prolonged_los", "readmission"]


def get_cohort_and_target_data(con, subject_ids: List[int]) -> Tuple[List[int], pd.DataFrame]:
    con.register("tmp_subject_ids", pd.DataFrame({"subject_id": subject_ids}))

    df = con.execute(ADMISSION_SQL).fetchdf()

    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)

    df = df.groupby("subject_id", as_index=False).head(1)

    age_ok = (get_time_difference(df["admittime"], df["dob"], SECONDS_PER_YEAR)).between(MIN_AGE, MAX_AGE)
    los_ok = (get_time_difference(df["dischtime"], df["admittime"], SECONDS_PER_HOUR)) >= MIN_LOS_HOURS
    dod_ok = df["dod"].isna() | (get_time_difference(df["dod"], df["admittime"], SECONDS_PER_HOUR)) >= MIN_LOS_HOURS
    chartevents_ok = df["has_chartevents_data"] == 1

    keep = age_ok & los_ok & dod_ok & chartevents_ok

    df = df[keep].reset_index(drop=True)

    df["mortality"] = df["dod"].notna() & (get_time_difference(df["dod"], df["dischtime"], SECONDS_PER_YEAR) <= MORTALITY_DAYS)
    df["prolonged_los"] = (get_time_difference(df["dischtime"], df["admittime"], SECONDS_PER_HOUR)) > PROLONGED_LOS_DAYS
    df["readmission"] = (df["next_admittime"].notna() & (get_time_difference(df["next_admittime"], df["dischtime"], SECONDS_PER_HOUR) <= READMISSION_DAYS))

    cohort = df["hadm_id"].tolist()
    targets = df[TARGET_COLUMNS].reset_index(drop=True)

    return cohort, targets
