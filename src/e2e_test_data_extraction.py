import datetime as dt
from typing import Any

import duckdb  # type: ignore
import pandas as pd

from data_extraction import extract_raw


def _ts(s: str) -> pd.Timestamp:

    return pd.Timestamp(s)


def create_in_memory_mimic() -> Any:

    con = duckdb.connect(database=":memory:")

    # Minimal schemas with only the columns referenced in queries
    con.execute(
        """
        CREATE TABLE admissions (
            subject_id INTEGER,
            hadm_id INTEGER,
            admittime TIMESTAMP,
            dischtime TIMESTAMP,
            admission_type VARCHAR,
            admission_location VARCHAR,
            insurance VARCHAR,
            language VARCHAR,
            religion VARCHAR,
            marital_status VARCHAR,
            ethnicity VARCHAR,
            edregtime TIMESTAMP,
            has_chartevents_data INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE patients (
            subject_id INTEGER,
            gender VARCHAR,
            dob TIMESTAMP,
            dod TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE icustays (
            hadm_id INTEGER,
            intime TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE chartevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            charttime TIMESTAMP,
            itemid INTEGER,
            valuenum DOUBLE,
            value VARCHAR,
            error INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE inputevents_cv (
            subject_id INTEGER,
            hadm_id INTEGER,
            charttime TIMESTAMP,
            itemid INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE inputevents_mv (
            subject_id INTEGER,
            hadm_id INTEGER,
            starttime TIMESTAMP,
            itemid INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE prescriptions (
            subject_id INTEGER,
            hadm_id INTEGER,
            startdate TIMESTAMP,
            drug VARCHAR
        );
        """
    )

    con.execute(
        """
        CREATE TABLE procedureevents_mv (
            subject_id INTEGER,
            hadm_id INTEGER,
            starttime TIMESTAMP,
            itemid INTEGER
        );
        """
    )

    con.execute(
        """
        CREATE TABLE microbiologyevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            spec_type_desc VARCHAR,
            org_name VARCHAR,
            charttime TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE labevents (
            subject_id INTEGER,
            hadm_id INTEGER,
            charttime TIMESTAMP,
            itemid INTEGER,
            valuenum DOUBLE
        );
        """
    )

    return con


def seed_synthetic_data(con: Any) -> None:

    # Subject 1: valid, will be included
    subject_id_1 = 1
    hadm_id_1 = 10
    admit_1 = _ts("2100-01-01 00:00:00")
    discharge_1 = admit_1 + pd.Timedelta(days=10)  # prolonged LOS

    admissions_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "admittime": admit_1,
                "dischtime": discharge_1,
                "admission_type": "EMERGENCY",
                "admission_location": "EMERGENCY ROOM ADMIT",
                "insurance": "MEDICARE",
                "language": "ENGL",
                "religion": "CATHOLIC",
                "marital_status": "MARRIED",
                "ethnicity": "WHITE",
                "edregtime": admit_1 - pd.Timedelta(hours=1),
                "has_chartevents_data": 1,
            },
            # Subject 2: excluded (age < 18)
            {
                "subject_id": 2,
                "hadm_id": 20,
                "admittime": _ts("2100-01-05 00:00:00"),
                "dischtime": _ts("2100-01-07 12:00:00"),
                "admission_type": "URGENT",
                "admission_location": "EMERGENCY ROOM ADMIT",
                "insurance": "PRIVATE",
                "language": "ENGL",
                "religion": "CATHOLIC",
                "marital_status": "SINGLE",
                "ethnicity": "WHITE",
                "edregtime": _ts("2100-01-04 22:00:00"),
                "has_chartevents_data": 1,
            },
        ]
    )
    con.register("admissions_df", admissions_df)
    con.execute("INSERT INTO admissions SELECT * FROM admissions_df")

    patients_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "gender": "M",
                "dob": _ts("2040-01-01 00:00:00"),  # age 60 at admit
                "dod": discharge_1 + pd.Timedelta(days=20),  # mortality within 30 days of discharge
            },
            {
                "subject_id": 2,
                "gender": "F",
                "dob": _ts("2090-01-01 00:00:00"),  # age 10 -> excluded
                "dod": pd.NaT,
            },
        ]
    )
    con.register("patients_df", patients_df)
    con.execute("INSERT INTO patients SELECT * FROM patients_df")

    icu_df = pd.DataFrame(
        [
            {"hadm_id": hadm_id_1, "intime": admit_1 + pd.Timedelta(hours=10)},
        ]
    )
    con.register("icu_df", icu_df)
    con.execute("INSERT INTO icustays SELECT * FROM icu_df")

    # Height (inches itemid -> will be converted to cm) and Weight (kg)
    chartevents_rows = [
        {
            "subject_id": subject_id_1,
            "hadm_id": hadm_id_1,
            "charttime": admit_1 + pd.Timedelta(hours=2),
            "itemid": 920,  # height in inches
            "valuenum": 70.0,
            "value": None,
            "error": 0,
        },
        {
            "subject_id": subject_id_1,
            "hadm_id": hadm_id_1,
            "charttime": admit_1 + pd.Timedelta(hours=3),
            "itemid": 762,  # weight kg
            "valuenum": 80.0,
            "value": None,
            "error": 0,
        },
        # Ventilation flags: oxygen device and settings within 48h
        {
            "subject_id": subject_id_1,
            "hadm_id": hadm_id_1,
            "charttime": admit_1 + pd.Timedelta(hours=5),
            "itemid": 467,  # oxygen device
            "valuenum": None,
            "value": "Ventilator",
            "error": 0,
        },
        {
            "subject_id": subject_id_1,
            "hadm_id": hadm_id_1,
            "charttime": admit_1 + pd.Timedelta(hours=6),
            "itemid": 223849,  # ventilator setting
            "valuenum": 1.0,
            "value": "AC",
            "error": 0,
        },
        # Vitals within metadata
        {
            "subject_id": subject_id_1,
            "hadm_id": hadm_id_1,
            "charttime": admit_1 + pd.Timedelta(hours=1),
            "itemid": 211,  # Heart Rate
            "valuenum": 80.0,
            "value": None,
            "error": 0,
        },
    ]
    chartevents_df = pd.DataFrame(chartevents_rows)
    con.register("chartevents_df", chartevents_df)
    con.execute("INSERT INTO chartevents SELECT * FROM chartevents_df")

    # Vasopressor within 48h
    inputevents_cv_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "charttime": admit_1 + pd.Timedelta(hours=8),
                "itemid": 221906,  # norepinephrine
            }
        ]
    )
    con.register("inputevents_cv_df", inputevents_cv_df)
    con.execute("INSERT INTO inputevents_cv SELECT * FROM inputevents_cv_df")

    inputevents_mv_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "starttime": admit_1 + pd.Timedelta(hours=9),
                "itemid": 221749,  # vasopressor item
            }
        ]
    )
    con.register("inputevents_mv_df", inputevents_mv_df)
    con.execute("INSERT INTO inputevents_mv SELECT * FROM inputevents_mv_df")

    # Antibiotic within 48h
    prescriptions_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "startdate": admit_1 + pd.Timedelta(hours=4),
                "drug": "Vancomycin",
            }
        ]
    )
    con.register("prescriptions_df", prescriptions_df)
    con.execute("INSERT INTO prescriptions SELECT * FROM prescriptions_df")

    # RRT within 48h
    procedureevents_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "starttime": admit_1 + pd.Timedelta(hours=7),
                "itemid": 225802,
            }
        ]
    )
    con.register("procedureevents_df", procedureevents_df)
    con.execute("INSERT INTO procedureevents_mv SELECT * FROM procedureevents_df")

    # Positive blood culture within 48h
    micro_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "spec_type_desc": "Blood Culture",
                "org_name": "E. coli",
                "charttime": admit_1 + pd.Timedelta(hours=12),
            }
        ]
    )
    con.register("micro_df", micro_df)
    con.execute("INSERT INTO microbiologyevents SELECT * FROM micro_df")

    # Labs within 48h using metadata itemids
    labevents_df = pd.DataFrame(
        [
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "charttime": admit_1 + pd.Timedelta(hours=2),
                "itemid": 50912,  # CREATININE
                "valuenum": 1.2,
            },
            {
                "subject_id": subject_id_1,
                "hadm_id": hadm_id_1,
                "charttime": admit_1 + pd.Timedelta(hours=3),
                "itemid": 51222,  # HEMOGLOBIN
                "valuenum": 13.5,
            },
        ]
    )
    con.register("labevents_df", labevents_df)
    con.execute("INSERT INTO labevents SELECT * FROM labevents_df")


def run_test() -> None:

    con = create_in_memory_mimic()
    seed_synthetic_data(con)

    results = extract_raw(
        con=con,
        initial_cohort_csv="csvs/initial_cohort.csv",
        labs_csv="csvs/labs_metadata.csv",
        vitals_csv="csvs/vital_metadata.csv",
    )

    # Basic structure
    assert set(results.keys()) == {"cohort", "labs", "vitals", "targets"}

    cohort = results["cohort"]
    targets = results["targets"]
    labs = results["labs"]
    vitals = results["vitals"]

    # Cohort should include only subject 1 (subject 2 excluded by age)
    assert len(cohort) == 1, f"Unexpected cohort size: {len(cohort)}"
    c_row = cohort.iloc[0]
    assert int(c_row.subject_id) == 1 and int(c_row.hadm_id) == 10

    # Added features exist and are populated
    for col in [
        "first_icu_intime",
        "height_cm",
        "weight_kg",
        "received_vasopressor",
        "received_sedation",
        "received_antibiotic",
        "was_mechanically_ventilated",
        "received_rrt",
        "positive_blood_culture",
    ]:
        assert col in cohort.columns, f"Missing column in cohort: {col}"

    assert pd.notnull(c_row.first_icu_intime)
    assert pd.notnull(c_row.height_cm) and c_row.height_cm > 100
    assert pd.notnull(c_row.weight_kg) and c_row.weight_kg > 0
    assert int(c_row.received_vasopressor) == 1
    # sedation not seeded, expect 0
    assert int(c_row.received_sedation) == 0
    assert int(c_row.received_antibiotic) == 1
    assert int(c_row.was_mechanically_ventilated) == 1
    assert int(c_row.received_rrt) == 1
    assert int(c_row.positive_blood_culture) == 1

    # Targets
    assert len(targets) == 1
    t_row = targets.iloc[0]
    assert int(t_row.subject_id) == 1 and int(t_row.hadm_id) == 10
    assert int(t_row.mortality) == 1
    assert int(t_row.prolonged_los) == 1
    # No readmission seeded, expect 0
    assert int(t_row.readmission_30d) == 0

    # Labs and Vitals within bounds and linked to hadm_id 10
    assert not labs.empty and labs["hadm_id"].unique().tolist() == [10]
    assert not vitals.empty and (vitals["hadm_id"] == 10).all()

    print("All tests passed for data_extraction.extract_raw")


if __name__ == "__main__":

    run_test()

