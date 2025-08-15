"""
Comprehensive test suite for data_extraction.py

This test suite validates all aspects of the data extraction pipeline:
- Cohort selection criteria
- Target variable calculations
- Static feature extraction
- Time-series data extraction
- Edge cases and boundary conditions
- Data quality validation
"""

import datetime as dt
from typing import Any, Dict, List
import pandas as pd
import duckdb  # type: ignore
import pytest
from data_extraction import (
    extract_raw,
    load_initial_subjects,
    load_metadata,
    query_base_admissions,
    create_cohort_and_targets,
    add_first_icu_intime,
    add_first_height,
    add_first_weight,
    add_received_vasopressor_flag,
    add_received_sedation_flag,
    add_received_antibiotic_flag,
    add_was_mechanically_ventilated_flag,
    add_received_rrt_flag,
    add_positive_blood_culture_flag,
    query_labs_48h,
    query_vitals_48h,
    normalize_categorical_enums,
    parse_enum,
    WINDOW_HOURS,
    GAP_HOURS,
    MIN_LOS_HOURS,
    MIN_AGE,
    MAX_AGE,
    PROLONGED_LOS_THRESHOLD_DAYS,
    READMISSION_WINDOW_DAYS,
    MORTALITY_WINDOW_DAYS,
)


def _ts(s: str) -> pd.Timestamp:
    """Helper to create timestamps."""
    return pd.Timestamp(s)


class TestDataExtractionComprehensive:
    """Comprehensive test suite for data extraction."""

    def setup_method(self):
        """Set up test database and data for each test."""
        self.con = duckdb.connect(database=":memory:")
        self._create_mimic_schema()
        self._seed_comprehensive_test_data()

    def _create_mimic_schema(self):
        """Create complete MIMIC schema with all required tables."""
        # Core tables
        self.con.execute("""
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
        """)

        self.con.execute("""
            CREATE TABLE patients (
                subject_id INTEGER,
                gender VARCHAR,
                dob TIMESTAMP,
                dod TIMESTAMP
            );
        """)

        self.con.execute("""
            CREATE TABLE icustays (
                hadm_id INTEGER,
                intime TIMESTAMP
            );
        """)

        self.con.execute("""
            CREATE TABLE chartevents (
                subject_id INTEGER,
                hadm_id INTEGER,
                charttime TIMESTAMP,
                itemid INTEGER,
                valuenum DOUBLE,
                value VARCHAR,
                error INTEGER
            );
        """)

        self.con.execute("""
            CREATE TABLE inputevents_cv (
                subject_id INTEGER,
                hadm_id INTEGER,
                charttime TIMESTAMP,
                itemid INTEGER
            );
        """)

        self.con.execute("""
            CREATE TABLE inputevents_mv (
                subject_id INTEGER,
                hadm_id INTEGER,
                starttime TIMESTAMP,
                itemid INTEGER
            );
        """)

        self.con.execute("""
            CREATE TABLE prescriptions (
                subject_id INTEGER,
                hadm_id INTEGER,
                startdate TIMESTAMP,
                drug VARCHAR
            );
        """)

        self.con.execute("""
            CREATE TABLE procedureevents_mv (
                subject_id INTEGER,
                hadm_id INTEGER,
                starttime TIMESTAMP,
                itemid INTEGER
            );
        """)

        self.con.execute("""
            CREATE TABLE microbiologyevents (
                subject_id INTEGER,
                hadm_id INTEGER,
                spec_type_desc VARCHAR,
                org_name VARCHAR,
                charttime TIMESTAMP
            );
        """)

        self.con.execute("""
            CREATE TABLE labevents (
                subject_id INTEGER,
                hadm_id INTEGER,
                charttime TIMESTAMP,
                itemid INTEGER,
                valuenum DOUBLE
            );
        """)

    def _seed_comprehensive_test_data(self):
        """Seed comprehensive test data covering all edge cases."""
        base_time = _ts("2100-01-01 00:00:00")
        
        # Test subjects with different scenarios
        # NOTE: Using subjects 1, 2, 3 to match initial_cohort.csv
        test_cases = [
            # Subject 1: Valid case - should be included
            {
                "subject_id": 1,
                "hadm_id": 101,
                "admittime": base_time,
                "dischtime": base_time + pd.Timedelta(days=10),  # Prolonged LOS
                "age_years": 65,  # Valid age
                "dod": base_time + pd.Timedelta(days=40),  # Mortality within 30 days of discharge
                "has_chartevents": 1,
                "next_admission": None,  # No readmission
            },
            # Subject 2: Too young - should be excluded
            {
                "subject_id": 2,
                "hadm_id": 102,
                "admittime": base_time + pd.Timedelta(days=1),
                "dischtime": base_time + pd.Timedelta(days=4),
                "age_years": 17,  # Too young
                "dod": None,
                "has_chartevents": 1,
                "next_admission": None,
            },
            # Subject 3: Valid with readmission - should be included
            {
                "subject_id": 3,
                "hadm_id": 103,
                "admittime": base_time + pd.Timedelta(days=2),
                "dischtime": base_time + pd.Timedelta(days=4, hours=6),  # 54 hours LOS (minimum required)
                "age_years": 50,  # Valid age
                "dod": None,
                "has_chartevents": 1,
                "next_admission": base_time + pd.Timedelta(days=20),  # Readmission within 30 days
            },
            # Second admission for subject 1 - should be excluded (first admission only)
            {
                "subject_id": 1,  # Same subject as first case
                "hadm_id": 108,
                "admittime": base_time + pd.Timedelta(days=30),  # Later admission
                "dischtime": base_time + pd.Timedelta(days=33),
                "age_years": 65,
                "dod": None,
                "has_chartevents": 1,
                "next_admission": None,
            },
        ]

        # Insert admissions
        admissions_data = []
        patients_data = []
        for case in test_cases:
            admissions_data.append({
                "subject_id": case["subject_id"],
                "hadm_id": case["hadm_id"],
                "admittime": case["admittime"],
                "dischtime": case["dischtime"],
                "admission_type": "EMERGENCY",
                "admission_location": "EMERGENCY ROOM ADMIT",
                "insurance": "MEDICARE",
                "language": "ENGL",
                "religion": "CATHOLIC",
                "marital_status": "MARRIED",
                "ethnicity": "WHITE",
                "edregtime": case["admittime"] - pd.Timedelta(hours=1),
                "has_chartevents_data": case["has_chartevents"],
            })
            
            # Only add patient data once per subject
            if not any(p["subject_id"] == case["subject_id"] for p in patients_data):
                patients_data.append({
                    "subject_id": case["subject_id"],
                    "gender": "M" if case["subject_id"] % 2 == 1 else "F",
                    "dob": case["admittime"] - pd.Timedelta(days=case["age_years"] * 365.25),
                    "dod": case["dod"],
                })

        # Insert second admissions for readmission testing
        for case in test_cases:
            if case["next_admission"]:
                admissions_data.append({
                    "subject_id": case["subject_id"],
                    "hadm_id": case["hadm_id"] + 1000,  # Different hadm_id
                    "admittime": case["next_admission"],
                    "dischtime": case["next_admission"] + pd.Timedelta(days=2),
                    "admission_type": "URGENT",
                    "admission_location": "EMERGENCY ROOM ADMIT",
                    "insurance": "PRIVATE",
                    "language": "ENGL",
                    "religion": "PROTESTANT_QUAKER",
                    "marital_status": "SINGLE",
                    "ethnicity": "BLACK_AFRICAN_AMERICAN",
                    "edregtime": case["next_admission"] - pd.Timedelta(hours=2),
                    "has_chartevents_data": 1,
                })

        # Insert data
        admissions_df = pd.DataFrame(admissions_data)
        self.con.register("admissions_df", admissions_df)
        self.con.execute("INSERT INTO admissions SELECT * FROM admissions_df")

        patients_df = pd.DataFrame(patients_data)
        self.con.register("patients_df", patients_df)
        self.con.execute("INSERT INTO patients SELECT * FROM patients_df")

        # Add comprehensive feature data for valid subjects (1, 3)
        valid_hadm_ids = [101, 103]
        self._add_icu_data(valid_hadm_ids, base_time)
        self._add_height_weight_data(valid_hadm_ids, base_time)
        self._add_intervention_data(valid_hadm_ids, base_time)
        self._add_labs_vitals_data(valid_hadm_ids, base_time)

    def _add_icu_data(self, hadm_ids: List[int], base_time: pd.Timestamp):
        """Add ICU data for testing."""
        icu_data = []
        for i, hadm_id in enumerate(hadm_ids):
            # Some have ICU within 48h, some don't
            if i % 2 == 0:  # Even indices get ICU within 48h
                icu_data.append({
                    "hadm_id": hadm_id,
                    "intime": base_time + pd.Timedelta(days=i, hours=10)
                })
        
        if icu_data:
            icu_df = pd.DataFrame(icu_data)
            self.con.register("icu_df", icu_df)
            self.con.execute("INSERT INTO icustays SELECT * FROM icu_df")

    def _add_height_weight_data(self, hadm_ids: List[int], base_time: pd.Timestamp):
        """Add height and weight data for testing."""
        chartevents_data = []
        for i, hadm_id in enumerate(hadm_ids):
            subject_id = hadm_id - 100  # Convert back to subject_id
            # Correct admit time calculation for each subject
            if subject_id == 1:
                admit_time = base_time
            elif subject_id == 3:
                admit_time = base_time + pd.Timedelta(days=2)
            else:
                admit_time = base_time + pd.Timedelta(days=i)
            
            # Height data (mix of inches and cm)
            if i % 2 == 0:  # Even indices get inches
                chartevents_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": admit_time + pd.Timedelta(hours=2),
                    "itemid": 920,  # height in inches
                    "valuenum": 70.0,
                    "value": None,
                    "error": 0,
                })
            else:  # Odd indices get cm
                chartevents_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": admit_time + pd.Timedelta(hours=2),
                    "itemid": 3485,  # height in cm
                    "valuenum": 175.0,
                    "value": None,
                    "error": 0,
                })

            # Weight data (mix of kg and lbs)
            if i % 2 == 0:  # Even indices get kg
                chartevents_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": admit_time + pd.Timedelta(hours=3),
                    "itemid": 762,  # weight kg
                    "valuenum": 80.0,
                    "value": None,
                    "error": 0,
                })
            else:  # Odd indices get lbs
                chartevents_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": admit_time + pd.Timedelta(hours=3),
                    "itemid": 3581,  # weight lbs
                    "valuenum": 180.0,
                    "value": None,
                    "error": 0,
                })

        if chartevents_data:
            chartevents_df = pd.DataFrame(chartevents_data)
            self.con.register("chartevents_df", chartevents_df)
            self.con.execute("INSERT INTO chartevents SELECT * FROM chartevents_df")

    def _add_intervention_data(self, hadm_ids: List[int], base_time: pd.Timestamp):
        """Add intervention data for comprehensive testing."""
        # Vasopressor data
        inputevents_cv_data = []
        inputevents_mv_data = []
        
        # Sedation data
        prescriptions_data = []
        
        # RRT data
        procedureevents_data = []
        
        # Microbiology data
        micro_data = []
        
        # Ventilation data
        vent_data = []

        for i, hadm_id in enumerate(hadm_ids):
            subject_id = hadm_id - 100
            # Correct admit time calculation for each subject
            if subject_id == 1:
                admit_time = base_time
            elif subject_id == 3:
                admit_time = base_time + pd.Timedelta(days=2)
            else:
                admit_time = base_time + pd.Timedelta(days=i)
            
            # Different patterns for different subjects
            if i == 0:  # Subject 1 - all interventions
                inputevents_cv_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "charttime": admit_time + pd.Timedelta(hours=8),
                    "itemid": 221906,  # norepinephrine
                })
                
                inputevents_mv_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "starttime": admit_time + pd.Timedelta(hours=9),
                    "itemid": 221749,  # vasopressor
                })
                
                prescriptions_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "startdate": admit_time + pd.Timedelta(hours=4),
                    "drug": "Vancomycin",
                })
                
                procedureevents_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "starttime": admit_time + pd.Timedelta(hours=7),
                    "itemid": 225802,  # RRT
                })
                
                micro_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "spec_type_desc": "Blood Culture",
                    "org_name": "E. coli",
                    "charttime": admit_time + pd.Timedelta(hours=12),
                })
                
                vent_data.extend([
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=5),
                        "itemid": 467,  # oxygen device
                        "valuenum": None,
                        "value": "Ventilator",
                        "error": 0,
                    },
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=6),
                        "itemid": 223849,  # ventilator setting
                        "valuenum": 1.0,
                        "value": "AC",
                        "error": 0,
                    }
                ])
                
            elif i == 1:  # Subject 7 - some interventions
                prescriptions_data.append({
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "startdate": admit_time + pd.Timedelta(hours=6),
                    "drug": "Piperacillin/Tazobactam",
                })
                
            # Other subjects get no interventions for testing 0 values

        # Insert all intervention data
        if inputevents_cv_data:
            df = pd.DataFrame(inputevents_cv_data)
            self.con.register("inputevents_cv_df", df)
            self.con.execute("INSERT INTO inputevents_cv SELECT * FROM inputevents_cv_df")

        if inputevents_mv_data:
            df = pd.DataFrame(inputevents_mv_data)
            self.con.register("inputevents_mv_df", df)
            self.con.execute("INSERT INTO inputevents_mv SELECT * FROM inputevents_mv_df")

        if prescriptions_data:
            df = pd.DataFrame(prescriptions_data)
            self.con.register("prescriptions_df", df)
            self.con.execute("INSERT INTO prescriptions SELECT * FROM prescriptions_df")

        if procedureevents_data:
            df = pd.DataFrame(procedureevents_data)
            self.con.register("procedureevents_df", df)
            self.con.execute("INSERT INTO procedureevents_mv SELECT * FROM procedureevents_df")

        if micro_data:
            df = pd.DataFrame(micro_data)
            self.con.register("micro_df", df)
            self.con.execute("INSERT INTO microbiologyevents SELECT * FROM micro_df")

        if vent_data:
            df = pd.DataFrame(vent_data)
            self.con.register("vent_df", df)
            self.con.execute("INSERT INTO chartevents SELECT * FROM vent_df")

    def _add_labs_vitals_data(self, hadm_ids: List[int], base_time: pd.Timestamp):
        """Add comprehensive labs and vitals data."""
        labevents_data = []
        chartevents_vitals_data = []

        for i, hadm_id in enumerate(hadm_ids):
            subject_id = hadm_id - 100
            # Correct admit time calculation for each subject
            if subject_id == 1:
                admit_time = base_time
            elif subject_id == 3:
                admit_time = base_time + pd.Timedelta(days=2)
            else:
                admit_time = base_time + pd.Timedelta(days=i)
            
            # Add multiple lab values across time
            lab_times = [2, 12, 24, 36, 47]  # Hours after admission
            for hour in lab_times:
                labevents_data.extend([
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=hour),
                        "itemid": 50912,  # CREATININE
                        "valuenum": 1.0 + (hour * 0.1),  # Gradually increasing
                    },
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=hour),
                        "itemid": 51222,  # HEMOGLOBIN
                        "valuenum": 12.0 + (i * 0.5),  # Varies by subject
                    }
                ])

            # Add vital signs
            vital_times = [1, 6, 12, 18, 24, 30, 36, 42, 47]
            for hour in vital_times:
                chartevents_vitals_data.extend([
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=hour),
                        "itemid": 211,  # Heart Rate
                        "valuenum": 70.0 + (hour * 0.5),  # Gradually increasing HR
                        "value": None,
                        "error": 0,
                    },
                    {
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                        "charttime": admit_time + pd.Timedelta(hours=hour),
                        "itemid": 51,  # Systolic BP
                        "valuenum": 120.0 + (i * 5),  # Varies by subject
                        "value": None,
                        "error": 0,
                    }
                ])

        # Insert labs and vitals
        if labevents_data:
            df = pd.DataFrame(labevents_data)
            self.con.register("labevents_df", df)
            self.con.execute("INSERT INTO labevents SELECT * FROM labevents_df")

        if chartevents_vitals_data:
            df = pd.DataFrame(chartevents_vitals_data)
            self.con.register("chartevents_vitals_df", df)
            self.con.execute("INSERT INTO chartevents SELECT * FROM chartevents_vitals_df")

    def test_constants_are_correct(self):
        """Test that all constants match the plan requirements."""
        assert WINDOW_HOURS == 48, "Window should be 48 hours"
        assert GAP_HOURS == 6, "Gap should be 6 hours"
        assert MIN_LOS_HOURS == 54, "Minimum LOS should be 54 hours (48h + 6h gap)"
        assert MIN_AGE == 18, "Minimum age should be 18"
        assert MAX_AGE == 89, "Maximum age should be 89"
        assert PROLONGED_LOS_THRESHOLD_DAYS == 7, "Prolonged LOS threshold should be 7 days"
        assert READMISSION_WINDOW_DAYS == 30, "Readmission window should be 30 days"
        assert MORTALITY_WINDOW_DAYS == 30, "Mortality window should be 30 days"

    def test_cohort_selection_criteria(self):
        """Test all cohort selection criteria are correctly applied."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Should include subjects: 1, 3
        # Should exclude: 2 (too young), second admission of 1
        expected_subjects = {1, 3}
        actual_subjects = set(cohort["subject_id"].tolist())
        
        assert actual_subjects == expected_subjects, (
            f"Expected subjects {expected_subjects}, got {actual_subjects}"
        )
        
        # Verify each subject meets criteria
        for _, row in cohort.iterrows():
            # Age criteria
            assert MIN_AGE <= row["admission_age"] <= MAX_AGE, (
                f"Subject {row['subject_id']} age {row['admission_age']} outside range"
            )

    def test_target_variable_calculations(self):
        """Test target variable calculations are correct."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        targets = results["targets"]
        
        # Test specific cases
        subject_1_targets = targets[targets["subject_id"] == 1].iloc[0]
        assert subject_1_targets["mortality"] == 1, "Subject 1 should have mortality=1"
        assert subject_1_targets["prolonged_los"] == 1, "Subject 1 should have prolonged_los=1"
        assert subject_1_targets["readmission_30d"] == 1, "Subject 1 should have readmission_30d=1 (second admission within 30 days)"
        
        subject_3_targets = targets[targets["subject_id"] == 3].iloc[0]
        assert subject_3_targets["mortality"] == 0, "Subject 3 should have mortality=0"
        assert subject_3_targets["prolonged_los"] == 0, "Subject 3 should have prolonged_los=0"
        assert subject_3_targets["readmission_30d"] == 1, "Subject 3 should have readmission_30d=1"

    def test_static_feature_extraction(self):
        """Test static feature extraction completeness and correctness."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Check all required columns exist
        required_columns = [
            "subject_id", "hadm_id", "admittime", "admission_type", "admission_location",
            "insurance", "language", "religion", "marital_status", "ethnicity",
            "edregtime", "gender", "admission_age", "first_icu_intime", "height_cm",
            "weight_kg", "received_vasopressor", "received_sedation", "received_antibiotic",
            "was_mechanically_ventilated", "received_rrt", "positive_blood_culture"
        ]
        
        for col in required_columns:
            assert col in cohort.columns, f"Missing required column: {col}"
        
        # Test specific feature values
        subject_1 = cohort[cohort["subject_id"] == 1].iloc[0]
        
        # Height conversion (70 inches -> cm)
        expected_height_cm = 70.0 * 2.54
        assert abs(subject_1["height_cm"] - expected_height_cm) < 0.1, (
            f"Height conversion incorrect: expected {expected_height_cm}, got {subject_1['height_cm']}"
        )
        
        # Weight should be in kg
        assert subject_1["weight_kg"] == 80.0, f"Weight incorrect: {subject_1['weight_kg']}"
        
        # Binary flags
        assert subject_1["received_vasopressor"] == 1, "Vasopressor flag incorrect"
        assert subject_1["received_antibiotic"] == 1, "Antibiotic flag incorrect"
        assert subject_1["was_mechanically_ventilated"] == 1, "Ventilation flag incorrect"
        assert subject_1["received_rrt"] == 1, "RRT flag incorrect"
        assert subject_1["positive_blood_culture"] == 1, "Blood culture flag incorrect"
        assert subject_1["received_sedation"] == 0, "Sedation flag should be 0"

    def test_time_series_data_extraction(self):
        """Test time-series data extraction within 48-hour window."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        labs = results["labs"]
        vitals = results["vitals"]
        
        # Check data exists
        assert not labs.empty, "Labs data should not be empty"
        assert not vitals.empty, "Vitals data should not be empty"
        
        # Check required columns
        lab_columns = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
        vital_columns = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
        
        for col in lab_columns:
            assert col in labs.columns, f"Missing lab column: {col}"
        
        for col in vital_columns:
            assert col in vitals.columns, f"Missing vital column: {col}"
        
        # Check 48-hour window constraint
        cohort = results["cohort"]
        for _, patient in cohort.iterrows():
            admit_time = patient["admittime"]
            window_end = admit_time + pd.Timedelta(hours=48)
            
            patient_labs = labs[labs["hadm_id"] == patient["hadm_id"]]
            patient_vitals = vitals[vitals["hadm_id"] == patient["hadm_id"]]
            
            # All timestamps should be within 48-hour window
            for _, lab in patient_labs.iterrows():
                assert admit_time <= lab["charttime"] <= window_end, (
                    f"Lab timestamp {lab['charttime']} outside 48h window for admission {admit_time}"
                )
            
            for _, vital in patient_vitals.iterrows():
                assert admit_time <= vital["charttime"] <= window_end, (
                    f"Vital timestamp {vital['charttime']} outside 48h window for admission {admit_time}"
                )

    def test_categorical_normalization(self):
        """Test categorical variable normalization."""
        # Test parse_enum function
        test_series = pd.Series(["EMERGENCY", "emergency", "ELECTIVE", "UNKNOWN_TYPE", None])
        valid_values = ["EMERGENCY", "URGENT", "ELECTIVE"]
        
        result = parse_enum(test_series, valid_values, "OTHER")
        expected = pd.Series(["EMERGENCY", "EMERGENCY", "ELECTIVE", "OTHER", "OTHER"])
        
        pd.testing.assert_series_equal(result, expected)

    def test_metadata_bounds_filtering(self):
        """Test that lab and vital values are filtered by metadata bounds."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        labs = results["labs"]
        vitals = results["vitals"]
        
        # Load metadata to check bounds
        labs_meta = load_metadata("csvs/labs_metadata.csv")
        vitals_meta = load_metadata("csvs/vital_metadata.csv")
        
        # Check lab bounds
        for _, lab in labs.iterrows():
            meta_row = labs_meta[labs_meta["itemid"] == lab["itemid"]]
            if not meta_row.empty:
                min_val, max_val = meta_row.iloc[0]["min"], meta_row.iloc[0]["max"]
                assert min_val <= lab["valuenum"] <= max_val, (
                    f"Lab value {lab['valuenum']} outside bounds [{min_val}, {max_val}] for itemid {lab['itemid']}"
                )
        
        # Check vital bounds
        for _, vital in vitals.iterrows():
            meta_row = vitals_meta[vitals_meta["itemid"] == vital["itemid"]]
            if not meta_row.empty:
                min_val, max_val = meta_row.iloc[0]["min"], meta_row.iloc[0]["max"]
                assert min_val <= vital["valuenum"] <= max_val, (
                    f"Vital value {vital['valuenum']} outside bounds [{min_val}, {max_val}] for itemid {vital['itemid']}"
                )

    def test_unit_conversions(self):
        """Test unit conversions for height and weight."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Test height conversion (inches to cm)
        subject_1 = cohort[cohort["subject_id"] == 1].iloc[0]
        expected_height = 70.0 * 2.54  # 70 inches to cm
        assert abs(subject_1["height_cm"] - expected_height) < 0.1
        
        # Test weight conversion (should be in kg)
        if len(cohort[cohort["subject_id"] == 3]) > 0:
            subject_3 = cohort[cohort["subject_id"] == 3].iloc[0]
            if pd.notnull(subject_3["weight_kg"]):
                # Subject 3 gets lbs input (180 lbs -> kg)
                expected_weight = 180.0 * 0.45359237
                assert abs(subject_3["weight_kg"] - expected_weight) < 0.1

    def test_first_admission_only(self):
        """Test that only first admission per subject is included."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Subject 1 has two admissions, should only include the first
        subject_1_admissions = cohort[cohort["subject_id"] == 1]
        assert len(subject_1_admissions) == 1, "Should only have one admission per subject"
        
        # Should be the earlier admission (hadm_id 101, not 108)
        assert subject_1_admissions.iloc[0]["hadm_id"] == 101

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Subject 3: normal LOS (48 hours) - should be included (meets minimum of 54h with our data)
        subject_3 = cohort[cohort["subject_id"] == 3]
        assert len(subject_3) == 1, "Subject 3 should be included"
        
        # All subjects should meet age criteria
        for _, row in cohort.iterrows():
            assert row["admission_age"] >= MIN_AGE, f"Subject {row['subject_id']} age too low"
            assert row["admission_age"] <= MAX_AGE, f"Subject {row['subject_id']} age too high"

    def test_data_types_and_structure(self):
        """Test that output data has correct types and structure."""
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        # Check return structure
        assert isinstance(results, dict)
        assert set(results.keys()) == {"cohort", "labs", "vitals", "targets"}
        
        # Check DataFrame types
        for key, df in results.items():
            assert isinstance(df, pd.DataFrame), f"{key} should be a DataFrame"
            assert not df.empty or key in ["labs", "vitals"], f"{key} should not be empty"
        
        # Check specific column types
        cohort = results["cohort"]
        assert cohort["subject_id"].dtype in [int, "int64"], "subject_id should be integer"
        assert cohort["hadm_id"].dtype in [int, "int64"], "hadm_id should be integer"
        assert pd.api.types.is_datetime64_any_dtype(cohort["admittime"]), "admittime should be datetime"

    def test_missing_data_handling(self):
        """Test handling of missing data scenarios."""
        # Test with subject that has no interventions
        results = extract_raw(
            con=self.con,
            initial_cohort_csv="csvs/initial_cohort.csv",
            labs_csv="csvs/labs_metadata.csv",
            vitals_csv="csvs/vital_metadata.csv",
        )
        
        cohort = results["cohort"]
        
        # Subject 3 should have limited interventions (only antibiotic)
        if len(cohort[cohort["subject_id"] == 3]) > 0:
            subject_3_data = cohort[cohort["subject_id"] == 3].iloc[0]
            
            # These should be 0 (no interventions seeded for subject 3 except antibiotic)
            assert subject_3_data["received_vasopressor"] == 0
            assert subject_3_data["received_sedation"] == 0
            assert subject_3_data["received_antibiotic"] == 1  # Subject 3 gets antibiotic
            assert subject_3_data["was_mechanically_ventilated"] == 0
            assert subject_3_data["received_rrt"] == 0
            assert subject_3_data["positive_blood_culture"] == 0


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    test_instance = TestDataExtractionComprehensive()
    
    # List of all test methods
    test_methods = [
        test_instance.test_constants_are_correct,
        test_instance.test_cohort_selection_criteria,
        test_instance.test_target_variable_calculations,
        test_instance.test_static_feature_extraction,
        test_instance.test_time_series_data_extraction,
        test_instance.test_categorical_normalization,
        test_instance.test_metadata_bounds_filtering,
        test_instance.test_unit_conversions,
        test_instance.test_first_admission_only,
        test_instance.test_edge_cases,
        test_instance.test_data_types_and_structure,
        test_instance.test_missing_data_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            # Set up for each test
            test_instance.setup_method()
            
            # Run the test
            test_method()
            
            print(f"✅ {test_method.__name__}")
            passed += 1
            
        except Exception as e:
            print(f"❌ {test_method.__name__}: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("🎉 All comprehensive tests passed! Data extraction is working correctly.")
    else:
        print(f"⚠️  {failed} tests failed. Please review the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    run_comprehensive_tests()