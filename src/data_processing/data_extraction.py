"""
Data Extraction Module for ICU Patient Data

This module provides the main entry point for extracting all required data
from the MIMIC-III database for a given set of patient subject IDs.

The module coordinates data extraction across three domains:
1. Cohort definition and target label generation
2. Static patient features (demographics, treatments, procedures)
3. Time-series features (vital signs and laboratory values)
"""
import duckdb
import numpy as np
from typing import List, Tuple
from .cohort_data import get_cohort_hadm_ids_and_targets
from .static_data import get_static_data
from .timeseries_data import get_timeseries_data
from .logging_utils import logger

# Path to the MIMIC-III DuckDB database file
DUCKDB_PATH = r"H:\My Drive\MIMIC-III\mimiciii.duckdb"


def extract_data(subject_ids: List[int]) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract comprehensive patient data from MIMIC-III database for given subject IDs.
    
    This is the main orchestrator function that coordinates extraction of cohort data,
    static features, and time-series features from the MIMIC-III database.
    
    Args:
        subject_ids (List[int]): List of patient subject IDs to extract data for
        
    Returns:
        Tuple containing:
            - hadm_ids (List[int]): Hospital admission IDs for patients meeting cohort criteria
            - static_data (np.ndarray): Static patient features (demographics, treatments, etc.)
            - timeseries_data (np.ndarray): Time-series data (vitals, labs) of shape (n_patients, 48_hours, n_features)
            - timeseries_missingness (np.ndarray): Binary missingness indicators for time-series data
            - targets (np.ndarray): Binary target labels for mortality, LOS, and readmission outcomes
            
    Note:
        This function manages the database connection lifecycle, opening it at the start
        and ensuring it's properly closed after all data extraction is complete.
    """
    logger.log_start("extract_data")
    
    # Establish database connection
    con = duckdb.connect(DUCKDB_PATH)
    
    # Extract cohort and target labels
    hadm_ids, targets = get_cohort_hadm_ids_and_targets(con, subject_ids)
    
    # Extract static patient features
    static_data = get_static_data(con, hadm_ids)
    
    # Extract time-series features and missingness indicators
    timeseries_data, timeseries_missingness = get_timeseries_data(con, hadm_ids)
    
    # Close database connection
    con.close()
    
    logger.log_end("extract_data")
    return hadm_ids, static_data, timeseries_data, timeseries_missingness, targets
