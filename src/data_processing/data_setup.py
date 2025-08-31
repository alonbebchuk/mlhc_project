"""
Data Setup and Pipeline Orchestration Module

This module provides the main data processing pipeline that coordinates:
1. Loading patient subject IDs from CSV files
2. Creating train/validation/test splits with integrated preprocessing
3. Saving processed datasets and fitted preprocessors to disk

The module handles the complete data pipeline from raw subject IDs to 
preprocessed, analysis-ready datasets for machine learning model training.

File structure:
- csvs/initial_cohort.csv: Patient IDs for training/validation cohort
- csvs/test_example.csv: Patient IDs for held-out test set
- data/: Directory for processed datasets and fitted preprocessors
"""
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np

from .integrated_data_preprocessor import IntegratedICUPreprocessor
from .logging_utils import logger

# Input CSV file paths
INITIAL_COHORT_CSV = "csvs/initial_cohort.csv"    # Training/validation patient IDs
TEST_EXAMPLE_CSV = "csvs/test_example.csv"        # Test set patient IDs

# Output directory for processed data
DATA_DIR = "data"

# Output file names for saved artifacts
PREPROCESSOR_FILE = "integrated_preprocessor.pkl"  # Fitted preprocessing pipeline
TRAIN_DATA_FILE = "train_data.pkl"                 # Training dataset
VAL_DATA_FILE = "val_data.pkl"                     # Validation dataset  
TEST_DATA_FILE = "test_data.pkl"                   # Test dataset


def create_data_directory():
    """
    Create the data directory for storing processed datasets and preprocessors.
    
    Creates the directory specified by DATA_DIR if it doesn't already exist.
    Uses exist_ok=True to avoid errors if directory already exists.
    """
    logger.log_start("create_data_directory")
    data_path = Path(DATA_DIR)
    data_path.mkdir(exist_ok=True)
    logger.log_end("create_data_directory")


def load_subject_ids(csv_path: str) -> List[int]:
    """
    Load patient subject IDs from a CSV file.
    
    Args:
        csv_path (str): Path to CSV file containing patient subject IDs
        
    Returns:
        List[int]: List of patient subject IDs extracted from the 'subject_id' column
        
    Note:
        The CSV file must contain a column named 'subject_id' with integer patient IDs.
    """
    logger.log_start("load_subject_ids")
    df = pd.read_csv(csv_path)
    subject_ids = df['subject_id'].tolist()
    logger.log_end("load_subject_ids")
    return subject_ids


def save_dataset(data_tuple: Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], filepath: str):
    """
    Save a complete dataset tuple to disk using pickle serialization.
    
    Packages all components of a processed dataset into a dictionary and saves
    it as a pickle file for later loading during model training or evaluation.
    
    Args:
        data_tuple: Tuple containing the complete dataset with components:
            - hadm_ids (List[int]): Hospital admission IDs
            - static_data (np.ndarray): Preprocessed static patient features
            - static_missingness (np.ndarray): Static feature missingness indicators
            - timeseries_data (np.ndarray): Preprocessed time-series features  
            - timeseries_missingness (np.ndarray): Time-series missingness indicators
            - targets (np.ndarray): Binary target labels for outcomes
        filepath (str): Path where the dataset pickle file should be saved
        
    Note:
        The saved dictionary contains keys: 'hadm_ids', 'static_data', 
        'static_missingness', 'timeseries_data', 'timeseries_missingness', 'targets'
    """
    logger.log_start("save_dataset")
    
    # Unpack the data tuple components
    hadm_ids, static_data, static_missingness, timeseries_data, timeseries_missingness, targets = data_tuple
    
    # Create dictionary with all dataset components
    dataset_dict = {
        'hadm_ids': hadm_ids,
        'static_data': static_data,
        'static_missingness': static_missingness,
        'timeseries_data': timeseries_data,
        'timeseries_missingness': timeseries_missingness,
        'targets': targets
    }
    
    # Save dataset to pickle file
    with open(filepath, 'wb') as f:
        pickle.dump(dataset_dict, f)
        
    logger.log_end("save_dataset")


def main():
    """
    Execute the complete data processing pipeline.
    
    This function orchestrates the entire data processing workflow:
    1. Creates output directory structure
    2. Loads patient subject IDs from CSV files
    3. Initializes integrated preprocessor and creates train/val/test splits
    4. Saves fitted preprocessor and all datasets to disk
    
    The function processes two cohorts:
    - Initial cohort: Used for training and validation (with stratified split)
    - Test example cohort: Used as held-out test set
    
    All outputs are saved to the data/ directory as pickle files for later use.
    """
    logger.log_start("main")
    
    # Create output directory for processed data
    create_data_directory()
    
    # Load patient subject IDs from CSV files
    initial_cohort_subject_ids = load_subject_ids(INITIAL_COHORT_CSV)
    test_example_subject_ids = load_subject_ids(TEST_EXAMPLE_CSV)
    
    # Initialize integrated preprocessor and create dataset splits
    preprocessor = IntegratedICUPreprocessor()
    train_data, val_data, test_data = preprocessor.create_train_val_test_splits(
        initial_cohort_subject_ids,
        test_example_subject_ids
    )
    
    # Save fitted preprocessor for later use
    preprocessor_path = Path(DATA_DIR) / PREPROCESSOR_FILE
    preprocessor.save(preprocessor_path)
    
    # Define output paths for datasets
    train_path = Path(DATA_DIR) / TRAIN_DATA_FILE
    val_path = Path(DATA_DIR) / VAL_DATA_FILE
    test_path = Path(DATA_DIR) / TEST_DATA_FILE
    
    # Save all processed datasets
    save_dataset(train_data, train_path)
    save_dataset(val_data, val_path)
    save_dataset(test_data, test_path)
    
    logger.log_end("main")


if __name__ == "__main__":
    main()
