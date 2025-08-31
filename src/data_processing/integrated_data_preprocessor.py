"""
Integrated Data Preprocessing Pipeline for ICU Patient Data

This module provides the main preprocessing pipeline that integrates static and 
time-series data preprocessing, handles train/validation/test splits, and manages
the complete transformation workflow for machine learning model preparation.

The IntegratedICUPreprocessor class coordinates:
1. Data extraction from the MIMIC-III database
2. Static feature preprocessing (encoding, imputation, scaling)
3. Time-series feature preprocessing (temporal imputation, standardization)
4. Stratified dataset splitting to maintain outcome distribution balance
5. Preprocessor persistence for consistent test-time transformations

Key features:
- Stratified splitting based on multi-label outcomes
- Consistent preprocessing pipeline between train/val/test
- Serializable preprocessor for deployment
"""
import numpy as np
import pandas as pd
import pickle
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from .static_data_preprocessor import StaticDataPreprocessor
from .timeseries_data_preprocessor import TimeSeriesDataPreprocessor
from .data_extraction import extract_data
from .logging_utils import logger

# Dataset splitting configuration
VALIDATION_SIZE = 0.1    # 10% of training data reserved for validation
RANDOM_SEED = 42         # Fixed seed for reproducible splits


class IntegratedICUPreprocessor:
    """
    Integrated preprocessing pipeline for ICU patient static and time-series data.
    
    This class provides a unified interface for preprocessing both static patient features
    (demographics, treatments, procedures) and time-series features (vitals, lab values).
    It handles the complete workflow from raw data extraction to analysis-ready datasets.
    
    The preprocessor manages:
    - Static data preprocessing (categorical encoding, numerical imputation/scaling)
    - Time-series data preprocessing (temporal imputation, feature standardization)
    - Stratified train/validation splitting for multi-label classification
    - Consistent transformations between training and test data
    
    Attributes:
        static_preprocessor (StaticDataPreprocessor): Handles static feature processing
        timeseries_preprocessor (TimeSeriesDataPreprocessor): Handles time-series processing
    """
    
    def __init__(self):
        """
        Initialize the integrated preprocessor with sub-preprocessors for each data type.
        
        Creates instances of StaticDataPreprocessor and TimeSeriesDataPreprocessor
        that will be fitted during the training phase and applied consistently
        to validation and test data.
        """
        logger.log_start("IntegratedICUPreprocessor.__init__")
        self.static_preprocessor = StaticDataPreprocessor()
        self.timeseries_preprocessor = TimeSeriesDataPreprocessor()
        logger.log_end("IntegratedICUPreprocessor.__init__")

    def _create_stratification_labels(self, targets: np.ndarray) -> np.ndarray:
        """
        Create stratification labels for multi-label train/validation splitting.
        
        Combines the three binary outcome labels (mortality, LOS, readmission) into 
        a single string label for each patient. This enables stratified splitting
        that maintains the distribution of outcome combinations across splits.
        
        Args:
            targets (np.ndarray): 2D array of shape (n_patients, 3) with binary outcomes
            
        Returns:
            np.ndarray: 1D array of string labels representing outcome combinations
                       (e.g., "000", "001", "010", "011", "100", "101", "110", "111")
                       
        Example:
            If a patient has [0, 1, 0] outcomes, the stratification label is "010"
        """
        logger.log_start("IntegratedICUPreprocessor._create_stratification_labels")
        stratification_labels = []
        
        # Create combined label string for each patient's outcome combination
        for row in targets:
            label = (str(int(row[0])) + str(int(row[1])) + str(int(row[2])))
            stratification_labels.append(label)
            
        logger.log_end("IntegratedICUPreprocessor._create_stratification_labels")
        return np.array(stratification_labels)

    def transform(self, subject_ids: List[int]) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply fitted preprocessing transformations to new patient data.
        
        Extracts raw data for given subject IDs and applies the previously fitted
        preprocessing transformations for both static and time-series features.
        This method should be used for test data or new predictions after the
        preprocessor has been fitted on training data.
        
        Args:
            subject_ids (List[int]): Patient subject IDs to extract and preprocess
            
        Returns:
            Tuple containing preprocessed data:
                - hadm_ids (List[int]): Hospital admission IDs
                - static_data (np.ndarray): Preprocessed static features
                - static_missingness (np.ndarray): Static feature missingness indicators
                - timeseries_data (np.ndarray): Preprocessed time-series features
                - timeseries_missingness (np.ndarray): Time-series missingness indicators  
                - targets (np.ndarray): Binary target labels
                
        Note:
            The preprocessor must be fitted before calling this method (via fit_transform
            or create_train_val_test_splits). Otherwise, encoding/scaling parameters
            will not be available.
        """
        logger.log_start("IntegratedICUPreprocessor.transform")
        
        # Extract raw data from database
        hadm_ids, static_data, timeseries_data, timeseries_missingness, targets = extract_data(subject_ids)
        
        # Apply fitted static data transformations
        static_data, static_missingness = self.static_preprocessor.transform(static_data)
        
        # Apply fitted time-series transformations
        timeseries_data = self.timeseries_preprocessor.transform(timeseries_data)
        
        logger.log_end("IntegratedICUPreprocessor.transform")
        return hadm_ids, static_data, static_missingness, timeseries_data, timeseries_missingness, targets

    def create_train_val_test_splits(self, initial_cohort_subject_ids: List[int], test_example_subject_ids: List[int]) -> Tuple[
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Create stratified train/validation/test splits with integrated preprocessing.
        
        This method performs the complete data preparation workflow:
        1. Extracts data for the initial cohort and creates stratified train/val split
        2. Fits preprocessing transformations on training data
        3. Applies transformations to training and validation data
        4. Processes test data using fitted transformations
        
        The stratification ensures balanced distribution of outcome combinations
        across training and validation sets for robust model evaluation.
        
        Args:
            initial_cohort_subject_ids (List[int]): Subject IDs for training/validation cohort
            test_example_subject_ids (List[int]): Subject IDs for held-out test set
            
        Returns:
            Tuple of three dataset tuples (train_data, val_data, test_data), where each contains:
                - hadm_ids (List[int]): Hospital admission IDs
                - static_data (np.ndarray): Preprocessed static features
                - static_missingness (np.ndarray): Static feature missingness indicators
                - timeseries_data (np.ndarray): Preprocessed time-series features
                - timeseries_missingness (np.ndarray): Time-series missingness indicators
                - targets (np.ndarray): Binary target labels
                
        Note:
            This method fits the preprocessors on training data and applies those same
            transformations to validation and test data to prevent data leakage.
        """
        logger.log_start("IntegratedICUPreprocessor.create_train_val_test_splits")
        
        # Extract raw data for initial cohort (will be split into train/val)
        cohort_hadm_ids, cohort_static_data, cohort_timeseries_data, cohort_timeseries_missingness, cohort_targets = extract_data(initial_cohort_subject_ids)
        
        # Create stratification labels for balanced splitting
        stratification_labels = self._create_stratification_labels(cohort_targets)
        
        # Perform stratified train/validation split
        train_indices, val_indices = train_test_split(
            range(len(cohort_hadm_ids)),
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_SEED,
            stratify=stratification_labels
        )

        # Split training data using stratified indices
        train_hadm_ids = [cohort_hadm_ids[i] for i in train_indices]
        train_static_data = cohort_static_data[train_indices]
        train_timeseries_data = cohort_timeseries_data[train_indices]
        train_timeseries_missingness = cohort_timeseries_missingness[train_indices]
        train_targets = cohort_targets[train_indices]

        # Split validation data using stratified indices
        val_hadm_ids = [cohort_hadm_ids[i] for i in val_indices]
        val_static_data = cohort_static_data[val_indices]
        val_timeseries_data = cohort_timeseries_data[val_indices]
        val_timeseries_missingness = cohort_timeseries_missingness[val_indices]
        val_targets = cohort_targets[val_indices]

        # Fit preprocessors on training data and transform training set
        self.static_preprocessor, train_static_data, train_static_missingness = self.static_preprocessor.fit_transform(train_static_data)
        self.timeseries_preprocessor, train_timeseries_data = self.timeseries_preprocessor.fit_transform(train_timeseries_data)

        # Package training data tuple
        train_data = (train_hadm_ids, train_static_data, train_static_missingness, train_timeseries_data, train_timeseries_missingness, train_targets)

        # Apply fitted transformations to validation data (no fitting)
        val_static_data, val_static_missingness = self.static_preprocessor.transform(val_static_data)
        val_timeseries_data = self.timeseries_preprocessor.transform(val_timeseries_data)

        # Package validation data tuple
        val_data = (val_hadm_ids, val_static_data, val_static_missingness, val_timeseries_data, val_timeseries_missingness, val_targets)

        # Process test data using fitted transformations
        test_data = self.transform(test_example_subject_ids)

        logger.log_end("IntegratedICUPreprocessor.create_train_val_test_splits")
        return train_data, val_data, test_data

    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk using pickle serialization.
        
        Saves the complete preprocessor instance including all fitted parameters
        (encoders, imputers, scalers) to enable consistent transformations
        during model inference.
        
        Args:
            filepath (str): Path where the preprocessor pickle file should be saved
        """
        logger.log_start("IntegratedICUPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("IntegratedICUPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'IntegratedICUPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Loads a previously saved IntegratedICUPreprocessor instance with all
        fitted parameters intact for applying transformations to new data.
        
        Args:
            filepath (str): Path to the saved preprocessor pickle file
            
        Returns:
            IntegratedICUPreprocessor: Loaded preprocessor instance ready for use
            
        Note:
            The loaded preprocessor retains all fitted parameters and can immediately
            be used to transform new data without refitting.
        """
        logger.log_start("IntegratedICUPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("IntegratedICUPreprocessor.load")
        return preprocessor
