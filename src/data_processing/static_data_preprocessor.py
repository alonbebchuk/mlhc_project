"""
Static Data Preprocessing for ICU Patient Features

This module provides preprocessing functionality for static patient features,
including categorical encoding, numerical imputation, and feature scaling.

The StaticDataPreprocessor class handles:
1. Categorical encoding with rare category grouping
2. Numerical imputation using median values
3. Feature standardization using z-score normalization
4. Missingness indicator creation for features with missing values

Key preprocessing steps:
- Rare categorical values (frequency < 1%) are grouped into 'other' category
- Missing numerical values are imputed with training set medians
- All numerical features are standardized to zero mean and unit variance
- Binary missingness indicators are created for features that can be missing
"""
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

from .static_data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_COLUMNS_WITH_MISSING,
    BINARY_COLUMNS,
    STATIC_COLUMNS
)
from .logging_utils import logger

# Minimum frequency threshold for categorical values (1%)
# Values occurring less frequently are grouped into 'other' category
MIN_CATEGORY_FREQ = 0.01


class StaticDataPreprocessor:
    """
    Preprocessor for static patient features with categorical encoding and numerical scaling.
    
    This class provides a complete preprocessing pipeline for static patient data,
    handling categorical variables through one-hot encoding with rare category grouping,
    numerical variables through median imputation and standardization, and creation
    of missingness indicators for features that may be absent.
    
    The preprocessor maintains fitted parameters (encoders, imputers, scalers) to ensure
    consistent transformations between training, validation, and test data.
    
    Attributes:
        categorical_encoders (Dict): Fitted categorical encoders for each categorical column
        numerical_imputers (Dict): Fitted median imputers for each numerical column
        numerical_scalers (Dict): Fitted standard scalers for each numerical column
    """
    
    def __init__(self):
        """
        Initialize the static data preprocessor with empty parameter dictionaries.
        
        Creates empty dictionaries to store fitted preprocessing parameters that
        will be populated during the fit_transform phase.
        """
        logger.log_start("StaticDataPreprocessor.__init__")
        self.categorical_encoders = {}  # Will store column -> {value: encoded_name} mappings
        self.numerical_imputers = {}    # Will store column -> median_value mappings
        self.numerical_scalers = {}     # Will store column -> StandardScaler mappings
        logger.log_end("StaticDataPreprocessor.__init__")

    def _get_column_lists(self) -> Tuple[list, list]:
        """
        Generate ordered column lists for processed data and missingness indicators.
        
        Creates the final column ordering for the preprocessed data matrix by combining
        encoded categorical features, numerical features, binary features, and
        missingness indicator columns.
        
        Returns:
            Tuple[list, list]: 
                - data_columns: List of all feature column names in final order
                - missingness_columns: List of missingness indicator column names
                
        Note:
            This method must be called after categorical encoders are fitted since
            it relies on the encoder mappings to determine the encoded column names.
        """
        logger.log_start("StaticDataPreprocessor._get_column_lists")
        
        data_columns = []
        # Add encoded categorical column names (sorted for consistency)
        for col in CATEGORICAL_COLUMNS:
            encoder = self.categorical_encoders[col]
            data_columns.extend(sorted(set(encoder.values())))
        
        # Add numerical and binary columns
        data_columns.extend(NUMERIC_COLUMNS)
        data_columns.extend(BINARY_COLUMNS)
        
        # Create missingness indicator column names
        missingness_columns = [f"{col}_missing" for col in NUMERIC_COLUMNS_WITH_MISSING]
        
        logger.log_end("StaticDataPreprocessor._get_column_lists")
        return data_columns, missingness_columns

    def _create_categorical_encoder(self, series: pd.Series, column_name: str) -> Dict[str, str]:
        """
        Create categorical encoder that groups rare values into 'other' category.
        
        Analyzes value frequencies in the training data and creates an encoding mapping
        that preserves frequent values as separate categories while grouping rare values
        (frequency < 1%) into a single 'other' category to prevent overfitting.
        
        Args:
            series (pd.Series): Training data for the categorical column
            column_name (str): Name of the categorical column being encoded
            
        Returns:
            Dict[str, str]: Mapping from original values to encoded column names
                          e.g., {'Emergency': 'admission_type_Emergency', 
                                'Rare_Value': 'admission_type_other'}
                                
        Note:
            The encoder includes a mapping for '__UNKNOWN__' values to handle
            unseen categories in validation/test data.
        """
        logger.log_start("StaticDataPreprocessor._create_categorical_encoder")
        
        # Calculate value frequencies
        value_counts = series.value_counts()
        total_count = len(series)
        rare_mask = (value_counts / total_count) < MIN_CATEGORY_FREQ
        
        encoder = {}
        
        # Map frequent values to individual encoded columns
        frequent_values = value_counts[~rare_mask].index.tolist()
        for value in frequent_values:
            encoder[value] = f"{column_name}_{value}"
        
        # Map rare values and unknown values to 'other' category
        rare_values = value_counts[rare_mask].index.tolist() + ['__UNKNOWN__']
        for value in rare_values:
            encoder[value] = f"{column_name}_other"
            
        logger.log_end("StaticDataPreprocessor._create_categorical_encoder")
        return encoder

    def _apply_categorical_encoding(self, df: pd.DataFrame, column_name: str, encoder: Dict[str, str]) -> pd.DataFrame:
        """
        Apply categorical encoding using fitted encoder to create one-hot encoded features.
        
        Transforms categorical values to encoded names using the fitted encoder,
        then creates one-hot encoded binary columns. Ensures all expected columns
        are present even if not observed in the current data.
        
        Args:
            df (pd.DataFrame): DataFrame containing the categorical column
            column_name (str): Name of the categorical column to encode
            encoder (Dict[str, str]): Fitted encoder mapping values to column names
            
        Returns:
            pd.DataFrame: One-hot encoded DataFrame with binary columns for each category
                         Columns are sorted alphabetically for consistency
                         
        Note:
            Unknown values (not seen during fitting) are mapped to the 'other' category.
            Missing columns are added with all zeros to maintain consistent shape.
        """
        logger.log_start("StaticDataPreprocessor._apply_categorical_encoding")
        
        # Map categorical values using fitted encoder, handle unknowns
        mapped_values = df[column_name].map(encoder).fillna(encoder['__UNKNOWN__'])
        
        # Create one-hot encoded columns
        encoded_df = pd.get_dummies(mapped_values)
        
        # Ensure all expected columns are present (add missing columns as zeros)
        expected_columns = sorted(list(set(encoder.values())))
        for col in expected_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        
        # Return columns in consistent sorted order
        encoded_df = encoded_df[expected_columns]
        
        logger.log_end("StaticDataPreprocessor._apply_categorical_encoding")
        return encoded_df

    def fit_transform(self, static_data: np.ndarray) -> Tuple['StaticDataPreprocessor', np.ndarray, np.ndarray]:
        """
        Fit preprocessing parameters on training data and transform it.
        
        Learns all preprocessing parameters (categorical encoders, numerical imputers,
        and scalers) from the training data and applies these transformations to
        produce analysis-ready features with missingness indicators.
        
        Args:
            static_data (np.ndarray): Raw static features of shape (n_samples, n_features)
                                     with columns ordered according to STATIC_COLUMNS
                                     
        Returns:
            Tuple containing:
                - self: The fitted preprocessor instance
                - static_data (np.ndarray): Preprocessed static features
                - static_missingness (np.ndarray): Binary missingness indicators
                
        Processing steps:
            1. Categorical encoding with rare value grouping and one-hot encoding
            2. Numerical imputation using median values for missing data
            3. Feature standardization using z-score normalization
            4. Creation of missingness indicator features
        """
        logger.log_start("StaticDataPreprocessor.fit_transform")
        processed_df = pd.DataFrame(static_data, columns=STATIC_COLUMNS)

        # Process categorical columns: fit encoders and apply encoding
        for col in CATEGORICAL_COLUMNS:
            # Fit categorical encoder on training data
            encoder = self._create_categorical_encoder(processed_df[col], col)
            self.categorical_encoders[col] = encoder
            
            # Apply encoding and add to dataframe
            encoded_df = self._apply_categorical_encoding(processed_df, col, encoder)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df = processed_df.drop(columns=[col])

        # Process numerical columns: fit imputers/scalers and apply transformations
        for col in NUMERIC_COLUMNS:
            processed_df[col] = pd.to_numeric(processed_df[col])

            # Handle missing values for applicable columns
            if col in NUMERIC_COLUMNS_WITH_MISSING:
                # Create missingness indicator before imputation
                processed_df[f"{col}_missing"] = processed_df[col].isna().astype(int)
                
                # Fit median imputer and apply imputation
                median_value = processed_df[col].median()
                self.numerical_imputers[col] = median_value
                processed_df[col] = processed_df[col].fillna(median_value)

            # Fit standard scaler and apply standardization
            scaler = StandardScaler()
            processed_df[col] = scaler.fit_transform(processed_df[col].values.reshape(-1, 1)).flatten()
            self.numerical_scalers[col] = scaler

        # Extract final feature matrices in consistent column order
        data_columns, missingness_columns = self._get_column_lists()
        static_data = processed_df[data_columns].values
        static_missingness = processed_df[missingness_columns].values

        logger.log_end("StaticDataPreprocessor.fit_transform")
        return self, static_data, static_missingness

    def transform(self, static_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply fitted preprocessing transformations to new data.
        
        Uses previously fitted preprocessing parameters to transform new static data
        with the same transformations learned from training data. Ensures consistent
        feature representation between training, validation, and test sets.
        
        Args:
            static_data (np.ndarray): Raw static features of shape (n_samples, n_features)
                                     with columns ordered according to STATIC_COLUMNS
                                     
        Returns:
            Tuple containing:
                - static_data (np.ndarray): Preprocessed static features
                - static_missingness (np.ndarray): Binary missingness indicators
                
        Note:
            This method requires the preprocessor to be fitted first (via fit_transform).
            All transformations use parameters learned from training data to prevent
            data leakage.
        """
        logger.log_start("StaticDataPreprocessor.transform")
        processed_df = pd.DataFrame(static_data, columns=STATIC_COLUMNS)

        # Apply fitted categorical encodings (no refitting)
        for col in CATEGORICAL_COLUMNS:
            encoder = self.categorical_encoders[col]
            encoded_df = self._apply_categorical_encoding(processed_df, col, encoder)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df = processed_df.drop(columns=[col])

        # Apply fitted numerical transformations (no refitting)
        for col in NUMERIC_COLUMNS:
            processed_df[col] = pd.to_numeric(processed_df[col])

            # Handle missing values using fitted parameters
            if col in NUMERIC_COLUMNS_WITH_MISSING:
                # Create missingness indicator
                processed_df[f"{col}_missing"] = processed_df[col].isna().astype(int)
                # Apply fitted median imputation
                processed_df[col] = processed_df[col].fillna(self.numerical_imputers[col])

            # Apply fitted standardization
            processed_df[col] = self.numerical_scalers[col].transform(processed_df[col].values.reshape(-1, 1)).flatten()

        # Extract final feature matrices in consistent column order
        data_columns, missingness_columns = self._get_column_lists()
        static_data = processed_df[data_columns].values
        static_missingness = processed_df[missingness_columns].values

        logger.log_end("StaticDataPreprocessor.transform")
        return static_data, static_missingness

    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk using pickle serialization.
        
        Saves the complete preprocessor instance including all fitted parameters
        (categorical encoders, numerical imputers, scalers) to enable consistent
        transformations during model inference.
        
        Args:
            filepath (str): Path where the preprocessor pickle file should be saved
        """
        logger.log_start("StaticDataPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("StaticDataPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'StaticDataPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Loads a previously saved StaticDataPreprocessor instance with all
        fitted parameters intact for applying transformations to new data.
        
        Args:
            filepath (str): Path to the saved preprocessor pickle file
            
        Returns:
            StaticDataPreprocessor: Loaded preprocessor instance ready for use
            
        Note:
            The loaded preprocessor retains all fitted parameters and can immediately
            be used to transform new data without refitting.
        """
        logger.log_start("StaticDataPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("StaticDataPreprocessor.load")
        return preprocessor
