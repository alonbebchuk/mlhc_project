"""
Static data preprocessor for ICU patient data.

This module provides a comprehensive preprocessing class for static patient features
including categorical encoding, numerical imputation and scaling, and missingness
indicator generation.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler

# Import column definitions from static_data module
from data_processing.static_data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_COLUMNS_WITH_MISSING,
    BINARY_COLUMNS
)

# Default minimum frequency threshold for categorical encoding
MIN_CATEGORY_FREQ = 0.01


class StaticDataPreprocessor:
    """
    Comprehensive preprocessor for static ICU patient data.

    This class handles categorical encoding, numerical imputation and scaling,
    and missingness indicator generation for static patient features.

    Key Features:
    - One-hot encoding for categorical variables with rare category aggregation
    - Mean imputation and z-score standardization for numerical variables
    - Missingness indicator generation
    - Handling of unseen categorical values in unseen data
    - Save/load functionality for model persistence
    """

    def __init__(self):
        """
        Initialize the static data preprocessor.

        The preprocessor handles:
        - Categorical variables: One-hot encoding with rare category aggregation
        - Numerical variables: Mean imputation (for columns with missing values) and z-score standardization  
        - Binary variables: Pass-through (no processing needed)
        - Missingness indicators: Generated for variables that can have missing values

        Note: The minimum category frequency threshold is set to MIN_CATEGORY_FREQ
              Categories appearing in less than MIN_CATEGORY_FREQ of training data are aggregated to 'other'.
        """
        self.is_fitted = False

        # Categorical encoding components
        # Dict[str, Dict[str, str]] - maps original values to encoded column names
        self.categorical_encoders = {}

        # Numerical processing components
        # Dict[str, float] - mean values for imputation
        self.numerical_imputers = {}
        # Dict[str, StandardScaler] - fitted scalers
        self.numerical_scalers = {}

    def _create_categorical_encoder(self, series: pd.Series, column_name: str) -> Dict[str, str]:
        """
        Create categorical encoder for a single column.

        This method handles all categorical encoding logic including:
        - Rare category aggregation (values < MIN_CATEGORY_FREQ → 'other')
        - Unknown value handling for test data (→ 'other')

        Args:
            series: Pandas series with categorical data
            column_name: Name of the column

        Returns:
            Dictionary mapping original values to encoded column names
        """
        # Calculate value counts and frequencies
        value_counts = series.value_counts()
        total_count = len(series)

        # Identify rare categories (less than MIN_CATEGORY_FREQ)
        rare_mask = (value_counts / total_count) < MIN_CATEGORY_FREQ

        # Create encoder mapping
        encoder = {}

        # Handle frequent values
        frequent_values = value_counts[~rare_mask].index.tolist()
        for value in frequent_values:
            encoder[value] = f"{column_name}_{value}"

        # Aggregate rare values to 'other'
        rare_values = value_counts[rare_mask].index.tolist() + ['__UNKNOWN__']
        for value in rare_values:
            encoder[value] = f"{column_name}_other"

        return encoder

    def _apply_categorical_encoding(self, df: pd.DataFrame, column_name: str, encoder: Dict[str, str]) -> pd.DataFrame:
        """
        Apply categorical encoding to a DataFrame column.

        This method ensures deterministic column ordering by sorting column names
        alphabetically. This is critical for one-hot encoding correctness - the
        same categorical value must always map to the same feature index.

        Args:
            df: DataFrame to process
            column_name: Column to encode
            encoder: Encoder mapping created by _create_categorical_encoder

        Returns:
            DataFrame with one-hot encoded columns in consistent sorted order
        """
        # Map values using encoder (unknown values get mapped to 'other')
        mapped_values = df[column_name].map(encoder).fillna(encoder['__UNKNOWN__'])

        # Create one-hot encoded DataFrame
        encoded_df = pd.get_dummies(mapped_values, prefix='', prefix_sep='')

        # Get expected columns (all possible encoder outputs) in deterministic order
        expected_columns = sorted(list(set(encoder.values())))

        # Ensure all expected columns exist (fill missing with 0)
        for col in expected_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0

        # Keep only expected columns in sorted order (ensures consistent ordering)
        encoded_df = encoded_df[expected_columns]

        return encoded_df

    def fit_transform(self, df: pd.DataFrame) -> Tuple['StaticDataPreprocessor', np.ndarray, np.ndarray]:
        """
        Fit the preprocessor on training data and transform it.

        This method performs the complete preprocessing pipeline:
        1. Categorical encoding: Creates one-hot encoders with rare category aggregation
        2. Numerical processing: Fits mean imputers and z-score scalers
        3. Binary processing: Pass-through (no fitting required)
        4. Missingness indicators: Creates indicators for missing values

        The fitted parameters are stored in the preprocessor instance for later use
        on validation/test data.

        Args:
            df (pd.DataFrame): Training DataFrame with static features.
                              Must contain columns defined in CATEGORICAL_COLUMNS,
                              NUMERIC_COLUMNS_WITHOUT_MISSING, and BINARY_COLUMNS.

        Returns:
            Tuple[StaticDataPreprocessor, np.ndarray, np.ndarray]: 
                - fitted_preprocessor: The fitted preprocessor instance
                - data_array: Preprocessed feature array, shape (n_samples, n_features)
                - missingness_array: Missingness indicators, shape (n_samples, n_missingness_features)
        """
        df = df.copy()
        data_parts = []
        missingness_parts = []

        # 1. Process categorical columns with one-hot encoding and rare category aggregation
        print("Processing categorical columns...")
        for col in sorted(CATEGORICAL_COLUMNS):
            print(f"  Encoding {col}...")
            # Create encoder that maps values to column names
            # Rare categories (< MIN_CATEGORY_FREQ) are aggregated to 'other'
            encoder = self._create_categorical_encoder(df[col], col)
            self.categorical_encoders[col] = encoder

            # Apply one-hot encoding using the created encoder
            encoded_df = self._apply_categorical_encoding(df, col, encoder)
            data_parts.append(encoded_df)

        # 2. Process numerical columns with imputation and standardization
        print("Processing numerical columns...")
        for col in sorted(NUMERIC_COLUMNS):
            print(f"  Processing {col}...")
            col_data = df[col].copy()

            # Handle missing values (only for missing values)
            if col in NUMERIC_COLUMNS_WITH_MISSING:
                # Create binary missingness indicator (1=missing, 0=observed)
                missingness_indicator = col_data.isna().astype(int)
                missingness_parts.append(pd.DataFrame({f"{col}_missing": missingness_indicator}))

                # Fit mean imputer and store for later use on test data
                mean_value = col_data.mean()
                self.numerical_imputers[col] = mean_value

                # Apply mean imputation to fill missing values
                col_data = col_data.fillna(mean_value)

            # Fit z-score standardizer (StandardScaler) and apply transformation
            scaler = StandardScaler()
            col_data_scaled = scaler.fit_transform(col_data.values.reshape(-1, 1)).flatten()
            self.numerical_scalers[col] = scaler

            # Store the standardized numerical data
            data_parts.append(pd.DataFrame({col: col_data_scaled}))

        # 3. Process binary columns (no preprocessing needed - pass through as-is)
        print("Processing binary columns...")
        for col in sorted(BINARY_COLUMNS):
            print(f"  Processing {col}...")
            # Binary features (0/1) don't need encoding, imputation, or scaling
            data_parts.append(pd.DataFrame({col: df[col]}))

        # 4. Combine all processed feature parts into final arrays
        print("Combining processed features...")

        # Concatenate all parts - column order is deterministic due to sorted categorical columns
        final_data = pd.concat(data_parts, axis=1)
        final_missingness = pd.concat(missingness_parts, axis=1)

        # Mark as fitted
        self.is_fitted = True

        print(f"Preprocessing complete. Features: {len(final_data.columns)}, Missingness indicators: {len(final_missingness.columns)}")

        return self, final_data.values, final_missingness.values

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using the fitted preprocessor.

        Applies the same preprocessing steps as fit_transform but using the fitted
        parameters (encoders, imputers, scalers) from the training data. This ensures
        consistent preprocessing between training and validation/test data.

        Key behaviors:
        - Categorical: Unknown values are mapped to 'other' category
        - Numerical: Uses fitted mean values for imputation and fitted scalers
        - Binary: Pass-through (no processing)
        - Missingness: Creates same indicators as training
        - Column order: Ensures same feature order as training data

        Args:
            df (pd.DataFrame): DataFrame to transform with the same column structure
                              as the training data.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - data_array: Preprocessed feature array, shape (n_samples, n_features)
                - missingness_array: Missingness indicators, shape (n_samples, n_missingness_features)

        Raises:
            ValueError: If the preprocessor hasn't been fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data. Call fit_transform first.")

        df = df.copy()
        data_parts = []
        missingness_parts = []

        # 1. Process categorical columns (in sorted order for consistency)
        for col in sorted(CATEGORICAL_COLUMNS):
            print(f"  Encoding {col}...")
            encoder = self.categorical_encoders[col]
            encoded_df = self._apply_categorical_encoding(df, col, encoder)
            data_parts.append(encoded_df)

        # 2. Process numerical columns (in sorted order for consistency)
        for col in sorted(NUMERIC_COLUMNS):
            print(f"  Processing {col}...")
            col_data = df[col].copy()

            # Handle missing values
            if col in NUMERIC_COLUMNS_WITH_MISSING:
                # Create missingness indicator
                missingness_indicator = col_data.isna().astype(int)
                missingness_parts.append(pd.DataFrame({f"{col}_missing": missingness_indicator}))

                # Apply imputation using fitted mean
                col_data = col_data.fillna(self.numerical_imputers[col])

            # Apply fitted scaler
            col_data_scaled = self.numerical_scalers[col].transform(col_data.values.reshape(-1, 1)).flatten()
            data_parts.append(pd.DataFrame({col: col_data_scaled}))

        # 3. Process binary columns (in sorted order for consistency)
        for col in sorted(BINARY_COLUMNS):
            print(f"  Processing {col}...")
            # Binary features (0/1) don't need encoding, imputation, or scaling
            data_parts.append(pd.DataFrame({col: df[col]}))

        # 4. Combine all processed parts
        print("Combining processed features...")

        # Concatenate all parts - column order is deterministic due to sorted categorical columns
        final_data = pd.concat(data_parts, axis=1)
        final_missingness = pd.concat(missingness_parts, axis=1)

        print(f"Preprocessing complete. Features: {len(final_data.columns)}, Missingness indicators: {len(final_missingness.columns)}")

        return final_data.values, final_missingness.values

    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.

        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor. Call fit_transform first.")

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'StaticDataPreprocessor':
        """
        Load a fitted preprocessor from disk.

        Args:
            filepath: Path to load the preprocessor from

        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)

        if not isinstance(preprocessor, cls):
            raise ValueError(f"Loaded object is not a {cls.__name__} instance")

        print(f"Preprocessor loaded from {filepath}")
        return preprocessor
