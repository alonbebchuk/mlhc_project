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

MIN_CATEGORY_FREQ = 0.01


class StaticDataPreprocessor:
    def __init__(self):
        logger.log_start("StaticDataPreprocessor.__init__")
        self.categorical_encoders = {}
        self.numerical_imputers = {}
        self.numerical_scalers = {}
        logger.log_end("StaticDataPreprocessor.__init__")

    def _get_column_lists(self) -> Tuple[list, list]:
        logger.log_start("StaticDataPreprocessor._get_column_lists")
        data_columns = []
        for col in CATEGORICAL_COLUMNS:
            encoder = self.categorical_encoders[col]
            data_columns.extend(sorted(set(encoder.values())))
        data_columns.extend(NUMERIC_COLUMNS)
        data_columns.extend(BINARY_COLUMNS)
        missingness_columns = [f"{col}_missing" for col in NUMERIC_COLUMNS_WITH_MISSING]
        logger.log_end("StaticDataPreprocessor._get_column_lists")
        return data_columns, missingness_columns

    def _create_categorical_encoder(self, series: pd.Series, column_name: str) -> Dict[str, str]:
        logger.log_start("StaticDataPreprocessor._create_categorical_encoder")
        value_counts = series.value_counts()
        total_count = len(series)
        rare_mask = (value_counts / total_count) < MIN_CATEGORY_FREQ
        encoder = {}
        frequent_values = value_counts[~rare_mask].index.tolist()
        for value in frequent_values:
            encoder[value] = f"{column_name}_{value}"
        rare_values = value_counts[rare_mask].index.tolist() + ['__UNKNOWN__']
        for value in rare_values:
            encoder[value] = f"{column_name}_other"
        logger.log_end("StaticDataPreprocessor._create_categorical_encoder")
        return encoder

    def _apply_categorical_encoding(self, df: pd.DataFrame, column_name: str, encoder: Dict[str, str]) -> pd.DataFrame:
        logger.log_start("StaticDataPreprocessor._apply_categorical_encoding")
        mapped_values = df[column_name].map(encoder).fillna(encoder['__UNKNOWN__'])
        encoded_df = pd.get_dummies(mapped_values)
        expected_columns = sorted(list(set(encoder.values())))
        for col in expected_columns:
            if col not in encoded_df.columns:
                encoded_df[col] = 0
        encoded_df = encoded_df[expected_columns]
        logger.log_end("StaticDataPreprocessor._apply_categorical_encoding")
        return encoded_df

    def fit_transform(self, static_data: np.ndarray) -> Tuple['StaticDataPreprocessor', np.ndarray, np.ndarray]:
        logger.log_start("StaticDataPreprocessor.fit_transform")
        processed_df = pd.DataFrame(static_data, columns=STATIC_COLUMNS)

        for col in CATEGORICAL_COLUMNS:
            encoder = self._create_categorical_encoder(processed_df[col], col)
            self.categorical_encoders[col] = encoder
            encoded_df = self._apply_categorical_encoding(processed_df, col, encoder)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df = processed_df.drop(columns=[col])

        for col in NUMERIC_COLUMNS:
            processed_df[col] = pd.to_numeric(processed_df[col])

            if col in NUMERIC_COLUMNS_WITH_MISSING:
                processed_df[f"{col}_missing"] = processed_df[col].isna().astype(int)
                median_value = processed_df[col].median()
                self.numerical_imputers[col] = median_value
                processed_df[col] = processed_df[col].fillna(median_value)

            scaler = StandardScaler()
            processed_df[col] = scaler.fit_transform(processed_df[col].values.reshape(-1, 1)).flatten()
            self.numerical_scalers[col] = scaler

        data_columns, missingness_columns = self._get_column_lists()
        static_data = processed_df[data_columns].values
        static_missingness = processed_df[missingness_columns].values

        logger.log_end("StaticDataPreprocessor.fit_transform")
        return self, static_data, static_missingness

    def transform(self, static_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        logger.log_start("StaticDataPreprocessor.transform")
        processed_df = pd.DataFrame(static_data, columns=STATIC_COLUMNS)

        for col in CATEGORICAL_COLUMNS:
            encoder = self.categorical_encoders[col]
            encoded_df = self._apply_categorical_encoding(processed_df, col, encoder)
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df = processed_df.drop(columns=[col])

        for col in NUMERIC_COLUMNS:
            processed_df[col] = pd.to_numeric(processed_df[col])

            if col in NUMERIC_COLUMNS_WITH_MISSING:
                processed_df[f"{col}_missing"] = processed_df[col].isna().astype(int)
                processed_df[col] = processed_df[col].fillna(self.numerical_imputers[col])

            processed_df[col] = self.numerical_scalers[col].transform(processed_df[col].values.reshape(-1, 1)).flatten()

        data_columns, missingness_columns = self._get_column_lists()
        static_data = processed_df[data_columns].values
        static_missingness = processed_df[missingness_columns].values

        logger.log_end("StaticDataPreprocessor.transform")
        return static_data, static_missingness

    def save(self, filepath: str) -> None:
        logger.log_start("StaticDataPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("StaticDataPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'StaticDataPreprocessor':
        logger.log_start("StaticDataPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("StaticDataPreprocessor.load")
        return preprocessor
