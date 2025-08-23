import numpy as np
import pickle
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from .logging_utils import logger


class TimeSeriesDataPreprocessor:
    def __init__(self):
        logger.log_start("TimeSeriesDataPreprocessor.__init__")
        self.global_medians = {}
        self.feature_scalers = {}
        logger.log_end("TimeSeriesDataPreprocessor.__init__")

    def _forward_fill(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor._forward_fill")
        n_patients, window_hours, n_features = timeseries_data.shape
        valid_mask = ~np.isnan(timeseries_data)
        time_indices = np.arange(window_hours)[None, :, None]
        time_indices = np.broadcast_to(time_indices, timeseries_data.shape)
        valid_time_indices = np.where(valid_mask, time_indices, -1)
        last_valid_indices = np.maximum.accumulate(valid_time_indices, axis=1)
        has_valid_predecessor = last_valid_indices >= 0
        patient_indices = np.arange(n_patients)[:, None, None]
        patient_indices = np.broadcast_to(patient_indices, timeseries_data.shape)
        feature_indices = np.arange(n_features)[None, None, :]
        feature_indices = np.broadcast_to(feature_indices, timeseries_data.shape)
        filled_data = np.where(
            has_valid_predecessor,
            timeseries_data[patient_indices, last_valid_indices, feature_indices],
            timeseries_data
        )
        logger.log_end("TimeSeriesDataPreprocessor._forward_fill")
        return filled_data

    def _fit_transform_temporal_imputation(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor._fit_transform_temporal_imputation")
        filled_data = self._forward_fill(timeseries_data)
        _, _, n_features = timeseries_data.shape
        reshaped_data = filled_data.reshape(-1, n_features)
        global_medians_array = np.nanmedian(reshaped_data, axis=0)
        self.global_medians = {}
        for feature_idx in range(n_features):
            if np.isnan(global_medians_array[feature_idx]):
                self.global_medians[feature_idx] = 0.0
            else:
                self.global_medians[feature_idx] = global_medians_array[feature_idx]
        remaining_nan_mask = np.isnan(filled_data)
        global_medians_broadcast = np.broadcast_to(
            global_medians_array[None, None, :],
            filled_data.shape
        )
        imputed_data = np.where(remaining_nan_mask, global_medians_broadcast, filled_data)
        logger.log_end("TimeSeriesDataPreprocessor._fit_transform_temporal_imputation")
        return imputed_data

    def _transform_temporal_imputation(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor._transform_temporal_imputation")
        filled_data = self._forward_fill(timeseries_data)
        _, _, n_features = timeseries_data.shape
        global_medians_array = np.array([self.global_medians[feature_idx] for feature_idx in range(n_features)])
        remaining_nan_mask = np.isnan(filled_data)
        global_medians_broadcast = np.broadcast_to(
            global_medians_array[None, None, :],
            filled_data.shape
        )
        imputed_data = np.where(remaining_nan_mask, global_medians_broadcast, filled_data)
        logger.log_end("TimeSeriesDataPreprocessor._transform_temporal_imputation")
        return imputed_data

    def _fit_transform_standardization(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor._fit_transform_standardization")
        _, _, n_features = timeseries_data.shape
        reshaped_data = timeseries_data.reshape(-1, n_features)
        feature_means = np.mean(reshaped_data, axis=0)
        feature_stds = np.std(reshaped_data, axis=0, ddof=0)
        zero_std_mask = feature_stds == 0
        if np.any(zero_std_mask):
            feature_stds[zero_std_mask] = 1.0
        self.feature_scalers = {}
        for feature_idx in range(n_features):
            scaler = StandardScaler()
            scaler.mean_ = feature_means[feature_idx]
            scaler.scale_ = feature_stds[feature_idx]
            scaler.var_ = feature_stds[feature_idx] ** 2
            scaler.n_features_in_ = 1
            self.feature_scalers[feature_idx] = scaler
        means_broadcast = np.broadcast_to(feature_means[None, None, :], timeseries_data.shape)
        stds_broadcast = np.broadcast_to(feature_stds[None, None, :], timeseries_data.shape)
        standardized_tensor = (timeseries_data - means_broadcast) / stds_broadcast
        logger.log_end("TimeSeriesDataPreprocessor._fit_transform_standardization")
        return standardized_tensor

    def _transform_standardization(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor._transform_standardization")
        _, _, n_features = timeseries_data.shape
        feature_means = np.array([self.feature_scalers[i].mean_ for i in range(n_features)])
        feature_stds = np.array([self.feature_scalers[i].scale_ for i in range(n_features)])
        means_broadcast = np.broadcast_to(feature_means[None, None, :], timeseries_data.shape)
        stds_broadcast = np.broadcast_to(feature_stds[None, None, :], timeseries_data.shape)
        standardized_tensor = (timeseries_data - means_broadcast) / stds_broadcast
        logger.log_end("TimeSeriesDataPreprocessor._transform_standardization")
        return standardized_tensor

    def fit_transform(self, timeseries_data: np.ndarray) -> Tuple['TimeSeriesDataPreprocessor', np.ndarray]:
        logger.log_start("TimeSeriesDataPreprocessor.fit_transform")
        imputed_data = self._fit_transform_temporal_imputation(timeseries_data)
        processed_data = self._fit_transform_standardization(imputed_data)
        logger.log_end("TimeSeriesDataPreprocessor.fit_transform")
        return self, processed_data

    def transform(self, timeseries_data: np.ndarray) -> np.ndarray:
        logger.log_start("TimeSeriesDataPreprocessor.transform")
        imputed_data = self._transform_temporal_imputation(timeseries_data)
        processed_data = self._transform_standardization(imputed_data)
        logger.log_end("TimeSeriesDataPreprocessor.transform")
        return processed_data

    def save(self, filepath: str) -> None:
        logger.log_start("TimeSeriesDataPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("TimeSeriesDataPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'TimeSeriesDataPreprocessor':
        logger.log_start("TimeSeriesDataPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("TimeSeriesDataPreprocessor.load")
        return preprocessor
