"""
Time-Series Data Preprocessing for ICU Patient Monitoring

This module provides preprocessing functionality for time-series patient monitoring data,
including temporal imputation and feature standardization to prepare data for machine
learning models.

The TimeSeriesDataPreprocessor class handles:
1. Forward-fill temporal imputation for missing time points
2. Global median imputation for values without temporal predecessors  
3. Feature-wise standardization using z-score normalization
4. Preservation of original missingness patterns for model awareness

Key preprocessing steps:
- Forward-fill imputation propagates the last observed value to fill gaps
- Global median imputation fills values at the start of time series
- All features are standardized to zero mean and unit variance
- Fitted parameters are saved for consistent test-time transformations

The preprocessing maintains temporal dependencies while ensuring numerical
stability for machine learning algorithms.
"""
import numpy as np
import pickle
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from .logging_utils import logger


class TimeSeriesDataPreprocessor:
    """
    Preprocessor for time-series patient monitoring data with temporal imputation and scaling.
    
    This class provides comprehensive preprocessing for time-series data including
    forward-fill temporal imputation, global median imputation for initial missing values,
    and feature-wise standardization. The preprocessor maintains temporal dependencies
    while ensuring numerical stability for machine learning models.
    
    The preprocessing pipeline follows these steps:
    1. Forward-fill imputation: Propagate last observed value forward in time
    2. Global median imputation: Fill remaining missing values with training medians  
    3. Feature standardization: Apply z-score normalization to each feature
    4. Parameter persistence: Save fitted parameters for consistent transformations
    
    Attributes:
        global_medians (Dict): Fitted median values for each feature for imputation
        feature_scalers (Dict): Fitted StandardScaler objects for each feature
    """
    
    def __init__(self):
        """
        Initialize the time-series preprocessor with empty parameter dictionaries.
        
        Creates empty dictionaries to store fitted preprocessing parameters that
        will be populated during the fit_transform phase.
        """
        logger.log_start("TimeSeriesDataPreprocessor.__init__")
        self.global_medians = {}    # Will store feature_idx -> median_value mappings
        self.feature_scalers = {}   # Will store feature_idx -> StandardScaler mappings
        logger.log_end("TimeSeriesDataPreprocessor.__init__")

    def _forward_fill(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Apply forward-fill imputation to time-series data.
        
        Propagates the last observed value forward in time to fill missing values,
        maintaining temporal dependencies and clinical plausibility. This approach
        assumes that patient states tend to persist between measurements.
        
        Args:
            timeseries_data (np.ndarray): 3D array of shape (n_patients, n_hours, n_features)
                                         containing time-series data with NaN for missing values
                                         
        Returns:
            np.ndarray: Forward-filled data with same shape, where missing values
                       are replaced by the most recent observed value for that feature
                       
        Note:
            Values without any prior observations remain as NaN and will be handled
            by subsequent global median imputation.
        """
        logger.log_start("TimeSeriesDataPreprocessor._forward_fill")
        
        n_patients, window_hours, n_features = timeseries_data.shape
        
        # Create mask for valid (non-NaN) values
        valid_mask = ~np.isnan(timeseries_data)
        
        # Create time index arrays for efficient indexing
        time_indices = np.arange(window_hours)[None, :, None]
        time_indices = np.broadcast_to(time_indices, timeseries_data.shape)
        
        # Get time indices for valid measurements, -1 for missing
        valid_time_indices = np.where(valid_mask, time_indices, -1)
        
        # Find the most recent valid time index for each position
        last_valid_indices = np.maximum.accumulate(valid_time_indices, axis=1)
        
        # Check if there's a valid predecessor for forward filling
        has_valid_predecessor = last_valid_indices >= 0
        
        # Create index arrays for advanced indexing
        patient_indices = np.arange(n_patients)[:, None, None]
        patient_indices = np.broadcast_to(patient_indices, timeseries_data.shape)
        feature_indices = np.arange(n_features)[None, None, :]
        feature_indices = np.broadcast_to(feature_indices, timeseries_data.shape)
        
        # Apply forward fill where valid predecessors exist
        filled_data = np.where(
            has_valid_predecessor,
            timeseries_data[patient_indices, last_valid_indices, feature_indices],
            timeseries_data
        )
        
        logger.log_end("TimeSeriesDataPreprocessor._forward_fill")
        return filled_data

    def _fit_transform_temporal_imputation(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Fit temporal imputation parameters on training data and apply imputation.
        
        Combines forward-fill imputation with global median imputation to create
        a complete imputation strategy. First applies forward-fill to maintain
        temporal dependencies, then uses global medians for remaining missing values.
        
        Args:
            timeseries_data (np.ndarray): Training time-series data of shape 
                                         (n_patients, n_hours, n_features)
                                         
        Returns:
            np.ndarray: Fully imputed data with no missing values
            
        Side effects:
            Fits and stores global median values for each feature in self.global_medians
        """
        logger.log_start("TimeSeriesDataPreprocessor._fit_transform_temporal_imputation")
        
        # Apply forward-fill imputation first
        filled_data = self._forward_fill(timeseries_data)
        
        # Calculate global medians for each feature from all non-NaN values
        _, _, n_features = timeseries_data.shape
        reshaped_data = filled_data.reshape(-1, n_features)
        global_medians_array = np.nanmedian(reshaped_data, axis=0)
        
        # Store global medians, using 0.0 for features with all NaN values
        self.global_medians = {}
        for feature_idx in range(n_features):
            if np.isnan(global_medians_array[feature_idx]):
                self.global_medians[feature_idx] = 0.0  # Fallback for features with no observations
            else:
                self.global_medians[feature_idx] = global_medians_array[feature_idx]
        
        # Apply global median imputation to remaining NaN values
        remaining_nan_mask = np.isnan(filled_data)
        global_medians_broadcast = np.broadcast_to(
            global_medians_array[None, None, :],
            filled_data.shape
        )
        imputed_data = np.where(remaining_nan_mask, global_medians_broadcast, filled_data)
        
        logger.log_end("TimeSeriesDataPreprocessor._fit_transform_temporal_imputation")
        return imputed_data

    def _transform_temporal_imputation(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Apply fitted temporal imputation to new data.
        
        Uses previously fitted global median values to impute missing values
        in new time-series data, ensuring consistent imputation between
        training and test data.
        
        Args:
            timeseries_data (np.ndarray): New time-series data to impute
            
        Returns:
            np.ndarray: Fully imputed data using fitted imputation parameters
            
        Note:
            This method requires the preprocessor to be fitted first. It applies
            the same forward-fill strategy followed by global median imputation
            using training-derived medians.
        """
        logger.log_start("TimeSeriesDataPreprocessor._transform_temporal_imputation")
        
        # Apply forward-fill imputation
        filled_data = self._forward_fill(timeseries_data)
        
        # Use fitted global medians for remaining missing values
        _, _, n_features = timeseries_data.shape
        global_medians_array = np.array([self.global_medians[feature_idx] for feature_idx in range(n_features)])
        
        # Apply global median imputation
        remaining_nan_mask = np.isnan(filled_data)
        global_medians_broadcast = np.broadcast_to(
            global_medians_array[None, None, :],
            filled_data.shape
        )
        imputed_data = np.where(remaining_nan_mask, global_medians_broadcast, filled_data)
        
        logger.log_end("TimeSeriesDataPreprocessor._transform_temporal_imputation")
        return imputed_data

    def _fit_transform_standardization(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Fit standardization parameters on training data and apply z-score normalization.
        
        Computes feature-wise means and standard deviations from training data and
        applies z-score standardization to center features at zero mean and unit variance.
        Handles edge cases where features have zero variance.
        
        Args:
            timeseries_data (np.ndarray): Imputed training time-series data
                                         of shape (n_patients, n_hours, n_features)
                                         
        Returns:
            np.ndarray: Standardized data with same shape, where each feature
                       has approximately zero mean and unit variance
                       
        Side effects:
            Fits and stores StandardScaler objects for each feature in self.feature_scalers
        """
        logger.log_start("TimeSeriesDataPreprocessor._fit_transform_standardization")
        
        # Reshape to 2D for statistical computation
        _, _, n_features = timeseries_data.shape
        reshaped_data = timeseries_data.reshape(-1, n_features)
        
        # Calculate feature-wise statistics
        feature_means = np.mean(reshaped_data, axis=0)
        feature_stds = np.std(reshaped_data, axis=0, ddof=0)
        
        # Handle features with zero variance (constant values)
        zero_std_mask = feature_stds == 0
        if np.any(zero_std_mask):
            feature_stds[zero_std_mask] = 1.0  # Prevent division by zero
        
        # Create and store StandardScaler objects for each feature
        self.feature_scalers = {}
        for feature_idx in range(n_features):
            scaler = StandardScaler()
            # Manually set fitted parameters to match our calculations
            scaler.mean_ = feature_means[feature_idx]
            scaler.scale_ = feature_stds[feature_idx]
            scaler.var_ = feature_stds[feature_idx] ** 2
            scaler.n_features_in_ = 1
            self.feature_scalers[feature_idx] = scaler
        
        # Apply standardization: (x - mean) / std
        means_broadcast = np.broadcast_to(feature_means[None, None, :], timeseries_data.shape)
        stds_broadcast = np.broadcast_to(feature_stds[None, None, :], timeseries_data.shape)
        standardized_tensor = (timeseries_data - means_broadcast) / stds_broadcast
        
        logger.log_end("TimeSeriesDataPreprocessor._fit_transform_standardization")
        return standardized_tensor

    def _transform_standardization(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Apply fitted standardization to new data.
        
        Uses previously fitted mean and standard deviation parameters to apply
        z-score standardization to new time-series data, ensuring consistent
        scaling between training and test data.
        
        Args:
            timeseries_data (np.ndarray): New imputed time-series data to standardize
            
        Returns:
            np.ndarray: Standardized data using fitted scaling parameters
            
        Note:
            This method requires the preprocessor to be fitted first. It applies
            the same standardization as used on training data.
        """
        logger.log_start("TimeSeriesDataPreprocessor._transform_standardization")
        
        # Extract fitted parameters from stored scalers
        _, _, n_features = timeseries_data.shape
        feature_means = np.array([self.feature_scalers[i].mean_ for i in range(n_features)])
        feature_stds = np.array([self.feature_scalers[i].scale_ for i in range(n_features)])
        
        # Apply standardization using fitted parameters
        means_broadcast = np.broadcast_to(feature_means[None, None, :], timeseries_data.shape)
        stds_broadcast = np.broadcast_to(feature_stds[None, None, :], timeseries_data.shape)
        standardized_tensor = (timeseries_data - means_broadcast) / stds_broadcast
        
        logger.log_end("TimeSeriesDataPreprocessor._transform_standardization")
        return standardized_tensor

    def fit_transform(self, timeseries_data: np.ndarray) -> Tuple['TimeSeriesDataPreprocessor', np.ndarray]:
        """
        Fit preprocessing parameters on training data and transform it.
        
        Learns all preprocessing parameters (imputation medians and standardization
        parameters) from the training data and applies the complete preprocessing
        pipeline to produce analysis-ready time-series features.
        
        Args:
            timeseries_data (np.ndarray): Raw training time-series data of shape
                                         (n_patients, n_hours, n_features) with NaN for missing values
                                         
        Returns:
            Tuple containing:
                - self: The fitted preprocessor instance
                - processed_data (np.ndarray): Fully processed time-series data with
                  no missing values and standardized features
                  
        Processing pipeline:
            1. Forward-fill temporal imputation
            2. Global median imputation for remaining missing values
            3. Feature-wise z-score standardization
        """
        logger.log_start("TimeSeriesDataPreprocessor.fit_transform")
        
        # Apply temporal imputation (forward-fill + global median)
        imputed_data = self._fit_transform_temporal_imputation(timeseries_data)
        
        # Apply feature standardization
        processed_data = self._fit_transform_standardization(imputed_data)
        
        logger.log_end("TimeSeriesDataPreprocessor.fit_transform")
        return self, processed_data

    def transform(self, timeseries_data: np.ndarray) -> np.ndarray:
        """
        Apply fitted preprocessing transformations to new data.
        
        Uses previously fitted preprocessing parameters to transform new time-series
        data with the same pipeline learned from training data, ensuring consistent
        feature representation between training, validation, and test sets.
        
        Args:
            timeseries_data (np.ndarray): New raw time-series data of shape
                                         (n_patients, n_hours, n_features) with NaN for missing values
                                         
        Returns:
            np.ndarray: Fully processed time-series data with no missing values
                       and features standardized using training parameters
                       
        Note:
            This method requires the preprocessor to be fitted first (via fit_transform).
            All transformations use parameters learned from training data to prevent
            data leakage.
        """
        logger.log_start("TimeSeriesDataPreprocessor.transform")
        
        # Apply fitted temporal imputation
        imputed_data = self._transform_temporal_imputation(timeseries_data)
        
        # Apply fitted standardization
        processed_data = self._transform_standardization(imputed_data)
        
        logger.log_end("TimeSeriesDataPreprocessor.transform")
        return processed_data

    def save(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk using pickle serialization.
        
        Saves the complete preprocessor instance including all fitted parameters
        (global medians and feature scalers) to enable consistent transformations
        during model inference.
        
        Args:
            filepath (str): Path where the preprocessor pickle file should be saved
        """
        logger.log_start("TimeSeriesDataPreprocessor.save")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.log_end("TimeSeriesDataPreprocessor.save")

    @classmethod
    def load(cls, filepath: str) -> 'TimeSeriesDataPreprocessor':
        """
        Load a fitted preprocessor from disk.
        
        Loads a previously saved TimeSeriesDataPreprocessor instance with all
        fitted parameters intact for applying transformations to new data.
        
        Args:
            filepath (str): Path to the saved preprocessor pickle file
            
        Returns:
            TimeSeriesDataPreprocessor: Loaded preprocessor instance ready for use
            
        Note:
            The loaded preprocessor retains all fitted parameters (global medians
            and feature scalers) and can immediately be used to transform new data
            without refitting.
        """
        logger.log_start("TimeSeriesDataPreprocessor.load")
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.log_end("TimeSeriesDataPreprocessor.load")
        return preprocessor
