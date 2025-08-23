"""
Gradient Boosted Machine baseline for ICU outcome prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import warnings
import logging

logger = logging.getLogger(__name__)

class GBMBaseline:
    """
    Gradient Boosted Machine baseline using XGBoost
    
    Uses flattened time-series features with summary statistics
    as described in the project plan
    """
    
    def __init__(self, target_columns: List[str]):
        """
        Args:
            target_columns: List of target column names
        """
        self.target_columns = target_columns
        self.models = {}
        self.feature_names = []
        
    def prepare_features(
        self, 
        features: np.ndarray, 
        missingness: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Prepare features for GBM by flattening time-series and computing summary statistics
        
        Args:
            features: [N, seq_len, feature_dim] - Time-series features
            missingness: [N, seq_len, feature_dim] - Missingness indicators (optional)
            
        Returns:
            Flattened features: [N, flattened_dim]
        """
        n_patients, seq_len, feature_dim = features.shape
        
        # Initialize feature list
        feature_list = []
        feature_names = []
        
        # 1. Static features (first timestep, assumed to be constant)
        static_features = features[:, 0, :]  # [N, feature_dim]
        feature_list.append(static_features)
        
        for i in range(feature_dim):
            feature_names.append(f'static_feature_{i}')
        
        # 2. Time-series summary statistics
        # Compute statistics only for time-varying features
        # Assuming static features are in the first part, time-series in the second part
        # This would need to be adjusted based on actual feature organization
        
        for i in range(feature_dim):
            feature_series = features[:, :, i]  # [N, seq_len]
            
            # Handle missing values for statistics computation
            if missingness is not None:
                missing_mask = missingness[:, :, i] == 1  # [N, seq_len]
                # Set missing values to NaN for proper statistics computation
                feature_series = feature_series.copy()
                feature_series[missing_mask] = np.nan
            
            # Compute summary statistics (ignoring NaN values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # Mean
                mean_vals = np.nanmean(feature_series, axis=1)
                mean_vals = np.nan_to_num(mean_vals, nan=0.0)
                feature_list.append(mean_vals.reshape(-1, 1))
                feature_names.append(f'feature_{i}_mean')
                
                # Median
                median_vals = np.nanmedian(feature_series, axis=1)
                median_vals = np.nan_to_num(median_vals, nan=0.0)
                feature_list.append(median_vals.reshape(-1, 1))
                feature_names.append(f'feature_{i}_median')
                
                # Min
                min_vals = np.nanmin(feature_series, axis=1)
                min_vals = np.nan_to_num(min_vals, nan=0.0)
                feature_list.append(min_vals.reshape(-1, 1))
                feature_names.append(f'feature_{i}_min')
                
                # Max
                max_vals = np.nanmax(feature_series, axis=1)
                max_vals = np.nan_to_num(max_vals, nan=0.0)
                feature_list.append(max_vals.reshape(-1, 1))
                feature_names.append(f'feature_{i}_max')
                
                # Standard deviation
                std_vals = np.nanstd(feature_series, axis=1)
                std_vals = np.nan_to_num(std_vals, nan=0.0)
                feature_list.append(std_vals.reshape(-1, 1))
                feature_names.append(f'feature_{i}_std')
                
                # Count of non-missing values
                if missingness is not None:
                    count_vals = np.sum(~missing_mask, axis=1).astype(float)
                    feature_list.append(count_vals.reshape(-1, 1))
                    feature_names.append(f'feature_{i}_count')
                    
                    # Fraction of missing values
                    missing_fraction = np.sum(missing_mask, axis=1) / seq_len
                    feature_list.append(missing_fraction.reshape(-1, 1))
                    feature_names.append(f'feature_{i}_missing_frac')
        
        # Concatenate all features
        combined_features = np.concatenate(feature_list, axis=1)
        
        # Store feature names for later use
        self.feature_names = feature_names
        
        logger.info(f"GBM features prepared: {combined_features.shape}")
        
        return combined_features
    
    def train_single_model(
        self,
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: np.ndarray,
        val_targets: np.ndarray,
        task_name: str
    ) -> xgb.XGBClassifier:
        """
        Train a single XGBoost model for one task
        
        Args:
            train_features: Training features
            train_targets: Training targets
            val_features: Validation features
            val_targets: Validation targets
            task_name: Name of the task
            
        Returns:
            Trained XGBoost model
        """
        logger.info(f"Training GBM model for task: {task_name}")
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'verbosity': 0
        }
        
        # Create and train model
        model = xgb.XGBClassifier(**params)
        
        model.fit(
            train_features, train_targets,
            eval_set=[(val_features, val_targets)],
            verbose=False
        )
        
        return model
    
    def evaluate_model(
        self,
        model: xgb.XGBClassifier,
        test_features: np.ndarray,
        test_targets: np.ndarray,
        task_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            test_features: Test features
            test_targets: Test targets
            task_name: Name of the task
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred_proba = model.predict_proba(test_features)[:, 1]
        
        # Calculate metrics
        try:
            auroc = roc_auc_score(test_targets, y_pred_proba)
        except ValueError:
            auroc = 0.5
            
        try:
            auprc = average_precision_score(test_targets, y_pred_proba)
        except ValueError:
            auprc = np.mean(test_targets)
            
        try:
            brier = brier_score_loss(test_targets, y_pred_proba)
        except ValueError:
            brier = np.mean((test_targets - y_pred_proba) ** 2)
        
        return {
            f'{task_name}_auroc': auroc,
            f'{task_name}_auprc': auprc,
            f'{task_name}_brier': brier
        }
    
    def get_feature_importance(self, task_name: str) -> pd.DataFrame:
        """
        Get feature importance for a specific task
        
        Args:
            task_name: Name of the task
            
        Returns:
            DataFrame with feature importance
        """
        if task_name not in self.models:
            raise ValueError(f"Model for task {task_name} not found")
        
        model = self.models[task_name]
        importance = model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def train_and_evaluate(
        self,
        train_features: np.ndarray,
        train_targets: Dict[str, np.ndarray],
        val_features: np.ndarray,
        val_targets: Dict[str, np.ndarray],
        test_features: np.ndarray,
        test_targets: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train and evaluate GBM models for all tasks
        
        Args:
            train_features: Training features
            train_targets: Training targets dictionary
            val_features: Validation features
            val_targets: Validation targets dictionary
            test_features: Test features
            test_targets: Test targets dictionary
            
        Returns:
            Dictionary with results for all tasks
        """
        results = {}
        
        for task_name in self.target_columns:
            if task_name in train_targets and task_name in test_targets:
                try:
                    # Train model
                    model = self.train_single_model(
                        train_features=train_features,
                        train_targets=train_targets[task_name],
                        val_features=val_features,
                        val_targets=val_targets[task_name],
                        task_name=task_name
                    )
                    
                    self.models[task_name] = model
                    
                    # Evaluate model
                    test_metrics = self.evaluate_model(
                        model=model,
                        test_features=test_features,
                        test_targets=test_targets[task_name],
                        task_name=task_name
                    )
                    
                    # Get feature importance
                    feature_importance = self.get_feature_importance(task_name)
                    
                    results[task_name] = {
                        'model': model,
                        'test_metrics': test_metrics,
                        'feature_importance': feature_importance
                    }
                    
                    logger.info(f"GBM {task_name} - AUROC: {test_metrics[f'{task_name}_auroc']:.4f}, "
                               f"AUPRC: {test_metrics[f'{task_name}_auprc']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to train GBM for task {task_name}: {str(e)}")
                    continue
        
        # Combine feature importance across all tasks
        if self.models:
            combined_importance = self._combine_feature_importance()
            results['feature_importance'] = combined_importance
        
        return results
    
    def _combine_feature_importance(self) -> pd.DataFrame:
        """
        Combine feature importance across all trained models
        
        Returns:
            DataFrame with combined feature importance
        """
        importance_dfs = []
        
        for task_name, model in self.models.items():
            df = self.get_feature_importance(task_name)
            df = df.rename(columns={'importance': f'{task_name}_importance'})
            importance_dfs.append(df[['feature', f'{task_name}_importance']])
        
        # Merge all importance dataframes
        combined_df = importance_dfs[0]
        for df in importance_dfs[1:]:
            combined_df = combined_df.merge(df, on='feature', how='outer')
        
        # Calculate average importance
        importance_cols = [col for col in combined_df.columns if col.endswith('_importance')]
        combined_df['avg_importance'] = combined_df[importance_cols].mean(axis=1)
        combined_df = combined_df.sort_values('avg_importance', ascending=False)
        
        return combined_df