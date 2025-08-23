"""
Main script to run all ICU outcome prediction experiments
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse
import json
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import (
    ICUDataPreprocessor, 
    create_project_specific_splits, 
    get_feature_lists
)
from common import (
    INITIAL_COHORT_PATH, TEST_EXAMPLE_PATH, VALIDATION_SIZE, RANDOM_STATE,
    DEFAULT_STRATIFY_STRATEGY, TARGET_COLUMNS, SEQUENCE_LENGTH, DATA_OUTPUT_DIR, DATA_FILES
)
from training import ExperimentRunner, create_summary_table
from gbm_baseline import GBMBaseline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_data(data_dir: str = DATA_OUTPUT_DIR) -> Dict[str, pd.DataFrame]:
    """Load raw data from CSV files"""
    logger.info("Loading raw data...")
    
    data = {}
    for key, filename in DATA_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            data[key] = pd.read_csv(filepath)
            logger.info(f"Loaded {key}: {len(data[key])} records")
        else:
            logger.warning(f"File not found: {filepath}")
            data[key] = pd.DataFrame()  # Empty dataframe as fallback
    
    return data

def preprocess_data(
    raw_data: Dict[str, pd.DataFrame],
    static_features: list,
    timeseries_features: list,
    target_columns: list = TARGET_COLUMNS
) -> Dict[str, Dict[str, np.ndarray]]:
    """Preprocess raw data into train/val/test splits"""
    logger.info("Creating train/validation/test splits...")
    
    # Create splits according to plan: 80/10/10 split
    train_raw, val_raw, test_raw = create_project_specific_splits(
        initial_cohort_path=INITIAL_COHORT_PATH,
        test_example_path=TEST_EXAMPLE_PATH,
        cohort_df=raw_data['cohort'],
        labs_df=raw_data['labs'],
        vitals_df=raw_data['vitals'],
        targets_df=raw_data['targets'],
        val_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify_strategy=DEFAULT_STRATIFY_STRATEGY,
        target_columns=target_columns
    )
    
    logger.info("Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = ICUDataPreprocessor(
        static_features=static_features,
        timeseries_features=timeseries_features,
        target_columns=target_columns,
        sequence_length=SEQUENCE_LENGTH
    )
    
    # Fit and transform training data
    train_processed = preprocessor.fit_transform(
        train_raw['cohort'], train_raw['labs'], train_raw['vitals'], train_raw['targets']
    )
    
    # Transform validation and test data
    val_processed = preprocessor.transform(
        val_raw['cohort'], val_raw['labs'], val_raw['vitals'], val_raw['targets']
    )
    
    test_processed = preprocessor.transform(
        test_raw['cohort'], test_raw['labs'], test_raw['vitals'], test_raw['targets']
    )
    
    logger.info(f"Preprocessed data shapes:")
    logger.info(f"  Features: {train_processed['features'].shape}")
    logger.info(f"  Targets: {[f'{k}: {v.shape}' for k, v in train_processed['targets'].items()]}")
    
    return {
        'train': train_processed,
        'val': val_processed,
        'test': test_processed
    }, preprocessor

def run_gbm_baseline(
    processed_data: Dict[str, Dict[str, np.ndarray]],
    target_columns: list
) -> Dict[str, Any]:
    """Run GBM baseline experiments"""
    logger.info("Running GBM baseline experiments...")
    
    gbm_baseline = GBMBaseline(target_columns=target_columns)
    
    # Prepare data for GBM (flatten time series)
    train_features_flat = gbm_baseline.prepare_features(
        processed_data['train']['features'],
        processed_data['train']['missingness']
    )
    val_features_flat = gbm_baseline.prepare_features(
        processed_data['val']['features'],
        processed_data['val']['missingness']
    )
    test_features_flat = gbm_baseline.prepare_features(
        processed_data['test']['features'],
        processed_data['test']['missingness']
    )
    
    # Train and evaluate
    gbm_results = gbm_baseline.train_and_evaluate(
        train_features=train_features_flat,
        train_targets=processed_data['train']['targets'],
        val_features=val_features_flat,
        val_targets=processed_data['val']['targets'],
        test_features=test_features_flat,
        test_targets=processed_data['test']['targets']
    )
    
    return gbm_results

def run_deep_learning_experiments(
    processed_data: Dict[str, Dict[str, np.ndarray]],
    input_dim: int,
    target_columns: list,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run deep learning experiments"""
    logger.info("Running deep learning experiments...")
    
    # Create experiment runner
    experiment_runner = ExperimentRunner(
        train_data=processed_data['train'],
        val_data=processed_data['val'],
        test_data=processed_data['test'],
        input_dim=input_dim,
        task_names=target_columns,
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4)
    )
    
    # Run all experiments
    dl_results = experiment_runner.run_all_experiments(
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.3),
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        num_epochs=config.get('num_epochs', 100),
        save_models=config.get('save_models', True)
    )
    
    return dl_results

def create_results_summary(
    gbm_results: Dict[str, Any],
    dl_results: Dict[str, Any]
) -> pd.DataFrame:
    """Create comprehensive results summary"""
    logger.info("Creating results summary...")
    
    # Combine all results
    all_results = {}
    
    # Add GBM results
    for task, result in gbm_results.items():
        if task != 'feature_importance':
            model_name = f"GBM-{task}"
            all_results[model_name] = {
                'model_name': model_name,
                'config': {'model_type': 'GBM', 'task_name': task},
                'test_metrics': result['test_metrics'],
                'final_epoch': 'N/A'
            }
    
    # Add deep learning results
    all_results.update(dl_results)
    
    # Create summary table
    summary_df = create_summary_table(all_results)
    
    return summary_df

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run ICU outcome prediction experiments')
    parser.add_argument('--data_dir', type=str, default='data', 
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--config_file', type=str, default=None,
                       help='JSON config file with hyperparameters')
    parser.add_argument('--skip_gbm', action='store_true',
                       help='Skip GBM baseline experiments')
    parser.add_argument('--skip_dl', action='store_true', 
                       help='Skip deep learning experiments')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load configuration
    default_config = {
        'batch_size': 32,
        'num_workers': 4,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'save_models': True
    }
    
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        default_config.update(config)
    
    config = default_config
    logger.info(f"Using configuration: {config}")
    
    # Load raw data
    raw_data = load_raw_data(args.data_dir)
    
    # Get feature lists
    static_features, timeseries_features = get_feature_lists()
    target_columns = ["mortality", "long_los", "readmission"]
    
    # Preprocess data
    processed_data, preprocessor = preprocess_data(
        raw_data=raw_data,
        static_features=static_features,
        timeseries_features=timeseries_features,
        target_columns=target_columns
    )
    
    input_dim = processed_data['train']['features'].shape[2]
    logger.info(f"Input dimension: {input_dim}")
    
    # Initialize results
    gbm_results = {}
    dl_results = {}
    
    # Run GBM baseline
    if not args.skip_gbm:
        try:
            gbm_results = run_gbm_baseline(processed_data, target_columns)
            logger.info("GBM baseline completed successfully")
        except Exception as e:
            logger.error(f"GBM baseline failed: {str(e)}")
    
    # Run deep learning experiments
    if not args.skip_dl:
        try:
            dl_results = run_deep_learning_experiments(
                processed_data, input_dim, target_columns, config
            )
            logger.info("Deep learning experiments completed successfully")
        except Exception as e:
            logger.error(f"Deep learning experiments failed: {str(e)}")
    
    # Create results summary
    if gbm_results or dl_results:
        summary_df = create_results_summary(gbm_results, dl_results)
        
        # Save results
        summary_path = os.path.join(args.output_dir, 'results_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Results summary saved to {summary_path}")
        
        # Save detailed results
        all_results = {'gbm': gbm_results, 'deep_learning': dl_results}
        results_path = os.path.join(args.output_dir, 'detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)
        
    else:
        logger.warning("No experiments were run successfully")

if __name__ == "__main__":
    main()