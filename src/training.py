"""
Training framework for ICU outcome prediction models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
import warnings
from tqdm import tqdm
import logging

from models.models import (
    MultiTaskModel, 
    SingleTaskModel, 
    UncertaintyWeightedLoss,
    create_model,
    get_model_name
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ICUDataset(Dataset):
    """Dataset class for ICU outcome prediction"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: Dict[str, np.ndarray],
        missingness: Optional[np.ndarray] = None,
        reconstruction_targets: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: [N, 48, D] - Feature tensor
            targets: Dictionary of target arrays [N,] for each task
            missingness: [N, 48, D] - Missingness indicators (optional)
            reconstruction_targets: [N, D] - Next timestep targets for reconstruction (optional)
        """
        self.features = torch.FloatTensor(features)
        self.targets = {k: torch.FloatTensor(v) for k, v in targets.items()}
        self.missingness = torch.FloatTensor(missingness) if missingness is not None else None
        self.reconstruction_targets = torch.FloatTensor(reconstruction_targets) if reconstruction_targets is not None else None
        
        # Validate shapes
        assert self.features.shape[0] == len(next(iter(self.targets.values()))), "Batch size mismatch"
        if self.missingness is not None:
            assert self.features.shape == self.missingness.shape, "Features and missingness shape mismatch"
        if self.reconstruction_targets is not None:
            assert self.features.shape[0] == self.reconstruction_targets.shape[0], "Reconstruction target batch size mismatch"
            
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            'features': self.features[idx],
            **{k: v[idx] for k, v in self.targets.items()}
        }
        
        if self.missingness is not None:
            item['missingness'] = self.missingness[idx]
            
        if self.reconstruction_targets is not None:
            item['reconstruction_target'] = self.reconstruction_targets[idx]
            
        return item


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score: Validation score (higher is better)
            model: Model to potentially save
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, task_name: str) -> Dict[str, float]:
        """Calculate binary classification metrics"""
        try:
            auroc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auroc = 0.5  # If only one class present
            
        try:
            auprc = average_precision_score(y_true, y_pred)
        except ValueError:
            auprc = np.mean(y_true)  # Baseline AUPRC
            
        try:
            brier = brier_score_loss(y_true, y_pred)
        except ValueError:
            brier = np.mean((y_true - y_pred) ** 2)
            
        return {
            f'{task_name}_auroc': auroc,
            f'{task_name}_auprc': auprc,
            f'{task_name}_brier': brier
        }
    
    @staticmethod
    def calculate_all_metrics(
        targets: Dict[str, np.ndarray], 
        predictions: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate metrics for all tasks"""
        all_metrics = {}
        
        for task_name in targets.keys():
            if task_name in predictions and task_name != 'reconstruction_target':
                metrics = MetricsCalculator.calculate_metrics(
                    targets[task_name], 
                    predictions[task_name], 
                    task_name
                )
                all_metrics.update(metrics)
                
        return all_metrics


class ModelTrainer:
    """Main training class"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        task_names: List[str],
        use_reconstruction: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_names = task_names
        self.use_reconstruction = use_reconstruction
        self.device = device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize loss function
        self.loss_fn = UncertaintyWeightedLoss(task_names, use_reconstruction)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            
            # Move to device
            features = batch['features'].to(self.device)
            missingness = batch.get('missingness', None)
            if missingness is not None:
                missingness = missingness.to(self.device)
                
            targets = {k: v.to(self.device) for k, v in batch.items() 
                      if k not in ['features', 'missingness']}
            
            # Forward pass
            predictions = self.model(features, missingness)
            
            # Calculate loss
            uncertainty_weights = self.model.get_uncertainty_weights()
            log_vars = self.model.get_log_vars()
            
            losses = self.loss_fn(predictions, targets, uncertainty_weights, log_vars)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate model on given data loader"""
        self.model.eval()
        total_loss = 0
        all_predictions = {task: [] for task in self.task_names}
        all_targets = {task: [] for task in self.task_names}
        
        if self.use_reconstruction:
            all_predictions['reconstruction'] = []
            all_targets['reconstruction_target'] = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                # Move to device
                features = batch['features'].to(self.device)
                missingness = batch.get('missingness', None)
                if missingness is not None:
                    missingness = missingness.to(self.device)
                    
                targets = {k: v.to(self.device) for k, v in batch.items() 
                          if k not in ['features', 'missingness']}
                
                # Forward pass
                predictions = self.model(features, missingness)
                
                # Calculate loss
                uncertainty_weights = self.model.get_uncertainty_weights()
                log_vars = self.model.get_log_vars()
                
                losses = self.loss_fn(predictions, targets, uncertainty_weights, log_vars)
                total_loss += losses['total_loss'].item()
                
                # Store predictions and targets
                for task in self.task_names:
                    if task in predictions:
                        all_predictions[task].extend(predictions[task].cpu().numpy())
                        all_targets[task].extend(targets[task].cpu().numpy())
                
                if self.use_reconstruction and 'reconstruction' in predictions:
                    all_predictions['reconstruction'].extend(predictions['reconstruction'].cpu().numpy())
                    all_targets['reconstruction_target'].extend(targets['reconstruction_target'].cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = {k: np.array(v) for k, v in all_predictions.items()}
        all_targets = {k: np.array(v) for k, v in all_targets.items()}
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(all_targets, all_predictions)
        
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, metrics
    
    def train(
        self, 
        num_epochs: int = 100,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            num_epochs: Maximum number of epochs
            save_path: Path to save the best model
            verbose: Whether to print progress
            
        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Calculate average AUROC for early stopping
            auroc_scores = [v for k, v in val_metrics.items() if k.endswith('_auroc')]
            avg_auroc = np.mean(auroc_scores) if auroc_scores else 0.5
            
            if verbose and epoch % 10 == 0:
                logger.info(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                           f"Val Loss = {val_loss:.4f}, Avg AUROC = {avg_auroc:.4f}")
            
            # Early stopping
            if self.early_stopping(avg_auroc, self.model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        test_loss, test_metrics = self.evaluate(self.test_loader)
        
        # Save model if path provided
        if save_path:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
                'test_metrics': test_metrics
            }, save_path)
            logger.info(f"Model saved to {save_path}")
        
        return {
            'history': self.history,
            'test_metrics': test_metrics,
            'final_epoch': epoch
        }


class ExperimentRunner:
    """Run all model experiments"""
    
    def __init__(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray],
        test_data: Dict[str, np.ndarray],
        input_dim: int,
        task_names: List[str] = ["mortality", "long_los", "readmission"],
        batch_size: int = 32,
        num_workers: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary  
            test_data: Test data dictionary
            input_dim: Input feature dimension
            task_names: List of task names
            batch_size: Batch size
            num_workers: Number of data loader workers
            device: Device to use
        """
        self.input_dim = input_dim
        self.task_names = task_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        # Create datasets
        self.train_dataset = ICUDataset(**train_data)
        self.val_dataset = ICUDataset(**val_data)
        self.test_dataset = ICUDataset(**test_data)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        self.results = {}
    
    def run_single_experiment(
        self,
        model_type: str,
        use_missingness: bool,
        use_reconstruction: bool,
        task_name: Optional[str] = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        save_models: bool = True
    ) -> Dict[str, Any]:
        """Run a single experiment configuration"""
        
        model_name = get_model_name(model_type, use_missingness, use_reconstruction, task_name)
        logger.info(f"Running experiment: {model_name}")
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_dim=self.input_dim,
            task_name=task_name,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_missingness=use_missingness,
            use_reconstruction=use_reconstruction,
            task_names=self.task_names
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            test_loader=self.test_loader,
            task_names=self.task_names if model_type == 'MTL' else [task_name],
            use_reconstruction=use_reconstruction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.device
        )
        
        # Train
        save_path = f"models/{model_name}.pt" if save_models else None
        if save_path:
            Path("models").mkdir(exist_ok=True)
            
        results = trainer.train(
            num_epochs=num_epochs,
            save_path=save_path,
            verbose=True
        )
        
        results['model_name'] = model_name
        results['config'] = {
            'model_type': model_type,
            'use_missingness': use_missingness,
            'use_reconstruction': use_reconstruction,
            'task_name': task_name,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay
        }
        
        return results
    
    def run_all_experiments(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 100,
        save_models: bool = True
    ) -> Dict[str, Any]:
        """
        Run all experiment configurations as specified in the paper:
        1. STM without reconstruction task and without missingness vector
        2. MTL without reconstruction task and without missingness vector  
        3. STM with reconstruction task and without missingness vector
        4. MTL with reconstruction task and without missingness vector
        5. STM with reconstruction task and with missingness vector
        6. MTL with reconstruction task and with missingness vector
        """
        
        experiments = [
            # 1. STM without reconstruction and without missingness
            *[{'model_type': 'STM', 'use_missingness': False, 'use_reconstruction': False, 'task_name': task}
              for task in self.task_names],
            
            # 2. MTL without reconstruction and without missingness  
            {'model_type': 'MTL', 'use_missingness': False, 'use_reconstruction': False, 'task_name': None},
            
            # 3. STM with reconstruction and without missingness
            *[{'model_type': 'STM', 'use_missingness': False, 'use_reconstruction': True, 'task_name': task}
              for task in self.task_names],
            
            # 4. MTL with reconstruction and without missingness
            {'model_type': 'MTL', 'use_missingness': False, 'use_reconstruction': True, 'task_name': None},
            
            # 5. STM with reconstruction and with missingness
            *[{'model_type': 'STM', 'use_missingness': True, 'use_reconstruction': True, 'task_name': task}
              for task in self.task_names],
            
            # 6. MTL with reconstruction and with missingness
            {'model_type': 'MTL', 'use_missingness': True, 'use_reconstruction': True, 'task_name': None},
        ]
        
        all_results = {}
        
        for exp_config in experiments:
            try:
                results = self.run_single_experiment(
                    **exp_config,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    num_epochs=num_epochs,
                    save_models=save_models
                )
                all_results[results['model_name']] = results
                
            except Exception as e:
                logger.error(f"Failed to run experiment {exp_config}: {str(e)}")
                continue
        
        # Save all results
        with open('experiment_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for k, v in all_results.items():
                json_results[k] = {
                    'model_name': v['model_name'],
                    'config': v['config'],
                    'test_metrics': v['test_metrics'],
                    'final_epoch': v['final_epoch']
                }
            json.dump(json_results, f, indent=2)
        
        return all_results


def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create summary table of results"""
    summary_data = []
    
    for model_name, result in results.items():
        row = {
            'Model': model_name,
            'Model Type': result['config']['model_type'],
            'Use Missingness': result['config']['use_missingness'],
            'Use Reconstruction': result['config']['use_reconstruction'],
            'Task': result['config'].get('task_name', 'All'),
            'Final Epoch': result['final_epoch']
        }
        
        # Add test metrics
        for metric_name, value in result['test_metrics'].items():
            row[metric_name.replace('_', ' ').title()] = value
            
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)