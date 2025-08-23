"""
ICU Outcome Prediction Models
Adapted from Dynamic-DeepHit for ICU binary outcome prediction with uncertainty weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

class LongitudinalAttention(nn.Module):
    """Longitudinal attention mechanism from Dynamic-DeepHit"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.W_att = nn.Linear(input_dim, hidden_dim)
        self.V_att = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, rnn_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rnn_outputs: [batch_size, seq_len, hidden_dim]
        Returns:
            attended_output: [batch_size, hidden_dim]
        """
        # Compute attention scores
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, hidden_dim]
        att_hidden = self.tanh(self.W_att(rnn_outputs))
        
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, 1]
        att_scores = self.V_att(att_hidden)
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len, 1]
        att_weights = F.softmax(att_scores, dim=1)
        
        # Weighted sum: [batch_size, seq_len, hidden_dim] * [batch_size, seq_len, 1]
        # -> [batch_size, hidden_dim]
        attended_output = torch.sum(att_weights * rnn_outputs, dim=1)
        
        return attended_output


class RNNBackbone(nn.Module):
    """RNN backbone with longitudinal attention (adapted from Dynamic-DeepHit)"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_missingness: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_missingness = use_missingness
        
        # If using missingness, double the input dimension
        effective_input_dim = input_dim * 2 if use_missingness else input_dim
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=effective_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Longitudinal attention
        self.attention = LongitudinalAttention(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        m: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - feature values
            m: [batch_size, seq_len, input_dim] - missingness indicators (optional)
        Returns:
            attended_features: [batch_size, hidden_dim]
        """
        if self.use_missingness:
            if m is None:
                raise ValueError("Missingness tensor required when use_missingness=True")
            # Concatenate features and missingness indicators
            inputs = torch.cat([x, m], dim=-1)
        else:
            inputs = x
            
        # GRU forward pass
        gru_outputs, _ = self.gru(inputs)  # [batch_size, seq_len, hidden_dim]
        
        # Apply longitudinal attention
        attended_output = self.attention(gru_outputs)  # [batch_size, hidden_dim]
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        return attended_output


class ReconstructionHead(nn.Module):
    """Reconstruction head for next timestep prediction"""
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TaskHead(nn.Module):
    """Binary classification head for each task"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MultiTaskModel(nn.Module):
    """Multi-Task Learning model for ICU outcomes"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_missingness: bool = False,
        use_reconstruction: bool = False,
        task_names: List[str] = ["mortality", "long_los", "readmission"]
    ):
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.use_reconstruction = use_reconstruction
        
        # Shared RNN backbone
        self.backbone = RNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_missingness=use_missingness
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: TaskHead(hidden_dim, dropout)
            for task in task_names
        })
        
        # Reconstruction head (optional)
        if use_reconstruction:
            self.reconstruction_head = ReconstructionHead(hidden_dim, input_dim)
        
        # Uncertainty parameters for loss weighting
        self.log_vars = nn.Parameter(torch.zeros(self.num_tasks + (1 if use_reconstruction else 0)))
        
    def forward(
        self, 
        x: torch.Tensor, 
        m: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            m: [batch_size, seq_len, input_dim] (optional)
        Returns:
            Dictionary with task predictions and reconstruction (if enabled)
        """
        # Get shared representation
        shared_features = self.backbone(x, m)
        
        # Task predictions
        outputs = {}
        for task_name in self.task_names:
            logits = self.task_heads[task_name](shared_features)
            outputs[task_name] = torch.sigmoid(logits.squeeze(-1))
            
        # Reconstruction prediction
        if self.use_reconstruction:
            # Predict next timestep (last hour features)
            outputs['reconstruction'] = self.reconstruction_head(shared_features)
            
        return outputs
    
    def get_uncertainty_weights(self) -> torch.Tensor:
        """Get uncertainty weights for loss computation"""
        return torch.exp(-self.log_vars)
    
    def get_log_vars(self) -> torch.Tensor:
        """Get log variance parameters"""
        return self.log_vars


class SingleTaskModel(nn.Module):
    """Single Task Learning model for one ICU outcome"""
    
    def __init__(
        self,
        input_dim: int,
        task_name: str,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_missingness: bool = False,
        use_reconstruction: bool = False
    ):
        super().__init__()
        self.task_name = task_name
        self.use_reconstruction = use_reconstruction
        
        # Shared RNN backbone
        self.backbone = RNNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_missingness=use_missingness
        )
        
        # Task head
        self.task_head = TaskHead(hidden_dim, dropout)
        
        # Reconstruction head (optional)
        if use_reconstruction:
            self.reconstruction_head = ReconstructionHead(hidden_dim, input_dim)
            
        # Uncertainty parameters for loss weighting (task + reconstruction if used)
        num_losses = 1 + (1 if use_reconstruction else 0)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(
        self, 
        x: torch.Tensor, 
        m: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            m: [batch_size, seq_len, input_dim] (optional)
        Returns:
            Dictionary with task prediction and reconstruction (if enabled)
        """
        # Get shared representation
        shared_features = self.backbone(x, m)
        
        # Task prediction
        logits = self.task_head(shared_features)
        outputs = {
            self.task_name: torch.sigmoid(logits.squeeze(-1))
        }
        
        # Reconstruction prediction
        if self.use_reconstruction:
            outputs['reconstruction'] = self.reconstruction_head(shared_features)
            
        return outputs
    
    def get_uncertainty_weights(self) -> torch.Tensor:
        """Get uncertainty weights for loss computation"""
        return torch.exp(-self.log_vars)
    
    def get_log_vars(self) -> torch.Tensor:
        """Get log variance parameters"""
        return self.log_vars


class UncertaintyWeightedLoss(nn.Module):
    """Uncertainty-weighted loss for multi-task learning"""
    
    def __init__(self, task_names: List[str], use_reconstruction: bool = False):
        super().__init__()
        self.task_names = task_names
        self.use_reconstruction = use_reconstruction
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        uncertainty_weights: torch.Tensor,
        log_vars: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-weighted loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainty_weights: exp(-log_vars)
            log_vars: Log variance parameters
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0
        loss_idx = 0
        
        # Task losses
        for task_name in self.task_names:
            if task_name in predictions and task_name in targets:
                task_loss = self.bce_loss(predictions[task_name], targets[task_name])
                
                # Apply uncertainty weighting
                weighted_loss = uncertainty_weights[loss_idx] * task_loss + 0.5 * log_vars[loss_idx]
                
                losses[f'{task_name}_loss'] = task_loss
                losses[f'{task_name}_weighted_loss'] = weighted_loss
                total_loss += weighted_loss
                loss_idx += 1
        
        # Reconstruction loss
        if self.use_reconstruction and 'reconstruction' in predictions and 'reconstruction_target' in targets:
            recon_loss = self.mse_loss(predictions['reconstruction'], targets['reconstruction_target'])
            
            # Apply uncertainty weighting
            weighted_recon_loss = uncertainty_weights[loss_idx] * recon_loss + 0.5 * log_vars[loss_idx]
            
            losses['reconstruction_loss'] = recon_loss
            losses['reconstruction_weighted_loss'] = weighted_recon_loss
            total_loss += weighted_recon_loss
            
        losses['total_loss'] = total_loss
        return losses


def create_model(
    model_type: str,
    input_dim: int,
    task_name: Optional[str] = None,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    use_missingness: bool = False,
    use_reconstruction: bool = False,
    task_names: List[str] = ["mortality", "long_los", "readmission"]
) -> Union[MultiTaskModel, SingleTaskModel]:
    """
    Factory function to create models
    
    Args:
        model_type: 'MTL' or 'STM'
        input_dim: Input feature dimension
        task_name: Required for STM models
        hidden_dim: Hidden dimension
        num_layers: Number of GRU layers
        dropout: Dropout rate
        use_missingness: Whether to use missingness indicators
        use_reconstruction: Whether to include reconstruction task
        task_names: List of task names for MTL
        
    Returns:
        Model instance
    """
    if model_type == 'MTL':
        return MultiTaskModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_missingness=use_missingness,
            use_reconstruction=use_reconstruction,
            task_names=task_names
        )
    elif model_type == 'STM':
        if task_name is None:
            raise ValueError("task_name required for STM models")
        return SingleTaskModel(
            input_dim=input_dim,
            task_name=task_name,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_missingness=use_missingness,
            use_reconstruction=use_reconstruction
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_model_name(
    model_type: str,
    use_missingness: bool,
    use_reconstruction: bool,
    task_name: Optional[str] = None
) -> str:
    """Generate model name based on configuration"""
    name_parts = [model_type]
    
    if use_missingness:
        name_parts.append("Full")
    else:
        name_parts.append("Baseline")
        
    if use_reconstruction:
        name_parts.append("Recon")
        
    if task_name:
        name_parts.append(task_name)
        
    return "-".join(name_parts)