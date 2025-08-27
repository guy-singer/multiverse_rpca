import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class RPCALossConfig:
    """Configuration for RPCA loss functions."""
    lambda_lowrank: float = 1.0      # Weight for low-rank component loss
    lambda_sparse: float = 1.0       # Weight for sparse component loss
    lambda_consistency: float = 0.1   # Weight for consistency loss (L + S = D)
    beta_nuclear: float = 0.01       # Nuclear norm regularization weight
    gamma_sparsity: float = 0.001    # Sparsity regularization weight
    
    # Advanced loss components
    use_perceptual_loss: bool = False
    use_temporal_consistency: bool = False
    temporal_weight: float = 0.1
    
    # Adaptive weighting
    use_adaptive_weights: bool = False
    adaptation_rate: float = 0.01


class NuclearNormLoss(nn.Module):
    """Differentiable nuclear norm loss for promoting low-rank structure."""
    
    def __init__(self, regularization_weight: float = 0.01):
        super().__init__()
        self.weight = regularization_weight
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute nuclear norm of input tensor.
        
        Args:
            X: Input tensor (B, T, C, H, W) or (B, C, H, W)
            
        Returns:
            Nuclear norm loss
        """
        if X.dim() == 5:
            # Handle video sequences: compute nuclear norm over spatial-temporal dims
            B, T, C, H, W = X.shape
            X_reshaped = X.reshape(B, T, C * H * W)
            
            total_nuclear_norm = 0
            for b in range(B):
                # Compute SVD for each batch element
                U, S, V = torch.svd(X_reshaped[b])  # (T, T), (T,), (C*H*W, T)
                total_nuclear_norm += torch.sum(S)
                
            return self.weight * total_nuclear_norm / B
            
        elif X.dim() == 4:
            # Handle single frames: compute nuclear norm over spatial dims
            B, C, H, W = X.shape
            X_reshaped = X.reshape(B, C, H * W)
            
            total_nuclear_norm = 0
            for b in range(B):
                U, S, V = torch.svd(X_reshaped[b])
                total_nuclear_norm += torch.sum(S)
                
            return self.weight * total_nuclear_norm / B
        else:
            raise ValueError(f"Unsupported tensor dimension: {X.dim()}")


class SparsityLoss(nn.Module):
    """Sparsity promoting loss for sparse component."""
    
    def __init__(self, regularization_weight: float = 0.001, norm: str = "l1"):
        super().__init__()
        self.weight = regularization_weight
        self.norm = norm
        
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Compute sparsity loss.
        
        Args:
            S: Sparse component tensor
            
        Returns:
            Sparsity loss
        """
        if self.norm == "l1":
            return self.weight * torch.mean(torch.abs(S))
        elif self.norm == "l0_approx":
            # Smooth approximation to L0 norm
            epsilon = 1e-6
            return self.weight * torch.mean(torch.sqrt(S**2 + epsilon))
        else:
            raise ValueError(f"Unknown sparsity norm: {self.norm}")


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video sequences."""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            X: Video tensor (B, T, C, H, W)
            
        Returns:
            Temporal consistency loss
        """
        if X.dim() != 5:
            return torch.tensor(0.0, device=X.device)
            
        # Compute frame-to-frame differences
        temporal_diff = X[:, 1:] - X[:, :-1]  # (B, T-1, C, H, W)
        
        # L2 norm of temporal differences
        temporal_loss = torch.mean(temporal_diff ** 2)
        
        return self.weight * temporal_loss


class AdaptiveLossWeights(nn.Module):
    """Adaptive loss weight adjustment during training."""
    
    def __init__(self, initial_weights: Dict[str, float], adaptation_rate: float = 0.01):
        super().__init__()
        self.adaptation_rate = adaptation_rate
        
        # Initialize learnable weights
        for name, weight in initial_weights.items():
            self.register_parameter(
                f"weight_{name}", 
                nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            )
            
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Get current adaptive weights."""
        weights = {}
        for name, param in self.named_parameters():
            if name.startswith("weight_"):
                weight_name = name[7:]  # Remove "weight_" prefix
                weights[weight_name] = torch.sigmoid(param)  # Ensure positive
        return weights
        
    def forward(self, loss_components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted loss with adaptive weights.
        
        Args:
            loss_components: Dictionary of individual loss components
            
        Returns:
            Weighted total loss
        """
        adaptive_weights = self.get_weights()
        
        total_loss = torch.tensor(0.0, device=next(iter(loss_components.values())).device)
        for name, loss_value in loss_components.items():
            if name in adaptive_weights:
                total_loss += adaptive_weights[name] * loss_value
                
        return total_loss


class RPCALossFunction(nn.Module):
    """
    Comprehensive RPCA loss function with multiple components.
    """
    
    def __init__(self, config: RPCALossConfig):
        super().__init__()
        self.config = config
        
        # Initialize loss components
        self.nuclear_norm_loss = NuclearNormLoss(config.beta_nuclear)
        self.sparsity_loss = SparsityLoss(config.gamma_sparsity)
        
        if config.use_temporal_consistency:
            self.temporal_loss = TemporalConsistencyLoss(config.temporal_weight)
        else:
            self.temporal_loss = None
            
        # Adaptive weights
        if config.use_adaptive_weights:
            initial_weights = {
                'lowrank': config.lambda_lowrank,
                'sparse': config.lambda_sparse,
                'consistency': config.lambda_consistency
            }
            self.adaptive_weights = AdaptiveLossWeights(initial_weights, config.adaptation_rate)
        else:
            self.adaptive_weights = None
            
    def forward(self, pred_L: torch.Tensor, pred_S: torch.Tensor,
                target_L: torch.Tensor, target_S: torch.Tensor,
                return_components: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive RPCA loss.
        
        Args:
            pred_L: Predicted low-rank component
            pred_S: Predicted sparse component
            target_L: Target low-rank component
            target_S: Target sparse component
            return_components: Whether to return individual loss components
            
        Returns:
            Dictionary containing total loss and optionally individual components
        """
        # Basic reconstruction losses
        loss_lowrank = F.mse_loss(pred_L, target_L)
        loss_sparse = F.l1_loss(pred_S, target_S)
        
        # Consistency loss
        pred_combined = pred_L + pred_S
        target_combined = target_L + target_S
        loss_consistency = F.mse_loss(pred_combined, target_combined)
        
        # Regularization losses
        nuclear_penalty = self.nuclear_norm_loss(pred_L)
        sparsity_penalty = self.sparsity_loss(pred_S)
        
        # Optional temporal consistency
        temporal_penalty = torch.tensor(0.0, device=pred_L.device)
        if self.temporal_loss is not None and pred_L.dim() == 5:
            temporal_penalty = self.temporal_loss(pred_L)
            
        # Compose loss components
        loss_components = {
            'lowrank': loss_lowrank,
            'sparse': loss_sparse, 
            'consistency': loss_consistency
        }
        
        # Compute total loss
        if self.adaptive_weights is not None:
            total_loss = self.adaptive_weights(loss_components)
        else:
            total_loss = (self.config.lambda_lowrank * loss_lowrank +
                         self.config.lambda_sparse * loss_sparse +
                         self.config.lambda_consistency * loss_consistency)
                         
        # Add regularization terms
        total_loss += nuclear_penalty + sparsity_penalty + temporal_penalty
        
        # Prepare return dictionary
        result = {'total_loss': total_loss}
        
        if return_components:
            result.update({
                'loss_lowrank': loss_lowrank,
                'loss_sparse': loss_sparse,
                'loss_consistency': loss_consistency,
                'nuclear_penalty': nuclear_penalty,
                'sparsity_penalty': sparsity_penalty,
                'temporal_penalty': temporal_penalty
            })
            
            # Add adaptive weights if used
            if self.adaptive_weights is not None:
                result.update({f'weight_{k}': v for k, v in self.adaptive_weights.get_weights().items()})
                
        return result


def create_rpca_loss(config_dict: Dict[str, Any]) -> RPCALossFunction:
    """
    Create RPCA loss function from configuration dictionary.
    
    Args:
        config_dict: Configuration parameters
        
    Returns:
        Configured RPCA loss function
    """
    config = RPCALossConfig(**config_dict)
    return RPCALossFunction(config)


class ProgressiveRPCALoss(nn.Module):
    """
    Progressive RPCA loss that gradually increases decomposition emphasis during training.
    """
    
    def __init__(self, base_loss: RPCALossFunction, 
                 warmup_epochs: int = 10, 
                 decomposition_weight_schedule: str = "linear"):
        super().__init__()
        self.base_loss = base_loss
        self.warmup_epochs = warmup_epochs
        self.schedule = decomposition_weight_schedule
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update current training epoch."""
        self.current_epoch = epoch
        
    def _get_decomposition_weight(self) -> float:
        """Get current decomposition weight based on schedule."""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            
            if self.schedule == "linear":
                return progress
            elif self.schedule == "cosine":
                return 0.5 * (1 - torch.cos(torch.pi * progress))
            else:
                return progress
        else:
            return 1.0
            
    def forward(self, pred_L: torch.Tensor, pred_S: torch.Tensor,
                target_L: torch.Tensor, target_S: torch.Tensor,
                original_pred: torch.Tensor, original_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute progressive RPCA loss.
        
        Args:
            pred_L, pred_S: Predicted RPCA components
            target_L, target_S: Target RPCA components  
            original_pred: Predicted combined frames (L + S)
            original_target: Target combined frames
            
        Returns:
            Loss dictionary
        """
        # Get decomposition weight
        decomp_weight = self._get_decomposition_weight()
        
        # Compute RPCA loss
        rpca_loss_dict = self.base_loss(pred_L, pred_S, target_L, target_S, return_components=True)
        
        # Compute standard reconstruction loss
        standard_loss = F.mse_loss(original_pred, original_target)
        
        # Progressive combination
        total_loss = ((1 - decomp_weight) * standard_loss + 
                     decomp_weight * rpca_loss_dict['total_loss'])
        
        # Update result
        result = rpca_loss_dict.copy()
        result.update({
            'total_loss': total_loss,
            'standard_loss': standard_loss,
            'decomposition_weight': decomp_weight
        })
        
        return result