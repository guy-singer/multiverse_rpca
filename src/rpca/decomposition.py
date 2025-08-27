import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, NamedTuple
from dataclasses import dataclass

from .core import RPCADecomposer, InexactALM
from .utils import (
    create_frame_matrix, 
    reconstruct_frames, 
    compute_compression_ratio,
    matrix_nuclear_norm_torch,
    soft_threshold_torch
)


@dataclass
class RPCAResult:
    """Results from RPCA decomposition."""
    L: np.ndarray  # Low-rank component
    S: np.ndarray  # Sparse component
    L_frames: Optional[np.ndarray] = None  # Reconstructed low-rank frames
    S_frames: Optional[np.ndarray] = None  # Reconstructed sparse frames
    compression_stats: Optional[Dict] = None
    metadata: Optional[Dict] = None


class FrameRPCA:
    """RPCA decomposition specialized for video frame sequences."""
    
    def __init__(self, method: str = "inexact_alm", 
                 lambda_coeff: Optional[float] = None,
                 temporal_mode: bool = True,
                 **kwargs):
        self.method = method
        self.lambda_coeff = lambda_coeff  
        self.temporal_mode = temporal_mode
        self.kwargs = kwargs
        
        # Initialize decomposer
        if method == "inexact_alm":
            self.decomposer = InexactALM(lambda_coeff=lambda_coeff, **kwargs)
        else:
            raise ValueError(f"Unknown RPCA method: {method}")
            
    def decompose_frames(self, frames: np.ndarray) -> RPCAResult:
        """
        Apply RPCA to frame sequence.
        
        Args:
            frames: Shape (T, C, H, W) - time, channels, height, width
            
        Returns:
            RPCAResult with low-rank and sparse components
        """
        # Convert frames to matrix
        matrix, shape_info = create_frame_matrix(frames, self.temporal_mode)
        
        # Apply RPCA  
        L, S = self.decomposer.fit_transform(matrix)
        
        # Reconstruct frames
        L_frames, S_frames = reconstruct_frames(L, S, shape_info)
        
        # Compute statistics
        compression_stats = compute_compression_ratio(L, S, frames.shape)
        
        metadata = {
            'original_shape': frames.shape,
            'matrix_shape': matrix.shape,
            'temporal_mode': self.temporal_mode,
            'method': self.method
        }
        
        return RPCAResult(
            L=L, S=S, 
            L_frames=L_frames, S_frames=S_frames,
            compression_stats=compression_stats,
            metadata=metadata
        )


def rpca_preprocessing(frames: torch.Tensor, 
                      method: str = "inexact_alm",
                      lambda_coeff: Optional[float] = None,
                      cache_decomposition: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess frame batch with RPCA decomposition.
    
    Args:
        frames: Input frames (B, T, C, H, W)
        method: RPCA method to use
        lambda_coeff: Regularization parameter
        cache_decomposition: Whether to cache results
        
    Returns:
        (L_frames, S_frames): Low-rank and sparse frame tensors
    """
    batch_size = frames.shape[0]
    L_batch = []
    S_batch = []
    
    rpca = FrameRPCA(method=method, lambda_coeff=lambda_coeff)
    
    for b in range(batch_size):
        # Convert to numpy and apply RPCA
        frame_seq = frames[b].detach().cpu().numpy()  # (T, C, H, W)
        
        result = rpca.decompose_frames(frame_seq)
        
        # Convert back to torch tensors
        L_batch.append(torch.from_numpy(result.L_frames).float())
        S_batch.append(torch.from_numpy(result.S_frames).float())
    
    L_tensor = torch.stack(L_batch, dim=0).to(frames.device)
    S_tensor = torch.stack(S_batch, dim=0).to(frames.device)
    
    return L_tensor, S_tensor


def rpca_loss(pred_L: torch.Tensor, pred_S: torch.Tensor,
              target_L: torch.Tensor, target_S: torch.Tensor,
              lambda_lowrank: float = 1.0,
              lambda_sparse: float = 1.0, 
              lambda_consistency: float = 0.1,
              beta_nuclear: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Compute RPCA-aware loss function.
    
    Args:
        pred_L, pred_S: Predicted low-rank and sparse components
        target_L, target_S: Target low-rank and sparse components
        lambda_*: Loss weight coefficients
        beta_nuclear: Nuclear norm regularization weight
        
    Returns:
        Dictionary of loss components
    """
    # Reconstruction losses
    loss_lowrank = F.mse_loss(pred_L, target_L)
    loss_sparse = F.l1_loss(pred_S, target_S) 
    
    # Consistency loss: ensure L + S reconstructs the original
    pred_recon = pred_L + pred_S
    target_recon = target_L + target_S
    loss_consistency = F.mse_loss(pred_recon, target_recon)
    
    # Nuclear norm regularization on predicted low-rank component
    # Flatten spatial dimensions for nuclear norm computation
    B, T, C, H, W = pred_L.shape
    pred_L_flat = pred_L.reshape(B, T, C * H * W)
    
    nuclear_penalty = 0
    for b in range(B):
        # Compute nuclear norm for each batch element
        nuclear_penalty += matrix_nuclear_norm_torch(pred_L_flat[b])
    nuclear_penalty = nuclear_penalty / B
    
    # Sparsity regularization on predicted sparse component
    sparsity_penalty = torch.mean(torch.abs(pred_S))
    
    # Total loss
    total_loss = (lambda_lowrank * loss_lowrank + 
                  lambda_sparse * loss_sparse +
                  lambda_consistency * loss_consistency +
                  beta_nuclear * nuclear_penalty)
    
    return {
        'total_loss': total_loss,
        'loss_lowrank': loss_lowrank,
        'loss_sparse': loss_sparse, 
        'loss_consistency': loss_consistency,
        'nuclear_penalty': nuclear_penalty,
        'sparsity_penalty': sparsity_penalty
    }


class DualBranchRPCA(torch.nn.Module):
    """
    Dual-branch architecture for processing low-rank and sparse components separately.
    """
    
    def __init__(self, shared_backbone: torch.nn.Module,
                 sparse_head_channels: int = 64,
                 fusion_method: str = "concat"):
        super().__init__()
        self.shared_backbone = shared_backbone
        self.fusion_method = fusion_method
        
        # Lightweight sparse processing head
        self.sparse_head = torch.nn.Sequential(
            torch.nn.Conv2d(3, sparse_head_channels//4, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(sparse_head_channels//4, sparse_head_channels//2, 3, padding=1),
            torch.nn.ReLU(),  
            torch.nn.Conv2d(sparse_head_channels//2, sparse_head_channels, 3, padding=1),
            torch.nn.ReLU()
        )
        
        # Fusion module
        if fusion_method == "concat":
            # Assume backbone outputs same channels as sparse head
            self.fusion = torch.nn.Conv2d(
                sparse_head_channels * 2, sparse_head_channels, 1
            )
        elif fusion_method == "add":
            self.fusion = None  # Simple addition
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
    def forward(self, L: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Process low-rank and sparse components.
        
        Args:
            L: Low-rank component (B, T, C, H, W) 
            S: Sparse component (B, T, C, H, W)
            
        Returns:
            Fused output tensor
        """
        B, T, C, H, W = L.shape
        
        # Process low-rank component through shared backbone
        L_flat = L.reshape(B * T, C, H, W)
        L_features = self.shared_backbone(L_flat)  # (B*T, feat_dim, H', W')
        
        # Process sparse component through lightweight head  
        S_flat = S.reshape(B * T, C, H, W)
        S_features = self.sparse_head(S_flat)  # (B*T, sparse_dim, H', W')
        
        # Ensure compatible spatial dimensions
        if L_features.shape[-2:] != S_features.shape[-2:]:
            S_features = F.interpolate(
                S_features, size=L_features.shape[-2:], 
                mode='bilinear', align_corners=False
            )
            
        # Fusion
        if self.fusion_method == "concat":
            fused = torch.cat([L_features, S_features], dim=1)
            output = self.fusion(fused)
        elif self.fusion_method == "add":
            # Ensure channel compatibility for addition
            if L_features.shape[1] != S_features.shape[1]:
                min_channels = min(L_features.shape[1], S_features.shape[1])
                L_features = L_features[:, :min_channels]
                S_features = S_features[:, :min_channels]
            output = L_features + S_features
        
        # Reshape back to (B, T, ...)
        out_shape = (B, T) + output.shape[1:]
        return output.reshape(out_shape)


def create_rpca_cache_key(frames: torch.Tensor, method: str, lambda_coeff: float) -> str:
    """Create cache key for RPCA decomposition results."""
    # Use tensor hash and parameters to create unique key
    tensor_hash = hash(frames.data_ptr())
    param_hash = hash((method, lambda_coeff, tuple(frames.shape)))
    return f"rpca_{tensor_hash}_{param_hash}"