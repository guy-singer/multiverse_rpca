import numpy as np
import torch
from scipy.linalg import svd
from typing import Tuple, Optional


def matrix_nuclear_norm(X: np.ndarray) -> float:
    """Compute nuclear norm (sum of singular values) of matrix X."""
    _, s, _ = svd(X, full_matrices=False)
    return np.sum(s)


def matrix_nuclear_norm_torch(X: torch.Tensor) -> torch.Tensor:
    """Compute nuclear norm using PyTorch for differentiability."""
    return torch.sum(torch.svd(X)[1])


def soft_threshold(X: np.ndarray, threshold: float) -> np.ndarray:
    """Apply soft thresholding (element-wise shrinkage) to matrix X."""
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def soft_threshold_torch(X: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply soft thresholding using PyTorch."""
    return torch.sign(X) * torch.clamp(torch.abs(X) - threshold, min=0)


def singular_value_shrinkage(X: np.ndarray, threshold: float) -> np.ndarray:
    """Apply singular value shrinkage (nuclear norm soft thresholding)."""
    U, s, Vt = svd(X, full_matrices=False)
    s_shrunk = np.maximum(s - threshold, 0)
    return U @ np.diag(s_shrunk) @ Vt


def singular_value_shrinkage_torch(X: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply singular value shrinkage using PyTorch."""
    U, s, V = torch.svd(X)
    s_shrunk = torch.clamp(s - threshold, min=0)
    return U @ torch.diag(s_shrunk) @ V.t()


def compute_rpca_lambda(m: int, n: int, method: str = "candès") -> float:
    """
    Compute recommended lambda parameter for RPCA.
    
    Args:
        m, n: Matrix dimensions
        method: 'candès' (1/sqrt(max(m,n))) or 'adaptive'
    """
    if method == "candès":
        return 1.0 / np.sqrt(max(m, n))
    elif method == "adaptive":
        return 1.0 / np.sqrt(m * n)
    else:
        raise ValueError(f"Unknown lambda method: {method}")


def check_convergence(L_new: np.ndarray, S_new: np.ndarray, 
                     L_old: np.ndarray, S_old: np.ndarray, 
                     tol: float = 1e-6) -> bool:
    """Check convergence of RPCA algorithm."""
    L_diff = np.linalg.norm(L_new - L_old, 'fro') / np.linalg.norm(L_old, 'fro')
    S_diff = np.linalg.norm(S_new - S_old, 'fro') / np.linalg.norm(S_old, 'fro')
    return max(L_diff, S_diff) < tol


def create_frame_matrix(frames: np.ndarray, temporal_mode: bool = True) -> Tuple[np.ndarray, Tuple]:
    """
    Convert frame sequence to matrix for RPCA.
    
    Args:
        frames: Shape (T, C, H, W) or (T, H, W, C)
        temporal_mode: If True, create (pixels, time) matrix. If False, create (time, pixels)
        
    Returns:
        Matrix for RPCA and shape info for reconstruction
    """
    if frames.ndim == 4:
        T, C, H, W = frames.shape
        # Reshape to (pixels*channels, time) or (time, pixels*channels)
        if temporal_mode:
            matrix = frames.reshape(T, C * H * W).T  # (pixels*channels, time)
            shape_info = (C, H, W, T, True)
        else:
            matrix = frames.reshape(T, C * H * W)    # (time, pixels*channels)
            shape_info = (C, H, W, T, False)
    else:
        raise ValueError(f"Unsupported frame shape: {frames.shape}")
    
    return matrix, shape_info


def reconstruct_frames(L: np.ndarray, S: np.ndarray, shape_info: Tuple) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct frames from RPCA matrix decomposition.
    
    Args:
        L, S: Low-rank and sparse matrices from RPCA
        shape_info: Shape information from create_frame_matrix
        
    Returns:
        (L_frames, S_frames) with original frame dimensions
    """
    C, H, W, T, temporal_mode = shape_info
    
    if temporal_mode:
        # L, S are (pixels*channels, time)
        L_frames = L.T.reshape(T, C, H, W)  # (T, C, H, W)
        S_frames = S.T.reshape(T, C, H, W)
    else:
        # L, S are (time, pixels*channels)
        L_frames = L.reshape(T, C, H, W)    # (T, C, H, W)
        S_frames = S.reshape(T, C, H, W)
        
    return L_frames, S_frames


def compute_compression_ratio(L: np.ndarray, S: np.ndarray, 
                            original_shape: Tuple, 
                            rank_threshold: float = 0.01) -> dict:
    """
    Compute memory compression statistics for RPCA decomposition.
    
    Args:
        L: Low-rank component
        S: Sparse component  
        original_shape: Original data shape
        rank_threshold: Threshold for computing effective rank
        
    Returns:
        Dictionary with compression statistics
    """
    m, n = L.shape
    
    # Compute effective rank
    _, s, _ = svd(L, full_matrices=False)
    effective_rank = np.sum(s > rank_threshold * s[0])
    
    # Compute sparsity
    sparsity = np.mean(np.abs(S) < 1e-6)
    
    # Memory usage (assuming float32)
    original_size = np.prod(original_shape) * 4  # bytes
    
    # Low-rank storage: U (m×r) + Σ (r) + V^T (r×n)
    lowrank_size = (m + n) * effective_rank * 4 + effective_rank * 4
    
    # Sparse storage: non-zero values + indices
    nnz = np.sum(np.abs(S) >= 1e-6)
    sparse_size = nnz * 8  # 4 bytes for value + 4 for index (simplified)
    
    compressed_size = lowrank_size + sparse_size
    compression_ratio = original_size / compressed_size
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': compression_ratio,
        'effective_rank': effective_rank,
        'sparsity': sparsity,
        'lowrank_fraction': lowrank_size / compressed_size,
        'sparse_fraction': sparse_size / compressed_size
    }