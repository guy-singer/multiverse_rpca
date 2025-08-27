"""
Memory optimization utilities for RPCA operations.
Implements compressed storage, streaming processing, and efficient batch handling.
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import gc
from pathlib import Path
import pickle
import lz4.frame
import threading
from concurrent.futures import ThreadPoolExecutor


@dataclass
class CompressionStats:
    """Statistics for RPCA compression."""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    effective_rank: int
    sparsity_ratio: float
    memory_saved_mb: float


class CompressedRPCAStorage:
    """
    Compressed storage for RPCA decomposition results.
    Stores only essential components to minimize memory usage.
    """
    
    def __init__(self, L: np.ndarray, S: np.ndarray, 
                 rank_threshold: float = 0.01,
                 sparsity_threshold: float = 1e-6):
        self.original_shape = L.shape
        self.rank_threshold = rank_threshold
        self.sparsity_threshold = sparsity_threshold
        
        # Compress low-rank component
        self.U, self.sigma, self.Vt = self._compress_lowrank(L)
        
        # Compress sparse component
        self.sparse_indices, self.sparse_values = self._compress_sparse(S)
        
        # Compute compression statistics
        self.stats = self._compute_compression_stats(L, S)
        
    def _compress_lowrank(self, L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compress low-rank component using SVD."""
        U, sigma, Vt = np.linalg.svd(L, full_matrices=False)
        
        # Determine effective rank
        effective_rank = np.sum(sigma > self.rank_threshold * sigma[0])
        effective_rank = max(1, effective_rank)  # Ensure at least rank 1
        
        # Truncate to effective rank
        U_compressed = U[:, :effective_rank].astype(np.float32)
        sigma_compressed = sigma[:effective_rank].astype(np.float32)
        Vt_compressed = Vt[:effective_rank, :].astype(np.float32)
        
        return U_compressed, sigma_compressed, Vt_compressed
        
    def _compress_sparse(self, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compress sparse component by storing only non-zero elements."""
        # Find significant sparse elements
        significant_mask = np.abs(S) > self.sparsity_threshold
        
        if not np.any(significant_mask):
            return np.array([]), np.array([])
            
        # Get indices and values of non-zero elements
        indices = np.where(significant_mask)
        values = S[significant_mask].astype(np.float32)
        
        # Convert to flat indices for more efficient storage
        flat_indices = np.ravel_multi_index(indices, S.shape).astype(np.int32)
        
        return flat_indices, values
        
    def _compute_compression_stats(self, L: np.ndarray, S: np.ndarray) -> CompressionStats:
        """Compute compression statistics."""
        original_size = (L.nbytes + S.nbytes)
        
        # Low-rank storage size
        lowrank_size = (self.U.nbytes + self.sigma.nbytes + self.Vt.nbytes)
        
        # Sparse storage size
        sparse_size = (self.sparse_indices.nbytes + self.sparse_values.nbytes)
        
        compressed_size = lowrank_size + sparse_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        effective_rank = len(self.sigma)
        sparsity_ratio = len(self.sparse_values) / L.size
        memory_saved_mb = (original_size - compressed_size) / (1024**2)
        
        return CompressionStats(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            effective_rank=effective_rank,
            sparsity_ratio=sparsity_ratio,
            memory_saved_mb=memory_saved_mb
        )
        
    def reconstruct(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct original L and S matrices."""
        # Reconstruct low-rank component
        L = self.U @ np.diag(self.sigma) @ self.Vt
        
        # Reconstruct sparse component
        S = np.zeros(self.original_shape, dtype=np.float32)
        if len(self.sparse_indices) > 0:
            flat_indices = self.sparse_indices
            indices = np.unravel_index(flat_indices, self.original_shape)
            S[indices] = self.sparse_values
            
        return L.astype(np.float64), S.astype(np.float64)
        
    def save_compressed(self, filepath: Path) -> None:
        """Save compressed representation to disk."""
        data = {
            'U': self.U,
            'sigma': self.sigma, 
            'Vt': self.Vt,
            'sparse_indices': self.sparse_indices,
            'sparse_values': self.sparse_values,
            'original_shape': self.original_shape,
            'stats': self.stats
        }
        
        # Use LZ4 compression for additional space savings
        with open(filepath, 'wb') as f:
            compressed_data = lz4.frame.compress(pickle.dumps(data))
            f.write(compressed_data)
            
    @classmethod
    def load_compressed(cls, filepath: Path) -> 'CompressedRPCAStorage':
        """Load compressed representation from disk."""
        with open(filepath, 'rb') as f:
            compressed_data = f.read()
            data = pickle.loads(lz4.frame.decompress(compressed_data))
            
        # Create instance with dummy data, then restore state
        instance = cls.__new__(cls)
        instance.U = data['U']
        instance.sigma = data['sigma']
        instance.Vt = data['Vt']
        instance.sparse_indices = data['sparse_indices']
        instance.sparse_values = data['sparse_values']
        instance.original_shape = data['original_shape']
        instance.stats = data['stats']
        
        return instance


class MemoryEfficientRPCABatch:
    """
    Memory-efficient batch processing for RPCA decompositions.
    """
    
    def __init__(self, max_memory_mb: float = 1000.0, 
                 compression_enabled: bool = True):
        self.max_memory_mb = max_memory_mb
        self.compression_enabled = compression_enabled
        self.batch_cache = {}
        self.memory_usage = 0.0
        self.cache_lock = threading.Lock()
        
    def _estimate_memory_usage(self, shape: Tuple[int, ...]) -> float:
        """Estimate memory usage for given tensor shape in MB."""
        return np.prod(shape) * 4 / (1024**2)  # Assume float32
        
    def _cleanup_cache(self) -> None:
        """Remove old entries from cache to free memory."""
        with self.cache_lock:
            # Simple LRU eviction
            if len(self.batch_cache) > 0:
                oldest_key = next(iter(self.batch_cache))
                del self.batch_cache[oldest_key]
                gc.collect()
                
    def process_batch_frames(self, frames_batch: torch.Tensor,
                           rpca_processor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of frames with memory optimization.
        
        Args:
            frames_batch: Input frames (B, T, C, H, W)
            rpca_processor: RPCA processor instance
            
        Returns:
            (L_batch, S_batch): Low-rank and sparse components
        """
        B, T, C, H, W = frames_batch.shape
        estimated_memory = self._estimate_memory_usage(frames_batch.shape)
        
        # Check memory constraints
        if estimated_memory > self.max_memory_mb:
            return self._process_large_batch(frames_batch, rpca_processor)
        else:
            return self._process_normal_batch(frames_batch, rpca_processor)
            
    def _process_normal_batch(self, frames_batch: torch.Tensor,
                             rpca_processor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch that fits in memory."""
        B = frames_batch.shape[0]
        L_batch = []
        S_batch = []
        
        for b in range(B):
            frames = frames_batch[b].detach().cpu().numpy()
            
            # Check cache first
            cache_key = self._get_cache_key(frames)
            
            with self.cache_lock:
                if cache_key in self.batch_cache:
                    compressed_result = self.batch_cache[cache_key]
                    L, S = compressed_result.reconstruct()
                else:
                    # Process with RPCA
                    result = rpca_processor.decompose_frames(frames)
                    L, S = result.L_frames, result.S_frames
                    
                    # Cache compressed result if enabled
                    if self.compression_enabled:
                        compressed_result = CompressedRPCAStorage(result.L, result.S)
                        self.batch_cache[cache_key] = compressed_result
                        
            L_batch.append(torch.from_numpy(L).float())
            S_batch.append(torch.from_numpy(S).float())
            
        return torch.stack(L_batch), torch.stack(S_batch)
        
    def _process_large_batch(self, frames_batch: torch.Tensor,
                            rpca_processor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process large batch using chunking."""
        B = frames_batch.shape[0]
        
        # Determine chunk size based on memory constraints
        single_frame_memory = self._estimate_memory_usage(frames_batch[0].shape)
        chunk_size = max(1, int(self.max_memory_mb / single_frame_memory))
        
        L_batch = []
        S_batch = []
        
        for start_idx in range(0, B, chunk_size):
            end_idx = min(start_idx + chunk_size, B)
            chunk = frames_batch[start_idx:end_idx]
            
            # Process chunk
            L_chunk, S_chunk = self._process_normal_batch(chunk, rpca_processor)
            L_batch.append(L_chunk)
            S_batch.append(S_chunk)
            
            # Force garbage collection
            gc.collect()
            
        return torch.cat(L_batch, dim=0), torch.cat(S_batch, dim=0)
        
    def _get_cache_key(self, frames: np.ndarray) -> str:
        """Generate cache key for frames."""
        # Use hash of first and last frame for speed
        first_frame_hash = hash(frames[0].tobytes())
        last_frame_hash = hash(frames[-1].tobytes())
        shape_hash = hash(frames.shape)
        return f"{first_frame_hash}_{last_frame_hash}_{shape_hash}"


class StreamingRPCAProcessor:
    """
    Streaming RPCA processor for online/real-time applications.
    Maintains running decomposition with minimal memory footprint.
    """
    
    def __init__(self, rank: int, ambient_dim: int,
                 forgetting_factor: float = 0.95,
                 adaptation_rate: float = 0.1):
        self.rank = rank
        self.ambient_dim = ambient_dim
        self.forgetting_factor = forgetting_factor
        self.adaptation_rate = adaptation_rate
        
        # Initialize subspace basis
        self.U = np.random.randn(ambient_dim, rank)
        self.U, _ = np.linalg.qr(self.U)  # Orthonormalize
        
        # Running statistics
        self.n_samples = 0
        self.total_sparse_ratio = 0.0
        
    def update(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update streaming decomposition with new observation.
        
        Args:
            observation: New observation vector (ambient_dim,)
            
        Returns:
            (low_rank_part, sparse_part): Decomposed components
        """
        # Project onto current subspace
        coefficients = self.U.T @ observation
        low_rank_reconstruction = self.U @ coefficients
        
        # Compute residual (potential sparse part)
        residual = observation - low_rank_reconstruction
        
        # Apply soft thresholding for sparsity
        threshold = self._compute_adaptive_threshold()
        sparse_part = self._soft_threshold(residual, threshold)
        
        # Refined low-rank part
        low_rank_part = observation - sparse_part
        
        # Update subspace if significant sparse component
        if np.linalg.norm(sparse_part) > 0.1 * np.linalg.norm(observation):
            self._update_subspace(low_rank_part)
            
        # Update statistics
        self.n_samples += 1
        sparse_ratio = np.linalg.norm(sparse_part) / np.linalg.norm(observation)
        self.total_sparse_ratio = (self.forgetting_factor * self.total_sparse_ratio + 
                                  sparse_ratio)
        
        return low_rank_part, sparse_part
        
    def _compute_adaptive_threshold(self) -> float:
        """Compute adaptive threshold based on running statistics."""
        if self.n_samples == 0:
            return 0.1
            
        avg_sparse_ratio = self.total_sparse_ratio / (1 - self.forgetting_factor**self.n_samples)
        return 0.1 * (1 + avg_sparse_ratio)
        
    def _soft_threshold(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """Apply soft thresholding."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
        
    def _update_subspace(self, low_rank_observation: np.ndarray) -> None:
        """Update subspace using gradient descent on Grassmann manifold."""
        # Compute gradient direction
        projection_error = low_rank_observation - self.U @ (self.U.T @ low_rank_observation)
        
        if np.linalg.norm(projection_error) > 1e-6:
            # Gradient on Grassmann manifold
            gradient_direction = projection_error / np.linalg.norm(projection_error)
            
            # Update with small step
            self.U = self.U + self.adaptation_rate * np.outer(gradient_direction, 
                                                             self.U.T @ low_rank_observation)
            
            # Re-orthonormalize
            self.U, _ = np.linalg.qr(self.U)
            
    def get_subspace_basis(self) -> np.ndarray:
        """Get current subspace basis."""
        return self.U.copy()
        
    def reset_subspace(self) -> None:
        """Reset subspace to random initialization."""
        self.U = np.random.randn(self.ambient_dim, self.rank)
        self.U, _ = np.linalg.qr(self.U)
        self.n_samples = 0
        self.total_sparse_ratio = 0.0


class MemoryMonitor:
    """Monitor and track memory usage during RPCA operations."""
    
    def __init__(self):
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self.memory_timeline = []
        
    def update_memory_usage(self) -> float:
        """Update current memory usage."""
        if torch.cuda.is_available():
            # GPU memory
            current_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            # CPU memory (simplified)
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)
            
        self.current_memory_mb = current_memory
        self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
        
        self.memory_timeline.append({
            'timestamp': len(self.memory_timeline),
            'memory_mb': current_memory
        })
        
        return current_memory
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'current_mb': self.current_memory_mb,
            'peak_mb': self.peak_memory_mb,
            'average_mb': np.mean([entry['memory_mb'] for entry in self.memory_timeline]) if self.memory_timeline else 0.0
        }
        
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on usage patterns."""
        suggestions = []
        
        if self.peak_memory_mb > 4000:  # > 4GB
            suggestions.append("Consider reducing batch size or using gradient checkpointing")
            
        if len(self.memory_timeline) > 10:
            # Check for memory leaks (steadily increasing memory)
            recent_trend = np.polyfit(
                range(len(self.memory_timeline[-10:])), 
                [entry['memory_mb'] for entry in self.memory_timeline[-10:]], 
                1
            )[0]
            
            if recent_trend > 10:  # Increasing by >10MB per step
                suggestions.append("Possible memory leak detected - check for unclosed resources")
                
        return suggestions


def optimize_rpca_memory_usage(frames: torch.Tensor, 
                              rpca_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze and optimize memory usage for RPCA processing.
    
    Args:
        frames: Input frames tensor
        rpca_config: RPCA configuration
        
    Returns:
        Optimized configuration suggestions
    """
    B, T, C, H, W = frames.shape
    
    # Estimate memory requirements
    frame_memory_mb = (B * T * C * H * W * 4) / (1024**2)  # float32
    
    suggestions = {}
    
    if frame_memory_mb > 2000:  # > 2GB
        # Suggest chunked processing
        optimal_chunk_size = max(1, int(1000 / frame_memory_mb * B))  # Target 1GB chunks
        suggestions['use_chunked_processing'] = True
        suggestions['optimal_chunk_size'] = optimal_chunk_size
        
    if rpca_config.get('cache_decompositions', True):
        # Estimate cache memory requirements
        cache_memory_estimate = frame_memory_mb * 0.3  # Compressed storage
        if cache_memory_estimate > 500:  # > 500MB
            suggestions['disable_caching'] = True
            
    # Temporal mode suggestions
    if T > 20:  # Long sequences
        suggestions['use_temporal_mode'] = True
        suggestions['compression_threshold'] = 0.05  # Higher threshold for better compression
        
    return suggestions