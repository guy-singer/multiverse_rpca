import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading

from ..rpca import FrameRPCA, RPCAResult
from .segment import Segment


class RPCAProcessor:
    """
    Handles RPCA preprocessing for frame sequences with caching and optimization.
    """
    
    def __init__(self, method: str = "inexact_alm", 
                 lambda_coeff: Optional[float] = None,
                 temporal_mode: bool = True,
                 cache_dir: Optional[Path] = None,
                 max_cache_size: int = 1000,
                 enable_parallel: bool = True):
        self.method = method
        self.lambda_coeff = lambda_coeff
        self.temporal_mode = temporal_mode
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size
        self.enable_parallel = enable_parallel
        
        # Initialize RPCA decomposer
        self.rpca = FrameRPCA(
            method=method, 
            lambda_coeff=lambda_coeff,
            temporal_mode=temporal_mode
        )
        
        # Cache for decomposition results
        self._cache = {}
        self._cache_lock = threading.Lock()
        
        # Setup cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _get_cache_key(self, frames: np.ndarray) -> str:
        """Generate cache key for frame sequence."""
        shape_str = "_".join(map(str, frames.shape))
        data_hash = hash(frames.data.tobytes())
        param_str = f"{self.method}_{self.lambda_coeff}_{self.temporal_mode}"
        return f"rpca_{shape_str}_{data_hash}_{param_str}"
        
    def _load_from_disk_cache(self, cache_key: str) -> Optional[RPCAResult]:
        """Load RPCA result from disk cache."""
        if self.cache_dir is None:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # Remove corrupted cache file
                cache_file.unlink()
        return None
        
    def _save_to_disk_cache(self, cache_key: str, result: RPCAResult) -> None:
        """Save RPCA result to disk cache."""
        if self.cache_dir is None:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            # Ignore cache save errors
            pass
            
    def _cleanup_cache(self) -> None:
        """Remove oldest entries from memory cache if size exceeds limit."""
        with self._cache_lock:
            while len(self._cache) > self.max_cache_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                
    def decompose_segment(self, segment: Segment) -> Segment:
        """
        Apply RPCA decomposition to a segment and return enhanced segment.
        
        Args:
            segment: Input segment with obs field
            
        Returns:
            Enhanced segment with RPCA components in info dict
        """
        frames = segment.obs.numpy()  # Shape: (T, C, H, W)
        
        # Generate cache key
        cache_key = self._get_cache_key(frames)
        
        # Try to load from cache
        result = None
        with self._cache_lock:
            if cache_key in self._cache:
                result = self._cache[cache_key]
                
        if result is None:
            result = self._load_from_disk_cache(cache_key)
            
        if result is None:
            # Compute RPCA decomposition
            result = self.rpca.decompose_frames(frames)
            
            # Save to caches
            with self._cache_lock:
                self._cache[cache_key] = result
                
            self._save_to_disk_cache(cache_key, result)
            self._cleanup_cache()
            
        # Create enhanced segment
        enhanced_segment = Segment(
            obs=segment.obs,
            act=segment.act, 
            rew=segment.rew,
            end=segment.end,
            trunc=segment.trunc,
            mask_padding=segment.mask_padding,
            info=segment.info.copy(),
            id=segment.id
        )
        
        # Add RPCA components to info
        enhanced_segment.info.update({
            'rpca_lowrank': torch.from_numpy(result.L_frames).float(),
            'rpca_sparse': torch.from_numpy(result.S_frames).float(), 
            'rpca_decomposed': True,
            'rpca_compression_stats': result.compression_stats,
            'rpca_metadata': result.metadata
        })
        
        return enhanced_segment
        
    def process_batch_parallel(self, segments: list) -> list:
        """Process multiple segments in parallel."""
        if not self.enable_parallel or len(segments) == 1:
            return [self.decompose_segment(seg) for seg in segments]
            
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self.decompose_segment, segments))


class RPCAConfig:
    """Configuration for RPCA preprocessing."""
    
    def __init__(self, enabled: bool = False,
                 method: str = "inexact_alm",
                 lambda_coeff: Optional[float] = None,
                 temporal_mode: bool = True,
                 cache_dir: Optional[str] = None,
                 max_cache_size: int = 1000,
                 enable_parallel: bool = True,
                 **kwargs):
        self.enabled = enabled
        self.method = method
        self.lambda_coeff = lambda_coeff
        self.temporal_mode = temporal_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cache_size = max_cache_size
        self.enable_parallel = enable_parallel
        self.kwargs = kwargs
        
    def create_processor(self) -> Optional[RPCAProcessor]:
        """Create RPCA processor if enabled."""
        if not self.enabled:
            return None
            
        return RPCAProcessor(
            method=self.method,
            lambda_coeff=self.lambda_coeff,
            temporal_mode=self.temporal_mode,
            cache_dir=self.cache_dir,
            max_cache_size=self.max_cache_size,
            enable_parallel=self.enable_parallel
        )


def apply_rpca_to_frames(frames: torch.Tensor, 
                        rpca_config: RPCAConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RPCA decomposition to frame tensor.
    
    Args:
        frames: Input frames (B, T, C, H, W) 
        rpca_config: RPCA configuration
        
    Returns:
        (L_frames, S_frames): Low-rank and sparse components
    """
    if not rpca_config.enabled:
        # Return frames as low-rank, zeros as sparse
        return frames, torch.zeros_like(frames)
        
    processor = rpca_config.create_processor()
    if processor is None:
        return frames, torch.zeros_like(frames)
        
    B, T, C, H, W = frames.shape
    L_batch = []
    S_batch = []
    
    for b in range(B):
        frame_seq = frames[b].detach().cpu().numpy()  # (T, C, H, W)
        result = processor.rpca.decompose_frames(frame_seq)
        
        L_batch.append(torch.from_numpy(result.L_frames).float())
        S_batch.append(torch.from_numpy(result.S_frames).float())
        
    L_tensor = torch.stack(L_batch, dim=0).to(frames.device)
    S_tensor = torch.stack(S_batch, dim=0).to(frames.device)
    
    return L_tensor, S_tensor


class MultiViewRPCA:
    """
    RPCA processor specialized for multi-view (dual-camera) frame sequences.
    Handles cross-view consistency as described in the RPCA paper.
    """
    
    def __init__(self, rpca_config: RPCAConfig, 
                 cross_view_mode: str = "stack_channels"):
        self.rpca_config = rpca_config
        self.cross_view_mode = cross_view_mode
        self.processor = rpca_config.create_processor()
        
    def decompose_dual_view(self, view1_frames: np.ndarray, 
                           view2_frames: np.ndarray) -> Tuple[RPCAResult, RPCAResult]:
        """
        Apply RPCA to dual-view frames for cross-view consistency.
        
        Args:
            view1_frames: First camera view (T, C, H, W)
            view2_frames: Second camera view (T, C, H, W)
            
        Returns:
            (result1, result2): RPCA results for each view
        """
        if self.processor is None:
            # Return identity decompositions
            result1 = RPCAResult(L=view1_frames, S=np.zeros_like(view1_frames))
            result2 = RPCAResult(L=view2_frames, S=np.zeros_like(view2_frames))
            return result1, result2
            
        if self.cross_view_mode == "stack_channels":
            # Stack both views as 6-channel sequence for shared decomposition
            T, C, H, W = view1_frames.shape
            combined_frames = np.concatenate([view1_frames, view2_frames], axis=1)  # (T, 2C, H, W)
            
            # Apply RPCA to combined sequence
            combined_result = self.processor.rpca.decompose_frames(combined_frames)
            
            # Split back into separate views
            L_combined = combined_result.L_frames
            S_combined = combined_result.S_frames
            
            result1 = RPCAResult(
                L=L_combined[:, :C],  # First C channels
                S=S_combined[:, :C],
                L_frames=L_combined[:, :C],
                S_frames=S_combined[:, :C]
            )
            
            result2 = RPCAResult(
                L=L_combined[:, C:],  # Next C channels  
                S=S_combined[:, C:],
                L_frames=L_combined[:, C:],
                S_frames=S_combined[:, C:]
            )
            
        elif self.cross_view_mode == "separate":
            # Process each view independently
            result1 = self.processor.rpca.decompose_frames(view1_frames)
            result2 = self.processor.rpca.decompose_frames(view2_frames)
            
        else:
            raise ValueError(f"Unknown cross-view mode: {self.cross_view_mode}")
            
        return result1, result2