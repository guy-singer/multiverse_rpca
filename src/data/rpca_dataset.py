from typing import Optional, Dict, Any
from pathlib import Path
import torch

from .dataset import Dataset, GameHdf5Dataset  
from .segment import Segment, SegmentId
from .rpca_processor import RPCAProcessor, RPCAConfig
from utils import StateDictMixin


class RPCADataset(Dataset):
    """
    Enhanced Dataset class with integrated RPCA preprocessing.
    """
    
    def __init__(self, directory: Path, 
                 dataset_full_res: Optional[torch.utils.data.Dataset],
                 rpca_config: Optional[RPCAConfig] = None,
                 name: Optional[str] = None,
                 cache_in_ram: bool = False,
                 use_manager: bool = False,
                 save_on_disk: bool = True):
        super().__init__(directory, dataset_full_res, name, cache_in_ram, use_manager, save_on_disk)
        
        self.rpca_config = rpca_config or RPCAConfig()
        self.rpca_processor = self.rpca_config.create_processor()
        
        # Track RPCA usage statistics
        self.rpca_stats = {
            'num_processed': 0,
            'cache_hits': 0,
            'total_compression_ratio': 0.0
        }
        
    def __getitem__(self, segment_id: SegmentId) -> Segment:
        # Get base segment
        segment = super().__getitem__(segment_id)
        
        # Apply RPCA processing if enabled
        if self.rpca_processor is not None:
            segment = self.rpca_processor.decompose_segment(segment)
            self._update_rpca_stats(segment)
            
        return segment
        
    def _update_rpca_stats(self, segment: Segment) -> None:
        """Update RPCA processing statistics."""
        if 'rpca_decomposed' in segment.info and segment.info['rpca_decomposed']:
            self.rpca_stats['num_processed'] += 1
            
            if 'rpca_compression_stats' in segment.info:
                compression_stats = segment.info['rpca_compression_stats']
                if 'compression_ratio' in compression_stats:
                    self.rpca_stats['total_compression_ratio'] += compression_stats['compression_ratio']
                    
    def get_rpca_statistics(self) -> Dict[str, Any]:
        """Get RPCA processing statistics."""
        stats = self.rpca_stats.copy()
        if stats['num_processed'] > 0:
            stats['avg_compression_ratio'] = stats['total_compression_ratio'] / stats['num_processed']
        else:
            stats['avg_compression_ratio'] = 0.0
        return stats
        
    def state_dict(self) -> Dict[str, Any]:
        """Include RPCA config in state dict."""
        state = super().state_dict()
        state['rpca_config'] = {
            'enabled': self.rpca_config.enabled,
            'method': self.rpca_config.method,
            'lambda_coeff': self.rpca_config.lambda_coeff,
            'temporal_mode': self.rpca_config.temporal_mode
        }
        state['rpca_stats'] = self.rpca_stats
        return state
        
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load RPCA config from state dict."""
        super().load_state_dict(state_dict)
        
        if 'rpca_config' in state_dict:
            rpca_config_dict = state_dict['rpca_config']
            self.rpca_config = RPCAConfig(**rpca_config_dict)
            self.rpca_processor = self.rpca_config.create_processor()
            
        if 'rpca_stats' in state_dict:
            self.rpca_stats = state_dict['rpca_stats']


class RPCAGameHdf5Dataset(GameHdf5Dataset):
    """
    Enhanced GameHdf5Dataset with RPCA preprocessing support.
    """
    
    def __init__(self, directory: Path, rpca_config: Optional[RPCAConfig] = None):
        super().__init__(directory)
        self.rpca_config = rpca_config or RPCAConfig()
        self.rpca_processor = self.rpca_config.create_processor()
        
    def __getitem__(self, segment_id: SegmentId) -> Segment:
        # Get base segment
        segment = super().__getitem__(segment_id)
        
        # Apply RPCA processing if enabled
        if self.rpca_processor is not None:
            segment = self.rpca_processor.decompose_segment(segment)
            
        return segment


def create_rpca_datasets(low_res_path: Path, full_res_path: Path, 
                        rpca_config: RPCAConfig) -> tuple:
    """
    Create RPCA-enhanced train and test datasets.
    
    Args:
        low_res_path: Path to low resolution data
        full_res_path: Path to full resolution data
        rpca_config: RPCA configuration
        
    Returns:
        (train_dataset, test_dataset, full_res_dataset)
    """
    # Create full resolution dataset with RPCA
    full_res_dataset = RPCAGameHdf5Dataset(full_res_path, rpca_config)
    
    # Create train/test datasets with RPCA
    train_dataset = RPCADataset(
        low_res_path / "train", 
        full_res_dataset, 
        rpca_config,
        "rpca_train_dataset",
        cache_in_ram=False
    )
    
    test_dataset = RPCADataset(
        low_res_path / "test",
        full_res_dataset,
        rpca_config, 
        "rpca_test_dataset",
        cache_in_ram=True
    )
    
    return train_dataset, test_dataset, full_res_dataset


class DualViewRPCADataset(Dataset):
    """
    Dataset for dual-view (multi-camera) RPCA processing.
    Handles cross-view consistency for racing game scenarios.
    """
    
    def __init__(self, directory: Path,
                 dataset_full_res: Optional[torch.utils.data.Dataset],
                 rpca_config: Optional[RPCAConfig] = None,
                 **kwargs):
        super().__init__(directory, dataset_full_res, **kwargs)
        
        self.rpca_config = rpca_config or RPCAConfig()
        
        # Import MultiViewRPCA here to avoid circular imports
        from .rpca_processor import MultiViewRPCA
        self.multiview_rpca = MultiViewRPCA(self.rpca_config)
        
    def __getitem__(self, segment_id: SegmentId) -> Segment:
        segment = super().__getitem__(segment_id)
        
        if not self.rpca_config.enabled:
            return segment
            
        # Extract dual views from frames
        frames = segment.obs.numpy()  # (T, C, H, W)
        
        # For racing game: assume frames contain both camera views
        # Split frames or extract views based on your specific data format
        if frames.shape[2] > frames.shape[3]:  # Height > Width suggests stacked views
            H, W = frames.shape[2], frames.shape[3]
            mid_height = H // 2
            
            view1_frames = frames[:, :, :mid_height, :]  # Top half
            view2_frames = frames[:, :, mid_height:, :]   # Bottom half
            
            # Apply multi-view RPCA
            result1, result2 = self.multiview_rpca.decompose_dual_view(view1_frames, view2_frames)
            
            # Combine results back into single frame format
            L_combined = torch.cat([
                torch.from_numpy(result1.L_frames),
                torch.from_numpy(result2.L_frames)
            ], dim=2)  # Stack along height
            
            S_combined = torch.cat([
                torch.from_numpy(result1.S_frames), 
                torch.from_numpy(result2.S_frames)
            ], dim=2)
            
            # Update segment info
            segment.info.update({
                'rpca_lowrank': L_combined.float(),
                'rpca_sparse': S_combined.float(),
                'rpca_decomposed': True,
                'dual_view_mode': True,
                'view1_compression': result1.compression_stats,
                'view2_compression': result2.compression_stats
            })
            
        return segment