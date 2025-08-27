"""
RPCA-enhanced trainer that extends the base trainer with RPCA capabilities.
"""

from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig
import torch

from trainer import Trainer
from data import RPCAConfig, RPCADataset, create_rpca_datasets
from models.rpca_agent import RPCAAgent, create_rpca_agent_config, RPCAModelFactory
from utils import build_ddp_wrapper


class RPCATrainer(Trainer):
    """
    Enhanced trainer with RPCA capabilities.
    Extends the base trainer to support RPCA preprocessing and dual-branch models.
    """
    
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        # Extract RPCA configuration
        self.rpca_config = self._create_rpca_config(cfg)
        
        # Initialize base trainer (but override dataset creation)
        super().__init__(cfg, root_dir)
        
        # Track RPCA-specific metrics
        self.rpca_metrics = {
            'total_decompositions': 0,
            'average_compression_ratio': 0.0,
            'cache_hit_rate': 0.0
        }
        
    def _create_rpca_config(self, cfg: DictConfig) -> RPCAConfig:
        """Create RPCA configuration from Hydra config."""
        if not hasattr(cfg.env, 'rpca') or not cfg.env.rpca.get('enabled', False):
            return RPCAConfig(enabled=False)
            
        rpca_cfg_dict = cfg.env.rpca
        
        return RPCAConfig(
            enabled=rpca_cfg_dict.get('enabled', False),
            method=rpca_cfg_dict.get('method', 'inexact_alm'),
            lambda_coeff=rpca_cfg_dict.get('lambda_coeff', None),
            temporal_mode=rpca_cfg_dict.get('temporal_mode', True),
            cache_dir=rpca_cfg_dict.get('cache_dir', None),
            max_cache_size=rpca_cfg_dict.get('max_cache_size', 1000),
            enable_parallel=rpca_cfg_dict.get('enable_parallel', True)
        )
        
    def _setup_datasets(self, cfg: DictConfig) -> None:
        """Setup datasets with RPCA support."""
        if cfg.env.train.id == "racing":
            assert cfg.env.path_data_low_res is not None and cfg.env.path_data_full_res is not None
            
            low_res_path = Path(cfg.env.path_data_low_res)
            full_res_path = Path(cfg.env.path_data_full_res)
            
            if self.rpca_config.enabled:
                # Use RPCA-enhanced datasets
                train_dataset, test_dataset, full_res_dataset = create_rpca_datasets(
                    low_res_path, full_res_path, self.rpca_config
                )
            else:
                # Use standard datasets
                from data import Dataset, GameHdf5Dataset
                full_res_dataset = GameHdf5Dataset(full_res_path)
                
                train_dataset = Dataset(
                    low_res_path / "train", 
                    full_res_dataset, 
                    "train_dataset",
                    cfg.training.cache_in_ram,
                    cfg.training.num_workers_data_loaders > 0 and cfg.training.cache_in_ram
                )
                
                test_dataset = Dataset(
                    low_res_path / "test",
                    full_res_dataset,
                    "test_dataset", 
                    cache_in_ram=True
                )
            
            # Set datasets
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset
            
            # Load dataset states
            self.train_dataset.load_from_default_path()
            self.test_dataset.load_from_default_path()
            
    def _setup_agent(self, cfg: DictConfig, num_actions: int) -> None:
        """Setup agent with optional RPCA enhancement."""
        from hydra.utils import instantiate
        
        # Create base agent config
        base_agent_config = instantiate(cfg.agent, num_actions=num_actions)
        
        # Create RPCA-enhanced agent if enabled
        if self.rpca_config.enabled:
            rpca_settings = self._extract_rpca_model_settings(cfg)
            self.agent = RPCAModelFactory.create_agent(base_agent_config, rpca_settings)
        else:
            from agent import Agent
            self.agent = Agent(base_agent_config)
            
        self.agent = self.agent.to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if self._world_size > 1 else self.agent
        
    def _extract_rpca_model_settings(self, cfg: DictConfig) -> dict:
        """Extract RPCA model settings from configuration."""
        rpca_cfg = cfg.env.rpca
        
        return {
            'enabled': True,
            'sparse_head_channels': rpca_cfg.get('sparse_head_channels', 64),
            'fusion_method': rpca_cfg.get('fusion_method', 'concat'),
            'lambda_lowrank': rpca_cfg.loss_weights.get('lambda_lowrank', 1.0),
            'lambda_sparse': rpca_cfg.loss_weights.get('lambda_sparse', 1.0),
            'lambda_consistency': rpca_cfg.loss_weights.get('lambda_consistency', 0.1),
            'beta_nuclear': rpca_cfg.loss_weights.get('beta_nuclear', 0.01),
            'use_rpca_denoiser': True
        }
        
    def train_component(self, name: str, steps: int) -> List[dict]:
        """Enhanced training component with RPCA tracking."""
        # Call base training
        logs = super().train_component(name, steps)
        
        # Add RPCA-specific metrics if available
        if self.rpca_config.enabled:
            rpca_stats = self._collect_rpca_stats()
            
            # Add RPCA metrics to logs
            for log_entry in logs:
                log_entry.update({f"rpca_{k}": v for k, v in rpca_stats.items()})
                
        return logs
        
    def _collect_rpca_stats(self) -> dict:
        """Collect RPCA-specific statistics."""
        stats = {}
        
        # Dataset statistics
        if hasattr(self.train_dataset, 'get_rpca_statistics'):
            dataset_stats = self.train_dataset.get_rpca_statistics()
            stats.update({f"dataset_{k}": v for k, v in dataset_stats.items()})
            
        # Model statistics
        if hasattr(self.agent, 'get_rpca_metrics'):
            model_stats = self.agent.get_rpca_metrics()
            stats.update({f"model_{k}": v for k, v in model_stats.items()})
            
        return stats
        
    def save_checkpoint(self) -> None:
        """Enhanced checkpoint saving with RPCA state."""
        super().save_checkpoint()
        
        # Save RPCA-specific checkpoint if applicable
        if self.rpca_config.enabled and hasattr(self.agent, 'save_rpca_checkpoint'):
            rpca_checkpoint_path = self._path_ckpt_dir / f"rpca_checkpoint_epoch_{self.epoch}.pt"
            self.agent.save_rpca_checkpoint(rpca_checkpoint_path)
            
        # Save RPCA configuration
        if self._rank == 0:
            rpca_config_path = self._path_ckpt_dir / "rpca_config.yaml"
            self._save_rpca_config(rpca_config_path)
            
    def _save_rpca_config(self, path: Path) -> None:
        """Save RPCA configuration to file."""
        import yaml
        
        rpca_config_dict = {
            'enabled': self.rpca_config.enabled,
            'method': self.rpca_config.method,
            'lambda_coeff': self.rpca_config.lambda_coeff,
            'temporal_mode': self.rpca_config.temporal_mode,
            'cache_dir': str(self.rpca_config.cache_dir) if self.rpca_config.cache_dir else None,
            'max_cache_size': self.rpca_config.max_cache_size,
            'enable_parallel': self.rpca_config.enable_parallel
        }
        
        with open(path, 'w') as f:
            yaml.dump({'rpca': rpca_config_dict}, f, default_flow_style=False)
            
    def load_rpca_checkpoint(self, checkpoint_path: Path) -> None:
        """Load RPCA-specific checkpoint."""
        if self.rpca_config.enabled and hasattr(self.agent, 'load_rpca_checkpoint'):
            if checkpoint_path.exists():
                self.agent.load_rpca_checkpoint(checkpoint_path)
                
    def toggle_rpca_mode(self, enable: bool = None) -> bool:
        """Toggle RPCA mode during training/evaluation."""
        if hasattr(self.agent, 'toggle_rpca'):
            return self.agent.toggle_rpca(enable)
        return False
        
    def get_rpca_summary(self) -> dict:
        """Get comprehensive RPCA training summary."""
        summary = {
            'rpca_enabled': self.rpca_config.enabled,
            'rpca_method': self.rpca_config.method if self.rpca_config.enabled else None,
            'lambda_coeff': self.rpca_config.lambda_coeff if self.rpca_config.enabled else None
        }
        
        if self.rpca_config.enabled:
            # Add dataset statistics
            if hasattr(self.train_dataset, 'get_rpca_statistics'):
                summary['dataset_stats'] = self.train_dataset.get_rpca_statistics()
                
            # Add model statistics
            if hasattr(self.agent, 'get_rpca_metrics'):
                summary['model_stats'] = self.agent.get_rpca_metrics()
                
        return summary


def create_rpca_trainer(cfg: DictConfig, root_dir: Path) -> Trainer:
    """
    Factory function to create appropriate trainer based on configuration.
    
    Args:
        cfg: Hydra configuration
        root_dir: Project root directory
        
    Returns:
        Trainer or RPCATrainer instance
    """
    # Check if RPCA is enabled
    rpca_enabled = (
        hasattr(cfg.env, 'rpca') and 
        cfg.env.rpca.get('enabled', False)
    )
    
    if rpca_enabled:
        return RPCATrainer(cfg, root_dir)
    else:
        return Trainer(cfg, root_dir)


# Monkey patch the original trainer creation if needed
def patch_trainer_creation():
    """Monkey patch trainer creation to use RPCA trainer when appropriate."""
    import sys
    import main
    
    # Store original trainer creation
    original_run = main.run
    
    def enhanced_run(cfg: DictConfig, root_dir: Path) -> None:
        trainer = create_rpca_trainer(cfg, root_dir) 
        trainer.run()
        
    # Replace the run function
    main.run = enhanced_run