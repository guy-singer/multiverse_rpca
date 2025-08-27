"""
Enhanced main script with RPCA support.

This script extends the original main.py to support RPCA-enhanced training.
It automatically detects RPCA configuration and uses the appropriate trainer.
"""

import os
from pathlib import Path
from typing import List, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from rpca_trainer import RPCATrainer, create_rpca_trainer
from trainer import Trainer
from utils import skip_if_run_is_over


OmegaConf.register_new_resolver("eval", eval)


@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main training function with RPCA support.
    Automatically selects appropriate trainer based on configuration.
    """
    setup_visible_cuda_devices(cfg.common.devices)
    world_size = torch.cuda.device_count()
    root_dir = Path(hydra.utils.get_original_cwd())
    
    # Print RPCA status
    rpca_enabled = hasattr(cfg.env, 'rpca') and cfg.env.rpca.get('enabled', False)
    print(f"RPCA Mode: {'ENABLED' if rpca_enabled else 'DISABLED'}")
    
    if rpca_enabled:
        rpca_method = cfg.env.rpca.get('method', 'inexact_alm')
        print(f"RPCA Method: {rpca_method}")
        print(f"RPCA Lambda: {cfg.env.rpca.get('lambda_coeff', 'auto')}")
        
    if world_size < 2:
        run(cfg, root_dir)
    else:
        mp.spawn(main_ddp, args=(world_size, cfg, root_dir), nprocs=world_size)


def main_ddp(rank: int, world_size: int, cfg: DictConfig, root_dir: Path) -> None:
    """Distributed training main function."""
    setup_ddp(rank, world_size)
    run(cfg, root_dir)
    destroy_process_group()


@skip_if_run_is_over
def run(cfg: DictConfig, root_dir: Path) -> None:
    """
    Enhanced run function that creates appropriate trainer.
    
    Args:
        cfg: Hydra configuration
        root_dir: Project root directory
    """
    # Create trainer (RPCA or standard based on configuration)
    trainer = create_rpca_trainer(cfg, root_dir)
    
    # Print trainer type
    trainer_type = "RPCA-Enhanced" if isinstance(trainer, RPCATrainer) else "Standard"
    print(f"Using {trainer_type} Trainer")
    
    # Run training
    trainer.run()
    
    # Print RPCA summary if applicable
    if isinstance(trainer, RPCATrainer):
        rpca_summary = trainer.get_rpca_summary()
        print(f"\nRPCA Training Summary:")
        for key, value in rpca_summary.items():
            print(f"  {key}: {value}")


def setup_ddp(rank: int, world_size: int) -> None:
    """Setup distributed data parallel training."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6006"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def setup_visible_cuda_devices(devices: Union[str, int, List[int]]) -> None:
    """Setup visible CUDA devices."""
    if isinstance(devices, str):
        if devices == "cpu":
            devices = []
        elif devices == "all":
            devices = list(range(torch.cuda.device_count()))
        else:
            raise ValueError(f"Invalid device specification: {devices}")
    elif isinstance(devices, int):
        devices = [devices]
        
    if isinstance(devices, list) and len(devices) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))


def validate_rpca_config(cfg: DictConfig) -> None:
    """Validate RPCA configuration."""
    if not hasattr(cfg.env, 'rpca'):
        return
        
    rpca_cfg = cfg.env.rpca
    
    if not rpca_cfg.get('enabled', False):
        return
        
    # Validate RPCA method
    valid_methods = ['inexact_alm', 'pcp', 'nonconvex']
    method = rpca_cfg.get('method', 'inexact_alm')
    if method not in valid_methods:
        raise ValueError(f"Invalid RPCA method: {method}. Valid options: {valid_methods}")
        
    # Validate fusion method
    valid_fusion = ['concat', 'add', 'attention']
    fusion = rpca_cfg.get('fusion_method', 'concat')
    if fusion not in valid_fusion:
        raise ValueError(f"Invalid fusion method: {fusion}. Valid options: {valid_fusion}")
        
    # Validate data paths
    if cfg.env.train.id == "racing":
        if not cfg.env.path_data_low_res or not cfg.env.path_data_full_res:
            raise ValueError("RPCA requires both path_data_low_res and path_data_full_res to be set")
            
    print("RPCA configuration validated successfully")


def print_rpca_info(cfg: DictConfig) -> None:
    """Print RPCA configuration information."""
    if not hasattr(cfg.env, 'rpca') or not cfg.env.rpca.get('enabled', False):
        print("RPCA: Disabled")
        return
        
    rpca_cfg = cfg.env.rpca
    
    print("RPCA Configuration:")
    print(f"  Method: {rpca_cfg.get('method', 'inexact_alm')}")
    print(f"  Lambda: {rpca_cfg.get('lambda_coeff', 'auto-computed')}")
    print(f"  Temporal Mode: {rpca_cfg.get('temporal_mode', True)}")
    print(f"  Fusion Method: {rpca_cfg.get('fusion_method', 'concat')}")
    print(f"  Cache Directory: {rpca_cfg.get('cache_dir', 'disabled')}")
    print(f"  Parallel Processing: {rpca_cfg.get('enable_parallel', True)}")
    
    if 'loss_weights' in rpca_cfg:
        print("  Loss Weights:")
        for key, value in rpca_cfg.loss_weights.items():
            print(f"    {key}: {value}")


# CLI for RPCA-specific operations
def rpca_cli():
    """Command-line interface for RPCA operations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RPCA Multiverse Operations")
    parser.add_argument('--validate-config', type=str, 
                       help='Validate RPCA configuration file')
    parser.add_argument('--preprocess-data', type=str,
                       help='Preprocess dataset with RPCA decomposition')
    parser.add_argument('--benchmark-methods', action='store_true',
                       help='Benchmark different RPCA methods')
    
    args = parser.parse_args()
    
    if args.validate_config:
        cfg = OmegaConf.load(args.validate_config)
        try:
            validate_rpca_config(cfg)
            print_rpca_info(cfg)
            print("Configuration is valid!")
        except Exception as e:
            print(f"Configuration error: {e}")
            
    elif args.preprocess_data:
        print(f"Preprocessing data: {args.preprocess_data}")
        # Add preprocessing logic here
        
    elif args.benchmark_methods:
        print("Benchmarking RPCA methods...")
        # Add benchmarking logic here


if __name__ == "__main__":
    # Check if running as CLI
    import sys
    if len(sys.argv) > 1 and sys.argv[1].startswith('--'):
        rpca_cli()
    else:
        # Run normal training
        main()