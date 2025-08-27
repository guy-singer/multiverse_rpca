#!/usr/bin/env python3
"""
RPCA Configuration Validation Script

This script validates RPCA configurations and provides helpful diagnostics.

Usage:
    python validate_rpca_config.py
    python validate_rpca_config.py --config config/env/racing.yaml
    python validate_rpca_config.py --fix-common-issues
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any

import yaml
from omegaconf import OmegaConf, DictConfig


class RPCAConfigValidator:
    """Validates and diagnoses RPCA configuration issues."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
    def validate_config(self, config_path: str) -> bool:
        """
        Validate RPCA configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Load configuration
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                with open(config_path, 'r') as f:
                    cfg_dict = yaml.safe_load(f)
                cfg = OmegaConf.create(cfg_dict)
            else:
                cfg = OmegaConf.load(config_path)
                
        except Exception as e:
            self.errors.append(f"Failed to load configuration: {e}")
            return False
            
        # Validate RPCA section
        self._validate_rpca_section(cfg)
        
        # Validate data paths
        self._validate_data_paths(cfg)
        
        # Validate model compatibility
        self._validate_model_compatibility(cfg)
        
        # Check for common issues
        self._check_common_issues(cfg)
        
        # Generate suggestions
        self._generate_suggestions(cfg)
        
        return len(self.errors) == 0
        
    def _validate_rpca_section(self, cfg: DictConfig) -> None:
        """Validate RPCA configuration section."""
        if not hasattr(cfg, 'rpca'):
            self.warnings.append("No RPCA configuration found. RPCA will be disabled.")
            return
            
        rpca_cfg = cfg.rpca
        
        # Check required fields
        if not isinstance(rpca_cfg.get('enabled'), bool):
            self.errors.append("RPCA 'enabled' field must be a boolean")
            
        # Validate method
        valid_methods = ['inexact_alm', 'pcp', 'nonconvex']
        method = rpca_cfg.get('method', 'inexact_alm')
        if method not in valid_methods:
            self.errors.append(f"Invalid RPCA method '{method}'. Valid options: {valid_methods}")
            
        # Validate lambda coefficient
        lambda_coeff = rpca_cfg.get('lambda_coeff')
        if lambda_coeff is not None:
            try:
                lambda_val = float(lambda_coeff)
                if lambda_val <= 0:
                    self.errors.append("RPCA lambda_coeff must be positive")
                elif lambda_val > 1:
                    self.warnings.append("RPCA lambda_coeff > 1 may over-emphasize sparsity")
            except (ValueError, TypeError):
                self.errors.append("RPCA lambda_coeff must be a number or null")
                
        # Validate fusion method
        valid_fusion = ['concat', 'add', 'attention']
        fusion_method = rpca_cfg.get('fusion_method', 'concat')
        if fusion_method not in valid_fusion:
            self.errors.append(f"Invalid fusion method '{fusion_method}'. Valid options: {valid_fusion}")
            
        # Validate loss weights
        if 'loss_weights' in rpca_cfg:
            self._validate_loss_weights(rpca_cfg.loss_weights)
            
    def _validate_loss_weights(self, loss_weights: DictConfig) -> None:
        """Validate RPCA loss weights."""
        required_weights = ['lambda_lowrank', 'lambda_sparse', 'lambda_consistency', 'beta_nuclear']
        
        for weight_name in required_weights:
            if weight_name not in loss_weights:
                self.warnings.append(f"Missing loss weight: {weight_name}")
            else:
                try:
                    weight_val = float(loss_weights[weight_name])
                    if weight_val < 0:
                        self.errors.append(f"Loss weight {weight_name} must be non-negative")
                except (ValueError, TypeError):
                    self.errors.append(f"Loss weight {weight_name} must be a number")
                    
        # Check for reasonable weight ranges
        if 'lambda_consistency' in loss_weights:
            consistency_weight = float(loss_weights['lambda_consistency'])
            if consistency_weight > 1.0:
                self.warnings.append("High consistency weight may dominate other loss terms")
                
    def _validate_data_paths(self, cfg: DictConfig) -> None:
        """Validate data paths for RPCA."""
        if not hasattr(cfg, 'path_data_low_res') or not hasattr(cfg, 'path_data_full_res'):
            self.errors.append("RPCA requires both path_data_low_res and path_data_full_res")
            return
            
        # Check if paths exist (if they're specified)
        low_res_path = cfg.get('path_data_low_res')
        full_res_path = cfg.get('path_data_full_res')
        
        if low_res_path and low_res_path != "null":
            if not Path(low_res_path).exists():
                self.warnings.append(f"Low resolution data path does not exist: {low_res_path}")
                
        if full_res_path and full_res_path != "null":
            if not Path(full_res_path).exists():
                self.warnings.append(f"Full resolution data path does not exist: {full_res_path}")
                
    def _validate_model_compatibility(self, cfg: DictConfig) -> None:
        """Validate model compatibility with RPCA."""
        # Check if denoiser configuration exists
        if not hasattr(cfg, 'denoiser'):
            self.errors.append("Denoiser configuration required for RPCA")
            return
            
        # Check batch size
        if hasattr(cfg, 'denoiser') and hasattr(cfg.denoiser, 'training'):
            batch_size = cfg.denoiser.training.get('batch_size', 32)
            if batch_size < 4:
                self.warnings.append("Small batch size may affect RPCA decomposition quality")
                
        # Check sequence length
        if hasattr(cfg, 'denoiser') and hasattr(cfg.denoiser.training, 'num_autoregressive_steps'):
            steps = cfg.denoiser.training.num_autoregressive_steps
            if steps < 4:
                self.warnings.append("Short sequences may not benefit from temporal RPCA")
                
    def _check_common_issues(self, cfg: DictConfig) -> None:
        """Check for common configuration issues."""
        # RPCA enabled but no cache directory
        if (hasattr(cfg, 'rpca') and cfg.rpca.get('enabled', False) and 
            not cfg.rpca.get('cache_dir')):
            self.warnings.append("RPCA cache directory not specified. Decompositions will not be cached.")
            
        # High memory usage warning
        if (hasattr(cfg, 'rpca') and cfg.rpca.get('enabled', False) and
            hasattr(cfg, 'denoiser') and cfg.denoiser.training.get('batch_size', 32) > 16):
            self.warnings.append("Large batch size with RPCA may cause memory issues")
            
        # Training configuration compatibility
        if (hasattr(cfg, 'training') and cfg.training.get('cache_in_ram', False) and
            hasattr(cfg, 'rpca') and cfg.rpca.get('enabled', False)):
            self.suggestions.append("Consider disabling cache_in_ram with RPCA to save memory")
            
    def _generate_suggestions(self, cfg: DictConfig) -> None:
        """Generate optimization suggestions."""
        if not hasattr(cfg, 'rpca') or not cfg.rpca.get('enabled', False):
            return
            
        rpca_cfg = cfg.rpca
        
        # Suggest lambda coefficient if not set
        if rpca_cfg.get('lambda_coeff') is None:
            self.suggestions.append("Consider setting explicit lambda_coeff for reproducible results")
            
        # Suggest fusion method based on data type
        fusion_method = rpca_cfg.get('fusion_method', 'concat')
        if fusion_method == 'concat':
            self.suggestions.append("Try 'add' fusion for memory efficiency or 'attention' for better performance")
            
        # Suggest temporal mode based on sequence length
        temporal_mode = rpca_cfg.get('temporal_mode', True)
        if not temporal_mode:
            self.suggestions.append("Enable temporal_mode for better compression on video sequences")
            
    def print_report(self) -> None:
        """Print validation report."""
        print("RPCA Configuration Validation Report")
        print("=" * 50)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if self.suggestions:
            print(f"\n💡 SUGGESTIONS ({len(self.suggestions)}):")
            for suggestion in self.suggestions:
                print(f"  • {suggestion}")
                
        if not self.errors and not self.warnings:
            print(f"\n✅ Configuration is valid!")
            
        print(f"\nSummary: {len(self.errors)} errors, {len(self.warnings)} warnings, {len(self.suggestions)} suggestions")


def create_default_rpca_config() -> Dict[str, Any]:
    """Create a default RPCA configuration."""
    return {
        'rpca': {
            'enabled': True,
            'method': 'inexact_alm',
            'lambda_coeff': None,
            'temporal_mode': True,
            'cache_dir': 'rpca_cache',
            'max_cache_size': 1000,
            'enable_parallel': True,
            'cross_view_mode': 'stack_channels',
            'compression_threshold': 0.01,
            'loss_weights': {
                'lambda_lowrank': 1.0,
                'lambda_sparse': 1.0,
                'lambda_consistency': 0.1,
                'beta_nuclear': 0.01
            }
        }
    }


def fix_common_issues(config_path: str) -> None:
    """Automatically fix common configuration issues."""
    print(f"Attempting to fix common issues in {config_path}")
    
    # Load existing config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Add RPCA section if missing
    if 'rpca' not in config:
        print("Adding RPCA configuration section...")
        config['rpca'] = create_default_rpca_config()['rpca']
        config['rpca']['enabled'] = False  # Default to disabled
        
    # Fix loss weights if missing
    if 'loss_weights' not in config.get('rpca', {}):
        print("Adding missing loss weights...")
        config['rpca']['loss_weights'] = create_default_rpca_config()['rpca']['loss_weights']
        
    # Create backup
    backup_path = config_path + '.backup'
    with open(backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Backup created: {backup_path}")
    
    # Save fixed config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print(f"Configuration updated: {config_path}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate RPCA configuration')
    parser.add_argument('--config', type=str, default='config/env/racing.yaml',
                       help='Path to configuration file')
    parser.add_argument('--fix-common-issues', action='store_true',
                       help='Automatically fix common configuration issues')
    parser.add_argument('--create-example', type=str,
                       help='Create example RPCA configuration file')
    
    args = parser.parse_args()
    
    if args.create_example:
        example_config = create_default_rpca_config()
        with open(args.create_example, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
        print(f"Example RPCA configuration created: {args.create_example}")
        return
        
    if args.fix_common_issues:
        if Path(args.config).exists():
            fix_common_issues(args.config)
        else:
            print(f"Configuration file not found: {args.config}")
        return
        
    # Validate configuration
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
        
    validator = RPCAConfigValidator()
    is_valid = validator.validate_config(args.config)
    validator.print_report()
    
    if not is_valid:
        print(f"\n❌ Validation failed. Use --fix-common-issues to auto-fix some issues.")
        sys.exit(1)
    else:
        print(f"\n✅ Configuration is valid and ready to use!")


if __name__ == "__main__":
    main()