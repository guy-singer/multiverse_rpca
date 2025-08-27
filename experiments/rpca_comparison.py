#!/usr/bin/env python3
"""
RPCA Multiverse Experiment Comparison Script

This script runs comparative experiments between:
1. Baseline Multiverse (original implementation)
2. RPCA-enhanced Multiverse (various configurations)

Usage:
    python experiments/rpca_comparison.py --config experiments/rpca_comparison_config.yaml
"""

import argparse
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import yaml

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.trainer import Trainer
from src.data import RPCAConfig, create_rpca_datasets
from src.models.rpca_agent import RPCAAgent, create_rpca_agent_config
from src.rpca import FrameRPCA, compute_compression_ratio
from omegaconf import DictConfig, OmegaConf


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    description: str
    
    # Model configuration
    use_rpca: bool = False
    rpca_method: str = "inexact_alm"
    lambda_coeff: Optional[float] = None
    fusion_method: str = "concat"
    
    # Training configuration  
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    
    # RPCA-specific settings
    rpca_loss_weights: Optional[Dict[str, float]] = None
    enable_preprocessing: bool = True
    cache_decompositions: bool = True
    
    # Evaluation settings
    eval_frequency: int = 5
    save_checkpoints: bool = True


@dataclass
class ComparisonResults:
    """Results from experiment comparison."""
    experiment_name: str
    
    # Training metrics
    final_loss: float
    convergence_epoch: int
    training_time: float
    
    # Model performance
    reconstruction_psnr: float
    reconstruction_ssim: float
    
    # RPCA-specific metrics
    compression_ratio: Optional[float] = None
    decomposition_quality: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Additional metrics
    inference_fps: float = 0.0
    model_parameters: int = 0


class RPCAExperimentRunner:
    """Manages and runs RPCA comparison experiments."""
    
    def __init__(self, base_config_path: str, 
                 data_path: str,
                 output_dir: str):
        self.base_config_path = Path(base_config_path)
        self.data_path = Path(data_path)  
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        self.base_config = OmegaConf.load(self.base_config_path)
        
        # Setup experiment tracking
        self.results: List[ComparisonResults] = []
        
    def setup_experiment_config(self, exp_config: ExperimentConfig) -> DictConfig:
        """Create Hydra config for specific experiment."""
        config = self.base_config.copy()
        
        # Update training settings
        config.training.num_final_epochs = exp_config.num_epochs
        config.denoiser.training.batch_size = exp_config.batch_size
        config.denoiser.optimizer.lr = exp_config.learning_rate
        
        # RPCA settings
        if exp_config.use_rpca:
            config.env.rpca.enabled = True
            config.env.rpca.method = exp_config.rpca_method
            config.env.rpca.lambda_coeff = exp_config.lambda_coeff
            config.env.rpca.cache_dir = str(self.output_dir / "rpca_cache" / exp_config.name)
            
            if exp_config.rpca_loss_weights:
                config.env.rpca.loss_weights.update(exp_config.rpca_loss_weights)
        else:
            config.env.rpca.enabled = False
            
        # Set data paths
        config.env.path_data_low_res = str(self.data_path / "low_res")
        config.env.path_data_full_res = str(self.data_path / "full_res")
        
        # Experiment-specific output directory
        exp_output_dir = self.output_dir / exp_config.name
        exp_output_dir.mkdir(exist_ok=True)
        
        return config
        
    def run_experiment(self, exp_config: ExperimentConfig) -> ComparisonResults:
        """Run a single experiment and return results."""
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_config.name}")
        print(f"Description: {exp_config.description}")
        print(f"RPCA enabled: {exp_config.use_rpca}")
        print(f"{'='*60}\n")
        
        # Setup configuration
        config = self.setup_experiment_config(exp_config)
        
        # Initialize wandb for this experiment
        wandb.init(
            project="multiverse-rpca-comparison",
            name=exp_config.name,
            config=asdict(exp_config),
            reinit=True
        )
        
        start_time = time.time()
        
        try:
            # Create trainer
            trainer = Trainer(config, Path.cwd())
            
            # Run training
            trainer.run()
            
            training_time = time.time() - start_time
            
            # Evaluate model
            results = self._evaluate_model(trainer, exp_config, training_time)
            
            # Save results
            self._save_experiment_results(exp_config, results)
            
        except Exception as e:
            print(f"Error in experiment {exp_config.name}: {e}")
            results = ComparisonResults(
                experiment_name=exp_config.name,
                final_loss=float('inf'),
                convergence_epoch=-1,
                training_time=time.time() - start_time,
                reconstruction_psnr=0.0,
                reconstruction_ssim=0.0
            )
        finally:
            wandb.finish()
            
        self.results.append(results)
        return results
        
    def _evaluate_model(self, trainer: Trainer, 
                       exp_config: ExperimentConfig, 
                       training_time: float) -> ComparisonResults:
        """Evaluate trained model and compute metrics."""
        
        # Basic training metrics
        final_loss = 0.0  # Extract from trainer logs
        convergence_epoch = trainer.epoch
        
        # Model size
        model_parameters = sum(p.numel() for p in trainer.agent.parameters())
        
        # Evaluation on test set
        trainer.agent.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = 0
        compression_ratios = []
        memory_usage = 0.0
        
        with torch.no_grad():
            # Memory usage baseline
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
            for batch in tqdm(trainer._data_loader_test.denoiser, desc="Evaluating"):
                batch = batch.to(trainer._device)
                
                # Forward pass
                loss, metrics, batch_data = trainer.agent.denoiser(batch)
                
                # Compute PSNR/SSIM
                pred_frames = batch_data['obs']
                target_frames = batch.obs[:, -pred_frames.shape[1]:]
                
                psnr = self._compute_psnr(pred_frames, target_frames)
                ssim = self._compute_ssim(pred_frames, target_frames)
                
                total_psnr += psnr
                total_ssim += ssim
                num_batches += 1
                
                # RPCA-specific metrics
                if exp_config.use_rpca and 'rpca_compression_stats' in batch.info[0]:
                    stats = batch.info[0]['rpca_compression_stats']
                    compression_ratios.append(stats.get('compression_ratio', 1.0))
                    
                if num_batches >= 10:  # Limit evaluation for speed
                    break
                    
            # Memory usage
            if torch.cuda.is_available():
                memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                
        # Compute averages
        avg_psnr = total_psnr / max(num_batches, 1)
        avg_ssim = total_ssim / max(num_batches, 1)
        avg_compression = np.mean(compression_ratios) if compression_ratios else None
        
        # Inference speed test
        inference_fps = self._measure_inference_speed(trainer.agent.denoiser, trainer._device)
        
        return ComparisonResults(
            experiment_name=exp_config.name,
            final_loss=final_loss,
            convergence_epoch=convergence_epoch,
            training_time=training_time,
            reconstruction_psnr=avg_psnr,
            reconstruction_ssim=avg_ssim,
            compression_ratio=avg_compression,
            memory_usage_mb=memory_usage,
            inference_fps=inference_fps,
            model_parameters=model_parameters
        )
        
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
        
    def _compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute Structural Similarity Index (simplified)."""
        # Simplified SSIM computation for demonstration
        # In practice, use a proper SSIM implementation
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
               
        return ssim.item()
        
    def _measure_inference_speed(self, model: torch.nn.Module, device: torch.device) -> float:
        """Measure inference speed in FPS."""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 10, 3, 48, 64).to(device)  # Small batch for speed
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(type('Batch', (), {'obs': dummy_input, 'act': None, 'mask_padding': torch.ones(1, 10).bool().to(device), 'info': [{}]})())
                
        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        num_frames = 100
        with torch.no_grad():
            for _ in range(num_frames):
                _ = model(type('Batch', (), {'obs': dummy_input, 'act': None, 'mask_padding': torch.ones(1, 10).bool().to(device), 'info': [{}]})())
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        
        return num_frames / elapsed
        
    def _save_experiment_results(self, exp_config: ExperimentConfig, 
                                results: ComparisonResults) -> None:
        """Save experiment results to disk."""
        exp_dir = self.output_dir / exp_config.name
        exp_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(exp_dir / "config.yaml", 'w') as f:
            yaml.dump(asdict(exp_config), f)
            
        # Save results
        with open(exp_dir / "results.json", 'w') as f:
            json.dump(asdict(results), f, indent=2)
            
    def run_comparison_suite(self, experiment_configs: List[ExperimentConfig]) -> None:
        """Run full comparison suite."""
        print(f"Starting RPCA Multiverse Comparison Suite")
        print(f"Total experiments: {len(experiment_configs)}")
        print(f"Output directory: {self.output_dir}")
        
        for exp_config in experiment_configs:
            try:
                self.run_experiment(exp_config)
            except Exception as e:
                print(f"Failed experiment {exp_config.name}: {e}")
                continue
                
        # Generate comparison report
        self.generate_comparison_report()
        
    def generate_comparison_report(self) -> None:
        """Generate comprehensive comparison report with visualizations."""
        print("\nGenerating comparison report...")
        
        # Create comparison plots
        self._create_performance_plots()
        self._create_efficiency_plots()
        self._create_summary_table()
        
        # Save consolidated results
        results_data = [asdict(result) for result in self.results]
        with open(self.output_dir / "comparison_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Report generated in: {self.output_dir}")
        
    def _create_performance_plots(self) -> None:
        """Create performance comparison plots."""
        if not self.results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        names = [r.experiment_name for r in self.results]
        psnr_values = [r.reconstruction_psnr for r in self.results]
        ssim_values = [r.reconstruction_ssim for r in self.results]
        training_times = [r.training_time / 3600 for r in self.results]  # Convert to hours
        final_losses = [r.final_loss for r in self.results]
        
        # PSNR comparison
        axes[0, 0].bar(names, psnr_values, color='skyblue')
        axes[0, 0].set_title('Reconstruction PSNR')
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        axes[0, 1].bar(names, ssim_values, color='lightcoral')
        axes[0, 1].set_title('Reconstruction SSIM')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time
        axes[1, 0].bar(names, training_times, color='lightgreen')
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Hours')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Final loss
        axes[1, 1].bar(names, final_losses, color='gold')
        axes[1, 1].set_title('Final Loss')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_efficiency_plots(self) -> None:
        """Create efficiency and resource usage plots."""
        if not self.results:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        names = [r.experiment_name for r in self.results]
        memory_usage = [r.memory_usage_mb or 0 for r in self.results]
        inference_fps = [r.inference_fps for r in self.results]
        compression_ratios = [r.compression_ratio or 1.0 for r in self.results]
        
        # Memory usage
        axes[0].bar(names, memory_usage, color='purple', alpha=0.7)
        axes[0].set_title('Memory Usage')
        axes[0].set_ylabel('Memory (MB)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Inference speed
        axes[1].bar(names, inference_fps, color='orange', alpha=0.7)
        axes[1].set_title('Inference Speed')
        axes[1].set_ylabel('FPS')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Compression ratio (RPCA experiments only)
        rpca_names = [name for name, ratio in zip(names, compression_ratios) if ratio > 1.0]
        rpca_ratios = [ratio for ratio in compression_ratios if ratio > 1.0]
        
        if rpca_ratios:
            axes[2].bar(rpca_names, rpca_ratios, color='cyan', alpha=0.7)
            axes[2].set_title('RPCA Compression Ratio')
            axes[2].set_ylabel('Compression Ratio')
            axes[2].tick_params(axis='x', rotation=45)
        else:
            axes[2].text(0.5, 0.5, 'No RPCA experiments', ha='center', va='center')
            axes[2].set_title('RPCA Compression Ratio')
            
        plt.tight_layout()
        plt.savefig(self.output_dir / "efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_summary_table(self) -> None:
        """Create summary table of all results."""
        import pandas as pd
        
        # Convert results to DataFrame
        data = []
        for result in self.results:
            data.append({
                'Experiment': result.experiment_name,
                'Final Loss': f"{result.final_loss:.4f}",
                'PSNR (dB)': f"{result.reconstruction_psnr:.2f}",
                'SSIM': f"{result.reconstruction_ssim:.3f}",
                'Training Time (h)': f"{result.training_time/3600:.2f}",
                'Memory (MB)': f"{result.memory_usage_mb:.1f}" if result.memory_usage_mb else "N/A",
                'Inference FPS': f"{result.inference_fps:.1f}",
                'Compression Ratio': f"{result.compression_ratio:.2f}" if result.compression_ratio else "N/A",
                'Parameters (M)': f"{result.model_parameters/1e6:.2f}"
            })
            
        df = pd.DataFrame(data)
        
        # Save as CSV
        df.to_csv(self.output_dir / "comparison_summary.csv", index=False)
        
        # Create formatted table image
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.savefig(self.output_dir / "summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_experiment_configs() -> List[ExperimentConfig]:
    """Create predefined experiment configurations."""
    configs = []
    
    # Baseline experiment (no RPCA)
    configs.append(ExperimentConfig(
        name="baseline",
        description="Original Multiverse implementation without RPCA",
        use_rpca=False,
        num_epochs=20
    ))
    
    # RPCA with default settings
    configs.append(ExperimentConfig(
        name="rpca_default",
        description="RPCA with default Inexact ALM and auto lambda",
        use_rpca=True,
        rpca_method="inexact_alm",
        lambda_coeff=None,
        fusion_method="concat",
        num_epochs=20
    ))
    
    # RPCA with different fusion methods
    configs.append(ExperimentConfig(
        name="rpca_fusion_add",
        description="RPCA with additive fusion",
        use_rpca=True,
        fusion_method="add",
        num_epochs=20
    ))
    
    # RPCA with different lambda values
    configs.append(ExperimentConfig(
        name="rpca_lambda_0p1",
        description="RPCA with lambda=0.1",
        use_rpca=True,
        lambda_coeff=0.1,
        num_epochs=20
    ))
    
    configs.append(ExperimentConfig(
        name="rpca_lambda_0p01",
        description="RPCA with lambda=0.01",
        use_rpca=True,
        lambda_coeff=0.01,
        num_epochs=20
    ))
    
    # RPCA with different loss weights
    configs.append(ExperimentConfig(
        name="rpca_weighted_sparse",
        description="RPCA with higher weight on sparse component",
        use_rpca=True,
        rpca_loss_weights={
            'lambda_lowrank': 1.0,
            'lambda_sparse': 2.0,
            'lambda_consistency': 0.1,
            'beta_nuclear': 0.01
        },
        num_epochs=20
    ))
    
    return configs


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='RPCA Multiverse Comparison Experiments')
    parser.add_argument('--config', type=str, default='config/trainer.yaml',
                       help='Path to base trainer configuration')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--output', type=str, default='./experiments/rpca_results',
                       help='Output directory for results')
    parser.add_argument('--experiments', type=str, nargs='+',
                       default=['baseline', 'rpca_default'],
                       help='Experiments to run')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = RPCAExperimentRunner(
        base_config_path=args.config,
        data_path=args.data,
        output_dir=args.output
    )
    
    # Get experiment configurations
    all_configs = create_experiment_configs()
    
    # Filter by requested experiments
    selected_configs = [cfg for cfg in all_configs if cfg.name in args.experiments]
    
    if not selected_configs:
        print(f"No matching experiments found. Available: {[cfg.name for cfg in all_configs]}")
        return
        
    # Run experiments
    runner.run_comparison_suite(selected_configs)
    
    print("\nExperiment comparison completed!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()