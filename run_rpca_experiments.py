#!/usr/bin/env python3
"""
Simple script to run RPCA experiments with different configurations.

Usage:
    python run_rpca_experiments.py --data /path/to/processed/data
    python run_rpca_experiments.py --data /path/to/data --experiments baseline rpca_default
    python run_rpca_experiments.py --help
"""

import argparse
import sys
from pathlib import Path

# Add experiments directory to path
sys.path.append(str(Path(__file__).parent / "experiments"))

from experiments.rpca_comparison import RPCAExperimentRunner, create_experiment_configs


def main():
    parser = argparse.ArgumentParser(
        description='Run RPCA Multiverse comparison experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all experiments
    python run_rpca_experiments.py --data ./data/processed
    
    # Run specific experiments
    python run_rpca_experiments.py --data ./data/processed --experiments baseline rpca_default
    
    # Run with custom output directory
    python run_rpca_experiments.py --data ./data/processed --output ./my_results
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to processed dataset directory containing low_res/ and full_res/ folders'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./experiment_results',
        help='Output directory for experiment results (default: ./experiment_results)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/trainer.yaml',
        help='Path to base trainer configuration (default: config/trainer.yaml)'
    )
    
    parser.add_argument(
        '--experiments', 
        type=str, 
        nargs='+',
        help='Specific experiments to run (default: run all). Available: baseline, rpca_default, rpca_fusion_add, rpca_lambda_0p1, rpca_lambda_0p01, rpca_weighted_sparse'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick experiments with fewer epochs for testing'
    )
    
    parser.add_argument(
        '--list', 
        action='store_true',
        help='List available experiments and exit'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data directory does not exist: {data_path}")
        sys.exit(1)
        
    if not (data_path / "low_res").exists() or not (data_path / "full_res").exists():
        print(f"Error: Data directory must contain 'low_res' and 'full_res' subdirectories")
        print(f"Found in {data_path}: {list(data_path.iterdir())}")
        sys.exit(1)
    
    # Get available experiment configurations
    all_configs = create_experiment_configs()
    
    if args.list:
        print("Available experiments:")
        for config in all_configs:
            print(f"  {config.name}: {config.description}")
        sys.exit(0)
    
    # Determine which experiments to run
    if args.experiments:
        # Validate experiment names
        available_names = {config.name for config in all_configs}
        invalid_names = set(args.experiments) - available_names
        
        if invalid_names:
            print(f"Error: Unknown experiments: {invalid_names}")
            print(f"Available experiments: {available_names}")
            sys.exit(1)
            
        # Filter to requested experiments
        selected_configs = [config for config in all_configs if config.name in args.experiments]
    else:
        # Run all experiments
        selected_configs = all_configs
    
    # Modify configs for quick run
    if args.quick:
        print("Running quick experiments (reduced epochs)")
        for config in selected_configs:
            config.num_epochs = 5  # Reduce epochs for quick testing
            config.eval_frequency = 2
    
    # Print experiment plan
    print(f"RPCA Multiverse Experiment Plan")
    print(f"Data directory: {data_path}")
    print(f"Output directory: {args.output}")
    print(f"Base config: {args.config}")
    print(f"Experiments to run: {len(selected_configs)}")
    
    for i, config in enumerate(selected_configs, 1):
        epochs_info = f" ({config.num_epochs} epochs)" if args.quick else ""
        print(f"  {i}. {config.name}: {config.description}{epochs_info}")
    
    # Confirm before starting
    if not args.quick:
        response = input(f"\nThis will run {len(selected_configs)} experiments. Continue? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Experiments cancelled.")
            sys.exit(0)
    
    # Create experiment runner
    print(f"\nInitializing experiment runner...")
    
    try:
        runner = RPCAExperimentRunner(
            base_config_path=args.config,
            data_path=str(data_path),
            output_dir=args.output
        )
        
        # Run experiments
        print(f"Starting experiments...")
        runner.run_comparison_suite(selected_configs)
        
        print(f"\n{'='*60}")
        print(f"All experiments completed!")
        print(f"Results saved to: {Path(args.output).absolute()}")
        print(f"{'='*60}")
        
        # Print summary
        if runner.results:
            print(f"\nExperiment Summary:")
            print(f"{'Name':<20} {'PSNR':<8} {'SSIM':<8} {'Time(h)':<10}")
            print(f"{'-'*50}")
            
            for result in runner.results:
                time_h = result.training_time / 3600
                print(f"{result.experiment_name:<20} {result.reconstruction_psnr:<8.2f} {result.reconstruction_ssim:<8.3f} {time_h:<10.2f}")
        
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        print(f"Make sure the base config file exists: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiments: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()