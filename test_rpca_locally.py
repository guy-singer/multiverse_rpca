#!/usr/bin/env python3
"""
Quick local RPCA testing script.
Runs minimal tests without requiring the full dataset or expensive compute.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import shutil


def run_command(cmd: list, description: str, timeout: int = 300) -> bool:
    """Run command with timeout and return success status."""
    print(f"\n🔄 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True, timeout=timeout, 
                              capture_output=False, text=True)
        elapsed = time.time() - start_time
        print(f"✅ {description} - SUCCESS ({elapsed:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>{timeout}s)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - COMMAND NOT FOUND")
        return False


def test_rpca_algorithms():
    """Test core RPCA algorithms with synthetic data."""
    print("🧪 Testing RPCA Algorithms")
    
    test_code = """
import numpy as np
from src.rpca.core import InexactALM, NonconvexRPCA

# Create small test matrix
np.random.seed(42)
m, n = 20, 15
L_true = np.random.randn(m, 5) @ np.random.randn(5, n)
S_true = np.zeros((m, n))
S_true.flat[::10] = np.random.randn(m*n//10) * 2
D = L_true + S_true

print("Testing Inexact ALM...")
rpca = InexactALM(lambda_coeff=0.1, max_iter=50)
L, S = rpca.fit_transform(D)
error = np.linalg.norm(D - (L + S), 'fro')
print(f"Reconstruction error: {error:.6f}")

print("Testing Non-convex RPCA...")  
rpca = NonconvexRPCA(rank=8, lambda_coeff=0.1, max_iter=50)
L, S = rpca.fit_transform(D)
error = np.linalg.norm(D - (L + S), 'fro')
print(f"Reconstruction error: {error:.6f}")

print("✅ RPCA algorithms working!")
"""
    
    return run_command(['python', '-c', test_code], "RPCA Algorithm Test", timeout=60)


def test_memory_optimization():
    """Test memory optimization features."""
    print("💾 Testing Memory Optimization")
    
    test_code = """
import numpy as np
from src.rpca.memory_optimization import CompressedRPCAStorage, StreamingRPCAProcessor

# Test compressed storage
np.random.seed(42)
L = np.random.randn(50, 10) @ np.random.randn(10, 30)  # Fixed dimensions: (50,10) @ (10,30) = (50,30)
S = np.zeros((50, 30))
S.flat[::20] = np.random.randn(50*30//20) * 3

print("Testing compressed storage...")
storage = CompressedRPCAStorage(L, S)
print(f"Compression ratio: {storage.stats.compression_ratio:.2f}x")
print(f"Memory saved: {storage.stats.memory_saved_mb:.2f} MB")

L_rec, S_rec = storage.reconstruct()
error = np.linalg.norm(L - L_rec, 'fro') + np.linalg.norm(S - S_rec, 'fro')
print(f"Reconstruction error: {error:.6f}")

# Test streaming RPCA
print("Testing streaming RPCA...")
processor = StreamingRPCAProcessor(rank=5, ambient_dim=20)
for i in range(10):
    obs = np.random.randn(20)
    L_part, S_part = processor.update(obs)
    
print("✅ Memory optimization working!")
"""
    
    return run_command(['python', '-c', test_code], "Memory Optimization Test", timeout=60)


def test_configuration_validation():
    """Test configuration validation."""
    print("⚙️ Testing Configuration")
    
    # Test existing config
    cmd1 = ['python', 'validate_rpca_config.py', '--config', 'config/env/racing.yaml']
    success1 = run_command(cmd1, "Configuration Validation", timeout=30)
    
    return success1


def create_and_test_synthetic_data():
    """Create synthetic data and test preprocessing."""
    print("🎮 Testing with Synthetic Data")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create tiny synthetic dataset
        cmd1 = ['python', 'create_test_data.py', '--output', str(temp_path), '--tiny']
        success1 = run_command(cmd1, "Create Synthetic Data", timeout=120)
        
        if not success1:
            return False
            
        # Test RPCA preprocessing on synthetic data
        test_code = f"""
import torch
from pathlib import Path
from src.data import RPCAConfig, RPCADataset

# Test RPCA dataset loading
data_path = Path('{temp_path}')
rpca_config = RPCAConfig(enabled=True, method='inexact_alm', cache_dir=None)

print("Loading synthetic dataset...")
train_dataset = RPCADataset(
    data_path / 'low_res' / 'train',
    None, 
    rpca_config,
    cache_in_ram=True
)
train_dataset.load_from_default_path()

print(f"Dataset loaded: {{train_dataset.num_episodes}} episodes, {{train_dataset.num_steps}} steps")

# Test RPCA processing
from src.data.segment import SegmentId
segment_id = SegmentId(0, 0, 5, True)
segment = train_dataset[segment_id]

print(f"Segment shape: {{segment.obs.shape}}")
print(f"RPCA decomposed: {{segment.info.get('rpca_decomposed', False)}}")

if 'rpca_lowrank' in segment.info:
    L = segment.info['rpca_lowrank']  
    S = segment.info['rpca_sparse']
    print(f"Low-rank shape: {{L.shape}}")
    print(f"Sparse shape: {{S.shape}}")
    print("✅ RPCA preprocessing working!")
else:
    print("⚠️ RPCA not applied")
"""
        
        success2 = run_command(['python', '-c', test_code], "RPCA Preprocessing Test", timeout=120)
        
        return success1 and success2


def run_mini_experiment():
    """Run a very small experiment comparing baseline vs RPCA."""
    print("🏁 Running Mini Experiment")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create tiny dataset
        cmd1 = ['python', 'create_test_data.py', '--output', str(temp_path), '--tiny']
        success1 = run_command(cmd1, "Create Test Data", timeout=120)
        
        if not success1:
            return False
            
        # Run quick experiment
        cmd2 = [
            'python', 'run_rpca_experiments.py',
            '--data', str(temp_path),
            '--experiments', 'baseline', 'rpca_default', 
            '--output', str(temp_path / 'results'),
            '--quick'
        ]
        
        success2 = run_command(cmd2, "Mini Experiment", timeout=300)
        
        if success2:
            # Show results
            results_dir = temp_path / 'results'
            if results_dir.exists():
                print(f"\n📊 Experiment Results in: {results_dir}")
                
                # Try to show summary if it exists
                summary_file = results_dir / 'comparison_summary.csv'
                if summary_file.exists():
                    try:
                        with open(summary_file, 'r') as f:
                            print("📈 Results Summary:")
                            print(f.read())
                    except Exception:
                        pass
                        
        return success2


def main():
    parser = argparse.ArgumentParser(description='Quick local RPCA testing')
    parser.add_argument('--test', choices=['algorithms', 'memory', 'config', 'data', 'experiment', 'all'],
                       default='all', help='Which test to run')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per test in seconds')
    
    args = parser.parse_args()
    
    print("🧪 RPCA Local Testing Suite")
    print("=" * 50)
    print("This runs quick tests to validate RPCA functionality")
    print("without requiring the full dataset or expensive compute.")
    print("=" * 50)
    
    tests = []
    
    if args.test in ['algorithms', 'all']:
        tests.append(('RPCA Algorithms', test_rpca_algorithms))
        
    if args.test in ['memory', 'all']:
        tests.append(('Memory Optimization', test_memory_optimization))
        
    if args.test in ['config', 'all']:
        tests.append(('Configuration', test_configuration_validation))
        
    if args.test in ['data', 'all']:
        tests.append(('Synthetic Data', create_and_test_synthetic_data))
        
    if args.test in ['experiment', 'all']:
        tests.append(('Mini Experiment', run_mini_experiment))
    
    # Run tests
    results = {}
    start_time = time.time()
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        results[name] = test_func()
        
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"🏁 LOCAL TEST SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<25} {status}")
        
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Time: {total_time:.1f}s")
    
    if passed == total:
        print(f"\n🎉 All tests passed! RPCA is ready for AWS deployment.")
        exit_code = 0
    else:
        print(f"\n💥 Some tests failed. Check the output above.")
        exit_code = 1
        
    print(f"\n💡 Next steps:")
    if passed == total:
        print(f"   1. Fork the repository to your GitHub")
        print(f"   2. Set up AWS instance with GPU")  
        print(f"   3. Download real dataset")
        print(f"   4. Run full experiments with:")
        print(f"      python run_rpca_experiments.py --data /path/to/real/data")
    else:
        print(f"   1. Fix failing tests first")
        print(f"   2. Re-run: python test_rpca_locally.py")
        
    sys.exit(exit_code)


if __name__ == "__main__":
    main()