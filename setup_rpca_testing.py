#!/usr/bin/env python3
"""
Quick setup script for RPCA testing.
Ensures all dependencies are installed and configuration is ready.
"""

import subprocess
import sys
from pathlib import Path
import importlib


def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'torch', 'numpy', 'scipy', 'scikit-learn', 
        'tqdm', 'yaml', 'h5py', 'pillow'
    ]
    
    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
            
    return missing


def install_missing_packages(missing):
    """Install missing packages."""
    if not missing:
        return True
        
    print(f"\n📦 Installing missing packages: {', '.join(missing)}")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install'
        ] + missing, check=True)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False


def check_torch_setup():
    """Check PyTorch installation and GPU availability."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU available: {gpu_count}x {gpu_name}")
            return "gpu"
        else:
            print("💻 CPU-only mode (GPU not available)")
            return "cpu"
    except ImportError:
        print("❌ PyTorch not available")
        return None


def validate_project_structure():
    """Check that required files exist."""
    required_files = [
        'src/rpca/__init__.py',
        'src/rpca/core.py',
        'config/env/racing.yaml',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")
            
    return len(missing_files) == 0


def create_minimal_test_config():
    """Create a minimal configuration for testing."""
    test_config = """
# Minimal test configuration
defaults:
  - _self_
  - env: racing
  - agent: racing
  - world_model_env: fast

# Minimal training settings for testing
training:
  should: true
  num_final_epochs: 2
  cache_in_ram: false
  num_workers_data_loaders: 0

denoiser:
  training:
    num_autoregressive_steps: 2
    steps_first_epoch: 2
    steps_per_epoch: 2
    batch_size: 2
    grad_acc_steps: 1
    
# Enable RPCA for testing
rpca:
  enabled: true
  method: "inexact_alm"
  lambda_coeff: 0.1
  cache_dir: "test_rpca_cache" 
  max_cache_size: 10
  enable_parallel: false
"""

    config_path = Path('config/test_rpca.yaml')
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(test_config)
        
    print(f"✅ Created test config: {config_path}")
    return config_path


def run_quick_validation():
    """Run a very quick validation test."""
    print("\n🔍 Running quick validation...")
    
    validation_code = '''
try:
    # Test basic imports
    from src.rpca.core import InexactALM
    from src.rpca.decomposition import FrameRPCA
    from src.data.rpca_processor import RPCAConfig
    
    # Test basic functionality
    import numpy as np
    np.random.seed(42)
    
    # Quick RPCA test
    D = np.random.randn(10, 8)
    rpca = InexactALM(lambda_coeff=0.1, max_iter=10)
    L, S = rpca.fit_transform(D)
    
    error = np.linalg.norm(D - (L + S))
    assert error < 1e-1, f"High reconstruction error: {error}"
    
    print("✅ Basic RPCA functionality working")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    exit(1)
'''
    
    try:
        subprocess.run([sys.executable, '-c', validation_code], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("🚀 RPCA Testing Setup")
    print("=" * 50)
    
    setup_steps = [
        ("Python Version", check_python_version),
        ("Project Structure", validate_project_structure),
        ("Dependencies", lambda: len(check_dependencies()) == 0),
        ("PyTorch Setup", lambda: check_torch_setup() is not None),
        ("Quick Validation", run_quick_validation)
    ]
    
    # Run dependency check and installation
    missing = check_dependencies()
    if missing:
        print(f"\n📦 Installing missing packages...")
        if not install_missing_packages(missing):
            print("❌ Setup failed - could not install dependencies")
            sys.exit(1)
    
    # Run setup steps
    results = {}
    for name, check_func in setup_steps:
        print(f"\n🔍 Checking {name}...")
        results[name] = check_func()
        
    # Create test configuration
    test_config_path = create_minimal_test_config()
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 SETUP SUMMARY")
    print(f"{'='*50}")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ OK" if passed else "❌ FAIL"
        print(f"{name:<20} {status}")
        if not passed:
            all_passed = False
            
    print(f"\n📁 Test config created: {test_config_path}")
    
    if all_passed:
        print(f"\n🎉 Setup complete! Ready for local testing.")
        print(f"\n🧪 Run quick tests with:")
        print(f"   python test_rpca_locally.py")
        print(f"\n🎮 Create synthetic data with:")
        print(f"   python create_test_data.py --tiny")
        print(f"\n🏁 Run mini experiment with:")
        print(f"   python test_rpca_locally.py --test experiment")
    else:
        print(f"\n💥 Setup incomplete. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()