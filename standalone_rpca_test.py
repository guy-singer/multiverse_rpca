#!/usr/bin/env python3
"""
Standalone RPCA test that doesn't depend on the full project structure.
Tests core RPCA functionality independently.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_rpca():
    """Test basic RPCA decomposition without external dependencies."""
    print("🔬 Testing Basic RPCA Decomposition")
    
    # Import RPCA core (should work independently)  
    try:
        from rpca.utils import singular_value_shrinkage, soft_threshold, compute_rpca_lambda
        print("✅ RPCA utilities imported")
    except ImportError as e:
        print(f"❌ Failed to import RPCA utilities: {e}")
        return False
        
    # Test utility functions
    np.random.seed(42)
    
    # Test soft thresholding
    X = np.random.randn(10, 8) * 2
    X_thresh = soft_threshold(X, threshold=1.0)
    sparsity = np.mean(np.abs(X_thresh) < 1e-6)
    print(f"✅ Soft thresholding: {sparsity*100:.1f}% sparsity")
    
    # Test SVD shrinkage  
    X_shrunk = singular_value_shrinkage(X, threshold=0.5)
    U, s, Vt = np.linalg.svd(X_shrunk, full_matrices=False)
    effective_rank = np.sum(s > 0.01)
    print(f"✅ SVD shrinkage: rank reduced to {effective_rank}")
    
    # Test lambda computation
    lambda_val = compute_rpca_lambda(10, 8)
    print(f"✅ Auto lambda: {lambda_val:.4f}")
    
    return True


def test_inexact_alm_standalone():
    """Test Inexact ALM algorithm with minimal dependencies."""
    print("\n🧮 Testing Inexact ALM Algorithm")
    
    try:
        from rpca.core import InexactALM
        print("✅ InexactALM imported")
    except ImportError as e:
        print(f"❌ Failed to import InexactALM: {e}")
        return False
        
    # Create synthetic low-rank + sparse data
    np.random.seed(42)
    m, n = 30, 20
    
    # Ground truth low-rank (rank 3)
    U_true = np.random.randn(m, 3)
    V_true = np.random.randn(3, n)
    L_true = U_true @ V_true
    
    # Ground truth sparse (5% non-zero)
    S_true = np.zeros((m, n))
    num_sparse = int(0.05 * m * n)
    sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
    S_true.flat[sparse_idx] = np.random.randn(num_sparse) * 5
    
    # Observed data
    D = L_true + S_true
    
    print(f"📊 Test data: {m}×{n}, true rank=3, {num_sparse} sparse elements")
    
    # Run RPCA
    start_time = time.time()
    rpca = InexactALM(lambda_coeff=0.1, max_iter=100, tol=1e-6)
    L_recovered, S_recovered = rpca.fit_transform(D)
    elapsed = time.time() - start_time
    
    # Check reconstruction
    reconstruction_error = np.linalg.norm(D - (L_recovered + S_recovered), 'fro')
    
    # Check low-rank recovery
    U_rec, s_rec, Vt_rec = np.linalg.svd(L_recovered, full_matrices=False)
    recovered_rank = np.sum(s_rec > 0.01 * s_rec[0])
    
    # Check sparsity recovery
    recovered_sparsity = np.mean(np.abs(S_recovered) > 0.01)
    
    print(f"✅ Reconstruction error: {reconstruction_error:.6f}")
    print(f"✅ Recovered rank: {recovered_rank} (true: 3)")
    print(f"✅ Recovered sparsity: {recovered_sparsity*100:.1f}% (true: {num_sparse/(m*n)*100:.1f}%)")
    print(f"✅ Time: {elapsed:.3f}s")
    
    # Success if reconstruction is good
    return reconstruction_error < 1e-3


def test_frame_processing():
    """Test frame processing functions."""
    print("\n🎬 Testing Frame Processing")
    
    try:
        from rpca.utils import create_frame_matrix, reconstruct_frames
        print("✅ Frame utilities imported")
    except ImportError as e:
        print(f"❌ Failed to import frame utilities: {e}")
        return False
        
    # Create synthetic video frames
    T, C, H, W = 8, 3, 16, 16
    frames = np.random.randn(T, C, H, W) * 0.5
    
    # Add global motion (low-rank structure)
    global_pattern = np.sin(np.linspace(0, 2*np.pi, H*W)).reshape(H, W)
    for t in range(T):
        for c in range(C):
            frames[t, c] += global_pattern * (0.3 + 0.1 * np.cos(t * 0.5))
            
    # Add sparse events
    frames[2, :, 5:8, 5:8] += 2.0  # Sparse event at frame 2
    frames[6, :, 10:13, 10:13] += 2.0  # Sparse event at frame 6
    
    print(f"📹 Test frames: {T} frames, {C} channels, {H}×{W}")
    
    # Test matrix conversion
    matrix, shape_info = create_frame_matrix(frames, temporal_mode=True)
    print(f"✅ Frame to matrix: {frames.shape} → {matrix.shape}")
    
    # Apply simple RPCA (using our tested InexactALM)
    from rpca.core import InexactALM
    rpca = InexactALM(lambda_coeff=0.1, max_iter=50)
    L_matrix, S_matrix = rpca.fit_transform(matrix)
    
    # Reconstruct frames
    L_frames, S_frames = reconstruct_frames(L_matrix, S_matrix, shape_info)
    
    print(f"✅ Matrix to frames: {L_matrix.shape} → {L_frames.shape}")
    
    # Check reconstruction
    reconstruction_error = np.mean((frames - (L_frames + S_frames))**2)
    print(f"✅ Frame reconstruction MSE: {reconstruction_error:.6f}")
    
    # Check that sparse events are captured
    sparse_energy = np.sum(S_frames**2, axis=(1, 2, 3))  # Energy per frame
    event_frames = [2, 6]
    non_event_frames = [0, 1, 3, 4, 5, 7]
    
    avg_event_energy = np.mean(sparse_energy[event_frames])
    avg_normal_energy = np.mean(sparse_energy[non_event_frames])
    
    print(f"✅ Sparse events detected: {avg_event_energy:.3f} vs {avg_normal_energy:.3f}")
    
    return reconstruction_error < 0.1 and avg_event_energy > 2 * avg_normal_energy


def test_memory_optimization_simple():
    """Test memory optimization without external dependencies."""
    print("\n💾 Testing Memory Optimization")
    
    try:
        from rpca.memory_optimization import CompressedRPCAStorage
        print("✅ Memory optimization imported")
    except ImportError as e:
        print(f"❌ Failed to import memory optimization: {e}")
        return False
        
    # Create test matrices
    np.random.seed(42)
    m, n = 100, 50
    
    # Create low-rank matrix
    L = np.random.randn(m, 5) @ np.random.randn(5, n)
    
    # Create sparse matrix
    S = np.zeros((m, n))
    num_sparse = int(0.1 * m * n)  # 10% sparse
    sparse_idx = np.random.choice(m * n, num_sparse, replace=False)
    S.flat[sparse_idx] = np.random.randn(num_sparse) * 3
    
    print(f"📊 Test matrices: {m}×{n}, {num_sparse} sparse elements")
    
    # Test compression
    storage = CompressedRPCAStorage(L, S, rank_threshold=0.01, sparsity_threshold=1e-6)
    
    print(f"✅ Compression ratio: {storage.stats.compression_ratio:.2f}x")
    print(f"✅ Memory saved: {storage.stats.memory_saved_mb:.3f} MB")
    print(f"✅ Effective rank: {storage.stats.effective_rank}")
    print(f"✅ Sparsity: {storage.stats.sparsity_ratio*100:.1f}%")
    
    # Test reconstruction
    L_rec, S_rec = storage.reconstruct()
    
    L_error = np.linalg.norm(L - L_rec, 'fro') / np.linalg.norm(L, 'fro')
    S_error = np.linalg.norm(S - S_rec, 'fro') / np.linalg.norm(S, 'fro')
    
    print(f"✅ L reconstruction error: {L_error:.6f}")
    print(f"✅ S reconstruction error: {S_error:.6f}")
    
    # Success criteria
    return (storage.stats.compression_ratio > 2.0 and 
            L_error < 0.1 and S_error < 0.1)


def main():
    """Run all standalone tests."""
    print("🧪 Standalone RPCA Test Suite")
    print("=" * 60)
    print("Testing core RPCA functionality without full project dependencies")
    print("=" * 60)
    
    tests = [
        ("Basic RPCA Utils", test_basic_rpca),
        ("Inexact ALM Algorithm", test_inexact_alm_standalone), 
        ("Frame Processing", test_frame_processing),
        ("Memory Optimization", test_memory_optimization_simple)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
            
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"🏁 STANDALONE TEST SUMMARY") 
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:<25} {status}")
        
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Time: {total_time:.1f}s")
    
    if passed == total:
        print(f"\n🎉 All core RPCA functionality is working!")
        print(f"\n💡 RPCA is ready for integration testing and AWS deployment")
        print(f"\nNext steps:")
        print(f"  1. ✅ Core algorithms: WORKING")  
        print(f"  2. ✅ Memory optimization: WORKING")
        print(f"  3. ✅ Frame processing: WORKING") 
        print(f"  4. 🔄 Ready for full integration tests")
        return 0
    else:
        print(f"\n💥 Some core functionality failed!")
        print(f"Fix these issues before proceeding to AWS")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)