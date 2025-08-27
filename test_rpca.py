#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import time

def rpca_alm(D, lambda_param=None, mu=None, tol=1e-7, max_iter=1000):
    """
    Robust Principal Component Analysis using Augmented Lagrange Multiplier method.
    
    Solves: min ||L||_* + λ||S||_1  subject to  D = L + S
    
    Args:
        D: Input data matrix (observations x features)
        lambda_param: Sparsity parameter (default: 1/sqrt(max(m,n)))
        mu: Penalty parameter (default: auto-computed)
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        L: Low-rank component
        S: Sparse component
        num_iter: Number of iterations
    """
    m, n = D.shape
    
    # Default parameters
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    if mu is None:
        mu = 0.25 / np.linalg.norm(D, ord=2)
    
    # Initialize variables
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    
    print(f"RPCA parameters: λ={lambda_param:.6f}, μ={mu:.6f}")
    
    for iter_num in range(max_iter):
        # Update L: SVD shrinkage
        U, sigma, Vt = randomized_svd(D - S + (1/mu) * Y, n_components=min(m, n))
        sigma_shrunk = np.maximum(sigma - 1/mu, 0)
        L = U @ np.diag(sigma_shrunk) @ Vt
        
        # Update S: soft thresholding
        temp = D - L + (1/mu) * Y
        S = np.sign(temp) * np.maximum(np.abs(temp) - lambda_param/mu, 0)
        
        # Update dual variable Y
        residual = D - L - S
        Y = Y + mu * residual
        
        # Check convergence
        primal_residual = np.linalg.norm(residual, 'fro')
        if primal_residual < tol:
            print(f"Converged after {iter_num + 1} iterations")
            break
            
        if (iter_num + 1) % 50 == 0:
            print(f"Iteration {iter_num + 1}: residual = {primal_residual:.2e}")
    
    return L, S, iter_num + 1

def load_frame_sequence(hdf5_path, start_frame=0, num_frames=50):
    """Load a sequence of frames from HDF5 dataset."""
    frames = []
    actions = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(start_frame, start_frame + num_frames):
            frame_key = f'frame_{i}_x'
            action_key = f'frame_{i}_y'
            
            if frame_key in f:
                frame = f[frame_key][:]  # Shape: (48, 64, 6)
                action = f[action_key][:]  # Shape: (66,)
                
                frames.append(frame)
                actions.append(action)
            else:
                print(f"Frame {i} not found, stopping at {len(frames)} frames")
                break
    
    return np.array(frames), np.array(actions)

def preprocess_frames_for_rpca(frames):
    """
    Reshape frames for RPCA analysis.
    Input: frames of shape (T, H, W, C)
    Output: matrix of shape (T, H*W*C) for temporal analysis
    """
    T, H, W, C = frames.shape
    # Flatten spatial and channel dimensions
    frame_matrix = frames.reshape(T, H * W * C)
    
    # Normalize to [0, 1] range
    frame_matrix = frame_matrix.astype(np.float32) / 255.0
    
    return frame_matrix

def visualize_rpca_results(frames, L, S, frame_indices=[0, 10, 20]):
    """Visualize original frames, low-rank and sparse components."""
    T, H, W, C = frames.shape
    
    # Reshape back to frame format
    L_frames = L.reshape(T, H, W, C)
    S_frames = S.reshape(T, H, W, C)
    
    fig, axes = plt.subplots(len(frame_indices), 4, figsize=(16, 4*len(frame_indices)))
    if len(frame_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, frame_idx in enumerate(frame_indices):
        # Use first 3 channels as RGB for visualization
        orig_rgb = frames[frame_idx, :, :, :3]
        L_rgb = np.clip(L_frames[frame_idx, :, :, :3] * 255, 0, 255).astype(np.uint8)
        S_rgb = np.clip(S_frames[frame_idx, :, :, :3] * 255, 0, 255).astype(np.uint8)
        
        # Compute difference for visualization
        diff_rgb = np.clip(np.abs(orig_rgb.astype(float) - L_rgb.astype(float)), 0, 255).astype(np.uint8)
        
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f'Original Frame {frame_idx}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(L_rgb)
        axes[i, 1].set_title('Low-rank (L)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(S_rgb)
        axes[i, 2].set_title('Sparse (S)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(diff_rgb)
        axes[i, 3].set_title('|Original - L|')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('rpca_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_rpca_components(L, S, frames_shape):
    """Analyze the RPCA decomposition results."""
    print("\n=== RPCA Analysis ===")
    
    # Basic statistics
    print(f"Low-rank component L:")
    print(f"  - Nuclear norm: {np.sum(np.linalg.svd(L, compute_uv=False)):.3f}")
    print(f"  - Frobenius norm: {np.linalg.norm(L, 'fro'):.3f}")
    print(f"  - Effective rank: {np.sum(np.linalg.svd(L, compute_uv=False) > 1e-10)}")
    
    print(f"\nSparse component S:")
    print(f"  - L1 norm: {np.sum(np.abs(S)):.3f}")
    print(f"  - Frobenius norm: {np.linalg.norm(S, 'fro'):.3f}")
    print(f"  - Sparsity: {np.mean(np.abs(S) < 1e-6)*100:.1f}% zeros")
    
    # Temporal analysis
    T = frames_shape[0]
    S_temporal = np.linalg.norm(S.reshape(T, -1), axis=1)
    sparse_peaks = np.where(S_temporal > np.percentile(S_temporal, 90))[0]
    print(f"\nTemporal analysis:")
    print(f"  - Frames with high sparse activity: {sparse_peaks}")
    print(f"  - Max sparse activity at frame: {np.argmax(S_temporal)}")

def main():
    dataset_path = '/Users/guysinger/Desktop/multiverse/dataset_multiplayer_racing_1.hdf5'
    
    print("Loading frame sequence...")
    frames, actions = load_frame_sequence(dataset_path, start_frame=0, num_frames=50)
    print(f"Loaded {len(frames)} frames with shape {frames.shape}")
    print(f"Actions shape: {actions.shape}")
    
    print("\nPreprocessing frames for RPCA...")
    frame_matrix = preprocess_frames_for_rpca(frames)
    print(f"Frame matrix shape: {frame_matrix.shape}")
    
    print("\nRunning RPCA decomposition...")
    start_time = time.time()
    L, S, num_iter = rpca_alm(frame_matrix, lambda_param=0.01, max_iter=100)
    elapsed_time = time.time() - start_time
    
    print(f"\nRPCA completed in {elapsed_time:.2f}s ({num_iter} iterations)")
    
    # Verify decomposition
    reconstruction_error = np.linalg.norm(frame_matrix - L - S, 'fro')
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    
    # Analysis
    analyze_rpca_components(L, S, frames.shape)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualize_rpca_results(frames, L, S, frame_indices=[0, 25, 49])
    
    print("\nRPCA test completed! Results saved to 'rpca_results.png'")

if __name__ == "__main__":
    main()