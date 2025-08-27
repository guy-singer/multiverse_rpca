#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
from scipy import ndimage
import time
import tqdm
import einops
import cv2

def rpca_alm(D, lambda_param=None, mu=None, tol=1e-7, max_iter=100):
    """Robust PCA with improved convergence."""
    m, n = D.shape
    
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    if mu is None:
        mu = 0.25 / np.linalg.norm(D, ord=2)
    
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    
    print(f"RPCA parameters: λ={lambda_param:.4f}, μ={mu:.6f}")
    
    for iter_num in tqdm.tqdm(range(max_iter)):
        # Update L: SVD shrinkage
        temp_L = D - S + (1/mu) * Y
        U, sigma, Vt = randomized_svd(temp_L, n_components=min(m, n, 50))
        sigma_shrunk = np.maximum(sigma - 1/mu, 0)
        L = U @ np.diag(sigma_shrunk) @ Vt
        
        # Update S: soft thresholding
        temp_S = D - L + (1/mu) * Y
        S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - lambda_param/mu, 0)
        
        # Update Y
        residual = D - L - S
        Y = Y + mu * residual
        
        # Check convergence
        primal_residual = np.linalg.norm(residual, 'fro')
        if primal_residual < tol:
            print(f"Converged after {iter_num + 1} iterations")
            break
            
        if (iter_num + 1) % 20 == 0:
            print(f"Iter {iter_num + 1}: residual = {primal_residual:.2e}")
    
    return L, S, iter_num + 1

def load_frame_sequence(hdf5_path, start_frame=0, num_frames=30):
    """Load frames with error handling."""
    frames = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(start_frame, start_frame + num_frames):
            frame_key = f'frame_{i}_x'
            
            if frame_key in f:
                frame1 = f[frame_key][:,:,:3]
                frame2 = f[frame_key][:,:,3:]
                # convert to grayscale
                frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)[:,:,None]
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)[:,:,None]

                frame = np.concatenate([frame1, frame2], axis=2)
                frames.append(frame)
            else:
                break
    
    return np.array(frames)

def upsample_frames(frames, scale_factor=2):
    """Upsample frames using bilinear interpolation."""
    T, H, W, C = frames.shape
    upsampled = np.zeros((T, H*scale_factor, W*scale_factor, C), dtype=frames.dtype)
    
    for t in range(T):
        for c in range(C):
            upsampled[t, :, :, c] = ndimage.zoom(frames[t, :, :, c], scale_factor, order=1)
    
    return upsampled

def preprocess_frames_rpca(frames, method='mean_center'):
    """
    Multiple preprocessing strategies for RPCA.
    
    Methods:
    - 'mean_center': Remove temporal mean (crucial for RPCA)
    - 'normalize': Min-max normalize to [-1, 1] 
    - 'standardize': Z-score normalization
    - 'temporal_diff': Use frame differences
    """
    T, H, W, C = frames.shape
    #frame_matrix = frames.reshape(T, H * W * C).astype(np.float32)
    frame_matrix = einops.rearrange(frames, 't h w (player color) -> t (h w player color)', color=1).astype(np.float32)

    if method == 'mean_center':
        # Convert to [0,1] then remove temporal mean
        frame_matrix = frame_matrix / 255.0
        temporal_mean = np.mean(frame_matrix, axis=0, keepdims=True)
        frame_matrix = frame_matrix - temporal_mean
        print(f"Mean-centered data. Mean removed: {np.mean(np.abs(temporal_mean)):.4f}")
        
    elif method == 'normalize':
        # Normalize to [-1, 1] range
        frame_matrix = (frame_matrix / 127.5) - 1.0
        
    elif method == 'standardize':
        # Z-score normalization
        frame_matrix = frame_matrix / 255.0
        mean = np.mean(frame_matrix)
        std = np.std(frame_matrix)
        frame_matrix = (frame_matrix - mean) / (std + 1e-8)
        
    elif method == 'temporal_diff':
        # Use temporal differences
        frame_matrix = frame_matrix / 255.0
        diff_matrix = np.diff(frame_matrix, axis=0)
        # Pad to maintain original size
        frame_matrix = np.vstack([diff_matrix, diff_matrix[-1:]])
        
    return frame_matrix

def test_lambda_values(frames, lambdas=[0.01, 0.05, 0.1, 0.3, 0.5]):
    """Test multiple lambda values to find optimal balance."""
    frame_matrix = preprocess_frames_rpca(frames, method='mean_center')
    
    results = {}
    
    for lam in lambdas:
        print(f"\n--- Testing λ = {lam} ---")
        L, S, iterations = rpca_alm(frame_matrix, lambda_param=lam, max_iter=50)
        
        # Calculate metrics
        nuclear_norm = np.sum(np.linalg.svd(L, compute_uv=False))
        l1_norm = np.sum(np.abs(S))
        sparsity = np.mean(np.abs(S) < 1e-6) * 100
        reconstruction_error = np.linalg.norm(frame_matrix - L - S, 'fro')
        
        results[lam] = {
            'L': L, 'S': S,
            'nuclear_norm': nuclear_norm,
            'l1_norm': l1_norm,
            'sparsity': sparsity,
            'recon_error': reconstruction_error,
            'iterations': iterations
        }
        
        print(f"  Nuclear norm: {nuclear_norm:.3f}")
        print(f"  L1 norm: {l1_norm:.3f}")
        print(f"  Sparsity: {sparsity:.1f}%")
        print(f"  Reconstruction error: {reconstruction_error:.2e}")
    
    return results


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
        orig_rgb = frames[frame_idx, :, :, :1]
        L_rgb = np.clip(L_frames[frame_idx, :, :, :1] * 255, 0, 255).astype(np.uint8)
        S_rgb = np.clip(S_frames[frame_idx, :, :, :1] * 255, 0, 255).astype(np.uint8)
        
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

def visualize_lambda_comparison(frames, results, lambdas):
    """Visualize results across different lambda values for both players."""
    T, H, W, C = frames.shape
    frame_idx = T // 2  # Middle frame
    
    # Create subplot layout: 6 rows (3 per player) x (len(lambdas) + 1) columns
    fig, axes = plt.subplots(6, len(lambdas) + 1, figsize=(4*(len(lambdas)+1), 18))
    
    # Original frames for both players
    orig_player1 = frames[frame_idx, :, :, :1]  # Player 1
    orig_player2 = frames[frame_idx, :, :, 1:2]  # Player 2
    
    # Player 1 original
    axes[0, 0].imshow(orig_player1)
    axes[0, 0].set_title('Original Player 1')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    
    # Player 2 original
    axes[3, 0].imshow(orig_player2)
    axes[3, 0].set_title('Original Player 2')
    axes[3, 0].axis('off')
    axes[4, 0].axis('off')
    axes[5, 0].axis('off')
    
    # Results for each lambda
    for i, lam in enumerate(lambdas):
        L = results[lam]['L'].reshape(T, H, W, C)
        S = results[lam]['S'].reshape(T, H, W, C)
        
        # Player 1 components (channel 0)
        L_vis_p1 = np.clip((L[frame_idx, :, :, :1] + 0.5) * 255, 0, 255).astype(np.uint8)
        S_enhanced_p1 = np.clip(np.abs(S[frame_idx, :, :, :1]) * 1000, 0, 255).astype(np.uint8)
        
        # Player 2 components (channel 1)
        L_vis_p2 = np.clip((L[frame_idx, :, :, 1:2] + 0.5) * 255, 0, 255).astype(np.uint8)
        S_enhanced_p2 = np.clip(np.abs(S[frame_idx, :, :, 1:2]) * 1000, 0, 255).astype(np.uint8)
        
        # Player 1 visualizations
        axes[0, i+1].imshow(L_vis_p1)
        axes[0, i+1].set_title(f'L P1 (λ={lam})')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(S_enhanced_p1)
        axes[1, i+1].set_title(f'S P1 (λ={lam})')
        axes[1, i+1].axis('off')
        
        # Player 2 visualizations
        axes[3, i+1].imshow(L_vis_p2)
        axes[3, i+1].set_title(f'L P2 (λ={lam})')
        axes[3, i+1].axis('off')
        
        axes[4, i+1].imshow(S_enhanced_p2)
        axes[4, i+1].set_title(f'S P2 (λ={lam})')
        axes[4, i+1].axis('off')
        
        # Show sparsity percentage for both players
        sparsity = results[lam]['sparsity']
        
        # Player 1 stats
        axes[2, i+1].text(0.5, 0.5, f'{sparsity:.1f}% sparse\nL1: {results[lam]["l1_norm"]:.2f}', 
                         ha='center', va='center', transform=axes[2, i+1].transAxes)
        axes[2, i+1].axis('off')
        
        # Player 2 stats (same overall stats but separate visualization)
        axes[5, i+1].text(0.5, 0.5, f'{sparsity:.1f}% sparse\nL1: {results[lam]["l1_norm"]:.2f}', 
                         ha='center', va='center', transform=axes[5, i+1].transAxes)
        axes[5, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lambda_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    dataset_path = '/home/ubuntu/multiplayer-racing-low-res/dataset_multiplayer_racing_470.hdf5'
    
    print("=== RPCA Parameter Optimization Test ===\n")
    
    # Load smaller sequence for faster testing
    print("Loading frames...")
    frames = load_frame_sequence(dataset_path, start_frame=0, num_frames=50)
    print(f"Loaded {len(frames)} frames with shape {frames.shape}")
    
    # Option 1: Test with original resolution
    print("\n1. Testing with original resolution (48x64)")
    lambdas_test = [0.005]
    results_orig = test_lambda_values(frames, lambdas_test)
    
    # Visualizations
    print("\n3. Generating comparison visualizations...")
    visualize_lambda_comparison(frames, results_orig, lambdas_test)
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    best_lambda = None
    best_balance = float('inf')
    
    for lam in lambdas_test:
        # Look for good balance: decent sparsity but not too high reconstruction error
        balance_score = results_orig[lam]['recon_error'] + (100 - results_orig[lam]['sparsity']) * 0.01
        if balance_score < best_balance:
            best_balance = balance_score
            best_lambda = lam
    
    print(f"Best lambda for balance: {best_lambda}")
    print(f"Sparsity: {results_orig[best_lambda]['sparsity']:.1f}%")
    print(f"Reconstruction error: {results_orig[best_lambda]['recon_error']:.2e}")
    
    # Alternative preprocessing strategies
    print("\n4. Testing alternative preprocessing...")
    frame_matrix_diff = preprocess_frames_rpca(frames, method='temporal_diff')
    L_diff, S_diff, _ = rpca_alm(frame_matrix_diff, lambda_param=0.1, max_iter=30)
    sparsity_diff = np.mean(np.abs(S_diff) < 1e-6) * 100
    print(f"Temporal differences approach - Sparsity: {sparsity_diff:.1f}%")

    # Visualization
    print("\nGenerating visualizations...")
    visualize_rpca_results(frames, L_diff, S_diff, frame_indices=[0, 25, 49])

if __name__ == "__main__":
    main()

