#!/usr/bin/env python3

import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import time
import cv2

def rpca_alm(D, lambda_param=None, mu=None, tol=1e-7, max_iter=100):
    """
    Robust PCA using Augmented Lagrange Multiplier method.
    Optimized for cross-view consistency analysis.
    """
    m, n = D.shape
    
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    
    if mu is None:
        mu = 0.25 / np.linalg.norm(D, ord=2)
    
    L = np.zeros_like(D)
    S = np.zeros_like(D)
    Y = np.zeros_like(D)
    
    print(f"RPCA parameters: λ={lambda_param:.4f}, μ={mu:.6f}")
    print(f"Matrix shape: {D.shape} ({m} observations, {n} features)")
    
    for iter_num in range(max_iter):
        # Update L: SVD shrinkage
        temp_L = D - S + (1/mu) * Y
        U, sigma, Vt = randomized_svd(temp_L, n_components=min(m, n, 100))
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
            
        if (iter_num + 1) % 25 == 0:
            sparsity = np.mean(np.abs(S) < 1e-6) * 100
            print(f"Iter {iter_num + 1}: residual = {primal_residual:.2e}, sparsity = {sparsity:.1f}%")
    
    return L, S, iter_num + 1

def load_dual_view_frames(hdf5_path, start_frame=0, num_frames=30):
    """
    Load frames and separate into two views.
    Returns frames with shape (T, H, W, 2) where last dim is [view1, view2]
    """
    frames = []
    
    with h5py.File(hdf5_path, 'r') as f:
        for i in range(start_frame, start_frame + num_frames):
            frame_key = f'frame_{i}_x'
            
            if frame_key in f:
                frame_data = f[frame_key][:]  # Shape: (48, 64, 6)
                
                # Split into two views (first 3 channels = view1, last 3 = view2)
                view1_rgb = frame_data[:, :, :3]
                view2_rgb = frame_data[:, :, 3:6]
                
                # Convert to grayscale for simpler analysis
                view1_gray = cv2.cvtColor(view1_rgb, cv2.COLOR_RGB2GRAY)
                view2_gray = cv2.cvtColor(view2_rgb, cv2.COLOR_RGB2GRAY)
                
                # Stack views as channels: (H, W, 2)
                dual_view_frame = np.stack([view1_gray, view2_gray], axis=2)
                frames.append(dual_view_frame)
            else:
                print(f"Frame {i} not found, stopping at {len(frames)} frames")
                break
    
    return np.array(frames)

def create_cross_view_matrix(frames):
    """
    Create the proper matrix structure for cross-view RPCA analysis.
    
    Input: frames of shape (T, H, W, 2) where last dim is [view1, view2]
    Output: matrix of shape (T×2, H×W) where rows alternate [view1_t0, view2_t0, view1_t1, view2_t1, ...]
    
    This structure allows RPCA to learn:
    - L: Shared scene structure appearing in both views
    - S: View-specific differences and temporal changes
    """
    T, H, W, num_views = frames.shape
    
    # Create alternating row structure
    cross_view_matrix = np.zeros((T * num_views, H * W), dtype=np.float32)
    
    for t in range(T):
        # View 1 at time t
        row_idx_v1 = t * 2
        cross_view_matrix[row_idx_v1, :] = frames[t, :, :, 0].flatten()
        
        # View 2 at time t  
        row_idx_v2 = t * 2 + 1
        cross_view_matrix[row_idx_v2, :] = frames[t, :, :, 1].flatten()
    
    # Normalize to [0, 1] range (NO mean centering to preserve shared background)
    cross_view_matrix = cross_view_matrix / 255.0
    
    print(f"Cross-view matrix structure:")
    print(f"  Shape: {cross_view_matrix.shape}")
    print(f"  Row 0: View 1, frame 0")
    print(f"  Row 1: View 2, frame 0") 
    print(f"  Row 2: View 1, frame 1")
    print(f"  Row 3: View 2, frame 1")
    print(f"  ...")
    
    return cross_view_matrix

def analyze_cross_view_results(frames, L, S):
    """
    Analyze RPCA results specifically for cross-view consistency.
    """
    T, H, W, num_views = frames.shape
    
    # Reshape L and S back to alternating view structure
    L_reshaped = L.reshape(T * 2, H, W)
    S_reshaped = S.reshape(T * 2, H, W)
    
    print("\n=== Cross-View RPCA Analysis ===")
    
    # Basic statistics
    print(f"Low-rank component L:")
    print(f"  - Nuclear norm: {np.sum(np.linalg.svd(L, compute_uv=False)):.3f}")
    print(f"  - Effective rank: {np.sum(np.linalg.svd(L, compute_uv=False) > 1e-6)}")
    
    print(f"\nSparse component S:")
    print(f"  - L1 norm: {np.sum(np.abs(S)):.3f}")
    print(f"  - Sparsity: {np.mean(np.abs(S) < 1e-6)*100:.1f}% zeros")
    print(f"  - Max sparse magnitude: {np.max(np.abs(S)):.4f}")
    
    # Cross-view consistency analysis
    print(f"\nCross-view consistency analysis:")
    
    view_consistency_scores = []
    sparse_activity_scores = []
    
    for t in range(T):
        # Extract L and S for both views at time t
        L_view1_t = L_reshaped[t * 2, :, :]      # Row t*2 = view1 at time t
        L_view2_t = L_reshaped[t * 2 + 1, :, :] # Row t*2+1 = view2 at time t
        
        S_view1_t = S_reshaped[t * 2, :, :]      
        S_view2_t = S_reshaped[t * 2 + 1, :, :]
        
        # Measure cross-view consistency in low-rank component
        consistency = np.corrcoef(L_view1_t.flatten(), L_view2_t.flatten())[0, 1]
        view_consistency_scores.append(consistency)
        
        # Measure sparse activity difference between views
        sparse_diff = np.mean(np.abs(S_view1_t - S_view2_t))
        sparse_activity_scores.append(sparse_diff)
    
    avg_consistency = np.nanmean(view_consistency_scores)
    avg_sparse_diff = np.mean(sparse_activity_scores)
    
    print(f"  - Average L cross-view correlation: {avg_consistency:.3f}")
    print(f"  - Average sparse activity difference: {avg_sparse_diff:.4f}")
    
    # Find frames with high/low consistency
    best_frame = np.nanargmax(view_consistency_scores)
    worst_frame = np.nanargmin(view_consistency_scores)
    
    print(f"  - Best consistency at frame {best_frame}: {view_consistency_scores[best_frame]:.3f}")
    print(f"  - Worst consistency at frame {worst_frame}: {view_consistency_scores[worst_frame]:.3f}")
    
    return view_consistency_scores, sparse_activity_scores

def visualize_cross_view_results(frames, L, S, frame_indices=[0, 10, 20]):
    """
    Visualize cross-view RPCA results showing both views and their decomposition.
    """
    T, H, W, num_views = frames.shape
    
    # Reshape L and S back to frame format
    L_frames = L.reshape(T * 2, H, W)
    S_frames = S.reshape(T * 2, H, W)
    
    fig, axes = plt.subplots(len(frame_indices), 6, figsize=(18, 3*len(frame_indices)))
    if len(frame_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= T:
            continue
            
        # Original frames
        orig_view1 = frames[frame_idx, :, :, 0]
        orig_view2 = frames[frame_idx, :, :, 1]
        
        # Decomposed components (convert back to [0, 255])
        L_view1 = np.clip(L_frames[frame_idx * 2, :, :] * 255, 0, 255).astype(np.uint8)
        L_view2 = np.clip(L_frames[frame_idx * 2 + 1, :, :] * 255, 0, 255).astype(np.uint8)
        
        # Enhanced sparse components for visibility
        S_view1 = np.clip(np.abs(S_frames[frame_idx * 2, :, :]) * 2000, 0, 255).astype(np.uint8)
        S_view2 = np.clip(np.abs(S_frames[frame_idx * 2 + 1, :, :]) * 2000, 0, 255).astype(np.uint8)
        
        # Plot original views
        axes[i, 0].imshow(orig_view1, cmap='gray')
        axes[i, 0].set_title(f'Original View 1\nFrame {frame_idx}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(orig_view2, cmap='gray')
        axes[i, 1].set_title(f'Original View 2\nFrame {frame_idx}')
        axes[i, 1].axis('off')
        
        # Plot low-rank components
        axes[i, 2].imshow(L_view1, cmap='gray')
        axes[i, 2].set_title('L View 1\n(Shared Structure)')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(L_view2, cmap='gray')
        axes[i, 3].set_title('L View 2\n(Shared Structure)')
        axes[i, 3].axis('off')
        
        # Plot sparse components
        axes[i, 4].imshow(S_view1, cmap='hot')
        axes[i, 4].set_title('S View 1\n(View-Specific)')
        axes[i, 4].axis('off')
        
        axes[i, 5].imshow(S_view2, cmap='hot') 
        axes[i, 5].set_title('S View 2\n(View-Specific)')
        axes[i, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig('cross_view_rpca_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def test_cross_view_lambdas(frames, lambdas=[0.01, 0.1, 0.3, 0.5, 1.0]):
    """
    Test different lambda values for cross-view RPCA analysis.
    """
    print("\n=== Testing Lambda Values for Cross-View RPCA ===")
    
    cross_view_matrix = create_cross_view_matrix(frames)
    results = {}
    
    for lam in lambdas:
        print(f"\n--- Testing λ = {lam} ---")
        L, S, iterations = rpca_alm(cross_view_matrix, lambda_param=lam, max_iter=75)
        
        # Calculate metrics
        sparsity = np.mean(np.abs(S) < 1e-6) * 100
        l1_norm = np.sum(np.abs(S))
        reconstruction_error = np.linalg.norm(cross_view_matrix - L - S, 'fro')
        effective_rank = np.sum(np.linalg.svd(L, compute_uv=False) > 1e-6)
        
        results[lam] = {
            'L': L, 'S': S,
            'sparsity': sparsity,
            'l1_norm': l1_norm,
            'recon_error': reconstruction_error,
            'effective_rank': effective_rank,
            'iterations': iterations
        }
        
        print(f"  Sparsity: {sparsity:.1f}%")
        print(f"  L1 norm: {l1_norm:.3f}")
        print(f"  Reconstruction error: {reconstruction_error:.2e}")
        print(f"  Effective rank: {effective_rank}")
        
        # Quick cross-view analysis
        T = frames.shape[0]
        L_reshaped = L.reshape(T * 2, frames.shape[1], frames.shape[2])
        
        # Sample consistency check for first frame
        if T > 0:
            consistency = np.corrcoef(
                L_reshaped[0, :, :].flatten(),    # View 1, frame 0
                L_reshaped[1, :, :].flatten()     # View 2, frame 0
            )[0, 1]
            print(f"  Sample cross-view correlation: {consistency:.3f}")
    
    return results

def main():
    dataset_path = '/Users/guysinger/Desktop/multiverse/dataset_multiplayer_racing_1.hdf5'
    
    print("=== Cross-View RPCA Analysis ===\n")
    
    # Load dual-view frames
    print("Loading dual-view frames...")
    frames = load_dual_view_frames(dataset_path, start_frame=0, num_frames=25)
    print(f"Loaded {len(frames)} frames with shape {frames.shape}")
    print(f"Each frame has 2 views: {frames.shape[3]} channels")
    
    # Test different lambda values
    lambdas_to_test = [0.01, 0.1, 0.3, 0.5, 1.0]
    results = test_cross_view_lambdas(frames, lambdas_to_test)
    
    # Find best lambda (balance between sparsity and reconstruction)
    best_lambda = None
    best_score = float('inf')
    
    print(f"\n=== Recommendation ===")
    for lam in lambdas_to_test:
        # Score: prioritize meaningful sparsity (not too high) and low reconstruction error
        sparsity_penalty = abs(results[lam]['sparsity'] - 80)  # Target ~80% sparsity
        score = results[lam]['recon_error'] + sparsity_penalty * 0.001
        
        print(f"λ = {lam}: sparsity = {results[lam]['sparsity']:.1f}%, "
              f"recon_err = {results[lam]['recon_error']:.2e}, score = {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_lambda = lam
    
    print(f"\nBest lambda: {best_lambda}")
    
    # Detailed analysis with best lambda
    print(f"\n=== Detailed Analysis (λ = {best_lambda}) ===")
    L_best = results[best_lambda]['L']
    S_best = results[best_lambda]['S']
    
    consistency_scores, sparse_diffs = analyze_cross_view_results(frames, L_best, S_best)
    
    # Visualization
    print(f"\nGenerating visualizations...")
    visualize_cross_view_results(frames, L_best, S_best, frame_indices=[0, 12, 24])
    
    print(f"\nCross-view RPCA analysis completed!")
    print(f"Results saved to 'cross_view_rpca_results.png'")
    
    return results, consistency_scores

if __name__ == "__main__":
    main()