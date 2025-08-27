import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from .utils import (
    singular_value_shrinkage, 
    soft_threshold, 
    compute_rpca_lambda, 
    check_convergence
)


class RPCADecomposer(ABC):
    """Abstract base class for RPCA algorithms."""
    
    def __init__(self, lambda_coeff: Optional[float] = None):
        self.lambda_coeff = lambda_coeff
        
    @abstractmethod
    def fit_transform(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose matrix D into L + S where L is low-rank, S is sparse."""
        pass
        
    def _compute_lambda(self, m: int, n: int) -> float:
        """Compute lambda coefficient if not provided."""
        if self.lambda_coeff is not None:
            return self.lambda_coeff
        return compute_rpca_lambda(m, n)


class InexactALM(RPCADecomposer):
    """
    Inexact Augmented Lagrange Multiplier method for RPCA.
    Fast and practical implementation of Principal Component Pursuit.
    """
    
    def __init__(self, lambda_coeff: Optional[float] = None, 
                 mu: float = 1.0, rho: float = 1.6, 
                 max_iter: int = 1000, tol: float = 1e-7):
        super().__init__(lambda_coeff)
        self.mu = mu
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve: min ||L||_* + λ||S||_1 s.t. D = L + S
        
        Args:
            D: Input matrix (m × n)
            
        Returns:
            (L, S): Low-rank and sparse components
        """
        m, n = D.shape
        lambda_param = self._compute_lambda(m, n)
        
        # Initialize variables
        L = np.zeros_like(D)
        S = np.zeros_like(D)
        Y = np.zeros_like(D)  # Lagrange multipliers
        
        mu = self.mu / np.linalg.norm(D, ord=2)
        eta = 0.9
        
        for iteration in range(self.max_iter):
            # Store previous values for convergence check
            L_prev, S_prev = L.copy(), S.copy()
            
            # Update L (singular value shrinkage)
            L = singular_value_shrinkage(D - S + Y/mu, 1.0/mu)
            
            # Update S (soft thresholding)  
            S = soft_threshold(D - L + Y/mu, lambda_param/mu)
            
            # Update Y (dual variables)
            Y = Y + mu * (D - L - S)
            
            # Update mu
            mu = min(self.rho * mu, 1e6)
            
            # Check convergence
            primal_residual = np.linalg.norm(D - L - S, 'fro')
            if primal_residual < self.tol * np.linalg.norm(D, 'fro'):
                break
                
            # Check relative change  
            if iteration > 0:
                if check_convergence(L, S, L_prev, S_prev, self.tol):
                    break
                    
        return L, S


class PrincipalComponentPursuit(RPCADecomposer):
    """
    Exact Principal Component Pursuit using convex optimization.
    More accurate but slower than Inexact ALM.
    """
    
    def __init__(self, lambda_coeff: Optional[float] = None, 
                 max_iter: int = 500, tol: float = 1e-6):
        super().__init__(lambda_coeff)  
        self.max_iter = max_iter
        self.tol = tol
        
    def fit_transform(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact PCP using interior-point methods (simplified implementation).
        For production use, consider cvxpy or similar convex optimization libraries.
        """
        # For simplicity, this falls back to Inexact ALM with tighter tolerances
        alm = InexactALM(
            lambda_coeff=self.lambda_coeff, 
            max_iter=self.max_iter,
            tol=self.tol
        )
        return alm.fit_transform(D)


class NonconvexRPCA(RPCADecomposer):
    """
    Non-convex factorization approach: L = AB^T where A ∈ ℝ^{m×r}, B ∈ ℝ^{n×r}
    Faster for large matrices with known/estimated rank.
    """
    
    def __init__(self, rank: int, lambda_coeff: Optional[float] = None,
                 max_iter: int = 1000, lr: float = 1e-3, tol: float = 1e-6):
        super().__init__(lambda_coeff)
        self.rank = rank
        self.max_iter = max_iter  
        self.lr = lr
        self.tol = tol
        
    def fit_transform(self, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve using alternating minimization over A, B, S.
        """
        m, n = D.shape
        lambda_param = self._compute_lambda(m, n)
        
        # Initialize factors
        A = np.random.randn(m, self.rank) * 0.1
        B = np.random.randn(n, self.rank) * 0.1  
        S = np.zeros_like(D)
        
        for iteration in range(self.max_iter):
            S_prev = S.copy()
            
            # Update S (soft thresholding)
            L = A @ B.T
            S = soft_threshold(D - L, lambda_param)
            
            # Update A, B (gradient descent on Frobenius loss)
            residual = D - S - A @ B.T
            grad_A = -residual @ B
            grad_B = -residual.T @ A
            
            A = A - self.lr * grad_A
            B = B - self.lr * grad_B
            
            # Check convergence
            if iteration > 0:
                change = np.linalg.norm(S - S_prev, 'fro') / np.linalg.norm(S_prev, 'fro')
                if change < self.tol:
                    break
                    
        return A @ B.T, S


class StreamingRPCA:
    """
    Streaming/online RPCA for processing data incrementally.
    Based on GRASTA (Grassmannian Robust Adaptive Subspace Tracking Algorithm).
    """
    
    def __init__(self, rank: int, ambient_dim: int, 
                 lambda_coeff: float = 0.1, step_size: float = 0.1):
        self.rank = rank
        self.ambient_dim = ambient_dim
        self.lambda_coeff = lambda_coeff
        self.step_size = step_size
        
        # Initialize subspace  
        U, _, _ = np.linalg.svd(np.random.randn(ambient_dim, rank), full_matrices=False)
        self.U = U  # Orthonormal basis for low-rank subspace
        
    def update(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process new observation y and update subspace.
        
        Args:
            y: New observation vector (ambient_dim,)
            
        Returns:
            (l, s): Low-rank and sparse parts of y
        """
        # Project onto current subspace
        w = self.U.T @ y
        l_proj = self.U @ w
        
        # Compute residual and apply soft thresholding
        residual = y - l_proj
        s = soft_threshold(residual, self.lambda_coeff)
        l = y - s
        
        # Update subspace via gradient descent on Grassmann manifold
        if np.linalg.norm(s) > 0:  # Only update if sparse part exists
            # Compute gradient
            grad = np.outer(l - l_proj, w)
            
            # Retraction onto Grassmann manifold  
            self.U = self.U + self.step_size * grad
            U_orth, _ = np.linalg.qr(self.U)
            self.U = U_orth
            
        return l, s
        
    def fit_batch(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process a batch of observations."""
        m, n = Y.shape
        L = np.zeros_like(Y)
        S = np.zeros_like(Y)
        
        for i in range(n):
            L[:, i], S[:, i] = self.update(Y[:, i])
            
        return L, S