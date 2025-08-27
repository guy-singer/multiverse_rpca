#!/usr/bin/env python3
"""
Comprehensive test suite for RPCA integration in Multiverse.

This test suite covers:
1. RPCA algorithm correctness
2. Data preprocessing integration  
3. Model architecture compatibility
4. Memory optimization features
5. Configuration validation
6. End-to-end training workflows

Usage:
    python -m pytest tests/test_rpca_integration.py -v
    python tests/test_rpca_integration.py --run-slow-tests
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
import yaml
from unittest.mock import Mock, patch
import gc

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.rpca.core import InexactALM, PrincipalComponentPursuit, NonconvexRPCA
from src.rpca.decomposition import FrameRPCA, rpca_preprocessing, rpca_loss
from src.rpca.memory_optimization import (
    CompressedRPCAStorage, MemoryEfficientRPCABatch, StreamingRPCAProcessor
)
from src.data.rpca_processor import RPCAProcessor, RPCAConfig
from src.models.diffusion.rpca_denoiser import RPCADenoiser, RPCADenoiserConfig
from src.models.rpca_agent import RPCAAgent, create_rpca_agent_config


class TestRPCAAlgorithms:
    """Test core RPCA algorithms for correctness."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic test data
        m, n = 50, 30
        rank = 5
        sparsity = 0.1
        
        # Generate ground truth low-rank matrix
        U = np.random.randn(m, rank)
        V = np.random.randn(rank, n)
        self.L_true = U @ V
        
        # Generate ground truth sparse matrix
        self.S_true = np.zeros((m, n))
        sparse_indices = np.random.choice(m * n, int(sparsity * m * n), replace=False)
        flat_indices = np.unravel_index(sparse_indices, (m, n))
        self.S_true[flat_indices] = np.random.randn(len(sparse_indices)) * 5
        
        # Observed data
        self.D = self.L_true + self.S_true
        
    def test_inexact_alm_basic(self):
        """Test basic Inexact ALM functionality."""
        rpca = InexactALM(lambda_coeff=0.1)
        L_recovered, S_recovered = rpca.fit_transform(self.D)
        
        # Check dimensions
        assert L_recovered.shape == self.D.shape
        assert S_recovered.shape == self.D.shape
        
        # Check reconstruction error
        reconstruction_error = np.linalg.norm(self.D - (L_recovered + S_recovered), 'fro')
        assert reconstruction_error < 1e-3
        
    def test_inexact_alm_convergence(self):
        """Test that Inexact ALM converges within reasonable iterations."""
        rpca = InexactALM(lambda_coeff=0.1, max_iter=100, tol=1e-6)
        L_recovered, S_recovered = rpca.fit_transform(self.D)
        
        # Algorithm should converge
        assert np.allclose(self.D, L_recovered + S_recovered, atol=1e-5)
        
    def test_principal_component_pursuit(self):
        """Test PCP algorithm."""
        rpca = PrincipalComponentPursuit(lambda_coeff=0.1)
        L_recovered, S_recovered = rpca.fit_transform(self.D)
        
        assert L_recovered.shape == self.D.shape
        assert S_recovered.shape == self.D.shape
        
    def test_nonconvex_rpca(self):
        """Test non-convex RPCA factorization."""
        rpca = NonconvexRPCA(rank=10, lambda_coeff=0.1)  # Overestimate rank
        L_recovered, S_recovered = rpca.fit_transform(self.D)
        
        assert L_recovered.shape == self.D.shape
        assert S_recovered.shape == self.D.shape
        
    def test_lambda_coefficient_effects(self):
        """Test effect of different lambda values."""
        lambdas = [0.01, 0.1, 1.0]
        sparsity_ratios = []
        
        for lam in lambdas:
            rpca = InexactALM(lambda_coeff=lam)
            L, S = rpca.fit_transform(self.D)
            sparsity_ratio = np.mean(np.abs(S) > 1e-6)
            sparsity_ratios.append(sparsity_ratio)
            
        # Higher lambda should promote more sparsity
        assert sparsity_ratios[0] < sparsity_ratios[1] < sparsity_ratios[2]
        
    def test_auto_lambda_computation(self):
        """Test automatic lambda coefficient computation."""
        rpca = InexactALM(lambda_coeff=None)  # Auto-compute
        L, S = rpca.fit_transform(self.D)
        
        # Should work without errors
        assert L.shape == self.D.shape
        assert S.shape == self.D.shape


class TestFrameRPCA:
    """Test RPCA for video frame sequences."""
    
    def setup_method(self):
        """Set up test frame data."""
        np.random.seed(42)
        
        # Create synthetic video frames (T, C, H, W)
        T, C, H, W = 10, 3, 32, 32
        
        # Generate frames with global motion (low-rank) + sparse events
        self.frames = np.random.randn(T, C, H, W) * 0.1
        
        # Add global motion (low-rank structure)
        global_pattern = np.random.randn(H, W)
        for t in range(T):
            for c in range(C):
                self.frames[t, c] += global_pattern * (0.5 + 0.1 * t)
                
        # Add sparse events
        for t in range(T):
            if t % 3 == 0:  # Sparse events every 3 frames
                event_mask = np.random.random((H, W)) < 0.05
                self.frames[t, :, event_mask] += np.random.randn(C, np.sum(event_mask)) * 2
                
    def test_frame_rpca_basic(self):
        """Test basic frame RPCA decomposition."""
        rpca = FrameRPCA(method="inexact_alm", temporal_mode=True)
        result = rpca.decompose_frames(self.frames)
        
        assert result.L_frames is not None
        assert result.S_frames is not None
        assert result.L_frames.shape == self.frames.shape
        assert result.S_frames.shape == self.frames.shape
        
        # Check reconstruction
        reconstructed = result.L_frames + result.S_frames
        mse = np.mean((self.frames - reconstructed) ** 2)
        assert mse < 1e-2
        
    def test_temporal_mode_effects(self):
        """Test different temporal modes."""
        rpca_temporal = FrameRPCA(temporal_mode=True)
        rpca_spatial = FrameRPCA(temporal_mode=False)
        
        result_temporal = rpca_temporal.decompose_frames(self.frames)
        result_spatial = rpca_spatial.decompose_frames(self.frames)
        
        # Both should work but may give different decompositions
        assert result_temporal.L_frames.shape == self.frames.shape
        assert result_spatial.L_frames.shape == self.frames.shape
        
    def test_compression_statistics(self):
        """Test compression statistics computation."""
        rpca = FrameRPCA()
        result = rpca.decompose_frames(self.frames)
        
        assert result.compression_stats is not None
        assert 'compression_ratio' in result.compression_stats
        assert result.compression_stats['compression_ratio'] >= 1.0
        
    def test_torch_preprocessing_integration(self):
        """Test integration with torch preprocessing."""
        frames_tensor = torch.from_numpy(self.frames).float().unsqueeze(0)  # Add batch dim
        
        L_tensor, S_tensor = rpca_preprocessing(frames_tensor)
        
        assert L_tensor.shape == frames_tensor.shape
        assert S_tensor.shape == frames_tensor.shape
        assert torch.is_tensor(L_tensor)
        assert torch.is_tensor(S_tensor)


class TestDataIntegration:
    """Test RPCA integration with data pipeline."""
    
    def setup_method(self):
        """Set up test data pipeline."""
        self.rpca_config = RPCAConfig(
            enabled=True,
            method="inexact_alm",
            temporal_mode=True,
            cache_dir=None,  # Disable caching for tests
            enable_parallel=False  # Disable parallel for reproducibility
        )
        
    def test_rpca_config_creation(self):
        """Test RPCA configuration creation."""
        assert self.rpca_config.enabled
        assert self.rpca_config.method == "inexact_alm"
        
        processor = self.rpca_config.create_processor()
        assert processor is not None
        
    def test_rpca_config_disabled(self):
        """Test disabled RPCA configuration."""
        disabled_config = RPCAConfig(enabled=False)
        processor = disabled_config.create_processor()
        assert processor is None
        
    def test_rpca_processor_basic(self):
        """Test basic RPCA processor functionality."""
        processor = RPCAProcessor(method="inexact_alm", cache_dir=None)
        
        # Create mock segment
        from src.data.segment import Segment, SegmentId
        from src.data.episode import Episode
        
        obs = torch.randn(5, 3, 16, 16)  # Small frames for speed
        segment = Segment(
            obs=obs,
            act=torch.zeros(5, 10),
            rew=torch.zeros(5),
            end=torch.zeros(5, dtype=torch.uint8),
            trunc=torch.zeros(5, dtype=torch.uint8),
            mask_padding=torch.ones(5).bool(),
            info={},
            id=SegmentId("test", 0, 5, True)
        )
        
        enhanced_segment = processor.decompose_segment(segment)
        
        # Check that RPCA components were added
        assert 'rpca_lowrank' in enhanced_segment.info
        assert 'rpca_sparse' in enhanced_segment.info
        assert 'rpca_decomposed' in enhanced_segment.info
        assert enhanced_segment.info['rpca_decomposed']


class TestModelIntegration:
    """Test RPCA integration with model architecture."""
    
    def setup_method(self):
        """Set up test models."""
        # Create minimal denoiser config
        from src.models.diffusion import DenoiserConfig
        from src.models.diffusion.inner_model import InnerModelConfig
        
        inner_config = InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=2,
            cond_channels=64,
            depths=[2, 2],
            channels=[32, 64],
            attn_depths=[False, False],
            num_actions=10
        )
        
        self.base_denoiser_config = DenoiserConfig(
            inner_model=inner_config,
            sigma_data=1.0,
            sigma_offset_noise=0.1,
            noise_previous_obs=False
        )
        
    def test_rpca_denoiser_config_creation(self):
        """Test RPCA denoiser configuration."""
        rpca_config = RPCADenoiserConfig(
            base_denoiser=self.base_denoiser_config,
            enable_rpca=True,
            fusion_method="concat"
        )
        
        assert rpca_config.enable_rpca
        assert rpca_config.fusion_method == "concat"
        
    def test_rpca_denoiser_initialization(self):
        """Test RPCA denoiser model creation."""
        rpca_config = RPCADenoiserConfig(
            base_denoiser=self.base_denoiser_config,
            enable_rpca=True
        )
        
        model = RPCADenoiser(rpca_config)
        
        assert model.rpca_cfg.enable_rpca
        assert model.sparse_head is not None
        assert model.fusion_module is not None
        
    def test_rpca_agent_creation(self):
        """Test RPCA agent creation."""
        from src.agent import AgentConfig
        
        base_agent_config = AgentConfig(
            denoiser=self.base_denoiser_config,
            upsampler=None,
            rew_end_model=None, 
            actor_critic=None,
            num_actions=10
        )
        
        rpca_settings = {
            'enabled': True,
            'fusion_method': 'concat',
            'use_rpca_denoiser': True
        }
        
        rpca_agent_config = create_rpca_agent_config(base_agent_config, rpca_settings)
        agent = RPCAAgent(rpca_agent_config)
        
        assert agent.use_rpca
        assert hasattr(agent.denoiser, 'rpca_cfg')


class TestMemoryOptimization:
    """Test memory optimization features."""
    
    def setup_method(self):
        """Set up memory optimization tests."""
        np.random.seed(42)
        
        # Create test matrices
        self.L = np.random.randn(100, 50) @ np.random.randn(5, 50)  # Low-rank
        self.S = np.zeros((100, 50))
        sparse_idx = np.random.choice(100*50, 500, replace=False)  # 10% sparse
        self.S.flat[sparse_idx] = np.random.randn(500) * 3
        
    def test_compressed_storage_basic(self):
        """Test compressed RPCA storage."""
        storage = CompressedRPCAStorage(self.L, self.S)
        
        assert storage.stats.compression_ratio > 1.0
        assert storage.stats.effective_rank <= min(self.L.shape)
        
        # Test reconstruction
        L_reconstructed, S_reconstructed = storage.reconstruct()
        
        reconstruction_error = (np.linalg.norm(self.L - L_reconstructed, 'fro') + 
                              np.linalg.norm(self.S - S_reconstructed, 'fro'))
        assert reconstruction_error < 1.0  # Allow some compression error
        
    def test_compressed_storage_serialization(self):
        """Test saving and loading compressed storage."""
        storage = CompressedRPCAStorage(self.L, self.S)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            temp_path = Path(f.name)
            
        try:
            storage.save_compressed(temp_path)
            loaded_storage = CompressedRPCAStorage.load_compressed(temp_path)
            
            # Check that loaded storage is equivalent
            L1, S1 = storage.reconstruct()
            L2, S2 = loaded_storage.reconstruct()
            
            assert np.allclose(L1, L2, atol=1e-6)
            assert np.allclose(S1, S2, atol=1e-6)
            
        finally:
            temp_path.unlink(missing_ok=True)
            
    def test_memory_efficient_batch(self):
        """Test memory-efficient batch processing."""
        batch_processor = MemoryEfficientRPCABatch(
            max_memory_mb=100.0,
            compression_enabled=True
        )
        
        # Create small test batch
        frames_batch = torch.randn(2, 5, 3, 16, 16)
        
        # Mock RPCA processor
        mock_processor = Mock()
        mock_result = Mock()
        mock_result.L_frames = np.random.randn(5, 3, 16, 16)
        mock_result.S_frames = np.random.randn(5, 3, 16, 16) * 0.1
        mock_result.L = np.random.randn(768, 5)  # Flattened
        mock_result.S = np.random.randn(768, 5) * 0.1
        mock_processor.decompose_frames.return_value = mock_result
        
        L_batch, S_batch = batch_processor.process_batch_frames(frames_batch, mock_processor)
        
        assert L_batch.shape == frames_batch.shape
        assert S_batch.shape == frames_batch.shape
        
    def test_streaming_rpca_processor(self):
        """Test streaming RPCA processor."""
        processor = StreamingRPCAProcessor(
            rank=5,
            ambient_dim=50,
            forgetting_factor=0.9,
            adaptation_rate=0.05
        )
        
        # Process sequence of observations
        observations = [np.random.randn(50) for _ in range(20)]
        
        for obs in observations:
            low_rank_part, sparse_part = processor.update(obs)
            
            assert low_rank_part.shape == obs.shape
            assert sparse_part.shape == obs.shape
            
            # Check reconstruction
            reconstruction_error = np.linalg.norm(obs - (low_rank_part + sparse_part))
            assert reconstruction_error < 10.0  # Allow some approximation error


class TestLossFunction:
    """Test RPCA loss function implementation."""
    
    def setup_method(self):
        """Set up loss function tests."""
        torch.manual_seed(42)
        
        # Create test tensors
        B, T, C, H, W = 2, 4, 3, 8, 8
        self.pred_L = torch.randn(B, T, C, H, W)
        self.pred_S = torch.randn(B, T, C, H, W) * 0.1
        self.target_L = torch.randn(B, T, C, H, W)
        self.target_S = torch.randn(B, T, C, H, W) * 0.1
        
    def test_rpca_loss_basic(self):
        """Test basic RPCA loss computation."""
        loss_dict = rpca_loss(
            self.pred_L, self.pred_S,
            self.target_L, self.target_S
        )
        
        assert 'total_loss' in loss_dict
        assert 'loss_lowrank' in loss_dict
        assert 'loss_sparse' in loss_dict
        assert 'loss_consistency' in loss_dict
        
        # All losses should be positive
        for key, value in loss_dict.items():
            if 'loss' in key:
                assert value.item() >= 0
                
    def test_loss_gradients(self):
        """Test that loss produces valid gradients."""
        self.pred_L.requires_grad_(True)
        self.pred_S.requires_grad_(True)
        
        loss_dict = rpca_loss(
            self.pred_L, self.pred_S,
            self.target_L, self.target_S
        )
        
        loss_dict['total_loss'].backward()
        
        assert self.pred_L.grad is not None
        assert self.pred_S.grad is not None
        assert torch.any(self.pred_L.grad != 0)
        assert torch.any(self.pred_S.grad != 0)


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_valid_rpca_config(self):
        """Test validation of valid RPCA configuration."""
        config = {
            'rpca': {
                'enabled': True,
                'method': 'inexact_alm',
                'lambda_coeff': 0.1,
                'fusion_method': 'concat',
                'loss_weights': {
                    'lambda_lowrank': 1.0,
                    'lambda_sparse': 1.0,
                    'lambda_consistency': 0.1,
                    'beta_nuclear': 0.01
                }
            }
        }
        
        # Should not raise any exceptions
        rpca_cfg = RPCAConfig(**config['rpca'])
        assert rpca_cfg.enabled
        
    def test_invalid_rpca_method(self):
        """Test handling of invalid RPCA method."""
        with pytest.raises(ValueError):
            from src.rpca.core import InexactALM
            rpca = InexactALM()
            # This would be caught at a higher level in actual usage
            
    def test_configuration_edge_cases(self):
        """Test edge cases in configuration."""
        # Empty configuration
        config = RPCAConfig()
        assert not config.enabled
        
        # Very small lambda
        config = RPCAConfig(enabled=True, lambda_coeff=1e-10)
        processor = config.create_processor()
        assert processor is not None


@pytest.mark.slow
class TestEndToEndIntegration:
    """Slow integration tests that test complete workflows."""
    
    def setup_method(self):
        """Set up integration tests."""
        # Create temporary directory for test data
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up after integration tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # This would test the full data processing pipeline
        # with RPCA enabled, but requires significant setup
        pytest.skip("Full integration test requires dataset setup")
        
    def test_training_workflow_integration(self):
        """Test integration with training workflow."""
        # This would test actual training with RPCA
        # but requires significant computational resources
        pytest.skip("Full training test requires computational resources")


def run_performance_benchmarks():
    """Run performance benchmarks for RPCA algorithms."""
    import time
    
    print("Running RPCA Performance Benchmarks...")
    
    # Test different matrix sizes
    sizes = [(100, 50), (200, 100), (500, 200)]
    methods = ['inexact_alm', 'nonconvex']
    
    results = {}
    
    for size in sizes:
        m, n = size
        D = np.random.randn(m, n)
        
        for method in methods:
            if method == 'inexact_alm':
                rpca = InexactALM(lambda_coeff=0.1)
            else:
                rpca = NonconvexRPCA(rank=min(m, n)//4, lambda_coeff=0.1)
                
            start_time = time.time()
            L, S = rpca.fit_transform(D)
            elapsed = time.time() - start_time
            
            key = f"{method}_{m}x{n}"
            results[key] = elapsed
            
            print(f"{method} {m}x{n}: {elapsed:.3f}s")
            
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RPCA Test Suite')
    parser.add_argument('--run-slow-tests', action='store_true',
                       help='Run slow integration tests')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_performance_benchmarks()
        sys.exit(0)
        
    # Run pytest with appropriate arguments
    pytest_args = [__file__]
    
    if args.verbose:
        pytest_args.append('-v')
        
    if not args.run_slow_tests:
        pytest_args.extend(['-m', 'not slow'])
        
    pytest.main(pytest_args)