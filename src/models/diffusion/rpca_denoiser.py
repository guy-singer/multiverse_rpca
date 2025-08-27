from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .denoiser import Denoiser, DenoiserConfig, Conditioners
from .inner_model import InnerModel, InnerModelConfig
from ..blocks import Conv3x3, GroupNorm
from utils import LossAndLogs, LossLogsData
from ...rpca.decomposition import rpca_loss, DualBranchRPCA


@dataclass
class RPCADenoiserConfig:
    """Configuration for RPCA-enhanced denoiser."""
    base_denoiser: DenoiserConfig
    
    # RPCA-specific settings
    enable_rpca: bool = True
    sparse_head_channels: int = 64
    fusion_method: str = "concat"  # "concat", "add", "attention"
    
    # Loss weights
    lambda_lowrank: float = 1.0
    lambda_sparse: float = 1.0
    lambda_consistency: float = 0.1
    beta_nuclear: float = 0.01
    
    # Dual-branch settings
    shared_backbone_layers: int = 3  # Number of layers to share
    sparse_head_depth: int = 2       # Depth of sparse processing head


@dataclass
class SparseHeadConfig:
    """Configuration for sparse component processing head."""
    input_channels: int
    output_channels: int
    depth: int = 2
    use_attention: bool = False


class SparseProcessingHead(nn.Module):
    """
    Lightweight head for processing sparse component.
    Designed to capture rare events and outliers.
    """
    
    def __init__(self, config: SparseHeadConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_channels = config.input_channels
        
        # Build sparse processing layers
        for i in range(config.depth):
            out_channels = config.output_channels if i == config.depth - 1 else config.output_channels // 2
            
            layers.extend([
                Conv3x3(in_channels, out_channels),
                GroupNorm(out_channels),
                nn.SiLU()
            ])
            
            in_channels = out_channels
            
        self.layers = nn.Sequential(*layers)
        
        # Optional attention mechanism for sparse events
        if config.use_attention:
            self.attention = nn.Sequential(
                Conv3x3(config.output_channels, config.output_channels // 4),
                nn.SiLU(),
                Conv3x3(config.output_channels // 4, 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None
            
    def forward(self, sparse_input: Tensor) -> Tensor:
        """
        Process sparse component.
        
        Args:
            sparse_input: Sparse component tensor (B, C, H, W)
            
        Returns:
            Processed sparse features
        """
        x = self.layers(sparse_input)
        
        if self.attention is not None:
            attention_weights = self.attention(x)
            x = x * attention_weights
            
        return x


class FusionModule(nn.Module):
    """
    Module for fusing low-rank and sparse component features.
    """
    
    def __init__(self, lowrank_channels: int, sparse_channels: int, 
                 output_channels: int, method: str = "concat"):
        super().__init__()
        self.method = method
        
        if method == "concat":
            self.fusion_conv = Conv3x3(lowrank_channels + sparse_channels, output_channels)
        elif method == "add":
            # Ensure channel compatibility
            if lowrank_channels != sparse_channels:
                self.channel_align = Conv3x3(sparse_channels, lowrank_channels)
            else:
                self.channel_align = nn.Identity()
            self.output_conv = Conv3x3(lowrank_channels, output_channels)
        elif method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=lowrank_channels, num_heads=8, batch_first=True
            )
            self.fusion_conv = Conv3x3(lowrank_channels + sparse_channels, output_channels)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
            
    def forward(self, lowrank_features: Tensor, sparse_features: Tensor) -> Tensor:
        """
        Fuse low-rank and sparse features.
        
        Args:
            lowrank_features: Features from shared backbone (B, C1, H, W)
            sparse_features: Features from sparse head (B, C2, H, W)
            
        Returns:
            Fused features (B, output_channels, H, W)
        """
        if self.method == "concat":
            # Ensure spatial compatibility
            if lowrank_features.shape[-2:] != sparse_features.shape[-2:]:
                sparse_features = F.interpolate(
                    sparse_features, size=lowrank_features.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            fused = torch.cat([lowrank_features, sparse_features], dim=1)
            return self.fusion_conv(fused)
            
        elif self.method == "add":
            aligned_sparse = self.channel_align(sparse_features)
            if aligned_sparse.shape[-2:] != lowrank_features.shape[-2:]:
                aligned_sparse = F.interpolate(
                    aligned_sparse, size=lowrank_features.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            added = lowrank_features + aligned_sparse
            return self.output_conv(added)
            
        elif self.method == "attention":
            # Flatten spatial dimensions for attention
            B, C1, H, W = lowrank_features.shape
            lr_flat = lowrank_features.view(B, C1, H*W).transpose(1, 2)  # (B, H*W, C1)
            
            B, C2, H2, W2 = sparse_features.shape
            if (H2, W2) != (H, W):
                sparse_features = F.interpolate(sparse_features, size=(H, W), mode='bilinear', align_corners=False)
            sp_flat = sparse_features.view(B, C2, H*W).transpose(1, 2)  # (B, H*W, C2)
            
            # Apply attention (sparse attends to low-rank)  
            attended, _ = self.attention(sp_flat, lr_flat, lr_flat)
            attended = attended.transpose(1, 2).view(B, C2, H, W)
            
            # Concatenate and fuse
            fused = torch.cat([lowrank_features, attended], dim=1)
            return self.fusion_conv(fused)


class RPCADenoiser(Denoiser):
    """
    RPCA-enhanced denoiser with dual-branch architecture.
    Processes low-rank and sparse components separately before fusion.
    """
    
    def __init__(self, cfg: RPCADenoiserConfig) -> None:
        # Initialize base denoiser
        super().__init__(cfg.base_denoiser)
        
        self.rpca_cfg = cfg
        
        if cfg.enable_rpca:
            # Sparse processing head
            sparse_head_cfg = SparseHeadConfig(
                input_channels=cfg.base_denoiser.inner_model.img_channels,
                output_channels=cfg.sparse_head_channels,
                depth=cfg.sparse_head_depth
            )
            self.sparse_head = SparseProcessingHead(sparse_head_cfg)
            
            # Fusion module 
            # Assume inner model output channels match img_channels
            backbone_channels = cfg.base_denoiser.inner_model.img_channels
            self.fusion_module = FusionModule(
                lowrank_channels=backbone_channels,
                sparse_channels=cfg.sparse_head_channels,
                output_channels=backbone_channels,
                method=cfg.fusion_method
            )
        else:
            self.sparse_head = None
            self.fusion_module = None
            
    def forward(self, batch: Batch) -> LossLogsData:
        """
        Enhanced forward pass with RPCA decomposition.
        
        Args:
            batch: Input batch, may contain pre-decomposed RPCA components
            
        Returns:
            Loss, metrics, and batch data
        """
        if not self.rpca_cfg.enable_rpca:
            # Fall back to standard denoiser
            return super().forward(batch)
            
        # Check if batch contains RPCA decomposition
        has_rpca_decomposition = (
            hasattr(batch, 'info') and 
            len(batch.info) > 0 and 
            'rpca_lowrank' in batch.info[0]
        )
        
        if has_rpca_decomposition:
            return self._forward_with_precomputed_rpca(batch)
        else:
            return self._forward_with_online_rpca(batch)
            
    def _forward_with_precomputed_rpca(self, batch: Batch) -> LossLogsData:
        """Forward pass using pre-computed RPCA components."""
        b, t, c, h, w = batch.obs.size()
        H, W = (self.cfg.upsampling_factor * h, self.cfg.upsampling_factor * w) if self.is_upsampler else (h, w)
        n = 0 if self.is_upsampler else self.context_indicies[-1] + 1
        seq_length = t - n
        
        # Extract RPCA components from batch info
        L_frames = torch.stack([info['rpca_lowrank'] for info in batch.info]).to(self.device)
        S_frames = torch.stack([info['rpca_sparse'] for info in batch.info]).to(self.device)
        
        if self.is_upsampler:
            all_obs_L = L_frames
            all_obs_S = S_frames
            low_res = F.interpolate(batch.obs.reshape(b * t, c, h, w), scale_factor=self.cfg.upsampling_factor,
                                    mode="bicubic").reshape(b, t, c, H, W)
            all_acts = None
        else:
            all_obs_L = L_frames
            all_obs_S = S_frames
            all_acts = batch.act.clone()
            
        total_loss = 0
        rpca_metrics = {}
        
        for i in range(seq_length):
            prev_obs_L, prev_act = self.get_prev_obs(all_obs_L, all_acts, i, b, self.cfg.inner_model.num_steps_conditioning, c, H, W)
            prev_obs_S, _ = self.get_prev_obs(all_obs_S, all_acts, i, b, self.cfg.inner_model.num_steps_conditioning, c, H, W)
            
            obs_L = all_obs_L[:, n + i]
            obs_S = all_obs_S[:, n + i] 
            mask = batch.mask_padding[:, n + i]
            
            # Apply noise to components separately if enabled
            if self.cfg.noise_previous_obs:
                sigma_cond = self.sample_sigma_training(b, self.device)
                prev_obs_L = self.apply_noise(prev_obs_L, sigma_cond, self.cfg.sigma_offset_noise)
                prev_obs_S = self.apply_noise(prev_obs_S, sigma_cond, self.cfg.sigma_offset_noise)
            else:
                sigma_cond = None
                
            if self.is_upsampler:
                prev_obs_L = torch.cat((prev_obs_L, low_res[:, n + i]), dim=1)
                # Don't add low_res to sparse component
                
            sigma = self.sample_sigma_training(b, self.device)
            
            # Apply noise to target components
            noisy_obs_L = self.apply_noise(obs_L, sigma, self.cfg.sigma_offset_noise)
            noisy_obs_S = self.apply_noise(obs_S, sigma, self.cfg.sigma_offset_noise)
            
            # Process through dual branches
            cs = self.compute_conditioners(sigma, sigma_cond)
            
            # Low-rank branch (use existing inner model)
            model_output_L = self.compute_model_output(noisy_obs_L, prev_obs_L, prev_act, cs)
            
            # Sparse branch
            model_output_S = self.sparse_head(noisy_obs_S)
            
            # Fusion
            if self.fusion_module is not None:
                fused_output = self.fusion_module(model_output_L, model_output_S)
            else:
                fused_output = model_output_L + model_output_S
                
            # Compute RPCA loss
            target_L = (obs_L - cs.c_skip * noisy_obs_L) / cs.c_out
            target_S = (obs_S - cs.c_skip * noisy_obs_S) / cs.c_out
            target_combined = target_L + target_S
            
            rpca_loss_dict = rpca_loss(
                pred_L=model_output_L[mask],
                pred_S=model_output_S[mask], 
                target_L=target_L[mask],
                target_S=target_S[mask],
                lambda_lowrank=self.rpca_cfg.lambda_lowrank,
                lambda_sparse=self.rpca_cfg.lambda_sparse,
                lambda_consistency=self.rpca_cfg.lambda_consistency,
                beta_nuclear=self.rpca_cfg.beta_nuclear
            )
            
            total_loss += rpca_loss_dict['total_loss']
            
            # Accumulate metrics
            for key, value in rpca_loss_dict.items():
                if key != 'total_loss':
                    rpca_metrics[key] = rpca_metrics.get(key, 0) + value.item()
                    
            # Wrap model outputs and update observations
            denoised_L = self.wrap_model_output(noisy_obs_L, model_output_L, cs)
            denoised_S = self.wrap_model_output(noisy_obs_S, model_output_S, cs)
            
            all_obs_L[:, n + i] = denoised_L
            all_obs_S[:, n + i] = denoised_S
            
        # Average metrics over sequence
        for key in rpca_metrics:
            rpca_metrics[key] /= seq_length
            
        # Combine low-rank and sparse for output
        combined_obs = all_obs_L + all_obs_S
        
        metrics = {
            "loss_denoising": total_loss.item() / seq_length,
            **{f"rpca_{k}": v for k, v in rpca_metrics.items()}
        }
        
        batch_data = {
            "obs": combined_obs[:, -seq_length:], 
            'act': batch.act[:, -seq_length:], 
            'mask_padding': batch.mask_padding[:, -seq_length:],
            'obs_lowrank': all_obs_L[:, -seq_length:],
            'obs_sparse': all_obs_S[:, -seq_length:]
        }
        
        return total_loss, metrics, batch_data
        
    def _forward_with_online_rpca(self, batch: Batch) -> LossLogsData:
        """Forward pass with online RPCA decomposition."""
        # For online RPCA, we'd need to decompose batch.obs on the fly
        # This is computationally expensive, so we'll fall back to standard denoiser
        # In production, pre-processing is recommended
        
        print("Warning: Online RPCA not implemented, falling back to standard denoiser")
        return super().forward(batch)