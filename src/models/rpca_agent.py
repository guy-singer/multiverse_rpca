from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import DenoiserConfig, SigmaDistributionConfig
from models.diffusion.rpca_denoiser import RPCADenoiser, RPCADenoiserConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from agent import Agent, AgentConfig
from utils import extract_state_dict


@dataclass  
class RPCAAgentConfig:
    """Configuration for RPCA-enhanced agent."""
    base_denoiser: DenoiserConfig
    rpca_denoiser: RPCADenoiserConfig
    upsampler: Optional[DenoiserConfig] 
    rew_end_model: Optional[RewEndModelConfig]
    actor_critic: Optional[ActorCriticConfig]
    num_actions: int
    
    # RPCA-specific settings
    use_rpca_denoiser: bool = True
    
    def __post_init__(self) -> None:
        # Set num_actions for all components
        self.base_denoiser.inner_model.num_actions = self.num_actions
        self.rpca_denoiser.base_denoiser.inner_model.num_actions = self.num_actions
        
        if self.upsampler is not None:
            self.upsampler.inner_model.num_actions = self.num_actions
        if self.rew_end_model is not None:
            self.rew_end_model.num_actions = self.num_actions
        if self.actor_critic is not None:
            self.actor_critic.num_actions = self.num_actions


class RPCAAgent(Agent):
    """
    Agent enhanced with RPCA denoiser capabilities.
    Can switch between standard and RPCA denoisers.
    """
    
    def __init__(self, cfg: Union[AgentConfig, RPCAAgentConfig]) -> None:
        if isinstance(cfg, AgentConfig):
            # Standard agent initialization
            super().__init__(cfg)
            self.rpca_denoiser = None
            self.use_rpca = False
        else:
            # RPCA agent initialization
            self.use_rpca = cfg.use_rpca_denoiser
            
            # Initialize base components (skip denoiser for now)
            nn.Module.__init__(self)
            
            if self.use_rpca:
                self.denoiser = RPCADenoiser(cfg.rpca_denoiser)
            else:
                from models.diffusion import Denoiser
                self.denoiser = Denoiser(cfg.base_denoiser)
                
            self.upsampler = Denoiser(cfg.upsampler) if cfg.upsampler is not None else None
            self.rew_end_model = RewEndModel(cfg.rew_end_model) if cfg.rew_end_model is not None else None  
            self.actor_critic = ActorCritic(cfg.actor_critic) if cfg.actor_critic is not None else None
            
    def toggle_rpca(self, enable: bool = None) -> bool:
        """
        Toggle between RPCA and standard denoiser.
        
        Args:
            enable: If provided, set RPCA state. If None, toggle current state.
            
        Returns:
            New RPCA state
        """
        if not hasattr(self, 'rpca_denoiser') or self.rpca_denoiser is None:
            print("Warning: RPCA denoiser not available")
            return False
            
        if enable is None:
            enable = not self.use_rpca
            
        self.use_rpca = enable
        
        if enable and hasattr(self, 'rpca_denoiser'):
            # Switch to RPCA denoiser
            self.denoiser, self.rpca_denoiser = self.rpca_denoiser, self.denoiser
        elif not enable and hasattr(self, 'rpca_denoiser'):
            # Switch to standard denoiser  
            self.denoiser, self.rpca_denoiser = self.rpca_denoiser, self.denoiser
            
        return self.use_rpca
        
    def get_rpca_metrics(self) -> dict:
        """Get RPCA-specific metrics from the denoiser."""
        if self.use_rpca and hasattr(self.denoiser, 'rpca_cfg'):
            return {
                'rpca_enabled': True,
                'fusion_method': self.denoiser.rpca_cfg.fusion_method,
                'lambda_lowrank': self.denoiser.rpca_cfg.lambda_lowrank,
                'lambda_sparse': self.denoiser.rpca_cfg.lambda_sparse
            }
        else:
            return {'rpca_enabled': False}
            
    def save_rpca_checkpoint(self, path: Path) -> None:
        """Save checkpoint with RPCA state information."""
        checkpoint = {
            'agent_state_dict': self.state_dict(),
            'rpca_enabled': self.use_rpca,
            'rpca_metrics': self.get_rpca_metrics()
        }
        
        if hasattr(self, 'rpca_denoiser') and self.rpca_denoiser is not None:
            checkpoint['rpca_denoiser_state_dict'] = self.rpca_denoiser.state_dict()
            
        torch.save(checkpoint, path)
        
    def load_rpca_checkpoint(self, path: Path) -> None:
        """Load checkpoint with RPCA state restoration."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['agent_state_dict'])
        
        if 'rpca_enabled' in checkpoint:
            self.use_rpca = checkpoint['rpca_enabled']
            
        if 'rpca_denoiser_state_dict' in checkpoint and hasattr(self, 'rpca_denoiser'):
            self.rpca_denoiser.load_state_dict(checkpoint['rpca_denoiser_state_dict'])


def create_rpca_agent_config(base_config: AgentConfig, 
                            rpca_settings: dict) -> RPCAAgentConfig:
    """
    Create RPCA agent configuration from base agent config.
    
    Args:
        base_config: Standard agent configuration
        rpca_settings: RPCA-specific settings
        
    Returns:
        RPCA agent configuration
    """
    # Create RPCA denoiser config
    rpca_denoiser_cfg = RPCADenoiserConfig(
        base_denoiser=base_config.denoiser,
        enable_rpca=rpca_settings.get('enable_rpca', True),
        sparse_head_channels=rpca_settings.get('sparse_head_channels', 64),
        fusion_method=rpca_settings.get('fusion_method', 'concat'),
        lambda_lowrank=rpca_settings.get('lambda_lowrank', 1.0),
        lambda_sparse=rpca_settings.get('lambda_sparse', 1.0),
        lambda_consistency=rpca_settings.get('lambda_consistency', 0.1),
        beta_nuclear=rpca_settings.get('beta_nuclear', 0.01)
    )
    
    return RPCAAgentConfig(
        base_denoiser=base_config.denoiser,
        rpca_denoiser=rpca_denoiser_cfg,
        upsampler=base_config.upsampler,
        rew_end_model=base_config.rew_end_model,
        actor_critic=base_config.actor_critic,
        num_actions=base_config.num_actions,
        use_rpca_denoiser=rpca_settings.get('use_rpca_denoiser', True)
    )


class RPCAModelFactory:
    """Factory for creating RPCA-enhanced models."""
    
    @staticmethod
    def create_agent(agent_config: AgentConfig, 
                    rpca_config: Optional[dict] = None) -> Union[Agent, RPCAAgent]:
        """
        Create agent with optional RPCA enhancement.
        
        Args:
            agent_config: Base agent configuration
            rpca_config: Optional RPCA configuration dict
            
        Returns:
            Agent or RPCAAgent instance
        """
        if rpca_config is None or not rpca_config.get('enabled', False):
            return Agent(agent_config)
        else:
            rpca_agent_config = create_rpca_agent_config(agent_config, rpca_config)
            return RPCAAgent(rpca_agent_config)
            
    @staticmethod
    def create_denoiser(denoiser_config: DenoiserConfig,
                       rpca_config: Optional[dict] = None) -> Union['Denoiser', RPCADenoiser]:
        """
        Create denoiser with optional RPCA enhancement.
        
        Args:
            denoiser_config: Base denoiser configuration
            rpca_config: Optional RPCA configuration dict
            
        Returns:
            Denoiser or RPCADenoiser instance
        """
        if rpca_config is None or not rpca_config.get('enabled', False):
            from models.diffusion import Denoiser
            return Denoiser(denoiser_config)
        else:
            rpca_denoiser_cfg = RPCADenoiserConfig(
                base_denoiser=denoiser_config,
                **rpca_config
            )
            return RPCADenoiser(rpca_denoiser_cfg)