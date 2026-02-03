"""
Spiking Dreamer: Model-based RL with Spiking Neural Networks.

A PyTorch implementation of TD3 with a spiking neural network world model
for sample-efficient reinforcement learning via imagination.
"""

from .surrogates import super_spike_fn, PseudoEncoderSpikeRegular, PseudoSpikeRect
from .neurons import AdaptiveLIFLayer
from .population_coding import PopulationEncoder, PopulationDecoder, PopSpikeEncoder, PopSpikeDecoder
from .multiscale import MultiScaleSNNBlock
from .world_model import ImprovedSpikingWorldModel
from .ensemble import ImprovedEnsembleSpikingWorldModel, FastEnsembleSpikingWorldModel
from .dreamer import EnhancedDreamer, create_improved_spiking_world_model
from .replay_buffer import ReplayBuffer
from .actor import SpikeMLP, SNN_Actor
from .critic import Critic
from .td3_agent import TD3_SpikingDreamer
from .eval import eval_policy

__version__ = "0.1.0"
__author__ = "Rohit"

__all__ = [
    # Surrogates
    "super_spike_fn",
    "PseudoEncoderSpikeRegular", 
    "PseudoSpikeRect",
    # Neurons
    "AdaptiveLIFLayer",
    # Population coding
    "PopulationEncoder",
    "PopulationDecoder", 
    "PopSpikeEncoder",
    "PopSpikeDecoder",
    # Network blocks
    "MultiScaleSNNBlock",
    # World models
    "ImprovedSpikingWorldModel",
    "ImprovedEnsembleSpikingWorldModel",
    "FastEnsembleSpikingWorldModel",
    # Dreamer
    "EnhancedDreamer",
    "create_improved_spiking_world_model",
    # Buffer
    "ReplayBuffer",
    # Actor/Critic
    "SpikeMLP",
    "SNN_Actor",
    "Critic",
    # Agent
    "TD3_SpikingDreamer",
    # Evaluation
    "eval_policy",
]
