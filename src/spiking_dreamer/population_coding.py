#!/usr/bin/env python3
"""
Population coding layers for encoding/decoding continuous values.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .surrogates import PseudoEncoderSpikeRegular, ENCODER_REGULAR_VTH


# ==============================================================================
# World Model Population Coding (Gaussian tuning curves)
# ==============================================================================

class PopulationEncoder(nn.Module):
    """
    Encode continuous values using population coding.
    
    Instead of single neurons, use a population where different neurons
    respond to different value ranges (like tuning curves in neuroscience).
    
    This gives much better precision than simple rate coding!
    """
    
    def __init__(self, input_dim: int, population_size: int = 10, value_range: float = 3.0):
        super().__init__()
        self.input_dim = input_dim
        self.population_size = population_size
        self.output_dim = input_dim * population_size
        
        # Centers for population coding (evenly spaced)
        centers = torch.linspace(-value_range, value_range, population_size)
        self.register_buffer('centers', centers)
        
        # Learnable width of tuning curves
        self.log_sigma = nn.Parameter(torch.zeros(input_dim) + math.log(value_range / population_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input using population coding.
        
        Args:
            x: Input values [batch, input_dim]
            
        Returns:
            encoded: Population activity [batch, input_dim * population_size]
        """
        batch_size = x.shape[0]
        sigma = torch.exp(self.log_sigma).clamp(min=0.1, max=2.0)  # [input_dim]
        
        # Expand for broadcasting
        x_exp = x.unsqueeze(-1)  # [batch, input_dim, 1]
        centers_exp = self.centers.view(1, 1, -1)  # [1, 1, population_size]
        sigma_exp = sigma.view(1, -1, 1)  # [1, input_dim, 1]
        
        # Gaussian tuning curves
        activity = torch.exp(-0.5 * ((x_exp - centers_exp) / sigma_exp) ** 2)
        
        # Flatten: [batch, input_dim * population_size]
        return activity.view(batch_size, -1)


class PopulationDecoder(nn.Module):
    """
    Decode population activity back to continuous values.
    Uses weighted average of population centers.
    """
    
    def __init__(self, output_dim: int, population_size: int = 10, value_range: float = 3.0):
        super().__init__()
        self.output_dim = output_dim
        self.population_size = population_size
        self.input_dim = output_dim * population_size
        
        centers = torch.linspace(-value_range, value_range, population_size)
        self.register_buffer('centers', centers)
        
        # Learnable bias for fine adjustment
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        """
        Decode population activity to continuous values.
        
        Args:
            activity: Population activity [batch, output_dim * population_size]
            
        Returns:
            decoded: Continuous values [batch, output_dim]
        """
        batch_size = activity.shape[0]
        
        # Reshape to [batch, output_dim, population_size]
        activity = activity.view(batch_size, self.output_dim, self.population_size)
        
        # Normalize to get weights
        weights = F.softmax(activity, dim=-1)
        
        # Weighted average of centers
        decoded = (weights * self.centers.view(1, 1, -1)).sum(dim=-1)
        
        return decoded + self.bias


# ==============================================================================
# PopSAN Spike Encoder/Decoder (Original implementation)
# ==============================================================================

class PopSpikeEncoder(nn.Module):
    """Population spike encoder for SNN actor."""
    
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply
        
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        obs = obs.view(-1, self.obs_dim, 1)
        pop_act = torch.exp(-(1./2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        
        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
        
        return pop_spikes


class PopSpikeDecoder(nn.Module):
    """Population spike decoder for SNN actor."""
    
    def __init__(self, act_dim, pop_dim):
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = nn.Tanh()

    def forward(self, pop_act):
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        return self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
