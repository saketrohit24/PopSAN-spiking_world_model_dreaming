#!/usr/bin/env python3
"""
Multi-scale temporal integration block for spiking neural networks.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .neurons import AdaptiveLIFLayer


class MultiScaleSNNBlock(nn.Module):
    """
    SNN block with multiple temporal scales.
    
    Different neurons operate at different timescales, capturing
    both fast and slow dynamics - crucial for world modeling!
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_scales: int = 3,
        tau_range: Tuple[float, float] = (1.5, 10.0),
    ):
        super().__init__()
        self.num_scales = num_scales
        self.features_per_scale = out_features // num_scales
        self.out_features = self.features_per_scale * num_scales  # Ensure divisible
        
        # Create layers with different timescales
        taus = torch.linspace(tau_range[0], tau_range[1], num_scales)
        
        self.layers = nn.ModuleList([
            AdaptiveLIFLayer(
                in_features=in_features,
                out_features=self.features_per_scale,
                tau_mem_init=taus[i].item(),
                use_recurrent=True,
            )
            for i in range(num_scales)
        ])
        
        # Combine scales
        self.combine = nn.Linear(self.out_features, out_features)
        self.ln = nn.LayerNorm(out_features)
    
    def forward(
        self,
        x: torch.Tensor,
        states: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        prev_spikes: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List, List, dict]:
        """
        Forward with multi-scale temporal integration.
        
        Returns:
            combined_spike: Combined spike output
            combined_membrane: Combined membrane readout  
            new_states: Updated states for each scale
            new_spikes: Spike outputs for each scale
            stats: Firing rate statistics per scale
        """
        scale_spikes = []
        scale_membranes = []
        new_states = []
        new_prev_spikes = []
        
        for i, layer in enumerate(self.layers):
            v, adapt, _ = states[i]
            prev_spike = prev_spikes[i]
            
            spike, v_new, adapt_new, membrane_out = layer(x, v, adapt, prev_spike)
            
            scale_spikes.append(spike)
            scale_membranes.append(membrane_out)
            new_states.append((v_new, adapt_new, spike))
            new_prev_spikes.append(spike)
        
        # Concatenate all scales
        combined_spike = torch.cat(scale_spikes, dim=-1)
        combined_membrane = torch.cat(scale_membranes, dim=-1)
        
        # Project to output dimension
        combined_spike = self.ln(self.combine(combined_spike))
        combined_membrane = self.ln(self.combine(combined_membrane))
        
        # Calculate firing rates per scale for logging
        scale_rates = []
        for spike in scale_spikes:
            scale_rates.append(spike.mean().detach())
            
        stats = {
            "rate_fast": scale_rates[0],
            "rate_med": scale_rates[1] if len(scale_rates) > 1 else torch.tensor(0.0),
            "rate_slow": scale_rates[-1] if len(scale_rates) > 1 else torch.tensor(0.0)
        }
        
        return combined_spike, combined_membrane, new_states, new_prev_spikes, stats
