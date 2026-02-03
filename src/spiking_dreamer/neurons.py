#!/usr/bin/env python3
"""
Adaptive Leaky Integrate-and-Fire (LIF) neuron implementations.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

from .surrogates import super_spike_fn


class AdaptiveLIFLayer(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire layer with:
    - Learnable time constants (per-neuron)
    - Learnable thresholds
    - Adaptive threshold (increases after spike)
    - Both spike output AND membrane potential output
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_mem_init: float = 2.0,
        tau_adapt_init: float = 10.0,
        threshold_init: float = 0.5,
        adapt_scale: float = 0.1,
        use_recurrent: bool = True,
    ):
        super().__init__()
        self.out_features = out_features
        self.adapt_scale = adapt_scale
        self.use_recurrent = use_recurrent
        
        # Main projection
        self.fc = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        
        # Recurrent connection (helps temporal dynamics)
        if use_recurrent:
            self.fc_rec = nn.Linear(out_features, out_features, bias=False)
            nn.init.orthogonal_(self.fc_rec.weight, gain=0.5)
        
        # Learnable time constants (per-neuron)
        self.log_tau_mem = nn.Parameter(torch.ones(out_features) * math.log(tau_mem_init))
        self.log_tau_adapt = nn.Parameter(torch.ones(out_features) * math.log(tau_adapt_init))
        
        # Learnable base threshold
        self.threshold_base = nn.Parameter(torch.ones(out_features) * threshold_init)
        
        # Surrogate gradient function
        self.spike_fn = super_spike_fn
    
    def forward(
        self, 
        x: torch.Tensor, 
        v: torch.Tensor, 
        adapt: torch.Tensor,
        prev_spike: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single timestep forward.
        
        Args:
            x: Input [batch, in_features]
            v: Membrane potential [batch, out_features]
            adapt: Adaptive threshold [batch, out_features]
            prev_spike: Previous spike output [batch, out_features]
            
        Returns:
            spike: Binary spike output
            v_new: Updated membrane potential
            adapt_new: Updated adaptive threshold
            v_out: Membrane potential (for continuous readout)
        """
        # Compute decay factors from learnable time constants
        tau_mem = torch.exp(self.log_tau_mem).clamp(min=1.0, max=20.0)
        tau_adapt = torch.exp(self.log_tau_adapt).clamp(min=5.0, max=50.0)
        
        decay_mem = 1.0 - 1.0 / tau_mem
        decay_adapt = 1.0 - 1.0 / tau_adapt
        
        # Current input (apply LN after adding recurrent for stability)
        current = self.fc(x)
        if self.use_recurrent:
            current = current + self.fc_rec(prev_spike)
        current = self.ln(current)
        
        # Update membrane potential (LIF dynamics)
        v_new = decay_mem * v + (1 - decay_mem) * current
        
        # Adaptive threshold
        threshold = self.threshold_base + adapt
        
        # Spike generation
        spike = self.spike_fn(v_new, threshold)  # Use per-neuron threshold
        
        # Hard reset after spike
        v_new = v_new * (1 - spike) + 0.0 * spike  # Reset to 0
        
        # Update adaptive threshold (increases after spike, decays otherwise)
        adapt_new = decay_adapt * adapt + self.adapt_scale * spike
        
        return spike, v_new, adapt_new, v_new + current * 0.1  # v_out includes some current for better gradient
