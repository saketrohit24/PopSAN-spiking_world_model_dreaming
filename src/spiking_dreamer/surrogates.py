#!/usr/bin/env python3
"""
Surrogate gradient functions for spiking neural networks.
"""

import torch

# ==============================================================================
# Constants (EXACT from original PopSAN)
# ==============================================================================
ENCODER_REGULAR_VTH = 0.999
SPIKE_PSEUDO_GRAD_WINDOW = 0.5
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2  # 0.5
NEURON_VDECAY = 3 / 4  # 0.75


# ==============================================================================
# Surrogate Gradient Functions
# ==============================================================================

def super_spike_fn(membrane, threshold=0.5, beta=10.0):
    """
    SuperSpike surrogate gradient for spiking threshold.
    
    Args:
        membrane: Membrane potential tensor
        threshold: Spike threshold
        beta: Sharpness parameter for surrogate gradient
        
    Returns:
        Spike output with surrogate gradient for backprop
    """
    # Heaviside step
    spike = (membrane >= threshold).float()
    
    # Smooth surrogate for gradients
    shifted = membrane - threshold
    surrogate = shifted / (1.0 + beta * shifted.abs())
    
    # Straight-through estimator:
    # Forward: spike (hard)
    # Backward: surrogate (soft)
    return spike.detach() - surrogate.detach() + surrogate


class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """Pseudo-spike for population encoder with regular threshold."""
    
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class PseudoSpikeRect(torch.autograd.Function):
    """Rectangular pseudo-gradient - EXACT from original PopSAN."""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_output * spike_pseudo_grad.float()
