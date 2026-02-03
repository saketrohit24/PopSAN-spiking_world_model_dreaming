#!/usr/bin/env python3
"""
SNN Actor with population spike encoding/decoding.
"""

import math
import torch
import torch.nn as nn

from .surrogates import PseudoSpikeRect, NEURON_CDECAY, NEURON_VDECAY
from .population_coding import PopSpikeEncoder, PopSpikeDecoder


class SpikeMLP(nn.Module):
    """Multi-layer spiking perceptron with CLIF neurons."""
    
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.out_pop_dim = out_pop_dim
        self.pseudo_spike = PseudoSpikeRect.apply
        
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        for layer in range(1, self.hidden_num):
            self.hidden_layers.append(nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer]))
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """CLIF neuron dynamics"""
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        hidden_states = [[torch.zeros(batch_size, hs, device=self.device) for _ in range(3)] 
                         for hs in self.hidden_sizes]
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device) for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        
        for step in range(self.spike_ts):
            x = in_pop_spikes[:, :, step]
            
            for i, layer in enumerate(self.hidden_layers):
                hidden_states[i][0], hidden_states[i][1], hidden_states[i][2] = self.neuron_model(
                    layer, x if i == 0 else hidden_states[i-1][2],
                    hidden_states[i][0], hidden_states[i][1], hidden_states[i][2]
                )
            
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
        
        return out_pop_act / self.spike_ts


class SNN_Actor(nn.Module):
    """Spiking Neural Network Actor with population encoding."""
    
    def __init__(self, obs_dim, act_dim, act_limit, en_pop_dim=10, de_pop_dim=10,
                 hidden_sizes=[256,256], mean_range=(-1,1), std=math.sqrt(0.05),
                 spike_ts=5, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.act_limit = act_limit
        self.encoder = PopSpikeEncoder(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)

    def forward(self, obs):
        batch_size = obs.size()[0]
        in_pop_spikes = self.encoder(torch.tanh(obs), batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        return self.act_limit * self.decoder(out_pop_activity)
