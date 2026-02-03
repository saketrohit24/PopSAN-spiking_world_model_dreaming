#!/usr/bin/env python3
"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) Critic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Twin Q-network critic for TD3.
    Two independent Q-networks to reduce overestimation bias.
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1 network
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 network
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = self.l3(F.relu(self.l2(q1)))
        
        q2 = F.relu(self.l4(sa))
        q2 = self.l6(F.relu(self.l5(q2)))
        
        return q1, q2

    def Q1(self, state, action):
        """Get Q1 value only (used for actor update)."""
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        return self.l3(F.relu(self.l2(q1)))
