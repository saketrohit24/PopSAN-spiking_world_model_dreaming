#!/usr/bin/env python3
"""
Policy evaluation utilities.
"""

import numpy as np
import gymnasium as gym


def eval_policy(policy, env_name, seed, episodes=10):
    """
    Evaluate a policy over multiple episodes.
    
    Args:
        policy: Policy with select_action method
        env_name: Gymnasium environment name
        seed: Random seed
        episodes: Number of evaluation episodes
        
    Returns:
        Tuple of (mean_reward, std_reward)
    """
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)
    
    avg_reward = 0.0
    episode_rewards = []
    
    for ep in range(episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        
        while not (done or truncated):
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        avg_reward += episode_reward
    
    eval_env.close()
    
    avg_reward /= episodes
    std_reward = np.std(episode_rewards)
    
    return avg_reward, std_reward
