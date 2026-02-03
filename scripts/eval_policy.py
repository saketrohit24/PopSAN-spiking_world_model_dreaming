#!/usr/bin/env python3
"""
Standalone policy evaluation script.

Usage:
    python eval_policy.py --checkpoint checkpoints/model_best.pt --env HalfCheetah-v4 --episodes 10
    python eval_policy.py --checkpoint model.pt --render  # Visualize
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiking_dreamer import TD3_SpikingDreamer, ReplayBuffer


def load_policy(checkpoint_path: str, env, device):
    """Load a trained policy from checkpoint."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Default config (just need enough for initialization)
    config = {
        "gamma": 0.99,
        "tau": 0.005,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "policy_freq": 2,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "en_pop_dim": 10,
        "de_pop_dim": 10,
        "hidden_sizes": [256, 256],
        "mean_range": [-1, 1],
        "std": 0.2236,
        "spike_ts": 5,
        "num_ensemble": 3,
        "wm_hidden_dim": 256,
        "wm_num_layers": 2,
        "wm_spike_steps": 8,
        "population_size": 8,
        "num_scales": 3,
        "wm_lr": 3e-4,
        "wm_weight_decay": 1e-5,
        "enable_dreaming": True,
        "dream_batch_size": 256,
        "dreams_per_phase": 4,
        "dream_horizon": 5,
        "epistemic_threshold": 0.5,
        "adaptive_threshold": True,
        "uncertainty_penalty": 1.0,
    }
    
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1000, device=device)
    policy = TD3_SpikingDreamer(state_dim, action_dim, max_action, device, config, replay_buffer)
    policy.load(checkpoint_path)
    
    return policy


def evaluate(policy, env, episodes=10, render=False):
    """Evaluate policy over multiple episodes."""
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0
        
        while not (done or truncated):
            action = policy.select_action(np.array(state))
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {episode_reward:.1f}, Length = {steps}")
    
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Spiking Dreamer policy")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--render", action="store_true",
                        help="Render environment")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Spiking Dreamer Policy Evaluation")
    print("=" * 50)
    
    # Setup
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)
    env.reset(seed=args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Environment: {args.env}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("-" * 50)
    
    # Load and evaluate
    policy = load_policy(args.checkpoint, env, device)
    rewards, lengths = evaluate(policy, env, args.episodes, args.render)
    
    # Summary
    print("=" * 50)
    print(f"Mean Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Max Reward:  {np.max(rewards):.1f}")
    print(f"Min Reward:  {np.min(rewards):.1f}")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()
