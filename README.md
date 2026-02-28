# PopSAN + Spiking World Model Dreaming

**PopSAN + spiking world model dreaming (ensemble + uncertainty filtering) for MuJoCo continuous control.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


This repository implements a **spiking neural network world model** with uncertainty-guided dreaming for sample-efficient reinforcement learning. It extends [PopSAN](https://github.com/xxx/popsan) (Population-coded Spiking Actor Network) with an ensemble of spiking world models for MBPO style imagined rollouts.

**[W&B Project](https://wandb.ai/rohit-deepa/Spiking-World-Model-Thesis/overview) ||
**[Powerpoint](https://docs.google.com/presentation/d/1fAt5grWoVqZ8tmExvKVxMFQmcHUWD0Hk07oeN7MOD0M/edit?usp=sharing)


---

## Install

**Requirements:** Python 3.9+, MuJoCo 2.3+

```bash
# Clone
git clone https://github.com/saketrohit24/PopSAN-spiking_world_model_dreaming.git
cd PopSAN-spiking_world_model_dreaming

# Create environment (recommended)
conda create -n snn_mujoco python=3.10
conda activate snn_mujoco

# Install MuJoCo (if not already installed)
# See: https://github.com/google-deepmind/mujoco

# Install dependencies
pip install -r requirements.txt
```

---

## Run

### Training with Dreaming (Default)

```bash
python scripts/train.py --config configs/halfcheetah.yaml --seed 0 --wandb
```

### Baseline (No Dreaming)

```bash
python scripts/train.py --config configs/halfcheetah.yaml --seed 0 --no-dreaming --wandb
```

### Ant Environment

```bash
python scripts/train.py --config configs/ant.yaml --seed 0 --wandb
```

### Humanoid (Full Observation)

```bash
python scripts/train.py --config configs/humanoid.yaml --seed 0 --wandb
```

### Humanoid Compact (45-D Observation)

`HumanoidCompactLite-v0` keeps only `qpos[2:] + qvel` (45 dims) from `Humanoid-v4` while keeping the same 17-D action space and full dynamics.

```bash
python scripts/train.py --config configs/humanoid_compact.yaml --seed 0 --wandb
```

### Evaluation

```bash
python scripts/eval_policy.py --checkpoint checkpoints/model_best.pt --env HalfCheetah-v4
```

---

## Results

| Environment | Method | Steps to 8K | Steps to 10K | Return @ 1M | Variance |
|-------------|--------|-------------|--------------|-------------|----------|
| HalfCheetah-v4 | Baseline | 480K | 920K | 9,605 | High |
| HalfCheetah-v4 | **Dreaming** | **360K** | **640K** | **10,206** | **24× lower** |


**Key findings:**
- Up to **31% fewer environment steps** to reach performance thresholds
- **6% higher mean return** at 1M steps on HalfCheetah
-  **24× lower cross-seed variance** for more reliable training

**Commands used for results:**
```bash
# HalfCheetah (5 seeds)
for seed in 0 1 2 3 4; do
  python scripts/train.py --config configs/halfcheetah.yaml --seed $seed --wandb
  python scripts/train.py --config configs/halfcheetah.yaml --seed $seed --no-dreaming --wandb
done
```

---

## Repo Structure

```
PopSAN-spiking_world_model_dreaming/
├── configs/                    # YAML hyperparameter configs
│   ├── default.yaml
│   ├── halfcheetah.yaml
│   ├── ant.yaml
│   ├── humanoid.yaml
│   └── humanoid_compact.yaml
├── src/spiking_dreamer/        # Main package
│   ├── surrogates.py           # Surrogate gradients (SuperSpike)
│   ├── neurons.py              # Adaptive LIF neurons
│   ├── population_coding.py    # Population encoder/decoder
│   ├── multiscale.py           # Multi-scale temporal blocks
│   ├── world_model.py          # Spiking dynamics model (single)
│   ├── ensemble.py             # Ensemble world models + FastEnsemble (vmap)
│   ├── dreamer.py              # MBPO/MOPO-style imagined rollouts + filtering
│   ├── replay_buffer.py        # GPU replay buffer with dream tracking
│   ├── actor.py                # PopSAN spiking actor
│   ├── critic.py               # TD3 twin critic
│   ├── td3_agent.py            # Training glue (TD3 + world model + dreaming)
│   ├── envs.py                 # Custom env wrappers + env factory
│   └── eval.py                 # Policy evaluation
├── scripts/
│   ├── train.py                # Main training script
│   └── eval_policy.py          # Standalone evaluation
├── checkpoints/                # Saved models
└── requirements.txt
```

### Module Overview

| Module | Description |
|--------|-------------|
| `world_model.py` | Spiking dynamics model with population coding, adaptive LIF neurons, multi-scale temporal integration, and hybrid spike-membrane readout |
| `ensemble.py` | Ensemble of world models for epistemic uncertainty via disagreement. `FastEnsembleSpikingWorldModel` uses `torch.vmap` for ~3× speedup |
| `dreamer.py` | MBPO/MOPO-style imagination with horizon-adaptive uncertainty filtering and pessimistic reward penalty |
| `td3_agent.py` | TD3 agent with 50/50 real/dream training split, world model training, and dream phase orchestration |


---

## License

MIT License
