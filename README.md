# PopSAN + Spiking World Model Dreaming

**PopSAN + spiking world model dreaming (ensemble + uncertainty filtering) for MuJoCo continuous control.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


This repository implements a **spiking neural network world model** with uncertainty-guided dreaming for sample-efficient reinforcement learning. It extends [PopSAN](https://github.com/xxx/popsan) (Population-coded Spiking Actor Network) with an ensemble of spiking world models for MBPO style imagined rollouts.

**[W&B Project](https://wandb.ai/rohit-deepa/Half_Cheetah_Dreaming?nw=nwusersaketrohit24)** 

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
│   └── ant.yaml
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


## Abstract

> Model-based reinforcement learning offers sample efficiency gains through synthetic experience generation, yet conventional approaches rely on energy-intensive artificial neural networks. Spiking neural networks (SNNs) provide a promising alternative due to their event-driven computation and compatibility with neuromorphic hardware.
>
> This thesis introduces a **spiking world model framework with uncertainty-guided dreaming** for continuous control tasks in MuJoCo. The proposed approach extends PopSAN by integrating an **ensemble of spiking world models** composed of adaptive current-based leaky integrate-and-fire neurons, multi-scale temporal integration, and hybrid spike–membrane readouts. **Epistemic uncertainty** is quantified through ensemble disagreement and used to selectively filter unreliable imagined transitions.
>
> On HalfCheetah-v4, uncertainty-guided dreaming **reduces environment steps by up to 31%**, improves mean return (10,206 vs. 9,605), and **reduces variance by 24×**. On Ant-v4, the method saves **200K–260K steps** and achieves up to **63% higher performance** at intermediate stages.

---

## License

MIT License
