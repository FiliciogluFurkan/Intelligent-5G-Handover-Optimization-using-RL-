# Intelligent 5G Handover Optimization using Reinforcement Learning

A simulation-based research project that applies Deep Reinforcement Learning (DQN and PPO) to optimize base station handover decisions in a 5G network environment. Built with Gymnasium, Stable-Baselines3, and a live Plotly Dash dashboard.

---

## Overview

In 5G networks, handover (the process of transferring a mobile user from one base station to another) directly impacts Quality of Service. Poor handover decisions cause signal drops, unnecessary ping-pong switching, and degraded throughput for emergency vehicles.

This project models the handover problem as a Markov Decision Process and trains RL agents to learn optimal handover policies from scratch — outperforming a greedy baseline on ping-pong elimination.

---

## Key Results

| Metric | Baseline (Greedy) | DQN | PPO |
|--------|:-----------------:|:---:|:---:|
| Avg Reward | ~30,000 | ~19,000 | ~26,000 |
| Handover Rate (per step) | 0.0 | 0.09 | 0.05 |
| Ping-Pong Rate | **18.2%** | **0%** | **0%** |
| Emergency Disconnections | varies | lower | lower |

> **Key insight:** The greedy baseline achieves higher raw reward because it never handovers (SINR gain >> handover cost). RL agents discover that *some* handovers are worth it while eliminating all ping-pong events — a critical real-world KPI.

---

## Architecture

```
500×500m simulation area
├── 3 Base Stations (fixed positions: [150,300], [300,200], [450,430])
├── 15 Users
│   ├── 7 Pedestrians   (5 km/h)
│   ├── 5 Vehicles      (60 km/h)
│   └── 3 Emergency     (120 km/h, priority protected)
└── Decision: per-user, per-step (cycles through all 15 users)
```

**Observation space** (8-dim, per user):
- SINR from each BS × 3 (clipped to [−120, 50] dB)
- Load of each BS × 3 (normalized to [0, 1])
- User velocity (normalized)
- User handover count (normalized)

**Action space:** `Discrete(4)` — connect to BS 0, 1, 2, or no change

**Reward function:**
```
R = SINR/10 − handover_penalty − ping_pong_penalty − energy_penalty − emergency_penalty
```

---

## Project Structure

```
├── environment.py        # HandoverEnv (Gymnasium) — core simulation loop
├── base_station.py       # BaseStation: SINR (3GPP path-loss), load tracking
├── users.py              # Pedestrian, Vehicle, EmergencyVehicle mobility models
├── agents.py             # Model loader (DQN / PPO / baseline policy)
├── train.py              # Training with CheckpointCallback + EvalCallback
├── evaluate.py           # 10-episode evaluation, normalized metrics, plots
├── callbacks.py          # Dash callback handlers (simulation step logic)
├── figures.py            # Plotly figure builders (network map, time-series)
├── layout.py             # Dash UI layout definition
├── app.py                # Application entry point
├── config/
│   └── settings.py       # Frozen dataclass config (SIM, BS, REWARD, TRAIN)
├── models/               # Saved model weights (created after training)
├── figures/              # Output charts (created after evaluation)
├── tests/                # pytest unit tests
└── docs/                 # Project documentation
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train agents

```bash
python train.py
```

Trains both DQN and PPO for **200,000 timesteps** each. Saves to `models/`:
- `dqn_handover.zip` / `ppo_handover.zip` — final models
- `best_dqn/` / `best_ppo/` — best checkpoint (by eval reward)
- `checkpoints/` — periodic snapshots every 20,000 steps
- `dqn_monitor.monitor.csv` / `ppo_monitor.monitor.csv` — training logs

Expected training time: ~20–25 minutes on CPU.

### 3. Evaluate and compare

```bash
python evaluate.py
```

Runs 10 episodes with fixed seeds for reproducible comparison. Prints a `mean ± std` table and saves two figures to `figures/`:
- `comparison_bar_charts.png` — 2×2 bar chart with error bars
- `training_curves.png` — episode reward vs episode with rolling mean

### 4. Launch live dashboard

```bash
python app.py
```

Open `http://127.0.0.1:8050` in your browser. Features:
- Live network map with user positions and BS coverage circles
- Real-time metrics: handovers, avg SINR, ping-pong count, emergency disconnections
- Base station load bars
- Time-series charts: handovers, SINR, energy consumption
- Algorithm selector: Baseline / DQN / PPO
- Simulation speed slider (1×–10×)

---

## Algorithms

### Baseline — Greedy SINR
Always connects each user to the BS with the highest SINR. No learning, no memory. Acts as the lower bound for intelligent decision-making quality.

### DQN — Deep Q-Network
Off-policy, experience replay buffer, target network for stability. Learns a Q-function mapping (state, action) → expected return. Suitable for discrete action spaces.

### PPO — Proximal Policy Optimization
On-policy, actor-critic architecture. Clips policy updates to prevent destructive large steps. More sample-efficient than vanilla policy gradient; generalizes well to unseen mobility patterns.

---

## SINR Model

Uses the 3GPP TR 36.839 simplified path-loss model:

```
PL(d) = 128.1 + 37.6 × log10(d / 1000)   [dB]
SINR   = P_tx − PL(d) − (N₀ + I)          [dBm]
```

Constants (from `config/settings.py`):
- Transmit power: 43 dBm
- Thermal noise floor: −130 dBm (10⁻¹³ mW)
- Interference floor: −100 dBm (10⁻¹⁰ mW)

---

## Configuration

All simulation parameters are centralized in `config/settings.py` as frozen dataclasses:

```python
SIM.area_size          = 500      # metres
SIM.num_base_stations  = 3
SIM.default_num_users  = 15
SIM.max_steps          = 1000

REWARD.handover_penalty          = 2.0
REWARD.ping_pong_penalty         = 5.0
REWARD.ping_pong_window_steps    = 10
REWARD.emergency_disconnect_penalty = 10.0
REWARD.sinr_scale                = 10.0

TRAIN.total_timesteps   = 200_000
TRAIN.dqn_learning_rate = 1e-4
TRAIN.ppo_learning_rate = 3e-4
```

---

## Tests

```bash
python -m pytest tests/ -v
```

Unit tests cover `BaseStation` (SINR calculations, user management, load bounds) and the `HandoverEnv` reset/step cycle.

---

## Requirements

```
gymnasium==0.28.1
stable-baselines3==1.8.0
numpy==1.21.6
matplotlib==3.5.3
plotly>=5.0
dash>=2.0
dash-bootstrap-components>=1.0
```

Full pinned versions in `requirements.txt`.

---

## Academic Context

This project was developed as part of a university **Computer Networks** course to demonstrate:
1. How cellular handover can be framed as a sequential decision problem (MDP)
2. Why model-free RL is appropriate when the transition dynamics are unknown
3. The trade-off between reward maximization and real-world KPIs like ping-pong rate
4. How modern RL libraries (Stable-Baselines3) integrate with custom Gym environments
