# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train both DQN and PPO agents (saves to models/)
python train.py

# Evaluate and compare Baseline vs DQN vs PPO (requires trained models)
python evaluate.py

# Launch Streamlit interactive dashboard
streamlit run dashboard.py

# Launch Flask web app (http://localhost:5000)
python web_app.py
```

`requirements.txt` pins exact versions: `gymnasium==0.28.1`, `stable-baselines3==1.8.0`, `numpy==1.21.6`, `matplotlib==3.5.3`, `flask==2.2.5`. Note: `dashboard.py` imports `streamlit` but it is not listed in `requirements.txt` — install it separately if needed.

## Architecture

The project simulates a 500×500m 5G cell with 3 base stations and trains RL agents to optimize handover decisions.

**Core simulation layer:**
- `base_station.py` — `BaseStation`: SINR computation via simplified path-loss model, load tracking, user connect/disconnect
- `users.py` — `User` base class + `Pedestrian` (5 km/h), `Vehicle` (60 km/h), `EmergencyVehicle` (120 km/h); random-walk movement with boundary reflection
- `environment.py` — `HandoverEnv(gym.Env)`: single-user-at-a-time decision loop (actions cycle through all 15 users each time step); 8-dim observation = [SINR×3, load×3, velocity, handover_count]; action space = {0,1,2=connect to BS, 3=no-change}; reward = SINR/10 − handover penalty − ping-pong penalty − energy − emergency disconnection penalty

**Training (`train.py`):** Uses `stable-baselines3` DQN and PPO with `MlpPolicy`. Trained models are saved as `models/dqn_handover.zip` and `models/ppo_handover.zip`. Evaluation (`evaluate.py`) loads these files — they must exist before running `evaluate.py`.

**Visualization layer (two alternatives):**
- `dashboard.py` — Streamlit app with real-time plots and sidebar controls
- `web_app.py` — Flask app serving `templates/index.html`; `/step/<algorithm>` drives the simulation step-by-step; DQN/PPO models are reloaded from disk on every request (no caching)

**Key design quirk:** The environment steps one user per `env.step()` call, not one full round. `time_step` only increments after all `num_users` have been processed. This means `total_handovers` and `ping_pong_count` accumulate across the whole episode without reset between users.
