# Constrained Multi-Agent RL for 5G RAN Slicing

A research codebase for learning **dynamic radio resource allocation (PRBs + transmit power)**
across 5G network slices using **Constrained Multi-Agent PPO (MAPPO)**, benchmarked against
fixed static-slicing baselines — all built on a custom, physically-grounded RAN simulator.

The core question this project explores: **can a learned, cooperative multi-agent policy
outperform simple fixed resource splits, while still respecting hard per-slice SLA constraints
(throughput, latency, reliability)?**

---

## Overview

Modern 5G networks slice a shared pool of radio resources across services with very
different requirements — high-throughput mobile broadband, ultra-low-latency critical
communications, and massive low-rate IoT. Deciding how much bandwidth and power to give
each slice, moment to moment, under time-varying traffic and channel conditions, is a
sequential decision-making problem well suited to reinforcement learning.

Everything lives in a single Jupyter notebook, organized into three self-contained
sections (cells):

| Component | Section | Description |
|---|---|---|
| **Environment** | `ImprovedRANSlicingEnv` | A [PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` simulating multi-slice, multi-UE RAN dynamics |
| **MAPPO Trainer** | `train_constrained_marl(...)` | Constrained Multi-Agent PPO with Lagrangian SLA-violation penalties |
| **Static Baseline** | `run_static_slicing(...)` | Fixed-allocation strategies evaluated through the identical env/metrics pipeline, for fair comparison |

Run the notebook's cells top to bottom: the environment class must be defined before
the training or baseline cells, since both depend on `ImprovedRANSlicingEnv`.

---

## Environment: `ImprovedRANSlicingEnv`

A custom multi-agent RL environment modeling 5G RAN slicing with realistic radio dynamics.

### Slices modeled

| Slice | Full Name | Priority | Key SLA |
|---|---|---|---|
| **eMBB** | Enhanced Mobile Broadband | Throughput | ≥ 50 Mbps, ≤ 500 ms latency |
| **URLLC** | Ultra-Reliable Low-Latency Communication | Latency & Reliability | ≤ 250 ms latency, ≤ 1e-4 packet loss |
| **mMTC** *(optional)* | massive Machine-Type Communication | Device density | ≥ 95% connection density, ≤ 1000 ms latency |

Each slice is controlled by its own agent (`embb_agent`, `urllc_agent`, optionally
`mmtc_agent`), which outputs a continuous 2D action `[PRB_fraction, tx_power_level]` every
timestep.

### Key modeling features

- **3GPP TR 38.901-compliant channel model** — Urban Macro (UMa) NLOS pathloss, correlated
  log-normal shadowing, and independent Rayleigh small-scale fading per UE, with optional
  random-walk UE mobility.
- **Multi-UE support per slice** — total traffic demand is fixed at the *slice* level, so
  adding more UEs increases contention per UE rather than scaling total demand — the
  physically correct behavior.
- **Fluid-queue dynamics** — per-UE queues updated each step
  (`Q[t] = clip(Q[t-1] + arrivals - service, 0, Q_max)`), with latency derived via
  Little's Law plus fixed transmission/processing delays, and packet loss from overflow.
- **Time-varying traffic** — diurnal (24-hour cyclic) load patterns for eMBB/URLLC,
  randomized per episode, plus periodic mMTC reporting bursts.
- **Blended reward design** — per-slice objective (e.g. URLLC uses a multiplicative
  "AND" of mean latency / tail latency / reliability rewards) plus a shared cooperative
  global term.
- **Separate constraint-cost signal** — log-scaled, per-slice, independent of reward —
  suited to constrained/safe RL formulations (Lagrangian methods, CMDPs).

### API

Implements the standard PettingZoo parallel API: `reset()`, `step()`,
`observation_space(agent)`, `action_space(agent)`, plus `state()` (global state for
centralized critics) and `render()` (debug printout).

---

## Training: Constrained MAPPO

`train_constrained_marl(...)` trains one MAPPO agent per slice using:

- **Centralized critics, decentralized actors** — actors see only local observations;
  critics see the full global state.
- **Dual value heads** — a standard reward critic and a separate cost critic.
- **Clipped PPO objective** extended with a cost-advantage penalty term.
- **Lagrangian dual variable `λ`**, updated each episode:

  ```
  λ ← clip(λ + lambda_lr * (avg_cost - cost_limit), 0, lambda_max)
  ```

  This automatically trades off performance against constraint satisfaction — no
  hand-tuned fixed penalty weight required.

Training progress (rewards, cost/λ, violation rates, per-slice throughput/latency/
reliability/PRB-efficiency, resource allocation) is logged every episode and can be
printed as a formatted summary or plotted as a 4×3 grid of training curves.

---

## Baseline: Static Slicing

To measure how much learning actually helps, `static_slicing_baseline.py` runs several
**fixed, non-learning** PRB/power splits through the *exact same* environment, reward,
cost, and logging pipeline as MAPPO:

| Strategy | eMBB (PRB/Power) | URLLC (PRB/Power) |
|---|---|---|
| `equal` | 50% / 50% | 50% / 50% |
| `embb_heavy` | 70% / 65% | 30% / 35% |
| `urllc_heavy` | 30% / 35% | 70% / 65% |
| `embb_priority` | 80% / 75% | 20% / 25% |
| `urllc_priority` | 20% / 25% | 80% / 75% |
| `custom` | user-defined | user-defined |

This provides an apples-to-apples comparison: same env dynamics, same metrics, same
constraint definitions — the only difference is a learned policy vs. a fixed rule.

---

## Getting Started

### Requirements

```bash
pip install numpy torch gymnasium pettingzoo matplotlib
```

### Running the notebook

Open the notebook in Jupyter, JupyterLab, VS Code, or Google Colab, and run the cells
in order:

1. **Environment cell** — defines `ImprovedRANSlicingEnv`. Run this first; the other
   two sections import/reuse this class.
2. **MAPPO training cell** — defines `ActorCritic`, `MAPPO`, `TrainingLogger`, and
   `train_constrained_marl(...)`. Calling `train_constrained_marl()` (or running the
   cell's `__main__`-style block, if present) starts training:

   ```python
   trained_agents, training_logger = train_constrained_marl(
       num_episodes=15000,
       max_steps=288,
       lambda_lr=0.002,
       lambda_max=5.0,
       cost_limit=0.15,      # allow up to 15% constraint-violation rate
       eval_freq=20,
       save_freq=5000,
       save_dir="./checkpoints",
   )
   ```

   Checkpoints and training plots are saved to `save_dir` (defaults to
   `./checkpoints`).

3. **Static baseline cell** — defines `StaticAgent`, `TrainingLogger`, and
   `run_static_slicing(...)`. Call it directly, e.g.:

   ```python
   logger = run_static_slicing(
       strategy="equal",          # or "embb_heavy", "urllc_heavy",
                                   # "embb_priority", "urllc_priority", "custom"
       num_episodes=7500,
       max_steps=100,
       cost_limit=0.15,
       eval_freq=20,
       save_plot="static_equal_results.png",
   )
   ```

   Results are saved as a PNG in the working directory.

### Using the environment on its own

```python
env = ImprovedRANSlicingEnv(
    max_steps=200,
    total_prbs=100,
    max_power_w=20.0,
    enable_mmtc=False,
    num_ues={"embb": 5, "urllc": 5},
)

obs, infos = env.reset(seed=0)
for _ in range(200):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    if all(terms.values()):
        break
```

---

## Configuration Notes

- All sections are configured via function arguments — there are no hardcoded personal
  file paths in the notebook.
- `save_dir` in the MAPPO training section defaults to a local `./checkpoints` folder.
  If running on **Google Colab** with Drive mounted, pass your own path, e.g.
  `save_dir="/content/drive/MyDrive/your_folder"`.
- Random seeds are configurable per environment instance (`seed=` argument) for
  reproducibility.

---

## Project Structure

```
.
├── ALMANS.ipynb   # Environment, MAPPO trainer, and static baseline (3 sections)
└── README.md
```



