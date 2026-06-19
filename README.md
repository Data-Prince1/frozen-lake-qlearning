# Frozen Lake from First Principles using Q-Learning

**Name:** Prince Obeng Nkoah
**Student ID:** 22424742
**Course:** DSCD 614 – Reinforcement Learning
**Assignment:** Programming Assignment 1

A complete, framework-free Reinforcement Learning solution to the 8×8 Frozen
Lake problem. The environment, the Q-Learning agent, the training loop and the
evaluation process are all implemented from scratch in pure Python — **no
Gymnasium, OpenAI Gym, Stable Baselines, RLlib or any other RL framework is
used.** The only third-party libraries are NumPy (array maths) and Matplotlib
(plotting the bonus graphs).

---

## Introduction

### What is Reinforcement Learning?
Reinforcement Learning (RL) is a branch of machine learning in which an **agent**
learns to make decisions by interacting with an **environment**. At each step the
agent observes a **state**, chooses an **action**, and receives a **reward** plus
the next state. The agent's goal is to learn a **policy** (a mapping from states
to actions) that maximises the expected cumulative reward over time. Unlike
supervised learning, the agent is never told the correct action — it must
discover good behaviour through trial and error, balancing **exploration**
(trying new actions) against **exploitation** (using what it already knows).

### What is Frozen Lake?
Frozen Lake is a classic grid-world problem. An agent must cross a frozen lake
from a **Start** cell (S) to a **Goal** cell (G) without falling into any
**Holes** (H); the remaining **Frozen** cells (F) are safe to walk on. Reaching
the goal yields a reward of `+1`; every other transition yields `0`. The map
used in this assignment is the standard 8×8 layout:

```
SFFFFFFF
FFFFFFFF
FFFHFFFF
FFFHFFFF
FFFHFFFF
FHHFFFHF
FHFFHFHF
FFFHFFFG
```

---

## Environment Design

### State representation
States are encoded as **single integer indices** `0 … 63`. An integer `s` maps to
grid coordinates by `row = s // 8`, `col = s % 8`. The start state is `0`
(top-left) and the goal state is `63` (bottom-right).

### Action representation
Four discrete actions, exactly as required by the assignment:

| Action | Meaning |
|:------:|:--------|
| `0`    | Left ←  |
| `1`    | Down ↓  |
| `2`    | Right → |
| `3`    | Up ↑    |

Movement is clipped at the grid boundaries — attempting to walk off the edge
leaves the agent in place.

### Reward structure
| Event                         | Reward | Terminates? |
|-------------------------------|:------:|:-----------:|
| Step onto the Goal (G)        | `+1`   | Yes         |
| Step into a Hole (H)          | `0`    | Yes         |
| Step onto a Frozen cell (F/S) | `0`    | No          |

A per-episode step budget (`max_steps`, default 200) guarantees termination.

### `FrozenLakeEnv` API (Part A)
`reset()`, `step(action)`, `render()`, `get_state()`, `is_terminal()`. The
environment maintains the current state, enforces movement boundaries, detects
holes and the goal, returns rewards and signals episode termination.

---

## Q-Learning Algorithm

### Description
Q-Learning is a model-free, off-policy, value-based RL algorithm. It learns a
**Q-table** `Q(s, a)` estimating the expected long-term return of taking action
`a` in state `s` and behaving greedily thereafter. The table is initialised to
zeros and refined from experience.

### The update equation
Implemented exactly as specified:

```
Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ' Q(s',a') − Q(s,a) ]
```

* `α` (learning rate) — how strongly new information overrides old estimates.
* `γ` (discount factor) — how much future rewards are valued relative to immediate ones.
* `r + γ · maxₐ' Q(s',a')` — the **TD target** (the bootstrapped estimate of return).
* The bracketed quantity is the **TD error**. When `s'` is terminal the
  bootstrap term is dropped (the future value is 0).

### Exploration strategy
Actions are selected with an **ε-greedy** policy: with probability `ε` a random
action is taken (exploration), otherwise the greedy action `argmaxₐ Q(s,a)` is
taken (exploitation). Greedy ties are broken randomly to avoid bias toward
action 0. `ε` starts high (1.0) and **decays** multiplicatively toward a floor
(`ε_min = 0.01`) so the agent explores early and exploits later.

---

## Training Procedure

### Hyperparameters used (main deterministic run)
| Hyperparameter        | Value   |
|-----------------------|---------|
| Episodes              | 15,000  |
| Learning rate α       | 0.1     |
| Discount factor γ     | 0.99    |
| Initial ε             | 1.0     |
| Minimum ε             | 0.01    |
| ε decay (per episode) | 0.9995  |
| Max steps per episode | 200     |
| Random seed           | 42      |

Statistics recorded each episode: episode reward, success flag, number of
successful episodes, and the ε value over time. Different learning rates,
discount factors and exploration rates can be supplied via command-line flags
(see below).

---

## Results

### Final success rate
* **Greedy evaluation (deterministic, 100 episodes): 100% success, average
  reward 1.0000, 0 failures.**
* Training success rate over the last 1,000 episodes reached ≈ 99.6%.
* Slippery / stochastic mode (Bonus A, 30,000 episodes): **88% greedy
  evaluation success**, which is strong given the noisy transitions.

### Learned policy
The extracted greedy policy (deterministic run) routes the agent down the first
column-block and then down the safe right-hand corridor to the goal, avoiding
the central column of holes:

```
↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
→  →  →  →  →  ↓  ↓  ↓
↑  ↑  ↑  H  →  →  →  ↓
↑  ↑  ↑  H  ↑  →  →  ↓
↑  ↑  ↑  H  →  →  →  ↓
↑  H  H  →  ↑  ↑  H  ↓
←  H  ←  ←  H  ↑  H  ↓
←  ←  ←  H  ←  ←  ←  G
```
Legend: `←` Left `↓` Down `→` Right `↑` Up `H` Hole `G` Goal.

### Discussion of performance
In the deterministic environment Q-Learning converges to an optimal,
hole-avoiding policy that reaches the goal every time. The moving-average reward
and success-rate curves (in `results/`) climb steadily as ε decays, illustrating
the classic exploration-to-exploitation transition. Under stochastic
transitions the same algorithm still learns a robust policy but cannot guarantee
success, because slips occasionally push the agent into holes regardless of the
chosen action.

See `results/` for the generated graphs:
`training_performance_main.png`, `training_performance_slippery.png`,
`strategy_comparison.png`.

---

## Bonus Tasks (all three implemented)

* **Option A – Stochastic transitions.** `FrozenLakeEnv(is_slippery=True)` makes
  the chosen action slip to a perpendicular direction with probability `2/3`
  total. Run: `python train.py --slippery --tag slippery --episodes 30000`.
* **Option B – Training graphs.** `train.py` saves reward, success-rate and ε
  plots to `results/` via Matplotlib.
* **Option C – Strategy comparison.** `compare_strategies.py` trains pure
  ε-greedy vs decaying ε-greedy under identical conditions and saves a
  comparison plot. Both reach 100% greedy evaluation here; decaying ε explores
  more aggressively early on.

---

## Project Structure
```
frozen-lake-qlearning/
├── environment.py        # Part A: FrozenLakeEnv (from scratch, optional slippery mode)
├── agent.py              # Part B: QLearningAgent (exact update equation)
├── train.py              # Part C: training loop + statistics + graphs (Bonus B)
├── evaluate.py           # Parts D & E: policy extraction + evaluation
├── compare_strategies.py # Bonus C: pure vs decaying ε-greedy
├── requirements.txt
├── README.md
├── report.pdf            # technical report
└── results/              # Q-tables, stats (JSON) and graphs
```

---

## Execution Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the agent (deterministic, default)**
   ```bash
   python train.py
   ```
   Saves `results/q_table_main.npy`, `results/training_stats_main.json` and
   `results/training_performance_main.png`.

3. **Extract the policy and evaluate (≥100 episodes)**
   ```bash
   python evaluate.py --tag main --episodes 100
   ```

4. **Bonus A – stochastic environment**
   ```bash
   python train.py --slippery --tag slippery --episodes 30000
   python evaluate.py --tag slippery --slippery --episodes 100
   ```

5. **Bonus C – compare exploration strategies**
   ```bash
   python compare_strategies.py
   ```

Useful training flags: `--episodes`, `--alpha`, `--gamma`, `--epsilon`,
`--epsilon-min`, `--epsilon-decay`, `--slippery`, `--seed`, `--tag`.
