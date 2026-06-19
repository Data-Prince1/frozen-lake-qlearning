# Frozen Lake from First Principles using Q-Learning
### Technical Report — DSCD 614 Reinforcement Learning, Assignment 1

**Name:** Prince Obeng Nkoah  **Student ID:** 22424742

---

## 1. Introduction
This report describes a complete Reinforcement Learning (RL) solution to the 8×8
Frozen Lake problem, built entirely from first principles in Python. No RL
framework (Gymnasium, OpenAI Gym, Stable Baselines, RLlib, etc.) is used; only
NumPy and Matplotlib support the maths and plotting. In RL an *agent* learns by
interacting with an *environment*: it observes a state, takes an action, and
receives a reward and the next state, aiming to learn a *policy* that maximises
cumulative reward. Frozen Lake casts this as crossing a frozen lake from a Start
cell (S) to a Goal cell (G) while avoiding Holes (H) and walking on Frozen
cells (F).

## 2. Environment Design
The environment is implemented as the `FrozenLakeEnv` class with the required API
`reset()`, `step(action)`, `render()`, `get_state()` and `is_terminal()`.

* **State representation:** single integer indices `0…63`, where `row = s // 8`
  and `col = s % 8`. Start = 0, Goal = 63.
* **Actions:** `0 = Left`, `1 = Down`, `2 = Right`, `3 = Up`. Movement is clipped
  at boundaries, so walking into a wall leaves the agent in place.
* **Reward structure:** `+1` for reaching the Goal, `0` otherwise (including
  holes). Holes and the Goal are terminal; a 200-step budget guarantees
  termination.

The class maintains the current state, enforces boundaries, detects holes and
goal, returns rewards and signals termination. It also supports an optional
`is_slippery` mode for the bonus stochastic experiment.

## 3. Q-Learning Algorithm
Q-Learning is a model-free, off-policy, value-based method. It maintains a
Q-table `Q(s,a)`, initialised to zeros, estimating the expected return of taking
action `a` in state `s`. After each transition the table is updated by the
required rule:

> Q(s,a) ← Q(s,a) + α [ r + γ · maxₐ' Q(s',a') − Q(s,a) ]

Here `α` is the learning rate, `γ` the discount factor, and
`r + γ · maxₐ' Q(s',a')` the temporal-difference (TD) target; the bracketed term
is the TD error. The bootstrap term is dropped when the next state is terminal.
**Exploration** uses an ε-greedy policy: a random action with probability `ε`,
otherwise the greedy action `argmaxₐ Q(s,a)` (ties broken randomly). `ε` starts
at 1.0 and decays multiplicatively toward 0.01, shifting the agent from
exploration to exploitation as learning progresses.

## 4. Training Methodology
The agent (`QLearningAgent`) is trained by `train.py`, which runs episodes,
applies the update rule on every transition, decays ε per episode, and records
episode rewards, success flags and ε over time. Main run hyperparameters:

| Parameter | Value | | Parameter | Value |
|---|---|---|---|---|
| Episodes | 15,000 | | Initial ε | 1.0 |
| Learning rate α | 0.1 | | Minimum ε | 0.01 |
| Discount factor γ | 0.99 | | ε decay | 0.9995 |
| Max steps/episode | 200 | | Seed | 42 |

Hyperparameters were explored via command-line flags; α = 0.1 with γ = 0.99 and a
slow ε decay gave reliable convergence. Statistics are saved to `results/` as
JSON, and the learned Q-table as `.npy`.

## 5. Experimental Results
Greedy evaluation over 100 episodes on the deterministic environment achieved
**100% success, average reward 1.0000, 0 failures, 100 successful runs**. The
training success rate over the final 1,000 episodes reached ≈ 99.6%. The
moving-average reward and success-rate curves rise steadily while ε decays
(`results/training_performance_main.png`).

**Bonus B (graphs):** reward, success-rate and ε curves are generated
automatically. **Bonus C (strategy comparison):** pure ε-greedy (ε = 0.1) and
decaying ε-greedy both reached 100% greedy evaluation; pure ε-greedy showed a
slightly higher *training* success rate (≈ 92% vs ≈ 87%) because it explores less
late in training, while decaying ε explores more early
(`results/strategy_comparison.png`). **Bonus A (stochastic):** with slippery
transitions (slip probability 2/3 total) trained for 30,000 episodes, greedy
evaluation reached **88% success** — strong given the noise.

## 6. Learned Policy
The greedy policy extracted from the Q-table (Part D) avoids the central hole
column and steers the agent down the safe right-hand corridor to the goal:

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
Legend: ← Left, ↓ Down, → Right, ↑ Up, H Hole, G Goal. States that are off the
optimal path may show arbitrary arrows since they are never visited under the
greedy policy from the start.

## 7. Challenges Encountered
* **Reward sparsity:** with reward only at the goal, early episodes rarely
  succeed; a slow ε decay and enough episodes were needed for the signal to
  propagate back through the Q-table.
* **Greedy tie-breaking:** an all-zeros initial Q-table makes every action a tie;
  breaking ties randomly (rather than always picking action 0) prevented a
  systematic bias and improved exploration.
* **Stochastic mode:** slippery transitions greatly increased variance and
  required ~2× the episodes to learn a robust policy.
* **Console encoding:** the arrow symbols required forcing UTF-8 output on
  Windows consoles.

## 8. Conclusion
A from-scratch tabular Q-Learning agent solves the deterministic 8×8 Frozen Lake
optimally (100% success) and handles the stochastic variant robustly (88%). The
implementation cleanly separates environment, agent, training and evaluation, and
includes all three bonus tasks. The results confirm the core RL principles:
value bootstrapping via the TD update and an ε-greedy schedule that transitions
from exploration to exploitation produce an optimal, hole-avoiding policy.
