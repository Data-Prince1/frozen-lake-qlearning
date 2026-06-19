"""
evaluate.py
===========
Policy extraction (Part D) and evaluation (Part E) for the trained agent.

* Extracts the greedy policy from a learned Q-table.
* Displays the policy in grid form using arrow symbols.
* Evaluates the agent greedily over at least 100 episodes and reports the
  success rate, average reward, number of failures and number of successes.

Usage
-----
    python evaluate.py
    python evaluate.py --tag main --episodes 100

Author : Prince Obeng Nkoah (ID: 22424742)
Course : DSCD 614 - Reinforcement Learning
"""

import argparse
import os
import sys

import numpy as np

from environment import FrozenLakeEnv, ACTION_ARROWS

# Ensure arrow symbols (←↓→↑) print correctly on Windows consoles (cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
from agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def extract_policy(env, q_table):
    """Return a list (length n_states) of the greedy action per state.

    Terminal cells (H/G) are marked with their letter instead of an action.
    """
    policy = []
    for s in range(env.n_states):
        cell = env._cell(s)
        if cell == "H":
            policy.append("H")
        elif cell == "G":
            policy.append("G")
        else:
            policy.append(int(np.argmax(q_table[s])))
    return policy


def render_policy(env, policy):
    """Return a printable grid of the policy using arrows / H / G."""
    lines = []
    for r in range(env.n_rows):
        row_syms = []
        for c in range(env.n_cols):
            s = env._to_state(r, c)
            entry = policy[s]
            if entry in ("H", "G"):
                row_syms.append(entry)
            else:
                row_syms.append(ACTION_ARROWS[entry])
        lines.append("  ".join(row_syms))
    return "\n".join(lines)


def print_policy_actions(env, policy):
    """Print the recommended action for every non-terminal state."""
    names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    print("Recommended action for every non-terminal state:")
    for s in range(env.n_states):
        entry = policy[s]
        if entry in ("H", "G"):
            continue
        r, c = env._to_coords(s)
        print(f"  state {s:>2} (row {r}, col {c}) -> {names[entry]} ({ACTION_ARROWS[entry]})")


def evaluate(env, agent, episodes=100, render_first=False):
    """Run ``episodes`` greedy episodes and return evaluation metrics."""
    successes = 0
    total_reward = 0.0
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.greedy_action(state)  # pure exploitation
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if render_first and ep == 0:
                env.render()
        rewards.append(ep_reward)
        total_reward += ep_reward
        if ep_reward > 0:
            successes += 1

    failures = episodes - successes
    return {
        "episodes": episodes,
        "successes": successes,
        "failures": failures,
        "success_rate": 100.0 * successes / episodes,
        "average_reward": total_reward / episodes,
        "rewards": rewards,
    }


def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate a trained Q-Learning agent.")
    p.add_argument("--tag", type=str, default="main", help="results tag to load")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--slippery", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    return p


def main():
    args = build_argparser().parse_args()

    q_path = os.path.join(RESULTS_DIR, f"q_table_{args.tag}.npy")
    if not os.path.exists(q_path):
        raise FileNotFoundError(
            f"No Q-table found at {q_path}. Run 'python train.py --tag {args.tag}' first."
        )

    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=args.seed)
    agent.load(q_path)
    agent.epsilon = 0.0  # fully greedy during evaluation

    # ---- Part D: policy extraction ---- #
    policy = extract_policy(env, agent.q_table)
    print("=" * 60)
    print("PART D - LEARNED POLICY")
    print("=" * 60)
    print("Legend:  ← Left   ↓ Down   → Right   ↑ Up   H Hole   G Goal\n")
    print(render_policy(env, policy))
    print()
    print_policy_actions(env, policy)

    # ---- Part E: evaluation ---- #
    print("\n" + "=" * 60)
    print(f"PART E - EVALUATION OVER {args.episodes} EPISODES (greedy)")
    print("=" * 60)
    metrics = evaluate(env, agent, episodes=args.episodes)
    print(f"  Success Rate (%)        : {metrics['success_rate']:.2f}")
    print(f"  Average Reward          : {metrics['average_reward']:.4f}")
    print(f"  Number of Successful Runs: {metrics['successes']}")
    print(f"  Number of Failures      : {metrics['failures']}")


if __name__ == "__main__":
    main()
