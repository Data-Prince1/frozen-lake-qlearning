# Policy extraction (Part D) and evaluation (Part E)
# Prince Obeng Nkoah - 22424742

import argparse
import os
import sys

import numpy as np

from environment import FrozenLakeEnv, ACTION_ARROWS
from agent import QLearningAgent

# make the arrows print properly on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# get the best action for each state from the q-table
def extract_policy(env, q_table):
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


# show the policy as a grid of arrows
def render_policy(env, policy):
    lines = []
    for r in range(env.n_rows):
        row = []
        for c in range(env.n_cols):
            s = env._to_state(r, c)
            entry = policy[s]
            if entry == "H" or entry == "G":
                row.append(entry)
            else:
                row.append(ACTION_ARROWS[entry])
        lines.append("  ".join(row))
    return "\n".join(lines)


def print_policy_actions(env, policy):
    names = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
    print("Recommended action for every non-terminal state:")
    for s in range(env.n_states):
        entry = policy[s]
        if entry == "H" or entry == "G":
            continue
        r, c = env._to_coords(s)
        print(f"  state {s} (row {r}, col {c}) -> {names[entry]} ({ACTION_ARROWS[entry]})")


# run the agent greedily and count how often it reaches the goal
def evaluate(env, agent, episodes=100):
    successes = 0
    total_reward = 0.0
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.greedy_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
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
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--slippery", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    return p


def main():
    args = build_argparser().parse_args()

    q_path = os.path.join(RESULTS_DIR, f"q_table_{args.tag}.npy")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"No q-table at {q_path}. Run train.py first.")

    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed)
    agent = QLearningAgent(env.n_states, env.n_actions, seed=args.seed)
    agent.load(q_path)
    agent.epsilon = 0.0

    # Part D - learned policy
    policy = extract_policy(env, agent.q_table)
    print("PART D - LEARNED POLICY")
    print("Legend:  ← Left   ↓ Down   → Right   ↑ Up   H Hole   G Goal\n")
    print(render_policy(env, policy))
    print()
    print_policy_actions(env, policy)

    # Part E - evaluation
    print("\nPART E - EVALUATION OVER", args.episodes, "EPISODES")
    metrics = evaluate(env, agent, episodes=args.episodes)
    print("  Success Rate (%)         :", round(metrics["success_rate"], 2))
    print("  Average Reward           :", round(metrics["average_reward"], 4))
    print("  Number of Successful Runs:", metrics["successes"])
    print("  Number of Failures       :", metrics["failures"])


if __name__ == "__main__":
    main()
