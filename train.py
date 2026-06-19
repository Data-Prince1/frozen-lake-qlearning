"""
train.py
========
Training loop for the Q-Learning agent on the custom Frozen Lake environment.

Running this module trains an agent, records training statistics
(episode rewards, success rate, number of successful episodes, epsilon over
time), saves the learned Q-table and statistics to ``results/`` and produces
training-performance graphs (Bonus B).

Usage
-----
    python train.py
    python train.py --episodes 20000 --alpha 0.1 --gamma 0.99 --slippery

Author : Prince Obeng Nkoah (ID: 22424742)
Course : DSCD 614 - Reinforcement Learning
"""

import argparse
import json
import os

import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def train(env, agent, episodes=15000, log_every=1000, verbose=True):
    """Train ``agent`` on ``env`` for ``episodes`` episodes.

    Returns a dictionary of recorded statistics.
    """
    episode_rewards = []      # total reward collected per episode
    episode_successes = []    # 1 if the goal was reached, else 0
    epsilon_history = []      # epsilon value at the start of each episode
    episode_steps = []        # steps taken per episode

    for ep in range(episodes):
        epsilon_history.append(agent.epsilon)
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # An episode is a "success" if it ended on the goal (reward == 1).
        success = 1 if total_reward > 0 else 0
        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_steps.append(env.steps)
        agent.decay()

        if verbose and (ep + 1) % log_every == 0:
            window = episode_successes[-log_every:]
            rate = 100.0 * sum(window) / len(window)
            print(f"Episode {ep + 1:>6}/{episodes} | "
                  f"epsilon={agent.epsilon:.3f} | "
                  f"success rate (last {log_every}) = {rate:5.1f}%")

    stats = {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "epsilon_history": epsilon_history,
        "episode_steps": episode_steps,
        "n_episodes": episodes,
        "n_successful_episodes": int(sum(episode_successes)),
        "overall_success_rate": 100.0 * sum(episode_successes) / episodes,
    }
    return stats


def save_results(agent, stats, hyperparams, tag="main"):
    """Persist the Q-table, statistics and hyper-parameters to ``results/``."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent.save(os.path.join(RESULTS_DIR, f"q_table_{tag}.npy"))

    # JSON-serialisable copy of the stats (lists of plain numbers).
    serialisable = {
        "hyperparameters": hyperparams,
        "n_episodes": stats["n_episodes"],
        "n_successful_episodes": stats["n_successful_episodes"],
        "overall_success_rate": stats["overall_success_rate"],
        "episode_rewards": [float(x) for x in stats["episode_rewards"]],
        "episode_successes": [int(x) for x in stats["episode_successes"]],
        "epsilon_history": [float(x) for x in stats["epsilon_history"]],
        "episode_steps": [int(x) for x in stats["episode_steps"]],
    }
    with open(os.path.join(RESULTS_DIR, f"training_stats_{tag}.json"), "w") as f:
        json.dump(serialisable, f)
    print(f"Saved Q-table and stats to {RESULTS_DIR} (tag='{tag}').")


def moving_average(values, window=100):
    """Return the simple moving average of ``values`` over ``window``."""
    if len(values) < window:
        window = max(1, len(values))
    cumsum = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training(stats, tag="main"):
    """Produce and save training-performance graphs (Bonus B)."""
    import matplotlib
    matplotlib.use("Agg")  # headless backend; no display required
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rewards = stats["episode_rewards"]
    successes = stats["episode_successes"]
    epsilons = stats["epsilon_history"]
    window = max(1, min(100, len(rewards)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(moving_average(rewards, window), color="tab:blue")
    axes[0].set_title(f"Episode Reward (moving avg, window={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(100.0 * moving_average(successes, window), color="tab:green")
    axes[1].set_title(f"Success Rate (moving avg, window={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epsilons, color="tab:red")
    axes[2].set_title("Epsilon over Time")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"training_performance_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved training graphs to {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="Train a Q-Learning agent on Frozen Lake.")
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    p.add_argument("--epsilon", type=float, default=1.0, help="initial epsilon")
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--slippery", action="store_true", help="enable stochastic transitions (Bonus A)")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--no-plot", action="store_true", help="skip generating graphs")
    return p


def main():
    args = build_argparser().parse_args()

    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        decay_epsilon=True,
        seed=args.seed,
    )

    hyperparams = {
        "episodes": args.episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon,
        "epsilon_min": args.epsilon_min,
        "epsilon_decay": args.epsilon_decay,
        "is_slippery": args.slippery,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    print("=" * 60)
    print("Training Q-Learning agent on Frozen Lake (8x8)")
    print("=" * 60)
    for k, v in hyperparams.items():
        print(f"  {k:>15} : {v}")
    print("-" * 60)

    stats = train(env, agent, episodes=args.episodes)

    print("-" * 60)
    print(f"Successful training episodes : {stats['n_successful_episodes']} / {args.episodes}")
    print(f"Overall training success rate: {stats['overall_success_rate']:.2f}%")

    save_results(agent, stats, hyperparams, tag=args.tag)
    if not args.no_plot:
        plot_training(stats, tag=args.tag)


if __name__ == "__main__":
    main()
