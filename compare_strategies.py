"""
compare_strategies.py
=====================
Bonus Task C: compare two exploration strategies on Frozen Lake.

    1. Pure epsilon-greedy     -> epsilon held constant.
    2. Decaying epsilon-greedy -> epsilon decays toward epsilon_min.

Both agents are trained under identical conditions (same episodes, alpha,
gamma, seed and environment) so the only difference is the exploration
schedule. The script prints a head-to-head summary and saves a comparison plot
to ``results/``.

Usage
-----
    python compare_strategies.py
    python compare_strategies.py --episodes 15000

Author : Prince Obeng Nkoah (ID: 22424742)
Course : DSCD 614 - Reinforcement Learning
"""

import argparse
import os

from environment import FrozenLakeEnv
from agent import QLearningAgent
from train import train, moving_average
from evaluate import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_strategy(name, decay_epsilon, fixed_epsilon, args):
    """Train and evaluate one strategy; return (stats, eval_metrics, agent)."""
    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=(fixed_epsilon if not decay_epsilon else 1.0),
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        decay_epsilon=decay_epsilon,
        seed=args.seed,
    )
    print(f"\n--- Training strategy: {name} ---")
    stats = train(env, agent, episodes=args.episodes, verbose=False)

    eval_env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed + 1)
    agent.epsilon = 0.0
    metrics = evaluate(eval_env, agent, episodes=args.eval_episodes)
    return stats, metrics, agent


def plot_comparison(pure_stats, decay_stats, tag="comparison"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    window = max(1, min(200, len(pure_stats["episode_successes"])))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(100.0 * moving_average(pure_stats["episode_successes"], window),
            label="Pure epsilon-greedy", color="tab:orange")
    ax.plot(100.0 * moving_average(decay_stats["episode_successes"], window),
            label="Decaying epsilon-greedy", color="tab:blue")
    ax.set_title(f"Exploration Strategy Comparison (success rate, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"strategy_{tag}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved comparison plot to {out}")


def build_argparser():
    p = argparse.ArgumentParser(description="Bonus C: compare exploration strategies.")
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--fixed-epsilon", type=float, default=0.1,
                   help="constant epsilon for the pure strategy")
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--slippery", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_argparser().parse_args()

    pure_stats, pure_eval, _ = run_strategy(
        f"Pure epsilon-greedy (epsilon={args.fixed_epsilon})",
        decay_epsilon=False, fixed_epsilon=args.fixed_epsilon, args=args)
    decay_stats, decay_eval, _ = run_strategy(
        "Decaying epsilon-greedy", decay_epsilon=True, fixed_epsilon=None, args=args)

    print("\n" + "=" * 64)
    print("EXPLORATION STRATEGY COMPARISON")
    print("=" * 64)
    header = f"{'Metric':<32}{'Pure eps-greedy':>15}{'Decaying':>15}"
    print(header)
    print("-" * 64)
    print(f"{'Training success rate (%)':<32}"
          f"{pure_stats['overall_success_rate']:>15.2f}"
          f"{decay_stats['overall_success_rate']:>15.2f}")
    print(f"{'Greedy eval success rate (%)':<32}"
          f"{pure_eval['success_rate']:>15.2f}"
          f"{decay_eval['success_rate']:>15.2f}")
    print(f"{'Greedy eval avg reward':<32}"
          f"{pure_eval['average_reward']:>15.4f}"
          f"{decay_eval['average_reward']:>15.4f}")
    print("-" * 64)

    plot_comparison(pure_stats, decay_stats)


if __name__ == "__main__":
    main()
