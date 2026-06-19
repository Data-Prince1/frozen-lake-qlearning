# Bonus C - compare pure epsilon-greedy vs decaying epsilon-greedy
# Prince Obeng Nkoah - 22424742

import argparse
import os

from environment import FrozenLakeEnv
from agent import QLearningAgent
from train import train, moving_average
from evaluate import evaluate

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# train and evaluate one strategy
def run_strategy(name, decay_epsilon, fixed_epsilon, args):
    env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed)
    if decay_epsilon:
        start_epsilon = 1.0
    else:
        start_epsilon = fixed_epsilon
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=start_epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        decay_epsilon=decay_epsilon,
        seed=args.seed,
    )
    print("Training strategy:", name)
    stats = train(env, agent, episodes=args.episodes, verbose=False)

    eval_env = FrozenLakeEnv(is_slippery=args.slippery, max_steps=args.max_steps, seed=args.seed + 1)
    agent.epsilon = 0.0
    metrics = evaluate(eval_env, agent, episodes=args.eval_episodes)
    return stats, metrics


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
    ax.set_title("Exploration strategy comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"strategy_{tag}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("Saved comparison plot to", out)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--fixed-epsilon", type=float, default=0.1)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--slippery", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_argparser().parse_args()

    pure_stats, pure_eval = run_strategy(
        "Pure epsilon-greedy", decay_epsilon=False,
        fixed_epsilon=args.fixed_epsilon, args=args)
    decay_stats, decay_eval = run_strategy(
        "Decaying epsilon-greedy", decay_epsilon=True,
        fixed_epsilon=None, args=args)

    print("\nEXPLORATION STRATEGY COMPARISON")
    print("Pure epsilon-greedy:")
    print("  training success rate (%):", round(pure_stats["overall_success_rate"], 2))
    print("  eval success rate (%)    :", round(pure_eval["success_rate"], 2))
    print("  eval avg reward          :", round(pure_eval["average_reward"], 4))
    print("Decaying epsilon-greedy:")
    print("  training success rate (%):", round(decay_stats["overall_success_rate"], 2))
    print("  eval success rate (%)    :", round(decay_eval["success_rate"], 2))
    print("  eval avg reward          :", round(decay_eval["average_reward"], 4))

    plot_comparison(pure_stats, decay_stats)


if __name__ == "__main__":
    main()
