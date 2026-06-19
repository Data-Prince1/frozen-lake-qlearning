# Training the Q-learning agent on Frozen Lake
# Prince Obeng Nkoah - 22424742

import argparse
import json
import os

import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# run the training loop and return the statistics we collected
def train(env, agent, episodes=15000, log_every=1000, verbose=True):
    episode_rewards = []
    episode_successes = []
    epsilon_history = []
    episode_steps = []

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

        # reward > 0 means we reached the goal
        success = 1 if total_reward > 0 else 0
        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_steps.append(env.steps)
        agent.decay()

        if verbose and (ep + 1) % log_every == 0:
            window = episode_successes[-log_every:]
            rate = 100.0 * sum(window) / len(window)
            print(f"Episode {ep + 1}/{episodes} | epsilon={agent.epsilon:.3f} | "
                  f"success rate (last {log_every}) = {rate:.1f}%")

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


# save the q-table and the stats so we can use them later
def save_results(agent, stats, hyperparams, tag="main"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    agent.save(os.path.join(RESULTS_DIR, f"q_table_{tag}.npy"))

    data = {
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
        json.dump(data, f)
    print("Saved q-table and stats to", RESULTS_DIR)


# simple moving average for smoother graphs
def moving_average(values, window=100):
    if len(values) < window:
        window = max(1, len(values))
    cumsum = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


# make the training graphs (bonus B)
def plot_training(stats, tag="main"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rewards = stats["episode_rewards"]
    successes = stats["episode_successes"]
    epsilons = stats["epsilon_history"]
    window = max(1, min(100, len(rewards)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(moving_average(rewards, window), color="tab:blue")
    axes[0].set_title("Average reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average reward")

    axes[1].plot(100.0 * moving_average(successes, window), color="tab:green")
    axes[1].set_title("Success rate")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success rate (%)")

    axes[2].plot(epsilons, color="tab:red")
    axes[2].set_title("Epsilon over time")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"training_performance_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("Saved graphs to", out_path)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=15000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--epsilon-decay", type=float, default=0.9995)
    p.add_argument("--slippery", action="store_true")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="main")
    p.add_argument("--no-plot", action="store_true")
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

    print("Training on Frozen Lake (8x8)")
    print("Hyperparameters:", hyperparams)

    stats = train(env, agent, episodes=args.episodes)

    print("Successful episodes:", stats["n_successful_episodes"], "/", args.episodes)
    print("Overall training success rate: %.2f%%" % stats["overall_success_rate"])

    save_results(agent, stats, hyperparams, tag=args.tag)
    if not args.no_plot:
        plot_training(stats, tag=args.tag)


if __name__ == "__main__":
    main()
