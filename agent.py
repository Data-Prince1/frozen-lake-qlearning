"""
agent.py
========
Q-Learning agent implemented from scratch (tabular Q-learning).

The update equation is implemented exactly as required by the assignment:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Author : Prince Obeng Nkoah (ID: 22424742)
Course : DSCD 614 - Reinforcement Learning
"""

import random

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with epsilon-greedy exploration.

    Parameters
    ----------
    n_states, n_actions : int
        Size of the discrete state and action spaces.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration rate.
    epsilon_min : float
        Lower bound on epsilon during decay.
    epsilon_decay : float
        Multiplicative decay applied to epsilon after each episode.
    decay_epsilon : bool
        If ``False`` epsilon is held constant (pure epsilon-greedy, Bonus C).
    seed : int or None
        Seed for reproducible exploration.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999,
                 decay_epsilon=True, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon = decay_epsilon

        self._rng = random.Random(seed)

        # Q-table initialization: all state-action values start at zero.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def choose_action(self, state):
        """Epsilon-greedy action selection.

        With probability ``epsilon`` explore (random action); otherwise exploit
        the greedy action. Ties among greedy actions are broken randomly to
        avoid a systematic bias toward action 0.
        """
        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.n_actions - 1)
        return self.greedy_action(state)

    def greedy_action(self, state):
        """Return the highest-value action for ``state`` (random tie-break)."""
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best.tolist()))

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(self, state, action, reward, next_state, done):
        """Apply the Q-learning update rule for one transition.

            Q(s,a) <- Q(s,a) + alpha * [ target - Q(s,a) ]

        where ``target = r + gamma * max_a' Q(s',a')`` and the bootstrap term is
        dropped when ``next_state`` is terminal.
        """
        current_q = self.q_table[state, action]
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q
        self.q_table[state, action] = current_q + self.alpha * td_error
        return td_error

    def decay(self):
        """Decay epsilon toward ``epsilon_min`` (no-op when decay disabled)."""
        if self.decay_epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def save(self, path):
        """Save the learned Q-table to ``path`` (.npy)."""
        np.save(path, self.q_table)

    def load(self, path):
        """Load a Q-table from ``path`` (.npy)."""
        self.q_table = np.load(path)
        return self.q_table
