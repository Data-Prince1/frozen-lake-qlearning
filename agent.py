# Q-Learning agent (written from scratch)
# Prince Obeng Nkoah - 22424742

import random
import numpy as np


class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999,
                 decay_epsilon=True, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon = decay_epsilon

        self.rng = random.Random(seed)

        # start the Q-table with all zeros
        self.q_table = np.zeros((n_states, n_actions))

    # epsilon-greedy: explore sometimes, otherwise pick the best action
    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, self.n_actions - 1)
        return self.greedy_action(state)

    def greedy_action(self, state):
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best.tolist()))

    # the Q-learning update rule:
    # Q(s,a) = Q(s,a) + alpha * [ r + gamma * max Q(s',a') - Q(s,a) ]
    def update(self, state, action, reward, next_state, done):
        current_q = self.q_table[state, action]
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        self.q_table[state, action] = current_q + self.alpha * (td_target - current_q)

    # reduce epsilon over time
    def decay(self):
        if self.decay_epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)
        return self.q_table
