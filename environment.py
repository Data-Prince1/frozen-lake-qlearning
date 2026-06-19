"""
environment.py
==============
Custom Frozen Lake environment implemented entirely from first principles.

No Reinforcement Learning frameworks (Gymnasium, OpenAI Gym, Stable Baselines,
RLlib, etc.) are used anywhere in this project.

Author : Prince Obeng Nkoah (ID: 22424742)
Course : DSCD 614 - Reinforcement Learning
"""

import random


# The standard 8x8 Frozen Lake map used throughout the assignment.
#   S : Start state
#   F : Frozen (safe) state
#   H : Hole (terminal, failure)
#   G : Goal (terminal, success)
DEFAULT_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action encoding required by the assignment.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

# Human-readable arrow symbols for rendering / policy display.
ACTION_ARROWS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}


class FrozenLakeEnv:
    """A grid-world environment where an agent walks from S to G avoiding holes.

    State representation
    --------------------
    States are single integer indices in ``range(0, n_rows * n_cols)``.
    The integer ``s`` maps to grid coordinates as::

        row = s // n_cols
        col = s %  n_cols

    Actions
    -------
    ``0 = Left, 1 = Down, 2 = Right, 3 = Up``

    Reward structure
    ----------------
    ``+1`` when the agent steps onto the Goal, ``0`` for every other transition
    (including falling into a hole).

    Stochasticity (Bonus A)
    -----------------------
    When ``is_slippery`` is ``True`` the chosen action only succeeds with
    probability ``1 - 2 * slip_prob``; otherwise the agent slips to one of the
    two perpendicular directions. With the default ``is_slippery=False`` the
    environment is fully deterministic.
    """

    def __init__(self, grid_map=None, is_slippery=False, slip_prob=1.0 / 3.0,
                 max_steps=200, seed=None):
        self.grid = [list(row) for row in (grid_map or DEFAULT_MAP)]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        self._rng = random.Random(seed)

        # Locate the start cell; default to top-left if none is marked.
        self.start_state = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == "S":
                    self.start_state = self._to_state(r, c)

        self.state = self.start_state
        self.steps = 0
        self.done = False

    # ------------------------------------------------------------------ #
    # Coordinate <-> state helpers
    # ------------------------------------------------------------------ #
    def _to_state(self, row, col):
        return row * self.n_cols + col

    def _to_coords(self, state):
        return divmod(state, self.n_cols)

    def _cell(self, state):
        row, col = self._to_coords(state)
        return self.grid[row][col]

    # ------------------------------------------------------------------ #
    # Core API required by the assignment
    # ------------------------------------------------------------------ #
    def reset(self):
        """Reset the agent to the start state and return that state."""
        self.state = self.start_state
        self.steps = 0
        self.done = False
        return self.state

    def get_state(self):
        """Return the agent's current integer state."""
        return self.state

    def is_terminal(self, state=None):
        """Return True if ``state`` (default: current) is a Hole or Goal."""
        state = self.state if state is None else state
        return self._cell(state) in ("H", "G")

    def _move(self, state, action):
        """Return the state reached by applying ``action`` from ``state``.

        Movement is clipped at the grid boundaries: walking into a wall keeps
        the agent in place.
        """
        row, col = self._to_coords(state)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n_rows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n_cols - 1)
        elif action == UP:
            row = max(row - 1, 0)
        else:
            raise ValueError(f"Invalid action: {action!r} (expected 0-3)")
        return self._to_state(row, col)

    def _resolve_action(self, action):
        """Apply slip noise (Bonus A) and return the action actually taken."""
        if not self.is_slippery:
            return action
        # Slip to a perpendicular direction with probability 2 * slip_prob.
        perpendicular = {
            LEFT:  (DOWN, UP),
            RIGHT: (DOWN, UP),
            DOWN:  (LEFT, RIGHT),
            UP:    (LEFT, RIGHT),
        }[action]
        if self._rng.random() < 2 * self.slip_prob:
            return self._rng.choice(perpendicular)
        return action

    def step(self, action):
        """Take ``action`` and return ``(next_state, reward, done, info)``.

        ``reward`` is ``1.0`` only on reaching the Goal; ``done`` is ``True`` for
        Hole, Goal, or when the step budget is exhausted.
        """
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        taken = self._resolve_action(action)
        self.state = self._move(self.state, taken)
        self.steps += 1

        cell = self._cell(self.state)
        if cell == "G":
            reward, self.done = 1.0, True
        elif cell == "H":
            reward, self.done = 0.0, True
        else:
            reward = 0.0

        # Enforce a step budget so episodes always terminate.
        if self.steps >= self.max_steps:
            self.done = True

        info = {"intended_action": action, "taken_action": taken, "steps": self.steps}
        return self.state, reward, self.done, info

    def render(self, mode="human"):
        """Print the grid with the agent's position marked by ``*``."""
        agent_row, agent_col = self._to_coords(self.state)
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                ch = self.grid[r][c]
                if r == agent_row and c == agent_col:
                    row_chars.append("*")  # current agent position
                else:
                    row_chars.append(ch)
            lines.append(" ".join(row_chars))
        out = "\n".join(lines)
        if mode == "human":
            print(out)
            print()
        return out


if __name__ == "__main__":
    # Quick manual smoke test of the environment.
    env = FrozenLakeEnv()
    env.reset()
    print("Initial grid (agent marked with *):")
    env.render()
    print(f"Start state      : {env.get_state()}")
    print(f"Is terminal?     : {env.is_terminal()}")
    ns, r, done, info = env.step(RIGHT)
    print(f"After RIGHT -> state={ns}, reward={r}, done={done}, info={info}")
