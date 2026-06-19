# Frozen Lake environment (built from scratch, no RL libraries used)
# Prince Obeng Nkoah - 22424742

import random

# The 8x8 map from the assignment
# S = start, F = frozen/safe, H = hole, G = goal
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

# Actions
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_ARROWS = {LEFT: "←", DOWN: "↓", RIGHT: "→", UP: "↑"}


class FrozenLakeEnv:
    def __init__(self, grid_map=None, is_slippery=False, slip_prob=1.0 / 3.0,
                 max_steps=200, seed=None):
        if grid_map is None:
            grid_map = DEFAULT_MAP
        self.grid = [list(row) for row in grid_map]
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        self.rng = random.Random(seed)

        # find the start cell
        self.start_state = 0
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == "S":
                    self.start_state = r * self.n_cols + c

        self.state = self.start_state
        self.steps = 0
        self.done = False

    # turn a state number into (row, col)
    def _to_coords(self, state):
        return state // self.n_cols, state % self.n_cols

    def _to_state(self, row, col):
        return row * self.n_cols + col

    def _cell(self, state):
        row, col = self._to_coords(state)
        return self.grid[row][col]

    def reset(self):
        self.state = self.start_state
        self.steps = 0
        self.done = False
        return self.state

    def get_state(self):
        return self.state

    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        return self._cell(state) == "H" or self._cell(state) == "G"

    # move one step, staying inside the grid
    def _move(self, state, action):
        row, col = self._to_coords(state)
        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self.n_rows - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n_cols - 1)
        elif action == UP:
            row = max(row - 1, 0)
        return self._to_state(row, col)

    # for the slippery (bonus) version the action can change
    def _resolve_action(self, action):
        if not self.is_slippery:
            return action
        if action == LEFT or action == RIGHT:
            sideways = [DOWN, UP]
        else:
            sideways = [LEFT, RIGHT]
        if self.rng.random() < 2 * self.slip_prob:
            return self.rng.choice(sideways)
        return action

    def step(self, action):
        taken = self._resolve_action(action)
        self.state = self._move(self.state, taken)
        self.steps += 1

        cell = self._cell(self.state)
        if cell == "G":
            reward = 1.0
            self.done = True
        elif cell == "H":
            reward = 0.0
            self.done = True
        else:
            reward = 0.0

        if self.steps >= self.max_steps:
            self.done = True

        info = {"taken_action": taken, "steps": self.steps}
        return self.state, reward, self.done, info

    def render(self):
        agent_row, agent_col = self._to_coords(self.state)
        for r in range(self.n_rows):
            row = ""
            for c in range(self.n_cols):
                if r == agent_row and c == agent_col:
                    row += "* "
                else:
                    row += self.grid[r][c] + " "
            print(row)
        print()


# quick test
if __name__ == "__main__":
    env = FrozenLakeEnv()
    env.reset()
    env.render()
    print("start state:", env.get_state())
    print("terminal?", env.is_terminal())
    ns, r, done, info = env.step(RIGHT)
    print("after right:", ns, r, done, info)
