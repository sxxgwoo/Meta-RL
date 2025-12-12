import gymnasium as gym
import numpy as np
from gymnasium import spaces

class GridWorldAlternate(gym.Env):
    """
    DynaMITE-RL GridWorld (Modified version)
    
    - Grid size: 5x5
    - Agent start position = ALWAYS (0,0)
    - Two goals: one rewarding (+1), one neutral (0)
    - Non-goal cells: reward = -0.1
    - Latent goal index (which goal gives +1) changes between sessions
    - Session switch occurs with probability `switch_prob`
    - When a session switches -> agent is teleported back to (0,0)
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=60, switch_prob=0.07):
        super().__init__()
        self.grid_size = 5
        self.max_steps = max_steps
        self.switch_prob = switch_prob

        # Observations: [ax, ay, g1x, g1y, g2x, g2y] / 4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # 4 directions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)

        self.goals = None
        self.agent_pos = None
        self.active_goal_idx = 0
        self.steps = 0

    def _get_obs(self):
        obs = np.concatenate([
            self.agent_pos,
            self.goals[0],
            self.goals[1]
        ]) / (self.grid_size - 1)
        return obs.astype(np.float32)
    
    def _sample_non_goal_position(self):
        """현재 goals를 기준으로 goal이 아닌 랜덤 위치 하나 샘플."""
        assert self.goals is not None, "goals must be set before sampling start position."
        positions = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if not np.array_equal([i, j], self.goals[0])
            and not np.array_equal([i, j], self.goals[1])
        ]
        positions = np.array(positions)
        idx = self.np_random.integers(len(positions))
        return positions[idx].astype(int)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        rng = self.np_random

        # Start is ALWAYS (0,0)
        self.agent_pos = np.array([0, 0], dtype=int)

        # Sample two goal positions (never equal to each other or to start)
        possible_positions = np.array([
            (i, j) for i in range(self.grid_size) for j in range(self.grid_size)
            # if not (i == 0 and j == 0)  # cannot be start
        ])
        selected = possible_positions[rng.choice(len(possible_positions), 2, replace=False)]

        self.goals = [
            selected[0].astype(int),
            selected[1].astype(int)
        ]
        # 2) 시작 위치는 goal이 아닌 칸 중 하나에서 샘플
        self.agent_pos = self._sample_non_goal_position()

        # Choose initial rewarding goal index
        self.active_goal_idx = rng.integers(0, 2)

        obs = self._get_obs()
        info = {
            "task_index": self.active_goal_idx,
            "session_changed": False
        }
        return obs, info

    def step(self, action):
        # movement: Up, Down, Left, Right
        moves = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
        }

        dy, dx = moves[int(action)]
        self.agent_pos[0] = np.clip(self.agent_pos[0] + dy, 0, self.grid_size - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + dx, 0, self.grid_size - 1)
        self.steps += 1

        reward = -0.1
        active_goal_pos = self.goals[self.active_goal_idx]

        # Reward rules
        if np.array_equal(self.agent_pos, active_goal_pos):
            reward = 1.0
        elif np.array_equal(self.agent_pos, self.goals[1 - self.active_goal_idx]):
            reward = 0.0
            # reward = -1.0

        # Check session switch
        session_changed = False
        if self.np_random.random() < self.switch_prob:
            session_changed = True

            # latent transition matrix
            if self.active_goal_idx == 0:
                self.active_goal_idx = self.np_random.choice([0, 1], p=[0.2, 0.8])
            else:
                self.active_goal_idx = self.np_random.choice([0, 1], p=[0.8, 0.2])

            # *** IMPORTANT ***
            # Return agent to start position (0,0)
            # self.agent_pos = np.array([0, 0], dtype=int)
            
            # 새 세션 시작 위치: goal이 아닌 칸에서 랜덤
            self.agent_pos = self._sample_non_goal_position()

        terminated = False
        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "task_index": self.active_goal_idx,
            "session_changed": session_changed
        }
        return obs, reward, terminated, truncated, info
