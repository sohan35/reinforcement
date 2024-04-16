import gym
import numpy as np

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(9)  # 9 possible actions (0-8)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float32)  # 3x3 board
        self.board = np.zeros((3, 3))
        self.current_player = 1  # Start with player 1
        self.game_over = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.game_over = False
        return self.board.copy()

    def step(self, action):
        row = action // 3
        col = action % 3
        if self.board[row][col] != 0:
            return self.board.copy(), -10, self.game_over, {}  # Invalid move penalty
        
        self.board[row][col] = self.current_player
        reward = self._calculate_reward()
        self.game_over = self._check_game_over()
        self.current_player *= -1  # Switch players
        return self.board.copy(), reward, self.game_over, {}

    def _calculate_reward(self):
        if self._check_winner(1):
            return 1
        elif self._check_winner(-1):
            return -1
        elif self._is_board_full():
            return 0
        else:
            return 0.01  # Small positive reward for continuing the game

    def _check_winner(self, player):
        for i in range(3):
            if all(self.board[i] == player) or all(self.board[:, i] == player):
                return True
        if all(np.diag(self.board) == player) or all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def _is_board_full(self):
        return not np.any(self.board == 0)

    def _check_game_over(self):
        return self._check_winner(1) or self._check_winner(-1) or self._is_board_full()
