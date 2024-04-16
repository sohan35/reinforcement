import numpy as np
from stable_baselines3 import PPO
from tic_tac_toe_env import TicTacToeEnv
from stable_baselines3.common.env_util import make_vec_env

def train_agent():
    # Create and train the RL agent with optimization techniques (PPO example)
    env = make_vec_env(TicTacToeEnv, n_envs=1)
    
    # Define PPO hyperparameters
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=0.0001,  # Lower learning rate for more stable training
                n_steps=2048,          # Number of steps to run for each environment per update
                batch_size=64,         # Number of samples per training batch
                n_epochs=10,           # Number of epochs to optimize the surrogate objective
                gamma=0.99,            # Discount factor
                gae_lambda=0.95,       # Lambda parameter for Generalized Advantage Estimation (GAE)
                clip_range=0.2,        # Clip range for the surrogate objective
                ent_coef=0.0,          # Entropy coefficient for encouraging exploration
                vf_coef=0.5,           # Value function coefficient
                max_grad_norm=0.5,     # Max norm of gradients
                use_sde=False)        # Whether to use SDE (Stochastic Differential Equation) for training

    model.learn(total_timesteps=10000)
    model.save("tic_tac_toe_model")

if __name__ == "__main__":
    train_agent()
