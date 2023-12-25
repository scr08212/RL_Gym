import sys
import os
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

env = gym.make('Breakout-v4', render_mode = 'human')
env = DummyVecEnv([lambda: env]) 
env = VecFrameStack(env, n_stack=4)

model_path = os.path.join('Atari', 'Saved Models','A2C_2M_model')
model = A2C.load(model_path, env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()