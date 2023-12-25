import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CarRacing-v2', render_mode='human')
DummyVecEnv([lambda: env])

log_path= os.path.join('CarRacing','Logs')
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

model_path = os.path.join('CarRacing','Saved Models','PPO_2m')
model = PPO.load(model_path)

evaluate_policy(model, env, n_eval_episodes=10,render=True)
env.close()