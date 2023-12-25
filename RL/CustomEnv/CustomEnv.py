import gymnasium as gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import os
import numpy as np
import random

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=0,high=100,shape=(1,))
        self.state=38+random.randint(-3,3)
        self.shower_length = 100

    def step(self, action):
        self.state += action-1
        self.shower_length -=1

        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done =True
        else:
            done = False
        
        info = {}

        return self.state, reward, done, info
    
    def render(self):
        pass

    def reset(self):
        self.state = np.array([38+random.randint(-3,3)]).astype(float)
        self.shower_length = 60
        return self.state


env = ShowerEnv()
'''
for episode in range(1, 6):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode{} score{}'.format(episode, score))
'''

'''
log_path = os.path.join('CustomEnv','Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)
'''


model_path = os.path.join('CustomEnv','Saved Models', 'PPO_Model')
model = PPO.load(model_path)

print(evaluate_policy(model, env, n_eval_episodes=10, render=False))

