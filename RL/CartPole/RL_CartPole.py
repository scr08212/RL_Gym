import os
import pygame
import gymnasium as gym
from stable_baselines3 import PPO # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
"""
# Load environment
env = gym.make('CartPole-v1',render_mode='human')
# Run without training
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    while not done:
       env.render()
       action = env.action_space.sample()
       n_state, reward, done, _, info = env.step(action)
       score+=reward
    print('Episodes:{} Score:{}'.format(episode, score))
env.close()

# Train RL model
env = gym.make('CartPole-v1',render_mode='human')
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps = 20000)

PPO_Path = os.path.join('RL','CartPole','Saved Models','PPO_Model_CartPole')
model.save(PPO_Path)
model = PPO.load(PPO_Path, env=env)
 
evaluate_policy(model, env, n_eval_episodes=10,render=True)

env.close()
"""


# Test Model
env = gym.make('CartPole-v1',render_mode='human')
PPO_Path = os.path.join('RL','CartPole','Saved Models','PPO_Model_CartPole')
model = PPO.load(PPO_Path, env=env)
episodes = 5
for episode in range(1, episodes+1):
    obs, info = env.reset()
    done = False
    score = 0
    
    while not done:
       env.render()
       action, _ = model.predict(obs)
       obs, reward, done, _, info = env.step(action)
       score+=reward
       if score>= 200:
           break
    print('Episodes:{} Score:{}'.format(episode, score))
env.close()