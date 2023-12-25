import os
#import pygame
import gymnasium as gym
#import tensorboard
from stable_baselines3 import PPO # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

log_Path = os.path.join('RL','CartPole','Logs')
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
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_Path)
model.learn(total_timesteps = 20000)

PPO_Path = os.path.join('RL','CartPole','Saved Models','PPO_Model_CartPole')
model.save(PPO_Path)
model = PPO.load(PPO_Path, env=env)
 
evaluate_policy(model, env, n_eval_episodes=10,render=True)

env.close()



"""
# Test Model
env = gym.make('CartPole-v1',render_mode='human')
PPO_Path = os.path.join('CartPole','Saved Models','PPO_Model_CartPole')
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

'''
training_log_path = os.path.join(log_Path, 'PPO_1')
# CMD::  PS C:\Users\scr08\Desktop\ML\RL\CartPole> tensorboard --logdir='Logs\PPO_1'

env = gym.make('CartPole-v1',render_mode='human')
save_path = os.path.join('RL','CartPole','Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=stop_callback, eval_freq=10000,best_model_save_path=save_path ,verbose=1)

#model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_Path)
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_Path, policy_kwargs={'net_arch':net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)
'''