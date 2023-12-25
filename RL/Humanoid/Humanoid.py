import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('Humanoid-v4', render_mode='human')
#env = gym.make('Humanoid-v4')

model_path = os.path.join('Humanoid','Saved Models', 'PPO_10M')
log_path = os.path.join('Humanoid', 'Logs')

'''
model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log=log_path)
model.learn(total_timesteps=10000000)
model.save(model_path)

del model
'''

model = PPO.load(model_path)
print(evaluate_policy(model, env, n_eval_episodes=10,render=True))

env.close()
