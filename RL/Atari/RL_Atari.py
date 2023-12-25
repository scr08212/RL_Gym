import sys
import os
from time import sleep
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

#reset variables

score = 0
done = False

env = make_atari_env('Breakout-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model_path = os.path.join('Atari', 'Saved Models','A2C_2M_model')
model = A2C.load(model_path, env)
vec_env = model.get_env() 
obs = vec_env.reset()
while not done:
    vec_env.render('human')
    action, _states = model.predict(obs, deterministic=True) 
    obs, rewards, done, info = vec_env.step(action) 
    score+=rewards
env.close()