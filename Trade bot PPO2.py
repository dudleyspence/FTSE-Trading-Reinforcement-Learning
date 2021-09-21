# gym stuff
import gym
import gym_anytrading

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def preprocess_original_data():
    # imports the original 20 years of data and preprocesses the columns into correct datatypes
    dataFile = 'ftse100-1m.csv'
    df = pd.read_csv(dataFile, delimiter=';', names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['DateTime'] = df['Date'] + ' ' + df['Time']
    del df['Date']
    del df['Time']
    df.to_csv('ftse100-1m_preprocessed.csv')


def check_environment_works():
    env = gym.make('stocks-v0', df=df, frame_bound=(200000, 1000000), window_size=30)

    state = env.reset()
    while True:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        if done:
            print("info", info)
            break

    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.show()


def train():
    env_maker = lambda: gym.make('stocks-v0', df=df, frame_bound=(255908, 3218898), window_size=30)
    env = DummyVecEnv([env_maker])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000000)
    model.save("PPO_ftse")
    return model
    # del model  # remove to demonstrate saving and loading


def test():
    model = PPO.load("PPO_ftse")
    env = gym.make('stocks-v0', df=df, frame_bound=(3218898, 4158436), window_size=30)
    obs = env.reset()
    while True:
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            print("info", info)
            break
    plt.figure(figsize=(15, 6))
    plt.cla()
    env.render_all()
    plt.show()


# preprocess_original_data()
preprocessedDataFile = 'ftse100-1m_preprocessed.csv'
df = pd.read_csv(preprocessedDataFile, parse_dates=['DateTime'], infer_datetime_format=True)
df.set_index('DateTime', inplace=True)
del df['Unnamed: 0']


# check_environment_works()

train()
test()


# 4158436 rows of data
# last date is 11/08/21
# first volume stat not until 255908
# line 3218898 is the start of 2018

