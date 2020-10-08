# Minor adjustments were made to this file to adapt to PyTorch and weighted ensemble technique
# See "CSCI-GA.3033-090" at https://cs.nyu.edu/dynamic/courses/schedule/?semester=fall_2020&level=GA
# for description of course which provided shell code

from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from model import *
from utils import *
import matplotlib.pyplot as plt
import argparse

# Gotta love argparse

CNN_AGENT = 0
CNN_HIS_AGENT = 1

parser = argparse.ArgumentParser()
parser.add_argument("model_num", type=int,
                    help="0=CNN_Agent, 1=CNN_History_Agent")
parser.add_argument("opt_his_num", type=int, nargs='?',
                    help="Optional history length")
parser.add_argument("--model", type=str,
                    help="Path to model file")
parser.add_argument("--results", type=str,
                    help="First part of path to results save location (data/time will be appended)")

args = parser.parse_args()
model_num = args.model_num
his_num = args.opt_his_num
if model_num == 1 and his_num == None:
    his_num = 3
model_file = args.model
if model_file == None:
    model_file = "./model/test.pth"
result_path = args.results
if result_path == None:
    result_path = "./results/results_bc_agent"

print("Args received:")
print("Model Number -", model_num)
print("History Length -", his_num)
print("Model File =", model_file)
print("Results Save Path -", result_path)

# Below is more or less that same as provided shell code, with small adaptations for PyTorch and original models
def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0
    state = env.reset()
    state_seq = torch.zeros(3, 1, 96, 96)
    while True:
        state = torch.tensor(rgb2gray(state))[None,None,:,:]
        state_seq = torch.cat((state_seq[1:], state), dim=0)
        if model_num == CNN_AGENT:
            a = agent(state)[0].detach().numpy()
        elif model_num == CNN_HIS_AGENT:
            a = agent(state_seq)[-1].detach().numpy()
        next_state, r, done, info = env.step(a)   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    if model_num == CNN_AGENT:
        agent = CNN_Agent()
    elif model_num == CNN_HIS_AGENT:
        agent = CNN_History_Agent(his_num)
    agent.load_state_dict(torch.load(model_file))
    agent.eval()

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = result_path + "-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
    env.close()
    print('... finished')
    
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
