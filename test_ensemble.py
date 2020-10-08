# Minor adjustments were made to this file to adapt to PyTorch and weighted ensemble technique
# See "CSCI-GA.3033-090" at https://cs.nyu.edu/dynamic/courses/schedule/?semester=fall_2020&level=GA
# for description of course which provided shell code
# Hard-coded to showcase agent that performs, on average, 800+ on OpenAI Gym CarRacing-v0

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
    

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0
    state = env.reset()
    state_seq = torch.zeros(3, 1, 96, 96)
    while True:
        state = torch.tensor(rgb2gray(state))[None,None,:,:]
        state_seq = torch.cat((state_seq[1:], state), dim=0)
        a = []
        for (agent, weight) in zip(models, weights):
            if type(agent) == CNN_Agent:
                a.append(torch.mul(agent(state)[0].detach(), weight))
            elif type(agent) == CNN_History_Agent:
                a.append(torch.mul(agent(state_seq)[-1].detach(), weight))
        a = torch.sum(torch.stack(a), dim=0).numpy()
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
    # Hard-coded depth parameter for CNN_History_Agent
    his_num = 3
    
    n_test_episodes = 15                  # number of episodes to test

    # Hard-coded model weights, chosen through empirical testing
    model_files = [
        "./model/urci_cnn_100_epochs.pth",
        "./model/urc_original_3hiscnn_100_epochs_slow_but_steady.pth",
        "./model/urc_agg_3hiscnn_45et_epochs_go_fast.pth"
    ]
    models = [
        # Basic agent, essentially CNN that outputs a 3-dimensional vector that corresponds to actions
        # in the CarRacing-v0 world
        CNN_Agent(),
        # More complex agent
        # Uses CNN similar to above to encode 2D image stream from game into feature fectors
        # Then performs a 1D convolution with a 2D kernel, taking sliding window of feature vectors,
        #   and passing the result through additional linear layers to produce 3-dim action vector
        # There may be better architectures than this, but this was enough to produce an agent
        #   that scores 800+ on average, which was the intent of this mini project
        # Weights loaded (see 2nd .pth file, above) make this agent perform very slow, but safely
        CNN_History_Agent(his_num),
        # Third agent is just like second, but loaded with different weights (trained on different data)
        # Weights loaded (see 3rd .pth file, above) make this agent perform very recklessly, but capable of high scores
        CNN_History_Agent(his_num)
    ]
    # This weighting scheme was determined through empirical testing, additional tuning might give better performance
    weights = [
        0.0, 0.75, 0.25
    ]
    for agent,model_file in zip(models, model_files):
        agent.load_state_dict(torch.load(model_file))
        agent.eval()

    for (agent,weight) in zip(models, weights):
        if weight == 0.0:
            continue
        print(" WEIGHT:", int(weight*100), "%")
        print("MODEL DIMENSIONS AND NUM OF PARAMETERS")
        print("--------------------------------------")
        test_input = torch.zeros(3,1,96,96)
        agent(test_input, verbose=True)
        
        
    # Below code starts the car racing game, runs episodes, and saves results to a .json file
    # No/minimal changes were made from here down from shell code that was given from NYU deep RL course
    # See "CSCI-GA.3033-090" at https://cs.nyu.edu/dynamic/courses/schedule/?semester=fall_2020&level=GA
    # for course description
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
 
    fname = "./results/ensemble_25_50_25-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
    env.close()
    print('... finished')
    
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()
