# Adjustments were made to this file to adapt to PyTorch and weighted ensemble technique
# See "CSCI-GA.3033-090" at https://cs.nyu.edu/dynamic/courses/schedule/?semester=fall_2020&level=GA
# for description of course which provided shell code

from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from statistics import mean
from model import *
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import argparse

CNN_AGENT = 0
CNN_HIS_AGENT = 1

# Gotta love argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_num", type=int,
                    help="0=CNN_Agent, 1=CNN_History_Agent")
parser.add_argument("opt_his_num", type=int, nargs='?',
                    help="Optional history length")
parser.add_argument("--data", type=str,
                    help="Name of data file within ./data")
parser.add_argument("--save", type=str,
                    help="Path to save model")
parser.add_argument("--epochs", type=int,
                    help="Max number of training epochs")
parser.add_argument("--lr", type=float,
                    help="Learning rate hyperparameter")
parser.add_argument("--momentum", type=float,
                    help="Momentum hyperparameter")
parser.add_argument("--log", type=int,
                    help="Logging frequency")
parser.add_argument("--cap", type=int,
                    help="Cap limit for data to train on")

args = parser.parse_args()

model_num = args.model_num
his_num = args.opt_his_num
if model_num == 1 and his_num == None:
    his_num = 3
data_file = args.data
if data_file == None:
    data_file = "data.pkl.gzip"
save_path = args.save
if save_path == None:
    save_path = "./model/test.pth"
epochs = args.epochs
if epochs == None:
    epochs = 20
lr = args.lr
if lr == None:
    lr = 0.0001
momentum = args.momentum
if momentum == None:
    momentum = 0.9
log_interval = args.log
if log_interval == None:
    log_interval = 10
cap = args.cap


print("Args recieved:")
print("Model Number -", model_num)
print("History Length -", his_num)
print("Data File -", data_file)
print("Save Path -", save_path)
print("Epochs -", epochs)
print("Learning Rate -", lr)
print("Momentum -", momentum)
print("Logging Interval -", log_interval)
print("Cap -", cap)
    
    
# Custom dataset to load from numpy arrays
class MyDataset(Dataset):
    def __init__(self,X,Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# More or less unchanged from given code, see above
# However, setting cap to non-null will set an upper limit to how much data the agent
# will test on (used for experiments)
def read_data(datasets_dir="./data", batch_size=64, frac = 0.1):
    global data_file
    
    print("... read data")
    data_file = os.path.join(datasets_dir, data_file)
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    Y = np.array(data["action"]).astype('float32')
    if cap != None:
        X = X[:cap]
        Y = Y[:cap]

    # split data into training and validation set
    n_samples = len(X)
    X_train, Y_train = X[:int((1-frac) * n_samples)], Y[:int((1-frac) * n_samples)]
    X_valid, Y_valid = X[int((1-frac) * n_samples):], Y[int((1-frac) * n_samples):]

    return X_train, Y_train, X_valid, Y_valid

# Just convert image stream from rgb to grayscale
def preprocessing(X_train, Y_train, X_valid, Y_valid):

    print("... preprocessing")
    X_train, X_valid = rgb2gray(X_train), rgb2gray(X_valid)
    return X_train, Y_train, X_valid, Y_valid

# Training loop for racing agents
def train_model(train_loader, val_loader,lr=0.01, momentum=0.9, epochs=0, log_interval=10):
    global model_num
    global his_num
    
    if model_num == CNN_AGENT:
        agent = CNN_Agent()
    elif model_num == CNN_HIS_AGENT:
        agent = CNN_History_Agent(his_num)
    else:
        agent = CNN_Agent()

    # This will print out a summary of the model to be trained
    print("MODEL DIMENSIONS AND NUM OF PARAMETERS")
    print("--------------------------------------")
    test_input = torch.zeros(64,1,96,96)
    output = agent(test_input, verbose=True)
    # Chosen optimizer and loss function
    optimizer = optim.SGD(agent.parameters(), lr=lr, momentum=momentum)
    Loss = nn.SmoothL1Loss(reduction='sum')

    # keeping track of metrics
    train_loss = []
    val_loss = []
    average_val_loss = 999999999
    # Window used for early termination determination
    WINDOW = 5

    for e in range(epochs+1):
        # Train
        agent.train()
        running_loss = 0
        for batch_idx, (states, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            actions = agent(states)
            loss = Loss(actions, targets)
            loss.backward()
            # Gradient clipping to avoid exploding gradient
            clip_grad_norm_(agent.parameters(), 0.5)
            running_loss += loss.detach().item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(states), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        train_loss.append(running_loss / len(train_loader.dataset))
        
        # Validate
        agent.eval()
        validation_loss = 0
        for batch_idx, (states, targets) in enumerate(val_loader):
            actions = agent(states)
            validation_loss += Loss(actions, targets).detach().item()
        validation_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}\n'.format(
            validation_loss), end='')
        val_loss.append(validation_loss)

        # Early termination test
        if e % WINDOW == 0 and e != 0:
            temp = mean(val_loss[-WINDOW:])
            print("Average loss over last", WINDOW, " epochs was",
                  temp/average_val_loss, "of previous average loss\n")
            if temp > average_val_loss:
                print("Stopping training now")
                break
            average_val_loss = temp

    return agent, train_loss, val_loss[1:]

if __name__ == "__main__":

    model_dir="./model"
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # read data    
    X_train, Y_train, X_valid, Y_valid = read_data()
    print("Training:", len(X_train), "samples")
    print("Validation:", len(X_valid), "samples")
    # preprocess data
    X_train, Y_train, X_valid, Y_valid = preprocessing(X_train, Y_train, X_valid, Y_valid)

    batch_size = 64
    train_loader = DataLoader(MyDataset(X_train, Y_train), batch_size=batch_size)
    val_loader = DataLoader(MyDataset(X_valid, Y_valid), batch_size=batch_size)
    # train model (you can change the parameters!)
    agent,train_loss,val_loss = train_model(train_loader, val_loader, epochs=epochs, lr=lr, momentum=momentum, log_interval=log_interval)
    print("Done")
    # Plot of loss for training / validation
    plt.plot(train_loss, label="Training")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    torch.save(agent.state_dict(), save_path)
 
