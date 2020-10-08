import torch
import torch.nn as nn


# Basic convolutional neural network, hand tuned for this mini-project
# Outputs a 3-dimension vector which is intended to be the action input 
# needed for OpenAI Gym CarRacing-v0 game.
class CNN_Agent(nn.Module):
    def __init__(self):
        super(CNN_Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*8*8, 100),
            #nn.ReLU(),
            nn.Linear(100, 3),
            #nn.Tanh()
        )

    def forward(self, x, verbose=False):

        if len(x.shape) == 2:
            x = x[None,...]

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        if verbose:
            print("Input", 
                  x.shape,
                  sep='\t\t|\t\t')
        total_params = 0
        
        for layer in self.net:
            x = layer(x)
            num_params = sum([p.numel() for p in layer.parameters()])
            total_params += num_params
            if verbose:
                print(layer.__class__.__name__, 
                      x.shape,
                      num_params,
                      sep='\t\t|\t\t')

        if verbose:
            print("Total Parameters:", total_params)
        return x

# Slightly more complex model. Similar to above CNN encodes input into feature vector. 1D convolution
# is then performed to give agent a sense of it's most immediate past actions (enough to get 800+ score, which
# was the goal of this mini-project). A couple more linear layers outputs a 3-dimension vector which is intended
# to be the action input needed for OpenAI Gym CarRacing-v0 game.
class CNN_History_Agent(nn.Module):
    def __init__(self, depth):
        super(CNN_History_Agent, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*8*8, 8*8),
            nn.ReLU()
            #nn.Linear(100, 3),
            #nn.Tanh()
        )
        self.conv1d = nn.Conv1d(1, 256, kernel_size=(depth, 64))
        self.net2 = nn.Sequential(
            nn.Linear(256, 1000),
            nn.ReLU(),
            nn.Linear(1000, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Tanh()
        )
        self.depth = depth
        
    def forward(self, x, verbose=False):

        if len(x.shape) == 2:
            x = x[None,...]

        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        if verbose:
            print("Input", 
                  x.shape,
                  sep='\t\t|\t\t')
        total_params = 0
        
        for layer in self.cnn:
            x = layer(x)
            num_params = sum([p.numel() for p in layer.parameters()])
            total_params += num_params
            if verbose:
                print(layer.__class__.__name__, 
                      x.shape,
                      num_params,
                      sep='\t\t|\t\t')

        # Pad feature vector to give action output for each frame in input
        # Only really necesarry for training to easily handle batches
        x = torch.cat((torch.zeros(self.depth-1, 64), x), dim=0)
        if verbose:
            print("Padding", 
                  x.shape,
                  num_params,
                  sep='\t\t|\t\t')
            
        x = x[None,None,...]
        x = self.conv1d(x)
        x = torch.transpose(x, 1, 3).squeeze()
        num_params = sum([p.numel() for p in self.conv1d.parameters()])
        total_params += num_params
        if verbose:
            print("Conv1d", 
                  x.shape,
                  num_params,
                  sep='\t\t|\t\t')

        for layer in self.net2:
            x = layer(x)
            num_params = sum([p.numel() for p in layer.parameters()])
            total_params += num_params
            if verbose:
                print(layer.__class__.__name__, 
                      x.shape,
                      num_params,
                      sep='\t\t|\t\t')
                
        if verbose:
            print("Total Parameters:", total_params)
        return x
