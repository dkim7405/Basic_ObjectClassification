import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init()

        ###
        ### YOUR CODE HERE
        ###

        self._body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=300, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self._head = nn.Sequential(
            nn.Linear(in_features=10800, out_features=256),
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=10)
        )
        
    def forward(self, x): 
        x = self._body(x)
        x = x.view(x.size()[0], -1)
        x = self._head(x)
        
        return x
