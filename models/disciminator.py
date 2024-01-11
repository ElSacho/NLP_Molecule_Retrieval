import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, parameters):
        super(Discriminator, self).__init__()
        # Define the architecture of the discriminator
        self.main = nn.Sequential(
            nn.Linear(parameters['nout'], 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, x):
        # Forward pass through the network
        output = self.main(x)
        return output
