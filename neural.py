from torch import nn
import copy

class CNN(nn.Module):
    def __init__(self, output_dim, if_target=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        if(if_target==True):
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, input):
        return self.net(input)
