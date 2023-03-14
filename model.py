# model 1 : CNN + Fully-connected layer #

import torch
import torch.nn as nn
import config as cfg

input_channels = 3
feature1_channels = 8
feature2_channels = 16
feature3_channels = 32
feature4_channels = 64
kernel_size = 5
pooling_size = 2
device = cfg.device

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        

        # define its modules
        self.module1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(input_channels, feature1_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature1_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(pooling_size)
        ) # output (8, 98, 98)

        self.module2 = nn.Sequential(
            nn.Conv2d(feature1_channels, feature1_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature1_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature1_channels, feature2_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature2_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(pooling_size)
        ) # output (16, 47, 47)

        self.module3 = nn.Sequential(
            nn.Conv2d(feature2_channels, feature2_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature2_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature2_channels, feature3_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature3_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(pooling_size)
        ) # output (32, 21, 21)

        self.module4 = nn.Sequential(
            nn.Conv2d(feature3_channels, feature3_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature3_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature3_channels, feature4_channels, kernel_size, padding = 1),
            nn.BatchNorm2d(feature4_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(pooling_size)
        ) # output (64, 8, 8)

        self.linearlayer = nn.Sequential(
            nn.Linear(4096, 2000),
            nn.Linear(2000, 500),
            nn.Linear(500, 50)
        )


    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        b, c, h, w = x.shape
        x = x.reshape((b, c *h *w)) # flatten
        #x = torch.flatten(x)
        x = self.linearlayer(x)
        return x
    


if __name__ == "__main__":
    model = CNN().to(device)
    FC = fullyconnected().to(device)
    data = torch.zeros((1, 3, 200, 200)).to(device)

    feature = model(data)
    print("feature.shape=", feature.shape)
    output = FC(feature)
    print("output.shape=", output.shape)
