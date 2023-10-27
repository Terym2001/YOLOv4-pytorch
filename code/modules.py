import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              stride=stride,
                              kernel_size=kernel_size,
                              padding=padding
                            ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.conv(x)
        return x

class CSPDarkNet53(nn.Module):
    def __init__(self, in_channels=3):
        super(CSPDarkNet53, self)

        self.cdb = nn.Sequential(
            # layer 1
            Conv(in_channels, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 2
            Conv(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # layer 3
            Conv(64, 128, kernel_size=3),
            Conv(128, 64, kernel_size=1),
            Conv(64, 128, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # layer 4
            Conv(128, 256, kernel_size=3),
            Conv(256, 128, kernel_size=1),
            Conv(128, 256, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.one_path = nn.Sequential(
            Conv(256, 512, kernel_size=3),
            Conv(512, 512, kernel_size=1),
            Conv(256, 512, kernel_size=3),
            Conv(512, 512, kernel_size=1),
        )

class YOLOv4(nn.Module):

    def __init__(self):
        super(YOLOv4, self).__init__()

