import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt


# def vgg_block(num_convs, out_channels):
#     layers = []
#     for _ in range(num_convs):
#         layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
#         layers.append(nn.ReLU())
#     layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#     return nn.Sequential(*layers)


class Conv2dMasked(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros", device=None, dtype=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, input):
        W = self.weight * self.weight_mask
        return F.conv2d(input, W, stride=self.stride, padding=self.padding, dilation=self.dilation)

class LinearMasked(nn.Linear):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__(in_f, out_f, bias)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))

    def forward(self, x):
        W = self.weight * self.weight_mask
        return F.linear(x, W, self.bias)

class VGG(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            Conv2dMasked(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            Conv2dMasked(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            Conv2dMasked(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            Conv2dMasked(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            Conv2dMasked(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Conv2dMasked(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearMasked(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            LinearMasked(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
