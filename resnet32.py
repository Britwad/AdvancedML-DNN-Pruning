# based on https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

# use kaiming normal to initialise Conv2d and Linear layer weights
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)    
    
class BasicBlock (nn.Module):
    expansion = 1

    def __init__ (self, in_planes, planes, stride = 1, option = 'A'):
        super(BasicBlock, self).__init__()

        # two 3x3 convolutions with batch normalisation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False);
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # downsample & zero-padding (CIFAR10 style)
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            # 1x1 convolution (more general)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    # standard residual block logic
    # conv -> conv -> add shortcut -> ReLU (rectified linear unit)
    def forward (self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out) # outputs the input directly if positive; 0 otherwise
        return out
    
class ResNet (nn.Module):
    def __init__ (self, block, num_blocks, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # conv1 = initial revolution from 3 input channels (RGB) to 16 feature maps
        # layer 1, 2, 3 = residual layers with increasing channels (16 -> 32 -> 64)
        # linear = final fully connected layer for classificiation
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    # builds a sequence of residual blocks  
    def _make_layer (self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    # standard ResNet foward
    # initial conv & ReLU -> pass through residual layers -> global average pooling (avg_pool2d over HxW) -> flatten & linear layer
    def forward (self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = self.linear(out)
        return out
    
    # creates 32-layer ResNet for CIFAR10 (3 layers, 5 blocks each)
    def resnet32():
        return ResNet(BasicBlock, [5, 5, 5])