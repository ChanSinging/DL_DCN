import torch
import torch.nn as nn
import math
from utils import *

channels = 64


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                               bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        # output *= 0.1
        output = torch.add(output, identity_data)
        return output


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()

        self.downscale = PixelUnshuffle(downscale_factor=2)

        self.conv_input = nn.Conv2d(in_channels=3 * 4, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                    bias=False)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1,
                                  bias=False)

        self.upscale2x = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

        self.conv_output = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1,
                                     bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downscale(x)
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        out = self.upscale2x(out)
        out = self.conv_output(out)
        return out
