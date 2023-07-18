import torch
import torch.nn as nn
from torch.nn import PixelUnshuffle


class pixeldowm(nn.Module):
    def __init__(self, n_channels=4):
        super(pixeldowm, self).__init__()
        self.downscale_2 = nn.Sequential(
            PixelUnshuffle(downscale_factor=2),
            nn.Conv2d(in_channels=n_channels * 2 * 2, out_channels=n_channels,
                      kernel_size=5, stride=1, padding=2, bias=False)
        )

    def forward(self, x):
        out = self.downscale_2(x)
        return out


class maxpooldown(nn.Module):
    def __init__(self, scale=2, n_channel=64):
        super(maxpooldown, self).__init__()
        # if scale == 1:
        #     self.convD = nn.Conv2d(n_channel, n_channel, 7, padding=7 // 2, stride=1)
        # else:
        self.convD = nn.Conv2d(n_channel, n_channel, 7, padding=7 // 2, stride=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv = nn.Conv2d(n_channel, n_channel, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # input: b * 64 * 64 *64
        # return: b * 64 * 16 * 16
        out = self.convD(x)
        out = self.pool(out)
        out = self.lrelu(self.conv(out))
        return out


class conv(nn.Module):
    def __init__(self, n_channel=4):
        super(conv, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, n_channel, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        return self.conv1(x)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=32, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DenseLayer(nn.Module):
    """ 定义denseNet的思想，使得密集块内的任何两层之间的直接连接 """
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # return torch.cat([x, self.relu(self.conv(x))], 1)
        # return self.relu(self.conv(x))
        out = self.conv(self.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

if __name__ == '__main__':
    input = torch.ones(1, 64, 4, 4)
    pixeldowmscale = pixeldowm()
    # print(pixeldowmscale(input))
    maxpooldownscale = maxpooldown()
    print(maxpooldownscale(input).size())
    # conv2d = conv()
    # print(conv2d(input))
    # rcan = CALayer(channel=32)
    # print(rcan(input))
    # resd = ResidualBlockNoBN()
    # print(resd(input))
    # DenLay = DenseLayer(32, 32)
    # print(DenLay(input).size())