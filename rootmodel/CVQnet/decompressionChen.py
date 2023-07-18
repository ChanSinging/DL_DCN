import torch
import torch.nn as nn
# Decompression processing module


class DenseLayer(nn.Module):
    """ 定义denseNet的思想，使得密集块内的任何两层之间的直接连接 """
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels  # =64
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """UpScaling with PixelShuffle v"""

    def __init__(self):
        super(Up, self).__init__()
        self.pixel_conv = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_conv(x)


class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)  # bs,c,1,1
        # print(y.squeeze(-1).shape)
        y = y.squeeze(-1).permute(0, 2, 1)  # bs,1,c , 把个数是1的维度删除
        y = self.conv(y)  # bs,1,c
        y = self.sigmoid(y)  # bs,1,c
        y = y.permute(0, 2, 1).unsqueeze(-1)  # bs,c,1,1
        return x*y.expand_as(x)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Conv(nn.Module):
    def __init__(self, num_feat):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class DPModule(nn.Module):  # min module
    def __init__(self, num_feat):
        super(DPModule, self).__init__()
        self.eca = ECAAttention(kernel_size=3)
        self.resBlock = ResidualBlockNoBN(num_feat)
        self.conv = Conv(num_feat=num_feat)

    def forward(self, x):
        x = self.eca(x)
        x = self.resBlock(x)
        logits = self.conv(x)
        return logits


class DPM(nn.Module):
    def __init__(self, n_channel=64):
        super(DPM, self).__init__()
        self.channel = n_channel

        self.DenseLayer11_12 = DenseLayer(in_channels=64, out_channels=64)
        self.DenseLayer21_22 = DenseLayer(in_channels=128, out_channels=128)
        self.DenseLayer12_13 = DenseLayer(in_channels=160, out_channels=160)

        self.DPM_11 = DPModule(num_feat=64)
        self.DPM_21 = DPModule(num_feat=128)
        self.DPM_31 = DPModule(num_feat=256)
        self.DPM_12 = DPModule(num_feat=160)
        self.DPM_22 = DPModule(num_feat=256)
        self.DPM_13 = DPModule(num_feat=512)

        self.inc = Conv(num_feat=64)
        # using same channel number
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(160, 192)

        self.up1 = Up()  # 256
        self.up2 = Up()  # 128
        self.up21_12 = Up()  # 64
        self.up3 = Up()  # 128
        self.out = nn.Conv2d(512, 64, kernel_size=1)

    def forward(self, x):  # x = [8,64,96,96]
        x1 = self.inc(x)  # [8,64,96,96]
        # downScale
        x1 = self.DPM_11(x1)  # [40, 64, 96, 96]
        # denseLayer

        # dense_x2 = self.DenseLayer12_13(dense_x1)  # [40, 256, 96, 96]
        x2 = self.down1(x1)  # [40, 128, 48, 48]
        x2 = self.DPM_21(x2)  # [40, 128, 48, 48]

        x3 = self.down2(x2)  # [40, 256, 24, 24]
        x3 = self.DPM_31(x3)  # [40, 256, 24, 24]

        # cross M12
        dense_x1 = self.DenseLayer11_12(x1)  # [40, 128, 96, 96]
        temp_x2 = self.up3(x2)  # [40, 32, 96, 96]
        _M12_in = torch.cat((dense_x1, temp_x2), dim=1)  # [40, 160, 96, 96]
        _M12_out = self.DPM_12(_M12_in)  # # [40, 160, 96, 96]

        # cross M22
        _x3 = self.up1(x3)  # [40, 64, 48, 48]
        _dense_x1 = self.down3(_M12_out)  # [40, 192, 48, 48] out_ch=192
        _x2 = self.DenseLayer21_22(x2)  # [40, 256, 48, 48]
        _x3 = torch.cat((_x3, _dense_x1), dim=1)  # [40, 256, 48, 48]
        _x3 = torch.add(_x3, _x2)  # [40, 256, 48, 48]
        _x3 = self.DPM_22(_x3)  # [40, 256, 48, 48]

        # cross M13
        _M12_out_dense = self.DenseLayer12_13(_M12_out)  # [40, 320, 96, 96]
        _M22_out_up = self.up3(_x3)  # # [40, 64, 48, 48]
        dense_x1 = dense_x1  # # [40, 128, 96, 96]
        _M13_in = torch.cat((_M12_out_dense, _M22_out_up, dense_x1), dim=1)
        _M13_out = self.DPM_13(_M13_in)  # [40, 512, 96, 96]

        logits = self.out(_M13_out)
        return logits


if __name__ == '__main__':
    inputIn = torch.ones(8, 5, 64, 96, 96)
    inputIn = inputIn.view(-1, 64, 96, 96)  # [8,64,96,96], [8,64,96,96]
    DPM = DPM()
    print(DPM(inputIn).shape)






