from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

# from NTIRE2021 VIP&DJI Team track1


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class QEnetwork(nn.Module):
    def __init__(self, in_nc=64, nf=64, out_nc=64, nb=5):
        super(QEnetwork, self).__init__()
        self.InputConv = InputConv(in_nc, nf, 2)
        self.ECA = nn.ModuleList([ECA(nf) for _ in range(21)])
        self.DownConv = nn.ModuleList([DownConv(nf, nf, nb) for _ in range(3)])
        self.UpConv_64 = nn.ModuleList([UpConv(nf*2, nf, nb) for _ in range(3)])  # 该模块有三个
        self.UpConv_96 = nn.ModuleList([UpConv(nf*3, nf, nb) for _ in range(2)])
        self.UpConv_128 = UpConv(nf*4, nf, nb)
        self.OutConv = OutConv(nf, out_nc)

    def forward(self, x):
        x00 = self.InputConv(x)

        x10 = self.DownConv[0](self.ECA[0](x00))
        x01 = self.UpConv_64[0](self.ECA[1](x00),
                                self.ECA[2](x10))

        x20 = self.DownConv[1](self.ECA[4](x10))
        x11 = self.UpConv_64[1](self.ECA[5](x10),
                                self.ECA[6](x20))
        x02 = self.UpConv_96[0](torch.cat((self.ECA[7](x00),
                                           self.ECA[8](x01)), dim=1),
                                self.ECA[9](x11))
        print("x20 :{}".format(x20.size()))  # x20 :torch.Size([1, 64, 160, 265])
        print("x20 ECA :{}".format(self.ECA[11](x20).size()))  # ([1, 64, 160, 265])
        x30 = self.DownConv[2](self.ECA[11](x20))  # x30 :torch.Size([1, 64, 80, 133])
        print("x30 :{}".format(x30.size()))  # # [1, 64, 80, 133]
        print("x20 ECA :{}".format(self.ECA[12](x20).size()))
        print("x30 ECA :{}".format(self.ECA[13](x30).size()))  # [1, 64, 80, 133]
        x21 = self.UpConv_64[2](self.ECA[12](x20),
                                self.ECA[13](x30))  # bug==========================
        x12 = self.UpConv_96[1](torch.cat((self.ECA[14](x10),
                                           self.ECA[15](x11)), dim=1),
                                self.ECA[16](x21))
        x03 = self.UpConv_128(torch.cat((self.ECA[17](x00),
                                         self.ECA[18](x01),
                                         self.ECA[19](x02)), dim=1),
                              self.ECA[20](x12))

        out = self.OutConv(x03)
        return out


class InputConv(nn.Module):
    def __init__(self, in_nc, outch, nb=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_nc, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBN(outch))
        self.InputConv = nn.Sequential(*body)

    def forward(self, x):
        x = self.head(x)
        out = self.InputConv(x)
        out = out + x
        return out


class DownConv(nn.Module):
    def __init__(self, inch, outch, nb=5):
        super().__init__()
        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBN(inch))
        self.body = nn.Sequential(*body)
        self.DownConv = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.body(x)
        y = y + x  # ([1, 64, 160, 265])
        return self.DownConv(y)  # torch.Size([1, 64, 80, 133]) 向上取整


class UpConv(nn.Module):
    def __init__(self, inch, outch, nb=5):
        super().__init__()
        self.UpConv = nn.Sequential(
            nn.Conv2d(outch, outch*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.head = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        body = []
        for i in range(1, nb):
            body.append(ResidualBlock_noBN(outch))
        self.conv = nn.Sequential(*body)

    def forward(self, x, x_l):
        x_l = self.UpConv(x_l)
        y = self.head(torch.cat((x, x_l), dim=1))  # bug
        out = self.conv(y)
        return out + y


class ECA(nn.Module):
    def __init__(self, inch):
        super().__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1)  # B inch 1 1
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.size()
        y = self.ave_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class OutConv(nn.Module):
    def __init__(self, inch, out_nc):
        super(OutConv, self).__init__()
        self.OutConv = nn.Conv2d(inch, out_nc, 3, 1, 1)

    def forward(self, x):
        return self.OutConv(x)


if __name__ == '__main__':
    inputIn = torch.ones(8, 5, 64, 96, 96)
    inputIn = inputIn.view(-1, 64, 96, 96)  # [8,64,96,96], [8,64,96,96]
    DPM = QEnetwork()
    print(DPM(inputIn).shape)

# ==========
# Spatio-temporal deformable fusion module
# ==========

