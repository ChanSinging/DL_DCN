import torch
import torch.nn as nn

class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=192, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class upscale(nn.Module):
    def __init__(self, n_channels=64):
        super(upscale, self).__init__()
        self.upscale = nn.Sequential(
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 4,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.upscale(x)


# input torch.Size([1, 64, 24, 24])
x = torch.ones(1, 64, 24, 24)
# high frequency part
avg3 = nn.AvgPool2d(3, stride=2, padding=3//2)
avg5 = nn.AvgPool2d(5, stride=2, padding=5//2)
avg7 = nn.AvgPool2d(7, stride=2, padding=7//2)
_t3 = avg3(x)  # torch.Size([1, 64, 12, 12])
_t5 = avg5(x)  # torch.Size([1, 64, 11, 11])
_t7 = avg7(x)  # torch.Size([1, 64, 10, 10])
_t = torch.cat([_t3, _t5, _t7], dim=1)  # torch.Size([1, 192, 12, 12]) high frequency part output


ResBlock = ResidualBlockNoBN()
y = ResBlock.forward(_t)
pixelshuffle = nn.PixelShuffle(2)
y = pixelshuffle(y)
print(y.shape)
x_high = torch.cat([y, x], dim=1)  # torch.Size([1, 112, 24, 24])


print('low low frequency part')
upsc = upscale()
_t3 = upsc.forward(_t3)
_xd3 = torch.subtract(x, _t3)

_t5 = upsc.forward(_t5)
_xd5 = torch.subtract(x, _t5)

_t7 = upsc.forward(_t7)
_xd7 = torch.subtract(x, _t7)

_xd = torch.cat([_xd3, _xd5, _xd7], dim=1)
_xd = ResBlock.forward(_xd)
_xd = torch.cat([x, _xd], dim=1)  # low frequency part output
print(_xd.shape)






