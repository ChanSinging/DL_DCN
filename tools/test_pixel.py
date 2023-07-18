import numpy as np
from torch import nn, autograd
import torch

# 传过来的数是一维，
# 生成1*4*16*16的张量
a = torch.arange(0, 256)
a = a.unsqueeze(0)
a = a.unsqueeze(1)
a = a.unsqueeze(2)
# 在第二个维度添加16个，改变形状

# new_a = torch.repeat_interleave(a, repeats=8, dim=3)
# [1, 256, 24, 24]--->[1,16,8,8]
new_a = torch.reshape(a, (1, 16, 4, 4))
print(new_a)
print("============")

ps = nn.PixelShuffle(2)
out = ps(new_a)
print(out, out.size())

