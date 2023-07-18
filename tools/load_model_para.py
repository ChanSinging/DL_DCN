import torch
# 引入torch.nn并指定别名
import torch.nn as nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--model', type=str, default='raw/EDVR_M_x4.pth',
                    help='model file to use')
opt = parser.parse_args()

# net = model = torch.load(opt.model, map_location='cpu')
save_model = torch.load('raw/EDVR_M_x4.pth')
state_dict = {k:v for k,v in save_model.items() if k in save_model.keys()}
# print(state_dict.keys())

# for k in state_dict.values():
#     with open('edvr.txt', 'w') as f:
#         f.write(str(k.items()))

'''odict_items  path is Collection.OrderedDict class

{'params': OrderedDict([('conv_first.weight', tensor([[[[-0.0203,  0.0433, -0.0341],
          [ 0.0360, -0.1656,  0.0601],
          [-0.0245,  0.1360, -0.0346]],

         [[ 0.0417, -0.0211,  0.0202],
          [-0.0877, -0.1982, -0.0556],
          [ 0.0508,  0.2178,  0.0569]],

         [[-0.0262,  0.0399,  0.0006],
          [ 0.0559, -0.1277,  0.0214],
          [-0.0295,  0.0669, -0.0048]]],

'''
for k in state_dict.values(): # value
    k["feature_extraction.0.conv1.weight"].requires_grad=False
    print('requires_grad success')
    print(k["feature_extraction.0.conv1.weight"])
    print('===========')
# print(net)
print()
