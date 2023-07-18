import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMOrg(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(LSTMOrg, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, align_feat, prev_hidden, prev_cell):
        '''
            align_feat is input feat from PCD alignment
            prev_hidden
            prev_cell is none
        '''

        # get batch and spatial sizes
        batch_size = align_feat.data.size()[0]
        spatial_size = align_feat.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))

        # print(prev_hidden.size())

        prev_hidden = prev_hidden.cuda().detach()
        prev_cell = prev_cell.cuda().detach()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((align_feat, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)  # 对gate进行分块

        # apply sigmoid non linearity  use
        in_gate = self.lrelu(in_gate)  # i_n
        remember_gate = self.lrelu(remember_gate)  # f_n
        out_gate = self.lrelu(out_gate)  # 最右边的sigmoid

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)  # 最下面的tanh

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        # compute current cell and hidden state
        hidden = out_gate * torch.tanh(cell)
        cell = cell + prev_cell
        return hidden, hidden, cell


class LSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, align_feat, input_weight, prev_hidden, prev_cell):  # align_feat is Sm(Yn)
        '''
            align_feat is input feat from PCD alignment
            input_weight is 来自空间网络的权重
            prev_hidden
            prev_cell is none
        '''

        # get batch and spatial sizes
        batch_size = align_feat.data.size()[0]
        spatial_size = align_feat.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))

        prev_hidden = prev_hidden.cuda().detach()
        prev_cell = prev_cell.cuda().detach()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((align_feat, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)  # 对gate进行分块

        # apply sigmoid non linearity  use
        in_gate = self.lrelu(in_gate)  # i_n
        remember_gate = self.lrelu(remember_gate)  # f_n
        out_gate = self.lrelu(out_gate)  # 最右边的sigmoid

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)  # 最下面的tanh

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        # compute current cell and hidden state
        hidden = out_gate * torch.tanh(cell)
        cell = cell + prev_cell
        return hidden, hidden, cell  # hidden is copy double


class BiLstm(nn.Module):
    '''
        双向lstm帧间融合，对齐
    '''
    def __init__(self, input_size=64, hidden_size=64, batch_first=True):
        super(BiLstm, self).__init__()
        self.ltsm_1 = LSTM(input_size, hidden_size, batch_first, babidirectional=True)
        self.ltsm_2 = LSTM(input_size, hidden_size, batch_first, babidirectional=True)

    def forward(self, feat):
        output, h_n, c_n = self.ltsm_1(feat)
        output = self.ltsm_2(output)
        return output


# x_out = nn.randint(1, 1, 96, 96)
#
# def generate_weight(x_out):
#     u_weight = x_out.unsqueeze(-1)
#     u_weight = x_out.unsqueeze(-1)
#     u_weight = x_out.unsqueeze(-1)
#
#     u_weight = torch.repeat_interleave(u_weight, repeats=4, dim=0)
#     f_weight = torch.ones()-u_weight
#     return f_weight, u_weight


a = torch.arange(0, 256)
a = a.unsqueeze(0)
a = a.unsqueeze(1)
a = a.unsqueeze(2)
new_a = torch.reshape(a, (1, 16, 4, 4))
batch_size = new_a.data.size()[2]  # new_a对应维数的数量
spatial_size = new_a.data.size()[2:]  # [4, 4]
print(batch_size)
print(spatial_size)


