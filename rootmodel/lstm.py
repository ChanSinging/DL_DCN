import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)
        self.lrelu = nn.Sigmoid()

    def forward(self, align_feat, prev_hidden, prev_cell):  # align_feat is Sm(Yn) 四维

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


class LSTMRen(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates_1 = nn.Conv2d(input_size, input_size, 3, padding=3 // 2)
        self.Gates_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=3 // 2)
        self.weight = nn.Conv2d(64, 64, 3, padding=3//2)
        self.sigmoid = nn.Sigmoid()  # can switch to

    def forward(self, align_feat, input_weight, prev_hidden, prev_cell):  # align_feat is Sm(Yn)

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
        # stacked_inputs = torch.cat((align_feat, prev_hidden), 1)
        gates_feat = self.Gates_1(align_feat)
        gates_hidden = self.Gates_2(prev_hidden)
        gates_weight = self.weight(input_weight)

        # apply sigmoid non linearity  use
        input_gate = torch.tanh(gates_feat)
        hidden_gate = self.sigmoid(gates_hidden)
        weight = self.sigmoid(gates_weight)

        # deal with f_n
        cell = weight*prev_cell

        # compute current cell and hidden state
        remember_gate = cell + input_gate*weight
        # output H_n
        hidden = torch.tanh(remember_gate) * hidden_gate
        return hidden, hidden, remember_gate  # hidden is copy double


class BiLstmRen(nn.Module):
    def __init__(self):
        super(BiLstm, self).__init__()
        self.ltsm_1 = LSTMRen()
        self.ltsm_2 = LSTMRen()

    def forward(self, feat, pre_hidden, pre_cell):
        output, h_n, c_n = self.ltsm_1(feat, pre_hidden, pre_cell)
        output, hidden, cell = self.ltsm_2(output, h_n, c_n)
        return output, hidden, cell


class BiLstm(nn.Module):
    def __init__(self):
        super(BiLstm, self).__init__()
        self.ltsm_1 = LSTM()
        self.ltsm_2 = LSTM()

    def forward(self, feat, pre_hidden, pre_cell):
        output, h_n, c_n = self.ltsm_1(feat, pre_hidden, pre_cell)
        output, hidden, cell = self.ltsm_2(output, h_n, c_n)
        return output, hidden, cell


class BiLstmDef(nn.Module):
    def __init__(self):
        super(BiLstmDef, self).__init__()
        self.ltsm = nn.LSTM(input_size=64, hidden_size=64, bidirectional=True)
        # self.fc = nn.Linear(128, 64)

    def forward(self, feat, pre_hidden, pre_cell):  # feat是四维张量  lstm需要三维的
        feat = feat[0, :, :, :]
        output, hidden, cell = self.ltsm(feat, (pre_hidden, pre_cell))
        return output, hidden, cell


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True, dropout=0.3)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        self.lstm1 = BidirectionalLSTM()  # LSTM()
        self.lstm2 = BidirectionalLSTM()
        self.lstm3 = BidirectionalLSTM()
        self.lstm4 = BidirectionalLSTM()
        self.lstm5 = BidirectionalLSTM()
        self.hidden = None
        self.cell = None
        # self.conv_meger_hidden = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.conv_meger_cell = nn.Conv2d(128, 64, kernel_size=3, padding=1)

    def forward(self, x):  # x [8,5,64,96,96] [8, 5, 64, 64, 64]
        x1 = x[:, 0, :, :, :]
        x2 = x[:, 1, :, :, :]
        x3 = x[:, 2, :, :, :]
        x4 = x[:, 3, :, :, :]
        x5 = x[:, 4, :, :, :]
        x1, hidden1, cell1 = self.lstm1(x1, self.hidden, self.cell)  # [8, 64, 64, 64]
        x2, hidden2, cell2 = self.lstm2(x2, hidden1, cell1)
        x3, hidden3, cell3 = self.lstm3(x3, hidden2, cell2)
        x4, hidden4, cell4 = self.lstm4(x4, hidden3, cell3)
        x5, hidden4, cell4 = self.lstm4(x5, hidden4, cell4)

        # hidden3 = torch.cat((hidden4, hidden2), dim=1)
        # hidden3 = self.conv_meger_hidden(hidden3)
        # cell3 = torch.cat((cell4, cell2), dim=1)
        # cell3 = self.conv_meger_cell(cell3)
        # x3, _, _ = self.lstm3(x3, hidden3, cell3)
        return x3




