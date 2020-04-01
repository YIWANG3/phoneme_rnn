from __future__ import absolute_import
import torch

__all__ = ['FourLstm']


class FourLstm(torch.nn.Module):
    def __init__(self, hidden_size):
        super(FourLstm, self).__init__()
        self.lstm1 = torch.nn.LSTM(40, hidden_size, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.lstm4 = torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, 47)

    def forward(self, X, lengths):
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm1(packed_X)[0]
        packed_out = self.lstm2(packed_out)[0]
        packed_out = self.lstm3(packed_out)[0]
        packed_out = self.lstm3(packed_out)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2)
        return out, out_lens
