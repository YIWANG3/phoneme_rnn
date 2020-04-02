from __future__ import absolute_import
import torch

__all__ = ['FourGRU']


class FourGRU(torch.nn.Module):
    def __init__(self, hidden_size):
        super(FourGRU, self).__init__()
        self.gru1 = torch.nn.GRU(40, hidden_size, bidirectional=True)
        self.gru2 = torch.nn.GRU(hidden_size * 2, hidden_size, bidirectional=True)
        self.gru3 = torch.nn.GRU(hidden_size * 2, hidden_size, bidirectional=True)
        self.gru4 = torch.nn.GRU(hidden_size * 2, hidden_size, bidirectional=True)
        self.output = torch.nn.Linear(hidden_size * 2, 47)

    def forward(self, X, lengths):
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.gru1(packed_X)[0]
        packed_out = self.gru2(packed_out)[0]
        packed_out = self.gru3(packed_out)[0]
        packed_out = self.gru4(packed_out)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2)
        return out, out_lens
