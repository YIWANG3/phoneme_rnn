from __future__ import absolute_import
import torch

__all__ = ['BaseLSTM']


class BaseLSTM(torch.nn.Module):
    def __init__(self, hidden_size):
        super(BaseLSTM, self).__init__()
        self.encoder = torch.nn.LSTM(40, hidden_size, bidirectional=True, num_layers=5, dropout=0.1)
        self.output = torch.nn.Linear(hidden_size * 2, 47)

    def forward(self, X, lengths):
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.encoder(packed_X)[0]
        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)
        out = self.output(out).log_softmax(2)
        return out, out_lens
