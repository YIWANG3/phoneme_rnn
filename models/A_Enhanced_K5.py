from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

__all__ = ['A_Enhanced_K5']


class A_Enhanced_K5(torch.nn.Module):
    def __init__(self, hidden_size):
        super(A_Enhanced_K5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=(hidden_size >> 2), kernel_size=5, padding=0, stride=2, bias=False),
            nn.BatchNorm1d(num_features=(hidden_size >> 2)),
            nn.ELU(),
            nn.Conv1d(in_channels=(hidden_size >> 2), out_channels=(hidden_size >> 1), kernel_size=5, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm1d(num_features=(hidden_size >> 1)),
            nn.ELU(),
        )

        self.lstm1 = torch.nn.LSTM(hidden_size >> 1, hidden_size, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.lstm3 = torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)

        self.transformer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.SELU()
        )

        self.classifier = nn.Linear(
            in_features=hidden_size,
            out_features=47
        )

    def calc_features_seq_len_batch(self, utterance_len_batch):
        return ((utterance_len_batch - 10) // 2) + 1

    def forward(self, utterance_batch, utterance_len_batch):
        utterance_batch = utterance_batch.permute(1, 2, 0)
        features_seq_batch = self.feature_extractor(utterance_batch)
        features_seq_len_batch = self.calc_features_seq_len_batch(utterance_len_batch)
        reshaped_features_seq_batch = features_seq_batch.permute(2, 0, 1)
        packed_X = torch.nn.utils.rnn.pack_padded_sequence(reshaped_features_seq_batch, features_seq_len_batch,
                                                           enforce_sorted=False)

        packed_out = self.lstm1(packed_X)[0]
        packed_out = self.lstm2(packed_out)[0]
        packed_out = self.lstm3(packed_out)[0]

        out, out_lens = torch.nn.utils.rnn.pad_packed_sequence(packed_out)

        return self.classifier(self.transformer(out)), out_lens.int()
