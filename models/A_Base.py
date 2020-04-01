from __future__ import absolute_import
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils

__all__ = ['A_Base']


class A_Base(torch.nn.Module):
    def __init__(self, hidden_size):
        super(A_Base, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=(hidden_size >> 2), kernel_size=3, padding=0, stride=2, bias=False),
            nn.BatchNorm1d(num_features=(hidden_size >> 2)),
            nn.ELU(),
            nn.Conv1d(in_channels=(hidden_size >> 2), out_channels=(hidden_size >> 1), kernel_size=3, padding=1,
                      stride=1, bias=False),
            nn.BatchNorm1d(num_features=(hidden_size >> 1)),
            nn.ELU(),
        )

        self.encoder = nn.Sequential(
            torch.nn.LSTM(hidden_size >> 1, hidden_size, bidirectional=True),
            torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True),
            torch.nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        )

        self.transformer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
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
        return ((utterance_len_batch - 3) // 2) + 1

    def forward(self, utterance_batch, utterance_len_batch):
        features_seq_batch = self.feature_extractor(utterance_batch.T)
        features_seq_len_batch = self.calc_features_seq_len_batch(utterance_len_batch.T)
        hidden_states_batch = self.encoder(
            rnn_utils.pack_padded_sequence(features_seq_batch.permute(2, 0, 1), features_seq_len_batch)
        )

        if isinstance(hidden_states_batch, rnn_utils.PackedSequence):
            hidden_states_batch, _ = rnn_utils.pad_packed_sequence(
                hidden_states_batch
            )
        return self.classifier(self.transformer(hidden_states_batch)), features_seq_len_batch.int()
