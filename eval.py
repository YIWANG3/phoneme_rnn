import importlib
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn

from lib.parse_args import parse
import models
import torch.optim as optim
import time
import torch
from ctcdecode import CTCBeamDecoder
from data.phoneme_list import PHONEME_MAP
import Levenshtein

CONFIG = None

device = torch.device("cuda")


class WSJ():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.dev_set = None
        self.train_set = None
        self.test_set = None

    @property
    def dev(self):
        if self.dev_set is None:
            self.dev_set = load_data(self.data_folder, 'wsj0_dev')
        return self.dev_set

    @property
    def train(self):
        if self.train_set is None:
            self.train_set = load_data(self.data_folder, 'wsj0_train')
        return self.train_set

    @property
    def test(self):
        if self.test_set is None:
            self.test_set = (
                np.load(os.path.join(self.data_folder, 'wsj0_test.npy'), encoding='bytes', allow_pickle=True),
                None)
        return self.test_set


def load_data(path, name):
    return (
        np.load(os.path.join(path, f'{name}.npy'), encoding='bytes', allow_pickle=True),
        np.load(os.path.join(path, f'{name}_merged_labels.npy'), encoding='bytes', allow_pickle=True)
    )


class MyDataset(Dataset):
    def __init__(self, features, labels, train_mode=True):
        self.train_mode = train_mode
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.train_mode:
            return self.features[index], self.labels[index]
        else:
            return self.features[index]


def custom_collate_train(batch):
    lst = []
    for samples in batch:
        cur_X = samples[0]
        cur_Y = samples[1]
        lst.append((cur_X, cur_Y))
    return lst


def custom_collate_test(batch):
    lstX = []

    for samples in batch:
        cur_X = samples
        lstX.append(cur_X)
    return lstX


def prepare_data():
    loader = WSJ("data")
    train_X, train_Y = loader.train
    train_loader = DataLoader(MyDataset(train_X, train_Y), shuffle=True, batch_size=CONFIG.batch_size,
                              collate_fn=custom_collate_train)

    dev_X, dev_Y = loader.dev
    dev_loader = DataLoader(MyDataset(dev_X, dev_Y), shuffle=False, batch_size=CONFIG.batch_size,
                            collate_fn=custom_collate_train)

    test_X, _ = loader.test
    test_loader = DataLoader(MyDataset(test_X, None, train_mode=False), shuffle=False, batch_size=CONFIG.batch_size,
                             collate_fn=custom_collate_test)
    return train_loader, dev_loader, test_loader


def process_train_lst(lst):
    X = [torch.FloatTensor([iv for iv in spell]) for spell, _ in lst]
    Y = [torch.LongTensor([ov for ov in speak]) for _, speak in lst]
    X_lens = torch.LongTensor([len(seq) for seq in X])
    Y_lens = torch.LongTensor([len(seq) for seq in Y])

    X = torch.nn.utils.rnn.pad_sequence(X)
    Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True)

    X = X.cuda()
    Y = Y.cuda()
    X_lens = X_lens.cuda()
    Y_lens = Y_lens.cuda()
    return X, X_lens, Y, Y_lens


def convert_to_phoneme(batch):
    phoneme_list = []
    for cur in batch:
        cur_line = ""
        for i in range(cur.shape[0]):
            if cur[i] >= 47:
                continue
            else:
                index = cur[i]
                cur_line = cur_line + PHONEME_MAP[index]
        phoneme_list.append(cur_line)
    return phoneme_list


def validate(model, dev_loader):
    decoder = CTCBeamDecoder(['$'] * 47, beam_width=100, log_probs_input=True, blank_id=46)
    with torch.no_grad():
        model.eval()
        model.cuda()
        count = 0
        dist_sum = 0
        for batch_idx, lst in enumerate(dev_loader):
            X, X_lens, Y, Y_lens = process_train_lst(lst)
            out, out_lens = model(X, X_lens)
            val_Y, _, _, val_Y_lens = decoder.decode(out.transpose(0, 1), out_lens)
            this_batch_size = val_Y.shape[0]

            predicted_list = [val_Y[i, 0, : val_Y_lens[i, 0]] for i in range(this_batch_size)]
            ground_truth_list = [Y[i, 0:Y_lens[i]] for i in range(this_batch_size)]
            ground_truth_phoneme_list = convert_to_phoneme(ground_truth_list)
            predicted_phoneme_list = convert_to_phoneme(predicted_list)

            for i in range(len(predicted_list)):
                count += 1
                cur_predicted_str = "".join(predicted_phoneme_list[i])
                cur_label_str = "".join(ground_truth_phoneme_list[i])
                cur_dist = Levenshtein.distance(cur_predicted_str, cur_label_str)
                dist_sum += cur_dist
            print(f"Batch: {batch_idx} | Avg Distance: {dist_sum / count}")
        print("Dev Avg Distance: {:.4f}".format(dist_sum / count))


def eval(model_path):
    train_loader, dev_loader, test_loader = prepare_data()
    model = torch.load(model_path)
    model.cuda()
    model.eval()
    validate(model, dev_loader)


if __name__ == "__main__":
    config_name = parse()
    print("Use config: " + config_name)
    CONFIG = (importlib.import_module("config." + config_name)).config

    model_path = os.path.join(CONFIG.MODEL_DIR,
                              "lstm_3_hidden_1024/lstm_3_hidden_1024__EP-4__04-02_22-52-41__7.241410488245931___c.model")
    eval(model_path)
