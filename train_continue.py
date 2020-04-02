import importlib
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import lr_scheduler
from lib.parse_args import parse
from lib import file_opt
import models
import torch.optim as optim
import time
import torch
import pandas
from ctcdecode import CTCBeamDecoder
from data.phoneme_list import PHONEME_MAP
from lib.optimizers import init_optim

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


def process_test_lst(lst):
    X = [torch.FloatTensor([iv for iv in spell]) for spell in lst]
    X_lens = torch.LongTensor([len(seq) for seq in X])
    X = torch.nn.utils.rnn.pad_sequence(X)
    X = X.cuda()
    X_lens = X_lens.cuda()
    return X, X_lens


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    start_time = time.time()
    cnt = 0

    for batch_idx, lst in enumerate(train_loader):
        cnt += 1

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

        optimizer.zero_grad()  # .backward() accumulates gradients
        out, out_lens = model(X, X_lens)

        loss = criterion(out, Y, out_lens, Y_lens)
        running_loss += loss.item()

        if (batch_idx + 1) % CONFIG.print_freq == 0:
            print(CONFIG.model_name,
                  ' Epoch: [{0}][{1}/{2}]\t Total Loss {cur_loss:.4f} ({running_loss:.4f})\t'.format(
                      epoch, batch_idx + 1, len(train_loader), cur_loss=loss.item(), running_loss=running_loss / cnt))
        loss.backward()
        optimizer.step()

    end_time = time.time()
    running_loss /= cnt
    print(CONFIG.model_name, ' Training Loss: ', running_loss, 'Total Time: ', end_time - start_time, 's ')
    return running_loss


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
    decoder = CTCBeamDecoder(['$'] * 47, beam_width=100, log_probs_input=True)
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
            print(CONFIG.model_name, f" Validating... | Batch: {batch_idx} | Avg Distance: {dist_sum / count}")
        print(CONFIG.model_name, " Validation Finished | Avg Distance: {:.4f}".format(dist_sum / count))
        return dist_sum / count


def gen_model_name(epoch, precision):
    file_opt.create_folder(CONFIG.MODEL_SAVE_PATH)
    cur_time = time.strftime("%m-%d_%H-%M-%S", time.localtime())
    model_name = CONFIG.MODEL_FOLDER_NAME + "__EP-" + str(epoch) + "__" + cur_time + "__" + str(
        precision) + "__" + "_c.model"
    path_name = os.path.join(CONFIG.MODEL_SAVE_PATH, model_name)
    return model_name, path_name


def export_to_csv(label, label_key, data, data_key, path):
    result = pandas.DataFrame()
    result[label_key] = label
    result[data_key] = data
    result.to_csv(path, index=False)


def predict(model, test_loader, result_path):
    decoder = CTCBeamDecoder(['$'] * 47, beam_width=100, log_probs_input=True)
    with torch.no_grad():
        model.eval()
        model.cuda()
        predicted_list = []
        for batch_idx, lst in enumerate(test_loader):
            X, X_lens = process_test_lst(lst)
            out, out_lens = model(X, X_lens)
            test_Y, _, _, test_Y_lens = decoder.decode(out.transpose(0, 1), out_lens)
            this_batch_size = test_Y.shape[0]

            predicted_list += [test_Y[i, 0, : test_Y_lens[i, 0]] for i in range(this_batch_size)]
            print(CONFIG.model_name, f" Predicting... | Batch: {batch_idx}")
        predicted_phoneme_list = convert_to_phoneme(predicted_list)
        labels = [i for i in range(len(predicted_list))]
        export_to_csv(labels, "id", predicted_phoneme_list, "Predicted", result_path)
        print(CONFIG.model_name, " Validation Finished")


def run():
    train_loader, dev_loader, test_loader = prepare_data()
    print("Data Loaded")
    num_epoch = CONFIG.epoch

    model = models.init_model(name=CONFIG.model_name, hidden_size=CONFIG.hidden_size)
    if "continue_model_path" in CONFIG:
        print("Load ", CONFIG.continue_model_path)
        model = torch.load(CONFIG.continue_model_path)

    model.cuda()
    criterion = nn.CTCLoss()

    optimizer = init_optim("sgd", model.parameters(), 0.01, CONFIG.wd)

    if "schedule" in CONFIG and CONFIG.schedule:
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9)

    for i in range(1, num_epoch + 1):
        start = time.time()
        running_loss = train(i, model, train_loader, criterion, optimizer)
        if "schedule" in CONFIG and CONFIG.schedule:
            scheduler.step()

        if i % CONFIG.val_freq == 0:
            avg_distance = validate(model, dev_loader)
        else:
            avg_distance = 999999
        end = time.time()
        print(CONFIG.model_name, ' Train Epoch: {} | Loss: {:.4f} | AVG_DIST: {:.4f} | Cost: {:.4f}s'.format(
            i, running_loss, avg_distance, (end - start)))

        if i % CONFIG.save_freq == 0:
            start = time.time()
            model_name, path_name = gen_model_name(i, avg_distance)
            torch.save(model, path_name)
            # torch.save(model.state_dict(), path_name + ".state")
            predict(model, test_loader, path_name + ".csv")
            end = time.time()
            print(CONFIG.model_name, ' Save Model and Predict Epoch: {} | Cost: {:.4f}s'.format(i, (end - start)))


if __name__ == "__main__":
    config_name = parse()
    print("Use config: " + config_name)
    CONFIG = (importlib.import_module("config." + config_name)).config
    run()
