import os
from easydict import EasyDict as edict

config = edict()

config.name = "base_gru"
config.batch_size = 32
config.epoch = 100
config.lr = 1e-4
config.wd = 5e-5
config.hidden_size = 1024
config.model_name = "base_gru"
config.band_width = 40
config.num_classes = 47
config.print_freq = 10
config.val_freq = 2
config.save_freq = 4
config.MODEL_DIR = 'saved_models'
config.MODEL_FOLDER_NAME = 'base_gru'
config.MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME)
config.optim = "adam"
