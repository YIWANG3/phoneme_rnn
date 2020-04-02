import os
from easydict import EasyDict as edict

config = edict()

config.name = "gru_4_hidden_1024"
config.batch_size = 32
config.epoch = 100
config.lr = 1e-4
config.wd = 5e-5
config.hidden_size = 1024
config.model_name = "fourgru"
config.band_width = 40
config.num_classes = 47
config.print_freq = 10
config.val_freq = 2
config.save_freq = 4
config.MODEL_DIR = 'saved_models'
config.MODEL_FOLDER_NAME = 'gru_4_hidden_1024'
config.MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME)
config.optim = "adam"
