import os
from easydict import EasyDict as edict

config = edict()

config.name = "baseline"
config.batch_size = 64
config.epoch = 100
config.lr = 2e-4
config.wd = 5e-5
config.hidden_size = 400
config.model_name = "baseline"
config.band_width = 40
config.num_classes = 47
config.print_freq = 10
config.val_freq = 5
config.save_freq = 1
config.MODEL_DIR = 'saved_models'
config.MODEL_FOLDER_NAME = 'baseline'
config.MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME)
config.optim = "sgd"
