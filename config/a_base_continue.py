import os
from easydict import EasyDict as edict

config = edict()

config.name = "a_base"
config.batch_size = 64
config.epoch = 100
config.lr = 0.001
config.wd = 5e-5
config.hidden_size = 512
config.model_name = "a_base"
config.band_width = 40
config.num_classes = 47
config.print_freq = 10
config.val_freq = 5
config.save_freq = 1
config.MODEL_DIR = 'saved_models'
config.MODEL_FOLDER_NAME = 'a_base'
config.MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME)
config.optim = "sgd"
config.schedule = False
config.continue_model_path = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME,
                                          "a_base__EP-15__04-02_02-47-11__10.510849909584087__.model")
