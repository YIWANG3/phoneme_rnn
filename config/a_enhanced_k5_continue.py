import os
from easydict import EasyDict as edict

config = edict()

config.name = "a_enhanced_k5"
config.batch_size = 64
config.epoch = 100
config.lr = 2e-4
config.wd = 5e-5
config.hidden_size = 800
config.model_name = "a_enhanced_k5"
config.band_width = 40
config.num_classes = 47
config.print_freq = 10
config.val_freq = 1
config.save_freq = 2
config.MODEL_DIR = 'saved_models'
config.MODEL_FOLDER_NAME = 'a_enhanced_k5'
config.MODEL_SAVE_PATH = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME)
config.optim = "adam"
config.continue_model_path = os.path.join(config.MODEL_DIR, config.MODEL_FOLDER_NAME,
                                          "a_enhanced_k5__EP-8__04-02_20-50-16__9.040687160940326__.model")