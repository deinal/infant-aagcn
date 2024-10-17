import os
import sys
import logging
from pytorch_lightning.callbacks import Callback


class LossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.losses = {'train': [], 'val': []}

    def on_train_epoch_end(self, trainer, pl_module):
        avg_train_loss = trainer.callback_metrics['train_loss']
        self.losses['train'].append(avg_train_loss.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        avg_val_loss = trainer.callback_metrics['val_loss']
        self.losses['val'].append(avg_val_loss.item())

def get_logger(name, stdout=sys.stdout, filename=None, loglevel=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    if stdout:
        console = logging.StreamHandler(stdout)
        console.setLevel(loglevel)
        console.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(console)
    if filename:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        logfile = logging.FileHandler(filename)
        logfile.setLevel(loglevel)
        logfile.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s'))
        logger.addHandler(logfile)
    
    return logger
