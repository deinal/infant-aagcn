import os
import sys
import logging
import networkx as nx
from modules.constants import NODE_INDEX, EDGE_LABELS
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

def get_infant_graph(frame):
    G = nx.Graph()

    bodyparts = iter(NODE_INDEX.keys())
    for bodypart in bodyparts:
        G.add_node(NODE_INDEX[bodypart], 
                   x=frame[f'{bodypart}_x'],
                   y=frame[f'{bodypart}_y'],
                   z=frame[f'{bodypart}_z'],
        )
    for label_0, label_1 in EDGE_LABELS:
        G.add_edge(NODE_INDEX[label_0], NODE_INDEX[label_1])
        
    return G
