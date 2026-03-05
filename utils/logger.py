import os
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Setup file logger
        self.logger = logging.getLogger('RiceDiseaseDetect')
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
            fh.setLevel(logging.INFO)
            
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir)

    def info(self, msg):
        self.logger.info(msg)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
