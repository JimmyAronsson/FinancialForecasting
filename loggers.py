import os
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger


class LoggerLSTM(TensorBoardLogger):

    @property
    def name(self):
        return "LSTMLogger"

    def make_log_dir(self):
        os.makedirs(self.log_dir)
