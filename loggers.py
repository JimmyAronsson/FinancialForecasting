import os
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger


class LoggerLSTM(TensorBoardLogger):

    @property
    def name(self):
        return "LSTMLogger"

    def make_log_dir(self):
        os.makedirs(self.log_dir)

    def log_config(self, config):
        with open(os.path.join(self.log_dir, 'config.txt'), 'w') as f:
            f.write(f'data_dir:\t{config.data_dir}')
            f.write(f'\nmodel_name:\t{config.get_model().__class__.__name__}')

            f.write(f'\nstart_date:\t{config.time_period[0]}')
            f.write(f'\nfinal_date:\t{config.time_period[1]}')
            f.write(f'\nforecast_steps:\t{config.forecast_steps}')
            f.write(f'\nbatch_size:\t{config.batch_size}')

            f.write(f'\nfilelist_train:\t{config.filelist["train"]}')
            f.write(f'\nfilelist_val:\t{config.filelist["val"]}')
