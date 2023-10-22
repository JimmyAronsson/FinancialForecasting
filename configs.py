import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import models


@dataclass
class Config:
    data_dir: str
    model_name: str
    # FIXME: time_period currently refers to historic data + future labels, but should only refer to historic data.
    time_period: Tuple[str, str]  # Example: ('2000-01-01', '2015-01-01')
    forecast_steps: int
    batch_size: int
    train_split: Optional[float]
    filelist: Optional[Dict[str, list]] = None

    def __post_init__(self):
        if self.filelist is None and self.train_split is not None:
            self._train_val_split()

    def _train_val_split(self):
        data_filelist = os.listdir(self.data_dir)
        random.shuffle(data_filelist)

        nfiles_train = round(self.train_split * len(data_filelist))
        self.filelist = {'train': data_filelist[:nfiles_train],
                         'val': data_filelist[nfiles_train:]
                         }

    def get_model(self):
        return getattr(models, self.model_name)()
