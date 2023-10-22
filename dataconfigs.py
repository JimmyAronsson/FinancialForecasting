import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import models


@dataclass
class FFConfig:
    data_dir: str
    model_name: str
    time_period: Tuple[str, str]  # Example: ('2000-01-01', '2015-01-01')
    forecast_steps: int
    batch_size: int
    train_split: Optional[float]
    filelist: Optional[Dict[str, list]] = None

    # FIXME: time_period currently refers to historic data + future labels,
    #        but should only refer to historic data.

    def __post_init__(self):
        if self.filelist is None and self.train_split is not None:
            self._train_val_split()

    def _train_val_split(self):
        filelist = os.listdir(self.data_dir)
        random.shuffle(filelist)

        nfiles_train = round(self.train_split * len(filelist))
        self.filelist = {'train': filelist[:nfiles_train],
                         'val': filelist[nfiles_train:]
                         }

    @property
    def get_model(self):
        return getattr(models, self.model_name)()
