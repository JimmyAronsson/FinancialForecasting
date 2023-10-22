import os
import ast
import torch
import matplotlib.pyplot as plt

import models
from configs import Config
from datasets import DatasetLSTM
from forecasters import FinancialForecaster


class EvaluateForecaster:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.config = self._create_config()

        self.model = self._load_model()
        self.dataset = DatasetLSTM(self.config, stage='val')
        self.forecaster = FinancialForecaster(self.config)

    def _create_config(self):
        with open(os.path.join(self.log_dir, 'config.txt')) as f:
            # Read from file and split each line into parameter, value pairs
            flist = [line.split(':\t') for line in f.read().splitlines()]
            fdict = dict((parameter, value) for parameter, value in flist)

            filelist = {'train': ast.literal_eval(fdict['filelist_train']),
                        'val': ast.literal_eval(fdict['filelist_val'])}

            # Turn string "['ABC', ..., 'XYZ']" into list ['ABC', ..., 'XYZ']
            config = Config(data_dir=fdict['data_dir'],
                            model_name=fdict['model_name'],
                            time_period=(fdict['start_date'], fdict['final_date']),
                            forecast_steps=int(fdict['forecast_steps']),
                            batch_size=int(fdict['batch_size']),
                            train_split=None,
                            filelist=filelist)
        return config

    def _load_model(self, ckpt_index=-1):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[ckpt_index])  # (default: latest checkpoint)
        ckpt = torch.load(ckpt_path)

        # FIXME: Used to remove 'model.' from all keys. Clean this up.
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            state_dict[key.removeprefix('model.')] = value
        ckpt['state_dict'] = state_dict

        model = self.config.get_model()  # Extracts and instantiates e.g. SmallLSTM from models.py

        model.load_state_dict(state_dict=ckpt['state_dict'])

        return model

    def eval(self, stock_id=0):
        inputs, labels = self.dataset.get_item(stock_id=stock_id, inputs_and_labels=True)

        print('size of inputs:', inputs.size())
        print('size of label:', labels.size())
        prediction = self.forecaster.forward(inputs)[0, :, :]  # TODO: Generalize forward to not need batch dimension
        print('size of prediction:', prediction.size())

        inputs = inputs.detach().numpy()
        labels = labels.detach().numpy()
        prediction = prediction.detach().numpy()

        axis = range(0, len(prediction[:, 0]))
        input_axis = axis[:len(inputs[:, 0])]
        label_axis = axis[len(inputs[:, 0]):]
        titles = ['open', 'high', 'low', 'close']

        fig, ax = plt.subplots(4)
        for i in range(4):
            ax[i].set_title(titles[i])
            ax[i].plot(input_axis, inputs[:, i], label='Input data')
            ax[i].plot(label_axis, labels[:, i], label='Forecast ground truth')
            ax[i].plot(axis, prediction[:, i], label='Forecast prediciton')
            ax[i].legend()

        plt.show()


def main():
    evaluator = EvaluateForecaster(log_dir='./logs/LSTMLogger/version_16')
    evaluator.eval(stock_id=0)


if __name__ == '__main__':
    main()
