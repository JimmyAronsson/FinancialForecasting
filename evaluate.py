import os
import ast
import torch
import matplotlib.pyplot as plt

import models
from datasets import DatasetLSTM
from run import FinancialForecaster  # TODO: Shouldn't need to import from run. Refactor.


class EvaluateForecaster:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.config = self._create_config()

        self.dataset = DatasetLSTM(data_dir=self.config['data_dir'],
                                   filelist=self.config['val_filelist'],
                                   start_date=self.config['start_date'],
                                   final_date=self.config['final_date'],
                                   forecast_steps=self.config['forecast_steps'])
        self.model = self._load_model()
        self.forecaster = FinancialForecaster(model=self.model, **self.config)

    def _create_config(self):

        with open(os.path.join(self.log_dir, 'run_info.txt')) as f:
            # Read from file and split each line into parameter, value pairs
            run_info = [line.split(':\t') for line in f.read().splitlines()]
            config = dict((parameter, value) for parameter, value in run_info)

            # Turn string "['ABC', ..., 'XYZ']" into list ['ABC', ..., 'XYZ']
            config['train_filelist'] = ast.literal_eval(config['train_filelist'])
            config['val_filelist'] = ast.literal_eval(config['val_filelist'])
            config['forecast_steps'] = int(config['forecast_steps'])
            config['batch_size'] = int(config['batch_size'])

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

        model = getattr(models, self.config['model_name'])()  # Extracts and instantiates e.g. SmallLSTM from models.py

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
    evaluator = EvaluateForecaster(log_dir='./logs/LSTMLogger/version_7')
    evaluator.eval(stock_id=0)


if __name__ == '__main__':
    main()
