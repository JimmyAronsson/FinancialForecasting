import ast
import os

import matplotlib.pyplot as plt
import torch

from dataconfigs import FFConfig
from datasets import StockDataset
from forecasters import FinancialForecaster


class EvaluateForecaster:
    def __init__(self, log_dir, stock_id):
        self.log_dir = log_dir
        self.stock_id = stock_id

        self.config = self._create_config()

        self.model = self._load_model()
        self.dataset = StockDataset(self.config, stage='val')
        self.forecaster = FinancialForecaster(self.config)

    def _create_config(self):
        with open(os.path.join(self.log_dir, 'config.txt')) as f:
            flist = [line.split(':\t') for line in f.read().splitlines()]
            fdict = dict((parameter, value) for parameter, value in flist)

            filelist = {'train': ast.literal_eval(fdict['filelist_train']),
                        'val': ast.literal_eval(fdict['filelist_val'])}

            config = FFConfig(data_dir=fdict['data_dir'],
                              model_name=fdict['model_name'],
                              time_period=(fdict['start_date'], fdict['final_date']),
                              forecast_steps=int(fdict['forecast_steps']),
                              batch_size=int(fdict['batch_size']),
                              train_split=None,
                              filelist=filelist)

        return config

    def _get_ckpt(self):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])
        ckpt = torch.load(ckpt_path)

        # FIXME: Used to remove 'model.' from all keys. Clean this up.
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            state_dict[key.removeprefix('model.')] = value
        ckpt['state_dict'] = state_dict

        return ckpt

    def _get_epoch(self):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        path = os.listdir(ckpt_dir)[-1]
        epoch = int(path[path.find('=')+1:path.find('-')])  # Example: epoch=123-step=1 --> 123

        return epoch

    def _get_ticker(self):
        filename = self.config.filelist['val'][self.stock_id]
        ticker = filename.removesuffix('.csv')

        return ticker

    def _load_model(self):
        ckpt = self._get_ckpt()
        model = self.config.get_model()
        model.load_state_dict(state_dict=ckpt['state_dict'])

        return model

    def eval(self):
        inputs, labels = self.dataset.get_item(stock_id=self.stock_id, inputs_and_labels=True)

        forecast = self.forecaster.forward(inputs)[0, :, :]

        inputs = inputs.detach().numpy()
        labels = labels.detach().numpy()
        forecast = forecast.detach().numpy()

        self.plot(inputs, labels, forecast)

    def plot(self, inputs, labels, forecast):

        axis = range(0, len(forecast[:, 0]))
        input_axis = axis[:len(inputs[:, 0])]
        label_axis = axis[len(inputs[:, 0]):]

        epoch = self._get_epoch()
        model = self.config.model_name
        ticker = self._get_ticker()

        fig, ax = plt.subplots(4)
        fig.suptitle(f'{ticker} forecast using {model} @ {epoch} epochs')

        titles = ['open', 'high', 'low', 'close']
        for i in range(4):
            ax[i].set_title(titles[i])
            ax[i].plot(input_axis, inputs[:, i], label='Input data')
            ax[i].plot(label_axis, labels[:, i], label='Forecast ground truth')
            ax[i].plot(axis, forecast[:, i], label='Forecast prediciton')
            ax[i].legend()

        plt.show()


def main():
    evaluator = EvaluateForecaster(log_dir='./logs/LSTMLogger/version_17', stock_id=0)
    evaluator.eval()


if __name__ == '__main__':
    main()
