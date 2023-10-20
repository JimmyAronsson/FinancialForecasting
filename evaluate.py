import os
import ast
import torch
from icecream import ic
import pytorch_lightning as pl
from run import ModelLSTM
from datasets import DatasetLSTM
from tools import Visualize

import matplotlib.pyplot as plt

from run import FinancialForecaster


def evaluate(start_month='2000-01',end_date='2022-07'):
    # Step 1: Load the checkpoint
    log_dir = './logs/LSTMLogger/version_18/'
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, os.listdir(ckpt_dir)[-1])
    hparams_file = os.path.join(log_dir, 'hparams.yaml')
    ckpt = torch.load(ckpt_path)

    with open(os.path.join(log_dir, 'run_info.txt')) as f:
        # Read from file and split each line into parameter, value pairs
        run_info = [line.split(':\t') for line in f.read().splitlines()]
        config = dict((parameter, value) for parameter, value in run_info)

        # Turn string "['ABC', ..., 'XYZ']" into list ['ABC', ..., 'XYZ']
        config['train_filelist'] = ast.literal_eval(config['train_filelist'])
        config['val_filelist'] = ast.literal_eval(config['val_filelist'])
        config['forecast_steps'] = int(config['forecast_steps'])
        config['batch_size'] = int(config['batch_size'])

    ic(config)

    # FIXME: Used to remove 'model.' from all keys. Clean this up.
    tmp = {}
    for key, value in ckpt['state_dict'].items():
        tmp[key.removeprefix('model.')] = value
    ckpt['state_dict'] = tmp

    # Step 2: Create the model
    model = ModelLSTM()
    model.load_state_dict(state_dict=ckpt['state_dict'])
    forecaster = FinancialForecaster(model=model,
                                     **config
                                     )
    """
    model = ModelLSTM.load_from_checkpoint(
        data_dir=config['data_dir'],
        debug=True,
        checkpoint_path=ckpt_path,
        hparams_file=hparams_file,
        map_location=None
    )"""

    dataset = DatasetLSTM(data_dir=config['data_dir'],
                          filelist=config['val_filelist'],
                          forecast_steps=config['forecast_steps'])

    """
    trainer = pl.Trainer()
    predictions = trainer.predict(forecaster, dataset)
    #print(data)
    print(predictions)

    print(predictions[0][0][0].size())
    print(predictions[0][1].size())
    """
    input, label = dataset.__getitem__(0)
    print('size of input:', input.size())
    print('size of label:', label.size())
    prediction = forecaster.forward(input.unsqueeze(0))[0, :, :]  # TODO: Generalize forward to not need batch dimension
    print('size of prediction:', prediction.size())

    input = input.detach().numpy()
    label = label.detach().numpy()
    prediction = prediction.detach().numpy()

    fig, ax = plt.subplots(4)

    axis = range(0, len(prediction[:, 0]))
    input_axis = axis[:len(input[:, 0])]
    label_axis = axis[len(input[:, 0]):]

    titles = ['open', 'high', 'low', 'close']
    for i in range(4):
        ax[i].set_title(titles[i])
        ax[i].plot(input_axis, input[:, i], label='Input data')
        ax[i].plot(label_axis, label[:, i], label='Forecast ground truth')
        ax[i].plot(axis, prediction[:, i], label='Forecast prediciton')
        ax[i].legend()

    plt.show()


def main():
    evaluate()

if __name__ == '__main__':
    main()
