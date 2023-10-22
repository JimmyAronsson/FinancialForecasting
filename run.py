import pytorch_lightning as pl

from configs import Config
from callbacks import CallbacksLSTM
from forecasters import FinancialForecaster
from loggers import LoggerLSTM


def main():
    config = Config(data_dir='data/monthly/NASDAQ-tiny/',
                    model_name='SmallLSTM',
                    time_period=('2020-01-01', '2022-01-01'),
                    forecast_steps=18,
                    batch_size=1,
                    train_split=0.5)

    logger = LoggerLSTM(save_dir="logs", name="LSTM")
    logger.make_log_dir()
    logger.log_config(config)

    forecaster = FinancialForecaster(config)

    trainer = pl.Trainer(accelerator='cpu',
                         callbacks=[CallbacksLSTM()],
                         logger=logger,
                         max_epochs=100,
                         )
    trainer.fit(forecaster, ckpt_path=None)


if __name__ == "__main__":
    main()
