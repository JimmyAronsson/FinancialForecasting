import os
import pandas as pd
from matplotlib import pyplot as plt


class Visualize:
    def __init__(self, load_dir, data='train', stock='SXP', channel='close'):
        self.load_dir = load_dir
        self.data = data
        self.stock = stock
        self.channel = channel

    def plot(self):
        train_loss = pd.read_csv(posixpath.join(self.load_dir, 'train_loss.csv'), index_col=0)
        val_loss = pd.read_csv(posixpath.join(self.load_dir, 'val_loss.csv'), index_col=0)

        data_dir = posixpath.join(self.load_dir, self.data)

        path_labels = posixpath.join(data_dir, f'{self.data}_labels_{self.stock}.csv')
        labels = pd.read_csv(path_labels, index_col="date", parse_dates=["date"])[self.channel]

        path_pred = posixpath.join(data_dir, f'{self.data}_pred_{self.stock}.csv')
        pred = pd.read_csv(path_pred, index_col="date", parse_dates=["date"])[self.channel]

        fig, ax = plt.subplots(3)
        ax[0].set_title(f"{self.stock} {self.channel} price performance on {self.data} data.")

        ax[0].plot(labels)
        ax[0].plot(pred)
        ax[0].legend(["ground truth", "prediction"])

        rel_error = abs((labels.values - pred.values) / labels.values)
        ax[1].plot(rel_error)
        ax[1].legend(["relative error"])

        ax[2].plot(range(len(train_loss.index)), train_loss.values)
        ax[2].plot(range(len(val_loss.index)), val_loss.values)
        ax[2].set_xlabel("epoch")
        ax[2].legend(["train loss", "val loss"])

        plt.show()


class DataPreprocessing:
    def __init__(self, load_path, save_path):
        self.load_path = load_path
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    @staticmethod
    def identify_daterange(path_to_dir='data/NASDAQ-full/original/'):
        """Saves the first and last dates of all stocks."""
        oldest_dates = []
        first_iteration = True
        for filename in os.listdir(path_to_dir):
            if first_iteration:
                df = pd.read_csv(path_to_dir + filename)
                oldest_dates.append(df["date"].values[0])
                most_recent_date = df["date"].values[-1]
                with open('data/most_recent_date.txt', 'w') as f:
                    f.write(most_recent_date)

                first_iteration = False
                continue

            df = pd.read_csv(path_to_dir + filename, nrows=1)  # First row contains first date
            oldest_dates.append(df["date"].values[0])

        with open('data/oldest_date.txt', 'w') as f:
            f.write(min(oldest_dates))
