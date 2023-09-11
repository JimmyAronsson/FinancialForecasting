import os

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from datatypes import Stock


def predict():
    pred_ndays = 14  # Number of days to predict.

    stock = Stock('data/NASDAQ-small/train/SXP.csv')
    stock.uniform_timespan()

    original_df = stock.df
    original_index = original_df.index
    val_index = original_index[-pred_ndays:]

    stock.log_normalize()
    lntd = stock.df.iloc[:-pred_ndays]  # log-normalized training data

    start = len(lntd.index)
    end = start + pred_ndays - 1

    k = 0
    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.suptitle("plot_acf for SXP")
    for i in range(2):
        for j in range(2):
            model = AutoReg(lntd[lntd.columns[k]], lags=30, trend='ct')
            results = model.fit()
            pred = results.predict(start=start, end=end).values

            stock.df = pd.concat([lntd, pd.DataFrame({'open': pred,
                                                      'high': pred,
                                                      'low': pred,
                                                      'close': pred}, index=val_index)])
            stock.log_denormalize()
            pred_values = stock.df[stock.df.columns[k]].iloc[-pred_ndays:].values

            train_include = pred_ndays + 90  # Number of days to include in plot
            original_values = original_df[original_df.columns[k]].values
            ax[i, j].plot(original_index[-train_include:],
                          original_values[-train_include:])
            ax[i, j].plot(val_index, pred_values)
            ax[i, j].axvline(x=val_index[0], color='k', linestyle='--')
            ax[i, j].legend(['data', 'prediction'])
            ax[i, j].set_title(original_df.columns[k])

            k += 1

    plt.show()


def adfuller_test():
    path = 'data/NASDAQ-small/train'
    for filename in os.listdir(path):
        stock = Stock(os.path.join(path, filename))

        dftest = adfuller(stock.df["close"], autolag='AIC')
        dfout = pd.Series(dftest[0:4], index=['ADF Test Statistic', 'p-value', '# lags', '# observations'])

        for key, val in dftest[4].items():
            dfout[f'critical value ({key})'] = val

        print()
        print(f"Test statistics for {stock.ticker} close price:")
        print()
        print(dfout)

        if dftest[1] <= 0.05:
            print()
            print("Strong evidence against the null hypothesis.")
            print("Reject the null hypothesis.")
            print("Data has no unit root and is stationary.")
        else:
            print()
            print("Weak evidence against the null hypothesis")
            print("Fail to reject the null hypothesis.")
            print("Data has a unit root and is non-stationary.")

        print()
        print('-' * 50)


def granger_causality_test():
    path = 'data/NASDAQ-small/train'
    filenames = os.listdir(path)
    filename_pairs = list(zip(filenames, filenames[1:] + filenames[:1]))

    for filename1, filename2 in filename_pairs:
        stock1 = Stock(os.path.join(path, filename1))
        stock1.uniform_timespan()

        stock2 = Stock(os.path.join(path, filename2))
        stock2.uniform_timespan()

        channel = 'close'

        df = pd.merge(stock1.df[channel], stock2.df[channel], right_index=True, left_index=True)

        test = grangercausalitytests(df, maxlag=5)

    print()
    print('-' * 50)


def main():
    granger_causality_test()


if __name__ == "__main__":
    main()
