from tools import Visualize


def main():
    visualize = Visualize(load_dir='results/2023-08-22--00-00_NASDAQ-small/',
                             data='train',
                             stock='TSLA',
                             channel='close')
    visualize.plot()


if __name__ == '__main__':
    main()
