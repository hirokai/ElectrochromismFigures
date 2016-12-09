import numpy as np
import matplotlib.pyplot as plt
from . import Kinetics, KineticsDataType, read_kinetics
import os


def plot_raw_traces(dat):
    assert isinstance(dat, Kinetics)
    pedot = 30
    rpm = 3000
    mode = 'ox'
    count = 1
    for rpm in [500, 1000, 2000, 3000, 4000, 5000]:
        for pedot in [20, 30, 40, 60, 80]:
            # for voltage in [0.2, 0.4, 0.6, 0.8]:
            voltage = 0.8
            plt.subplot(6, 5, count)
            v = dat.get_data(pedot, rpm, mode, voltage)
            plt.plot(v[:, 0], v[:, 1])
            plt.ylim([0, 70])
            plt.title('%d %%, %d rpm' % (pedot, rpm))
            count += 1
            # plt.figure(figsize=(10, 10))
            # plt.show()


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    print('Loading data...')
    dat1 = read_kinetics(KineticsDataType.SplitTraces)
    dat2 = read_kinetics(KineticsDataType.RawSplitTraces)
    print('Loading done.')

    plot_raw_traces(dat1)
    plot_raw_traces(dat2)
    plt.show()


if __name__ == "__main__":
    main()
