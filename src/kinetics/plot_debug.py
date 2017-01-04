import numpy as np
import matplotlib.pyplot as plt
from . import Kinetics, KineticsDataType, read_kinetics
from util.data_tools import colors10
import os

rpms = [500, 1000, 2000, 3000, 4000, 5000]
voltages = [0.2, 0.4, 0.6, 0.8]
pedots = [20, 30, 40, 60, 80]


def plot_raw_traces(dat, pedot, color='b'):
    assert isinstance(dat, Kinetics)
    mode = 'ox'
    count = 1
    for rpm in rpms:
        # for pedot in [20, 30, 40, 60, 80]:
        for voltage in voltages:
            plt.subplot(len(rpms), len(voltages), count)
            vs = dat.get_data(pedot, rpm, mode, voltage)
            for i, v in enumerate(vs):
                plt.plot(v[:, 0], v[:, 1], c=color, ls=['solid', 'dashed'][i % 2])
            plt.ylim([0, 70])
            plt.title('%.1f V, %d rpm' % (voltage, rpm))
            count += 1
            # plt.figure(figsize=(10, 10))
            # plt.show()


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

    print('Loading data...')
    dat_corrected = read_kinetics(KineticsDataType.SplitTraces)
    dat_raw = read_kinetics(KineticsDataType.RawSplitTraces)
    print('Loading done. Plotting...')

    for v in pedots:
        plot_raw_traces(dat_raw, v, color=colors10[0])
        plot_raw_traces(dat_corrected, v, color=colors10[1])
        fig = plt.gcf()
        fig.canvas.set_window_title('%d' % v)
        plt.show()


if __name__ == "__main__":
    main()
