import matplotlib.pyplot as plt
from kinetics import Kinetics, KineticsDataType, read_kinetics
from common.data_tools import colors10
from common.util import chdir_root


rpms = [500, 1000, 2000, 3000, 4000, 5000]
voltages = [0.2, 0.4, 0.6, 0.8]
pedots = [20, 30, 40, 60, 80]


def plot_raw_traces(dat, rpm, color='b'):
    assert isinstance(dat, Kinetics)
    mode = 'ox'
    count = 1
    print(dat)
    for pedot in pedots:
        # for pedot in [20, 30, 40, 60, 80]:
        for voltage in voltages:
            plt.subplot(len(rpms), len(voltages), count)
            print(pedot, rpm, mode, voltage)
            vs = dat.get_data(pedot, rpm, mode, voltage)
            for i, v in enumerate(vs):
                plt.plot(v[:, 0], v[:, 1], c=color, ls=['solid', 'dashed'][i % 2])
            plt.ylim([0, 70])
            plt.title('%.1f V, %d%% pedot' % (voltage, pedot))
            count += 1
            # plt.figure(figsize=(10, 10))
            # plt.show()


def main():
    print('Loading data...')
    dat_corrected = read_kinetics(KineticsDataType.SplitTraces)
    dat_raw = read_kinetics(KineticsDataType.RawSplitTraces)
    print('Loading done. Plotting...')

    for v in [2000]:
        plot_raw_traces(dat_raw, v, color=colors10[0])
        plot_raw_traces(dat_corrected, v, color=colors10[1])
        fig = plt.gcf()
        fig.canvas.set_window_title('%d' % v)
        plt.show()


if __name__ == "__main__":
    chdir_root()
    main()
