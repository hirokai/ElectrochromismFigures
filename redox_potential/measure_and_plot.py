import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.optimize import curve_fit
import os
from common.util import chdir_root, ensure_folder_exists
from common.figure_tools import colors10, plot_and_save, set_format, set_common_format
from kinetics import read_kinetics, KineticsDataType, read_final_l_predefined_dates


def plot_redox_potentials(ys, es):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ratios = [20, 40, 60, 80]
    plt.scatter(ratios, ys, color=colors10[0:4])
    (_, caps, _) = plt.errorbar(ratios, ys, es, fmt='none', lw=1, elinewidth=1, c=colors10[0:4])
    for i,cap in enumerate(caps):
        cap.set_markeredgewidth(1)
    plt.xlim([0, 100])
    plt.ylim([-0.4, 0.4])
    ax.xaxis.set_major_locator(MultipleLocator(20))
    # ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    outpath = os.path.join('redox_potential', 'dist', 'redox_potential.pdf')
    ensure_folder_exists(outpath)
    # plt.savefig(outpath)
    plt.show()


def sigmoid(x, x0, k, y0, y1):
    y = (y1 - y0) / (1 + np.exp(-k * (x - x0))) + y0
    return y


def main():
    chdir_root()
    dat = read_final_l_predefined_dates()

    pedots = [20, 40, 60, 80]
    voltages = [-0.5, -0.2, 0.2, 0.4, 0.6, 0.8]
    ys = []
    es = []
    for pedot in pedots:
        ls = []
        vs = []
        for voltage in voltages:
            a = dat.get_data(pedot, 2000, 'ox' if voltage > 0 else 'red', voltage)
            if a is not None:
                a = [np.mean(a)]
                ls += a
                vs += [voltage] * len(a)
        popt, pcov = curve_fit(sigmoid, vs, ls, [0.0, 0.2, min(ls), max(ls)])
        x = np.linspace(-1, 1, 100)
        perr = np.sqrt(np.diag(pcov))
        # print popt, perr
        ys.append(popt[0])
        es.append(perr[0])
        y = sigmoid(x, *popt)
        plt.plot(x, y)

        plt.scatter(vs, ls)
        plt.show()
        # print(vs, ls)
    print(ys, es)
    plot_redox_potentials(ys, es)


if __name__ == "__main__":
    main()
