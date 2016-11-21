import csv
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os
from util import sort_with_order
import glob
from data_tools import colors10
from util import ensure_folder_exists


def correct_cielab(in_csv, scale_csv, out_csv):
    with open(in_csv) as f:
        reader = csv.reader(f)
        vs = np.array([map(float, r) for r in reader])
    with open(scale_csv) as f:
        reader = csv.reader(f)
        ss = np.array([map(float, r) for r in reader]).flatten()
    print(np.prod([vs, ss]))
    return


def scaled(ls, ref_ls):
    def func(k):
        ds = ref_ls - k * ls
        return np.sum(np.square(ds))

    return func


def sort_and_plot(rs):
    lss = rs.transpose()
    l_mean = np.mean(lss, axis=1)
    chart_order = [i[0] for i in sorted(enumerate(l_mean), key=lambda x: x[1])]
    lss2 = np.zeros((len(chart_order), lss.shape[1]))
    for i in range(lss.shape[1]):
        lss2[:, i] = sort_with_order(lss[:, i], chart_order)
        # lss2[:, i] /= lss2[10, i]
    plt.plot(lss2, alpha=0.5)
    plt.show()


def read_l_values(csv_paths):
    vss = []
    for path in csv_paths:
        with open(path) as f:
            reader = csv.reader(f)
            reader.next()
            vs = np.array([map(float, row[3:]) for row in reader])
            vss.append(vs)
    return vss


def optimize_with_values(ls, ref_ls):
    k, v, _, _ = optimize.brent(scaled(ls, ref_ls), full_output=True)
    return k


# Do calibration with a set of color charts.
def calibration(colorcharts):
    ls_mean = np.mean(colorcharts, axis=0)
    ks = []
    calibrated = []
    for ls in colorcharts:
        k = optimize_with_values(ls, ls_mean)
        ks.append(k)
        calibrated.append(k * ls)
    return np.array(calibrated), ks


def get_outpath(s):
    return s.replace('colorchart', 'correction')


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    files = glob.glob(os.path.join('data', 'kinetics', 'colorchart', '20161019', '*.csv'))
    print('%d files.' % len(files))
    lsss = read_l_values(files)
    # for lss in lsss:
    #     plt.plot(lss)

    colorcharts = np.concatenate(lsss)
    ls_mean = np.mean(colorcharts, axis=0)
    calibrated, factors = calibration(colorcharts)
    count = 0
    for in_path, lss in zip(files, lsss):
        outpath = get_outpath(in_path)
        ensure_folder_exists(outpath)
        with open(outpath, 'wb') as f:
            writer = csv.writer(f)
            for i in range(len(lss)):
                writer.writerow([i+1,factors[count]])
                count += 1
    return
    plt.subplot(3, 1, 1)
    ls_mean_for_plot = np.repeat([ls_mean], colorcharts.shape[0], axis=0).transpose()
    for i, ls in enumerate(ls_mean_for_plot):
        plt.plot(ls, c=colors10[i % 10], lw=2)
    for i, lss in enumerate(colorcharts.transpose()):
        plt.plot(lss, c=colors10[i % 10])

    plt.subplot(3, 1, 2)
    for i, ls in enumerate(ls_mean_for_plot):
        plt.plot(ls, c=colors10[i % 10], lw=2)
    for i, lss in enumerate(calibrated.transpose()):
        plt.plot(lss, c=colors10[i % 10])

    plt.subplot(3, 1, 3)
    plt.plot(range(len(factors)), factors)
    plt.ylim([0, 2])

    plt.show()

    for i, ls in enumerate(ls_mean_for_plot):
        plt.plot(ls, c=colors10[i % 10], lw=2)
        plt.plot(calibrated[:, i], c=colors10[i % 10])
        if i % 5 == 4:
            plt.show()
    plt.show()


if __name__ == "__main__":
    main()
