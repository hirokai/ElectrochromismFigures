import csv
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os
from common.util import sort_with_order


def calc_calibration_scales(all_vs, vs, debug=False):
    res = find_best_fitting(all_vs)

    # Plot: Scaling by best fitting
    ks, rs, _ = optimize_with_vref(vs, all_vs[res[0][0],:])

    if debug:
        # plt.plot(all_vs.transpose())
        # plt.show()
        plt.plot(rs.transpose(), alpha=0.7)
        plt.title('calc_calibration_scales()')
        plt.show()
    return ks


def correct_cielab(in_csv, scale_csv, out_csv):
    with open(in_csv) as f:
        reader = csv.reader(f)
        vs = np.array([map(float, r) for r in reader])
    with open(scale_csv) as f:
        reader = csv.reader(f)
        ss = np.array([map(float, r) for r in reader]).flatten()
    print(np.prod([vs, ss]))
    return


def scaled(vs, ref, num):
    def func(k):
        ds = vs[ref, :] - k * vs[num, :]
        return np.sum(np.square(ds))

    return func


def scaled2(vs, v_ref, num):
    def func(k):
        ds = v_ref - k * vs[num, :]
        return np.sum(np.square(ds))

    return func


def optimize_with_vref(vs, v_ref):
    v_total = 0
    rs = np.zeros(vs.shape)
    ks = np.zeros(vs.shape[0])
    for i in range(0, vs.shape[0]):
        k, v, _, _ = optimize.brent(scaled2(vs, v_ref, i), full_output=True)
        rs[i, :] = vs[i, :] * k
        ks[i] = k
        v_total += v
    # print(k,v)
    return ks, rs, v_total


def optimize_with_ref(vs, ref):
    v_total = 0
    rs = np.zeros(vs.shape)
    ks = np.zeros(vs.shape[0])
    for i in range(0, vs.shape[0]):
        k, v, _, _ = optimize.brent(scaled(vs, ref, i), full_output=True)
        rs[i, :] = vs[i, :] * k
        ks[i] = k
        v_total += v
    # print(k,v)
    return ks, rs, v_total


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


def find_best_fitting(vs):
    v_totals = []
    for ref in range(0, vs.shape[0]):
        ks, rs, v_total = optimize_with_ref(vs, ref)
        # print(ref,v_total)
        v_totals.append((ref, v_total))
    # Order by how well the fitting is.
    return sorted(v_totals, key=lambda x: x[1])


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    with open('data/kinetics/20161013 calibration l values.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        vs = np.array([map(float, r) for r in reader])
        idxs = map(int, vs[:, 0])
        vs = vs[:, 1:]

    # Plot: Without scaling
    plt.plot(vs.transpose(), alpha=0.7)
    plt.show()

    res = find_best_fitting(vs)

    # Plot: Scaling by best fitting
    ks, rs, _ = optimize_with_ref(vs, res[0][0])
    print(ks)
    np.savetxt('data/kinetics/20161013 calibration scale.csv', np.array([idxs, ks]).transpose())
    plt.plot(rs.transpose(), alpha=0.7)
    plt.show()

    # Plot: Scaling by worst fitting
    _, rs2, _ = optimize_with_ref(vs, res[-1][0])
    plt.plot(rs2.transpose(), alpha=0.7)
    plt.show()

    # Plot: scale factors.
    plt.plot(ks)
    plt.ylim([0, 1.5])
    plt.show()

    sort_and_plot(rs)
    sort_and_plot(rs2)


if __name__ == "__main__":
    main()
