import csv
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import os
from util import sort_with_order


def scaled(vs, ref, num):
    def func(k):
        ds = vs[ref, :] - k * vs[num, :]
        return np.sum(np.square(ds))

    return func


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


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    with open('data/20161013 calibration l values.csv') as f:
        reader = csv.reader(f, delimiter=' ')
        vs = np.array([map(float, r) for r in reader])

    v_totals = []
    for ref in range(0, 72):
        ks, rs, v_total = optimize_with_ref(vs, ref)
        # print(ref,v_total)
        v_totals.append((ref, v_total))
    # Order by how well the fitting is.
    res = sorted(v_totals, key=lambda x: x[1])
    # Plot: Without scaling
    plt.plot(vs.transpose(), alpha=0.7)
    plt.show()
    # Plot: Scaling by best fitting
    ks, rs, _ = optimize_with_ref(vs, res[0][0])
    plt.plot(rs.transpose(), alpha=0.7)
    print(ks)
    np.savetxt('data/20161013 calibration scale.txt',ks)
    plt.show()
    # Plot: Scaling by worst fitting
    _, rs2, _ = optimize_with_ref(vs, res[-1][0])
    plt.plot(rs2.transpose(), alpha=0.7)
    plt.show()

    sort_and_plot(rs)
    sort_and_plot(rs2)


if __name__ == "__main__":
    main()
