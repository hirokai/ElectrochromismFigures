import csv

import luigi
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from common.data_tools import *
from figure_tools import colors10
from luigi_tools import cleanup

#
# Parameters for curve fitting
#

# Time for splitting traces
split_t = {
    '01': [2, 62, 122, 182, 242, 302],
    '02': [2, 62, 122, 182, 242, 302, 422, 482, 542, 602, 662],
    '03': [2, 62, 122, 182, 242, 302, 422, 482, 542, 602, 662],
    '04': [2, 62, 122, 182, 242, 302],
    '07': [2, 62, 122, 182, 242, 302],
    '08': [2, 62, 122, 182, 242, 302],
    '11': [2, 62, 122, 182, 242, 302],
    '14': [2, 62, 122, 182, 242, 302],
}
# Fill with default
for i in range(1, 17):
    k = '%02d' % i
    if k not in split_t:
        split_t[k] = map(lambda a: a + 2, range(0, 60 * 12 + 1, 60))

# Indices of sections used for curve fitting.
used_sections_idxlist = {
    '01': [1, 3, 5],
    '02': [3, 5, 7, 9],
    '03': [3, 5, 7, 9],
    '04': [1, 3, 5],
    '07': [1, 3, 5],
    '08': [1, 3, 5],
    '11': [1, 3, 5],
    '14': [1, 3, 5],
}
# Fill with default
for i in range(1, 17):
    k = '%02d' % i
    if k not in used_sections_idxlist:
        used_sections_idxlist[k] = [3, 5, 7, 9, 11]


def plot_offset(num, i):
    return 2


def curvefit_range(num, i):
    d = {'1:0': range(5, 20), '2:0': range(5, 10),
         '5:0': range(3, 60), '5:1': range(15, 60), '5:2': range(7, 60), '5:3': range(5, 60), '5:4': range(4, 60)
         }
    k = '%d:%d' % (num, i)
    if num == 1:
        return range(8, 16)
    elif num == 2:
        return range(5, 60)
    elif num == 3:
        return range(5, 60)
    elif num == 4:
        return range(0, 60) if i == 0 else range(3, 60)
    elif num == 6:
        return range(5, 30)
    elif num == 8:
        return range(0, 60) if i == 0 else range(3, 60)
    elif num == 9:
        return range(3, 60)
    elif num == 10:
        return range(5, 60)
    elif num == 11:
        return range(3, 60)
    elif num == 12:
        return range(5, 60)
    elif num == 13:
        return range(3, 60)
    elif num == 14:
        return range(0, 15) if i == 0 else range(3, 15)
    elif num == 15:
        return range(5, 60)
    else:
        return d[k] if k in d else range(3, 30)


def curvefit_t0(num, i):
    return 4


# First-order reaction.
def get_first_order(t0):
    def first_order(t, a_i, a_f, k):
        y = a_i + (a_f - a_i) * (1 - np.exp(-k * (t - t0)))
        return y

    return first_order


# Curve fitting with a first-order reaction (An exponentially converging function).
def do_fit(ts, ys, t0, color, fit_range=None, plotting=False):
    # Extract range
    x_data = ts if fit_range is None else by_indices(ts, fit_range)
    y_data = ys if fit_range is None else by_indices(ys, fit_range)

    # Scatter plot
    marker = None
    if plotting:
        marker = plt.scatter(ts, ys, c=color, lw=0, s=10)
    try:
        first_order = get_first_order(t0)
        popt, _ = curve_fit(first_order, x_data, y_data, [y_data[0], y_data[-1], 0.03])
        if plotting:
            xs = np.linspace(0, 60, 5 * 60 + 1)
            ys = first_order(xs, *popt)
            plt.plot(xs, ys, c=color)
        return popt, marker, x_data, y_data
    except:
        return None, marker, x_data, y_data


def do_curve_fit(input_paths, out_path):
    # Load csv data and split into sections.
    # Return used and unused sections.
    def get_and_split_dataset(num):
        # Read a csv file into a DataFrame
        path = input_paths['%02d' % num].path
        name = basename_noext(path)
        df = pd.read_csv(path, names=['index', 'file', 'l', 'a', 'b'])

        # Split traces into sections with `times`, manually entered ranges.
        ts = np.array(map(lambda a: a - 1, df['index']))
        ys = df['l']
        split_ts = split_t['%02d' % num]
        tss, yss = split_trace(ts, ys, split_ts)
        vss = zip(tss, yss)

        # Will plot in "used" and "unused" sections.
        used_sections_idxs = used_sections_idxlist['%02d' % num]
        used_sections = by_indices(vss, used_sections_idxs)
        unused_sections = by_indices(vss, set(range(len(split_ts) + 1)) - set(used_sections_idxs))
        return used_sections, unused_sections, name

    def vs_key(num, i, axis):
        return '%d:%d:%s' % (num, i, axis)

    def curve_fitting_all(vs, writer):
        for num in range(1, 17):
            if num != 15:
                continue
            ps = []
            count = 0
            for i in range(20):
                key_t = vs_key(num, i, 't')
                if num == 15 and i == 0:
                    continue
                if key_t in vs:
                    r = curvefit_range(num, i)
                    t0 = curvefit_t0(num, i)
                    p, m, _, ys = do_fit(vs[key_t], vs[vs_key(num, i, 'y')], t0, colors10[i % 10], fit_range=r)
                    ps.append(p)
                    plt.scatter(vs[vs_key(num, i, 't')] - t0, vs[vs_key(num, i, 'y')], c=colors10[count % 10], s=6)
                    count += 1
            for i, p in enumerate(ps):
                # if num == 15 and i == 0:
                #     continue
                if p is not None:
                    print('%d\t%d\t%s' % (num, i, '\t'.join(map(str, p))))
                    pedot = ([0] + [80] * 3 + [30] * 4 + [60] * 3 + [40] * 3 + [20] * 3)[num]
                    mode = \
                        ['ox', 'ox', 'ox', 'red', 'ox', 'ox', 'red', 'ox', 'ox', 'ox', 'red', 'ox', 'ox', 'red', 'ox',
                         'ox',
                         'red'][num]
                    voltage = 0
                    writer.writerow(map(str, [num, i, pedot, mode, voltage] + list(p)))

    vs = {}
    # Plot all images in sequence
    for num in range(1, 17):
        used, unused, name = get_and_split_dataset(num)
        for i, sec in enumerate(used):
            min_t = min(sec[0])
            vs['%d:%d:%s' % (num, i, 't')] = np.array(map(lambda t: t - min_t, sec[0]))
            vs['%d:%d:%s' % (num, i, 'y')] = sec[1]

    # Curve fitting for them
    ensure_exists('curvefit')
    with open(out_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['movie', 'section', 'PEDOT ratio', 'mode', 'voltage', 'L_i', 'L_f', 'k'])
        curve_fitting_all(vs, writer)


# Collect L traces for various PEDOT and voltage
class CollectCIELabStub(luigi.Task):
    def output(self):
        return {'%02d' % i: luigi.LocalTarget(
            '../../Suda Electrochromism data/20160512-13 Suda EC slices/analysis_0525/cielab/%02d.csv' % i) for i in
                range(1, 17)}


class CurveFitStub(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            '../data/curvefit_external.csv')


class CurveFit(luigi.Task):
    def requires(self):
        return CollectCIELabStub()

    def output(self):
        return [luigi.LocalTarget(
            '../data/curvefit_pedot_voltage.csv')]

    def run(self):
        do_curve_fit(self.input(), self.output()[0].path)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    cleanup(CurveFit())
    luigi.run(['CurveFit'])
