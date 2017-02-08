#
# Correspondence between L values and absorption for a revised manuscript.
#

import csv
import os

import luigi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.image_tools import get_cie_rois

from src.measure_colorchart import read_csv, measure_color_chart
from src.test_calibration import calc_calibration_scales


# Get absorbance at `wl` nm wavelength from .txt data file.
def get_abs(path, wl):
    v = None
    with open(path) as f:
        line = f.readline()
        while True:
            if f.readline().find('>>> Results <<<') >= 0:
                break
        for line in f.readlines():
            s = '  %.1f nm' % wl
            if line.find(s) >= 0:
                v = float(line.split('  ')[2].split(' ')[0])
    return v


def measure_all(image_list, sample_csv, roi_csv, uvvis_list, out_csv_sample, out_csv_calibration, out_csv_uvvis, w=4.0):
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    with open(image_list) as f:
        img_paths = map(lambda s: s.strip(), f.readlines())
    with open(uvvis_list) as f:
        uvvis_paths = map(lambda s: s.strip(), f.readlines())
    abs_570 = [get_abs(n, 570) for n in uvvis_paths]
    with open(sample_csv) as f:
        reader = csv.reader(f)
        reader.next()
        sample_rois = [map(float, r)[1:] for r in reader]
    crs = read_csv(roi_csv)
    calibration_rois = []
    for k, v in crs.iteritems():
        calibration_rois.append(v[0])
    ls = []
    ls_cals = []
    for path, sample_roi, cal_roi in zip(img_paths, sample_rois, calibration_rois):
        r = get_cie_rois(path, [sample_roi])[0, 0]
        ls_cal = measure_color_chart(path, [cal_roi], cell_width=w, debug_drawcells=False)[0][:, 0]
        print('%.4f: %s' % (r, path))
        print(ls_cal)
        ls_cals.append(ls_cal)
        ls.append(r)

    out_folder = os.path.dirname(out_csv_uvvis)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(out_csv_uvvis, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['uvvis_path', 'Abs at 570 nm'])
        for i, a in enumerate(abs_570):
            writer.writerow([uvvis_paths[i], '%03.4f' % a])

    ls_cals = np.array(ls_cals)

    out_folder = os.path.dirname(out_csv_sample)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(out_csv_sample, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'L'])
        for i, l in enumerate(ls):
            writer.writerow([img_paths[i], '%03.4f' % l])

    out_folder = os.path.dirname(out_csv_calibration)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    with open(out_csv_calibration, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path'] + ['L%d' % i for i in range(1, 25)])
        for i, ls2 in enumerate(ls_cals):
            writer.writerow([img_paths[i]] + list(map(lambda l: ('%03.4f' % l).rjust(8), ls2)))


old_data_str = """0.249	44.011
0.472	23.788
0.278	46.198
0.403	30.196
0.253	48.461
0.439	26.477"""

old_data = np.array([[float(s) for s in l.split('\t')] for l in old_data_str.split('\n')])


def plot_all(all_cs, sample_path, cal_path, abs_path, color='blue'):
    print(sample_path)
    sns.set_style('ticks')

    abs_570 = np.genfromtxt(abs_path, delimiter=',')[1:, 1]
    ls = np.genfromtxt(sample_path, delimiter=',')[1:, 1]

    cs = np.genfromtxt(cal_path, delimiter=',')[1:, 1:]
    ks = calc_calibration_scales(all_cs, cs, debug=False)

    # print(abs_570, ls)
    print(ks)
    plt.subplot(1, 2, 1)
    plt.scatter(abs_570, ls, c=color)
    plt.xlabel('Absorbance at 570 nm')
    plt.ylabel('L* value')
    plt.title('Before calibration')
    plt.xlim([0, 1])
    plt.ylim([0, 50])
    plt.subplot(1, 2, 2)
    plt.scatter(abs_570, ls * ks, c=color)
    plt.xlabel('Absorbance at 570 nm')
    plt.ylabel('Corrected L* value')
    plt.title('After calibration')
    plt.xlim([0, 1])
    plt.ylim([0, 50])
    return np.array([abs_570, ls * ks])


def my_savefig(out_path):
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    plt.savefig(out_path)


class PlotColorCharts(luigi.Task):
    def requires(self):
        return {
            'new': MeasureColorCharts(name='20161101'),
            'old': MeasureColorCharts(name='20160523')}

    def run(self):
        colors = ['blue', 'red']
        css = []
        for i, n in enumerate(['new']):
            cs = np.genfromtxt(self.input()[n]['calibration'].path, delimiter=',')[1:, 1:]
            css.append(cs)
        css = np.concatenate(css)

        plt.figure(figsize=(10, 4))
        rs = []
        for i, n in enumerate(['new', 'old']):
            rs.append(plot_all(css, self.input()[n]['sample'].path,
                               self.input()[n]['calibration'].path,
                               self.input()[n]['abs'].path,
                               color=colors[i]))
        rs = np.concatenate(rs, axis=1)

        my_savefig(self.output()['final'].path)
        plt.show()

        plt.figure(figsize=(4,4))
        plt.scatter(rs[0, :], rs[1, :])
        from scipy import stats
        rs2 = []
        for i in range(rs.shape[1]):
            if rs[0, i] < 0.6:
                rs2.append(rs[:, i])
        rs2 = np.array(rs2).transpose()
        slope, intercept, r_value, p_value, std_err = stats.linregress(rs2[0, :], rs2[1, :])
        xs = np.linspace(0, 0.6, 10)
        ys = slope * xs + intercept
        plt.plot(xs, ys, lw=1)
        plt.xlim([0, 0.8])
        plt.ylim([0, 50])
        my_savefig(self.output()['correction'].path)
        print(slope, intercept, np.power(r_value, 2), std_err)

        plt.show()

    def output(self):
        return {'correction': luigi.LocalTarget('dist/lab-uvvis/l vs abs correction.pdf'),
                'final': luigi.LocalTarget('dist/lab-uvvis/l vs abs.pdf')}


class MeasureColorCharts(luigi.Task):
    name = luigi.Parameter()

    def run(self):
        imgs = 'parameters/%s/list of images.txt' % self.name
        sample_roi = 'parameters/%s/sample rois.csv' % self.name
        cal_roi = 'parameters/%s/calibration rois.csv' % self.name
        uvvis = 'parameters/%s/list of uvvis.txt' % self.name
        measure_all(imgs, sample_roi, cal_roi, uvvis, self.output()['sample'].path,
                    self.output()['calibration'].path,
                    self.output()['abs'].path,
                    w=30.0 if self.name == '20160523' else 4.0
                    )

    def output(self):
        return {'sample': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' sample l values.csv'),
                'calibration': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' calibration l values.csv'),
                'abs': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' absorption.csv')}


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # cleanup(MeasureColorCharts(name='20160523'))
    # cleanup(PlotColorCharts())
    luigi.run(['PlotColorCharts'])


if __name__ == "__main__":
    main()
