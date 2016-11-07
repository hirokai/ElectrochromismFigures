#
# Correspondence between L values and absorption for a revised manuscript.
#

import csv
import os
from image_tools import get_cie_rois
import matplotlib.pyplot as plt
from measure_calibration_lab import read_csv, mk_cells, measure_color_chart
import numpy as np
import luigi
from luigi_tools import cleanup
from test_calibration_fit import calc_calibration_scales


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


def measure_all(image_list, sample_csv, roi_csv, uvvis_list, out_csv_sample, out_csv_calibration, out_csv_uvvis):
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
        ls_cal = measure_color_chart(path, [cal_roi])[0][:, 0]
        print('%.4f: %s' % (r, path))
        print(ls_cal)
        ls_cals.append(ls_cal)
        ls.append(r)
    print(abs_570)
    print(ls)

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


def plot_all(sample_path, cal_path, abs_path, out_path):
    out_folder = os.path.dirname(out_path)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    from numpy import genfromtxt
    abs_570 = genfromtxt(abs_path, delimiter=',')[1:, 1]
    ls = genfromtxt(sample_path, delimiter=',')[1:, 1]

    cs = genfromtxt(cal_path, delimiter=',')[1:, 1:]
    ks = calc_calibration_scales(cs, debug=True)

    # plt.scatter(abs_570, ls, c='gray')
    plt.scatter(abs_570, ls * ks, c='b')
    plt.savefig(out_path)
    plt.show()


class PlotColorCharts(luigi.Task):
    name = luigi.Parameter()

    def requires(self):
        return MeasureColorCharts(name=self.name)

    def run(self):
        plot_all(self.input()['sample'].path,
                 self.input()['calibration'].path,
                 self.input()['abs'].path, self.output().path)

    def output(self):
        return luigi.LocalTarget('dist/lab-uvvis/l vs abs.pdf')


class MeasureColorCharts(luigi.Task):
    name = luigi.Parameter()

    def run(self):
        imgs = 'parameters/20161101/list of images.txt'
        sample_roi = 'parameters/20161101/sample rois.csv'
        cal_roi = 'parameters/20161101/calibration rois.csv'
        uvvis = 'parameters/20161101/list of uvvis.txt'
        measure_all(imgs, sample_roi, cal_roi, uvvis, self.output()['sample'].path,
                    self.output()['calibration'].path,
                    self.output()['abs'].path
                    )

    def output(self):
        return {'sample': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' sample l values.csv'),
                'calibration': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' calibration l values.csv'),
                'abs': luigi.LocalTarget('data/lab-uvvis/' + str(self.name) + ' absorption.csv')}


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # cleanup(MeasureColorCharts(name='20161101'))
    cleanup(PlotColorCharts(name='20161101'))
    luigi.run(['PlotColorCharts', '--name', '20161101'])


if __name__ == "__main__":
    main()
