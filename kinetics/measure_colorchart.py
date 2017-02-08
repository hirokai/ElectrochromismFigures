# -*- coding: utf-8 -*-

import csv
import os
import sys

import luigi
import matplotlib.pyplot as plt
import numpy as np
from src.image_tools import get_cie_rois
from skimage import color
from skimage import io

from src.util import basename_noext, ensure_folder_exists


class RectROI:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def values(self):
        return [self.x, self.y, self.w, self.h]

    def __repr__(self):
        return 'Rect(%d,%d,%d,%d)' % (self.x, self.y, self.w, self.h)


class PointROI:
    def __init__(self, points):
        self.points = points

    def __getitem__(self, item):
        return self.points[item]

    def rect(self):  # Return bounding rect.
        xs = map(lambda a: a[0], self.points)
        ys = map(lambda a: a[1], self.points)
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        return [x, y, w, h]

    def __repr__(self):
        return 'PointROI(%s)' % self.points.__repr__()


def read_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.reader(f)
        reader.next()
        rows = [map(float, r) for r in reader]
    vs = {}
    for r in rows:
        i = int(r[0])
        if i not in vs:
            vs[i] = []
        vs[i].append([r[1], r[2]])
    rois = {}
    for k, v in vs.iteritems():
        if len(v) % 4 != 0:
            raise ValueError('the number of points must be multiple of 4')
        rs = []
        for i in range(len(v) / 4):
            rs.append(PointROI(v[i * 4:i * 4 + 4]))
        rois[k] = rs
    return rois


# McBeth color chart values from Wikipedia.
chart_colors_str = """735244 c29682 627a9d 576c43 8580b1 67bdaa d67e2c 505ba6 c15a63 5e3c6c 9dbc40 e0a32e """ + \
                   """383d96 469449 af363c e7c71f bb5695 0885a1 f3f3f2 c8c8c8 a0a0a0 7a7a79 555555 343434"""


def extract_l_from_str(s):
    def h(s):
        return float(int(s, 16)) / 255

    rgb = np.zeros((1, 1, 3))
    rgb[0, 0, :] = [h(s[0:2]), h(s[2:4]), h(s[4:6])]
    lab = color.rgb2lab(rgb)
    return lab[0, 0, 0]


chart_l_values = map(extract_l_from_str, chart_colors_str.split(" "))


def sort_with_order(vs, order):
    r = np.zeros(len(order))
    for i, o in enumerate(order):
        r[i] = vs[o]
    return r


def get_image_path(root_folder, name, num):
    return os.path.join(root_folder, 'slices', basename_noext(name), 'out%04d.png' % num)


def mk_cells(ps, s=4.0):
    # Two vectors
    x0 = ps[0][0]
    y0 = ps[0][1]
    ax = (ps[1][0] - x0) / 4
    ay = (ps[1][1] - y0) / 4
    bx = (ps[3][0] - x0) / 6
    by = (ps[3][1] - y0) / 6

    cells = []
    w = 0.1  # width ratio
    for ai in range(4):
        for bi in range(6):
            cells.append(
                RectROI(*map(int, [x0 + ax * (ai + 0.4) - s / 2 + bx * (bi + 0.4),
                                   y0 + ay * (ai + 0.3) - s / 2 + by * (bi + 0.3),
                                   s,
                                   s])))
    return cells


def find_surrounding_rect(rois):
    x_min = min(map(lambda a: a[0], rois))
    x_max = max(map(lambda a: a[0] + a[2], rois))
    y_min = min(map(lambda a: a[1], rois))
    y_max = max(map(lambda a: a[1] + a[3], rois))
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def draw_cells(cellss, img=None, count=None):
    from itertools import chain

    cells = list(chain.from_iterable(cellss))
    for c in cells:
        r = plt.Rectangle(xy=[c.x + c.w / 2, c.y + c.h / 2], width=c.w, height=c.h, facecolor='none', edgecolor='red')
        plt.gca().add_patch(r)
    if img is None:
        plt.xlim([0, 500])
        plt.ylim([0, 500])
    else:
        plt.imshow(img)
    if count is not None:
        plt.title(str(count))
    plt.show()


def get_cie_rois_custom(path, rois):
    img = io.imread(path)
    labs = []
    img2 = np.zeros((240, 10, 3))
    count = 0
    for roi in rois:
        rgb = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        img2[count * 10:count * 10 + roi[3], 0:roi[2], :] = rgb
        lab = color.rgb2lab(rgb)
        labs.append(np.mean(lab, axis=(0, 1)))
        count += 1
    plt.imshow(img2)
    plt.show()
    return np.array(labs)


def extract_used_l(ls):
    return [ls[23], ls[22], ls[21], ls[18]]


def mk_rgb_column(labs):
    lab = np.zeros((240, 10, 3))
    for i in range(24):
        lab[(i * 10):(i * 10 + 10), :, 0] = labs[i, 0]
        lab[(i * 10):(i * 10 + 10), :, 1] = labs[i, 1]
        lab[(i * 10):(i * 10 + 10), :, 2] = labs[i, 2]
    rgb = color.lab2rgb(lab)
    return rgb


def measure_color_chart(path, rois, cell_width=4.0, debug_drawcells=False):
    # There are 24 cells for each ROI of color chart.
    cellss = map(lambda roi: mk_cells(roi, s=cell_width), rois)

    if debug_drawcells:
        # Draw 24 cells for measurement.
        print(zip(rois, cellss))
        img2 = io.imread(path)
        draw_cells(cellss, img2)

    labss = []

    for cells in cellss:
        # Collect 24 mean (L,a,b) values of each of 24 cells.
        labs = get_cie_rois(path, map(lambda cell: cell.values(), cells))
        labss.append(labs)

    return labss


# Measure L values from ROIs of 24 cells of the color chart for each image.
def process_movies(root_folder, movie_conditions_path, roi_path, out_folder, slice_nums=[1],
                   debug_show_samples=False,
                   debug_show_l_plots=False):
    with open(movie_conditions_path) as f:
        reader = csv.reader(f)
        reader.next()
        rows = [r for r in reader]
        movie_names = [r[0] for r in rows]

    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))

    chart_order = [i[0] for i in sorted(enumerate(chart_l_values), key=lambda x: x[1])]
    # print(list(enumerate(chart_order)))
    # plt.plot(chart_l_values, marker='o')
    # plt.show()
    ordered_chart_values = sort_with_order(chart_l_values, chart_order)
    # plt.plot(ordered_chart_values, marker='o')
    # plt.show()

    roiss = read_csv(roi_path)
    count_rois = len(reduce(lambda a, b: a + b, roiss.values()))
    if debug_show_samples:
        img_samples = np.zeros((10 * 24, 10 * count_rois, 3))  # for Lab samples
    count = 0
    line_counts = []  # Will be used for vertical lines between images.
    slices = []
    movie_name_per_slice = []

    if not slice_nums:  # If empty.
        slice_nums = range(1, 10000)  # Sufficiently large number

    # Loop over movies.
    for k, rois in roiss.iteritems():
        sys.stdout.write('Processing: %s ' % movie_names[int(k) - 1])
        roi_idxs = []
        nums = []
        lss = []

        result = []
        for _ in range(len(rois)):
            result.append([])
        # Loop over slices of the movie.
        for slice_count, slice_num in enumerate(slice_nums):
            if slice_count % 10 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            path = get_image_path(root_folder, movie_names[int(k) - 1], slice_num)
            if not os.path.exists(path):
                break

            # Collect 24 mean (L,a,b) values of each of 24 cells, possibly for multiple ROIs.
            labss = measure_color_chart(path, rois)

            roi_idx = 1
            for labs in labss:
                slices.append(k)  # slices has 1-based index of image.
                lss.append(labs[:, 0])
                result[roi_idx - 1].append(labs[:, 0])
                nums.append(slice_num)
                roi_idxs.append(roi_idx)
                if np.isnan(labs[0, 0]):
                    raise ValueError("NaN in Lab measurement.")
                if debug_show_samples:
                    img_samples[:, (count * 10):(count * 10 + 10), :] = mk_rgb_column(labs)
                movie_name_per_slice.append(movie_names[len(line_counts)])
                count += 1
                roi_idx += 1
        lss = np.array(lss)
        l_mean = np.mean(lss, axis=0)
        chart_order = [i[0] for i in sorted(enumerate(l_mean), key=lambda x: x[1])]
        lss2 = np.zeros((len(chart_order), lss.shape[0]))
        if debug_show_l_plots:
            plt.plot(ordered_chart_values, linewidth=3)
            for i in range(lss.shape[1]):
                lss2[:, i] = sort_with_order(lss[:, i], chart_order)
                plt.plot(lss2)
                plt.show()
        for roi_idx in range(len(result)):
            m = basename_noext(movie_names[int(k) - 1])
            out_path = os.path.join(out_folder, "%s_%d.csv" % (m,roi_idx+1))
            ensure_folder_exists(out_path)
            with open(out_path, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(['movie_name', 'movie_slice', 'roi_index'] + ['L%d' % i for i in range(1, 25)])
                for slice_idx, ls in enumerate(result[roi_idx]):
                    writer.writerow(
                        [movie_name_per_slice[slice_idx], '%04d' % (slice_idx + 1), '%d' % (roi_idx + 1)] +
                        list(map(lambda l: ('%03.4f' % l).rjust(8), ls))
                    )
        sys.stdout.write('\n')


class MeasureLValuesOfColorCharts(luigi.Task):
    name = luigi.Parameter()
    folder = luigi.Parameter()
    roipath = luigi.Parameter()

    def run(self):
        movie_condition_path = os.path.join('parameters', self.name, 'movie_conditions.csv')
        out_path = os.path.join(self.output().path)
        print('Output: ' + out_path)
        process_movies(self.folder, movie_condition_path, self.roipath, out_path,
                       slice_nums=[],  # All frames
                       debug_show_samples=False)

    def output(self):
        # Folders
        return luigi.LocalTarget(os.path.join('data', 'kinetics', 'colorchart', str(self.name)))


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    # shutil.rmtree(os.path.join('data', 'kinetics', 'colorchart', '20161013'), ignore_errors=True)
    luigi.run(
        ['MeasureLValuesOfColorCharts', '--name', '20161019',
         '--folder', '/Volumes/ExtWork/Suda Electrochromism/20161019/',
         '--roipath', 'parameters/20161019/calibration_rois.csv'])


if __name__ == "__main__":
    main()
