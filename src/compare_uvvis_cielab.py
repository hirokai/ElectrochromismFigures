import csv
import os
from image_tools import get_cie_rois
import matplotlib.pyplot as plt
from measure_calibration_lab import read_csv,mk_cells

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


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
    with open('parameters/20161101/list of images.txt') as f:
        img_paths = map(lambda s: s.strip(), f.readlines())
    with open('parameters/20161101/list of uvvis.txt') as f:
        uvvis_paths = map(lambda s: s.strip(), f.readlines())
    print(uvvis_paths)
    abs_570 = [get_abs(n, 570) for n in uvvis_paths]
    with open('parameters/20161101/sample rois.csv') as f:
        reader = csv.reader(f)
        reader.next()
        sample_rois = [map(float, r)[1:] for r in reader]
    crs = read_csv('parameters/20161101/calibration rois.csv')
    calibration_rois = []
    for k,v in crs.iteritems():
        calibration_rois.append(v[0])
    ls = []
    for path, sample_roi, cal_roi in zip(img_paths, sample_rois,calibration_rois):
        r = get_cie_rois(path, [sample_roi])[0, 0]
        cells = mk_cells(cal_roi)
        ls2 = get_cie_rois(path, map(lambda cell: cell.values(), cells))
        print(r,cal_roi)
        ls.append(r)
    print(abs_570)
    print(ls)
    plt.scatter(abs_570, ls)
    plt.show()


if __name__ == "__main__":
    main()
