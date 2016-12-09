from skimage import io, color
import numpy as np
import sys
import scipy.stats


def get_cie_roi(path, roi):
    rgb = io.imread(path)[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
    lab = color.rgb2lab(rgb)
    return np.mean(lab, axis=(0, 1))


def get_cie_rois(path, rois, mode='mean'):
    img = io.imread(path)
    labs = []
    for roi in rois:
        rgb = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        lab = color.rgb2lab(rgb)
        if mode == 'mean':
            vs = np.mean(lab, axis=(0, 1))
        elif mode == 'mode':
            raise ValueError('No implementation yet.')
        else:
            raise ValueError('Invalid param.')
        labs.append(vs)
    return np.array(labs)


def do_cie_analysis(i, path, roi, show_progress=True):
    if show_progress and i > 0 and i % 100 == 0:
        print('%d files processed' % i)
    if show_progress:
        sys.stdout.write('.')
        sys.stdout.flush()
    return np.concatenate((np.array([i]), get_cie_roi(path, roi)))


def get_cie_l_rois(img_path, rois):
    rgb_whole = io.imread(img_path)
    ms = []
    for roi in rois:
        rgb = rgb_whole[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
        lab = color.rgb2lab(rgb)
        m = np.mean(lab[:, :, 0])
        ms.append(m)
    return ms
