import sys

import luigi
import numpy as np
from skimage import io, color

from src.make_slices import MakeSlicesStub
from src.data_tools import save_csv


def run_set(slices_folder, rois):
    slices_name = os.path.basename(slices_folder)
    rows = []
    if not os.path.exists(slices_folder):
        print('Folder not found')
        return
    for i in range(1, 2000):
        path = '%s/output%04d.png' % (slices_folder, i)
        if os.path.exists(path):
            rgb_whole = io.imread(path)
            sys.stdout.write('.')
            sys.stdout.flush()
            ms = []
            for roi in rois:
                rgb = rgb_whole[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2], :]
                lab = color.rgb2lab(rgb)
                m = np.mean(lab[:, :, 0])
                ms.append(m)
            rows.append([i, path] + ms)
    sys.stdout.write('\n')
    file_name = '../data/cielab_space/%s.csv' % slices_name
    save_csv(os.path.join(file_name), rows)


class CollectCIESpace(luigi.Task):
    movie_name = luigi.Parameter()

    with open('./data/CollectCIESpace_rois.txt', 'r') as content_file:
        rois_str = content_file.read()

    rois = map(lambda l: map(int, l.split('\t')), rois_str.split('\n'))

    base_folder = '/Users/hiroyuki/Downloads'

    def requires(self):
        return MakeSlicesStub(os.path.join(self.base_folder, self.movie_name))

    def output(self):
        return [luigi.LocalTarget('../data/cielab_space/%s.csv' % self.movie_name)]

    def run(self):
        run_set(self.input().path, self.rois)


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    try:
        map(lambda o: os.remove(o.path), CollectCIESpace(movie_name='04 MVI_0785 10fps').output())
    except:
        pass
    luigi.run(['CollectCIESpace', '--movie-name', '04 MVI_0785 10fps'])
