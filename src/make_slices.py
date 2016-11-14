import luigi
import os
from subprocess import call
from util import ensure_exists, basename_noext


class MakeSlicesStub(luigi.Task):
    folder = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.folder)


def mk_slices(path):
    base_folder = os.path.dirname(path)
    slices_base_folder = os.path.join(base_folder, 'slices')
    ensure_exists(slices_base_folder)
    out_folder = os.path.join(slices_base_folder, basename_noext(path))
    ensure_exists(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_path = os.path.join(out_folder, "out%04d.png")
    print(out_path)
    cmd = ["/Users/hiroyuki/bin/ffmpeg", "-i", path, "-vf", "fps=1", out_path]
    print(cmd)
    call(cmd)


def all_make_slices(folder):
    files = filter(lambda n: n.find('.MOV') >= 0, os.listdir(folder))
    print(files)
    for n in files:
        path = os.path.join(folder, n)
        if os.path.isfile(path):
            mk_slices(path)
