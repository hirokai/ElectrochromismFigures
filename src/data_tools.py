import numpy as np
import util
import os


#
# File manipulation
#

def ambiguous_path(path_head):
    import glob
    files = glob.glob(path_head)
    assert len(files) == 1, 'Only one file must be found (%d files found).' % len(files)
    return files[0]


#
# CSV read/write
#

def save_csv(path, rows):
    import csv

    folder = os.path.dirname(path)
    util.ensure_exists(folder)

    with open(path, 'wb') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(map(str, row))


def load_csv(path, skip_rows=0,numpy=False):
    import csv

    def may_float(s):
        try:
            return float(s)
        except ValueError:
            return s

    with open(path) as f:
        reader = csv.reader(f)
        for _ in range(skip_rows):
            reader.next()
        r = [row for row in reader]
        if numpy:
            return np.array(r).astype(np.float)
        else:
            return r


#
# Table data processing
#


def test_split():
    split_t = map(lambda a: a + 2, range(0, 361, 60))
    ts = np.arange(0, 400, 0.2)
    ys = ts * 2
    print split_trace(ts, ys, split_t)


def split_trace(ts_o, ys_o, split_t):
    # For oxidation
    ts = np.array(ts_o)
    ys = np.array(ys_o)
    assert (len(ts) == len(ys))
    tis = [0]
    ts_s = []
    ys_s = []
    for t in split_t:
        try:
            idx = np.argmin(np.abs(ts - t))
            tis.append(idx)
        except:
            pass
    tis.append(len(ts) - 1)
    for i in range(len(tis) - 1):
        ts_s.append(ts[tis[i]:tis[i + 1]])
        ys_s.append(ys[tis[i]:tis[i + 1]])
    return filter(lambda a: a.size != 0, ts_s), filter(lambda a: a.size != 0, ys_s)


#
# Plotting
#

colors10 = """#1f77b4
#ff7f0e
#2ca02c
#d62728
#9467bd
#8c564b
#e377c2
#7f7f7f
#bcbd22
#17becf""".split('\n')
