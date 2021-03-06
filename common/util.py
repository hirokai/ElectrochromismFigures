import os
import itertools
import numpy as np


def flatten(vs):
    return reduce(lambda a, b: a + b, vs)


# http://stackoverflow.com/questions/14423794/equivalent-of-haskell-scanl-in-python
def scanl(f, base, l):
    for x in l:
        base = f(base, x)
        yield base


def transpose(l):
    return map(list, zip(*l))


# http://stackoverflow.com/questions/4998427/how-to-group-elements-in-python-by-n-elements
def grouper(n, iterable, fillvalue=None):
    """grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"""
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)


# Extract elements by indices.
def by_indices(xs, idxs):
    return [xs[i] for i in filter(lambda i: i < len(xs), idxs)]


# Permutation of list
def sort_with_order(vs, order):
    r = np.zeros(len(order))
    for i, o in enumerate(order):
        r[i] = vs[o]
    return r


def ensure_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def ensure_folder_exists(path):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def basename_noext(path):
    return os.path.splitext(os.path.basename(path))[0]


# For making folder
def get_condition_concat_values(G, n, params):
    param_strs = map(lambda p: str(p[2]), get_conditions(G, n, params))
    if len(param_strs) == 0:
        return 'default'
    else:
        return '_'.join(param_strs)


def get_conditions(G, node, all_params):
    import networkx as nx
    prevs = filter(lambda a: G.node[a].get('type') == 'process', nx.ancestors(G, node)) + [node]
    nodes = sorted(prevs, key=lambda a: G.node[a].get('rank'))

    def f(n):
        ps = all_params.get(n) or []
        return map(lambda p: [n] + p, ps)

    return reduce(lambda a, b: a + b, map(f, nodes))


def get_params_dict(all_params):
    l = reduce(lambda a, b: a + b, map(lambda a: a[1], all_params.items()))
    print(l)
    d = {}
    for l2 in l:
        d[l2[0]] = l2[1]
    return d


def get_condition_keys(G, node, all_params):
    import networkx as nx
    prevs = filter(lambda a: G.node[a].get('type') == 'process', nx.ancestors(G, node)) + [node]
    nodes = sorted(prevs, key=lambda a: G.node[a].get('rank'))

    def f(n):
        ps = all_params.get(n) or []
        return map(lambda p: [n] + p[0:1], ps)

    return reduce(lambda a, b: a + b, map(f, nodes))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def chdir_root():
    os.chdir(os.path.join(os.path.dirname(__file__), os.pardir))
