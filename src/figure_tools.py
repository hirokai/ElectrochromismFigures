import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os


def ensure_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def figure(name, show=False):
    def _deco_save_fig(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            plt.clf()
            res = func(*args, **kwargs)
            ensure_exists('../dist')
            plt.savefig('../dist/Fig ' + name + '.pdf')
            if show:
                plt.show()
            return res

        return wrapper

    return _deco_save_fig


def set_format(ax, xticks, yticks, x_minor, y_minor):
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor))
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(15)
