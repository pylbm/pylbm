# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

# pylint: disable=invalid-name, missing-docstring

"""
utils module
"""

import sys
import logging
from colorama import Fore, Style, Back  # pylint: disable=unused-import

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


def header_string(title):
    barre = '+' + '-'*(len(title)+2) + '+'
    output = '\n| %s |\n' % title
    return Fore.BLUE + barre + output + barre + Fore.RESET


class itemproperty(property):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj)


class item2property(property):
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        else:
            return bounditemproperty(self, obj, True)


class bounditemproperty:
    def __init__(self, item_property, instance, nextItem=False):
        self.__item_property = item_property
        self.__instance = instance
        self.nextItem = nextItem

    def __getitem__(self, key):
        fget = self.__item_property.fget
        if fget is None:
            raise AttributeError("unreadable attribute item")
        if self.nextItem:
            return bound2itemproperty(
                self.__item_property,
                self.__instance,
                key
            )
        else:
            return fget(self.__instance, key)

    def __setitem__(self, key, value):
        fset = self.__item_property.fset
        if fset is None:
            raise AttributeError("can't set attribute item")
        fset(self.__instance, key, value)


class bound2itemproperty:
    def __init__(self, item_property, instance, key):
        self.__item_property = item_property
        self.__instance = instance
        self.key = key

    def __getitem__(self, key):
        fget = self.__item_property.fget
        if fget is None:
            raise AttributeError("unreadable attribute item")
        return fget(self.__instance, self.key, key)

    def __setitem__(self, key, value):
        fset = self.__item_property.fset
        if fset is None:
            raise AttributeError("can't set attribute item")
        fset(self.__instance, self.key, key, value)


def hsl_to_rgb(h, s, l):
    """
    Converts an HSL color value to RGB. Conversion formula
    adapted from http://en.wikipedia.org/wiki/HSL_color_space.
    Assumes h, s, and l are contained in the set [0, 1] and
    the output r, g, and b are in the set [0, 1].

    Parameters
    ----------

    h : double
        the hue
    s : double
        the saturation
    l : double
        the lightness

    Returns
    -------

    tuple
        the color in RGB format
    """

    if s == 0:  # achromatic
        r = l
        g = l
        b = l
    else:
        def hue2rgb(p, q, t):
            t = t % 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)

    return r, g, b


def print_progress(iteration, total,
                   prefix='', suffix='',
                   decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals
                                in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percents = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    p_bar = '*' * filledLength + '-' * (barLength - filledLength)
    # pylint: disable=expression-not-assigned
    sys.stdout.write(
        '\r%s |%s| %s%s %s' % (prefix, p_bar, percents, '%', suffix)
    ),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


class pylbm_progress_bar:
    def __init__(self, nb_total, title=None):
        log.warning(
            "module `alive_progress' not found\n"
            "replaced by my poor own\n"
        )
        self.nb_total = nb_total
        if title is None:
            self.title = ''
        else:
            self.title = title
        self.compt = 0
        print_progress(self.compt, self.nb_total, prefix=self.title)

    def __enter__(self):
        return self.pbar

    def pbar(self):
        self.compt += 1
        print_progress(self.compt, self.nb_total, prefix=self.title)

    # pylint: disable=redefined-builtin
    def __exit__(self, type, value, traceback):
        pass


try:
    from alive_progress import alive_bar, config_handler
    config_handler.set_global(
        spinner='waves', bar='smooth'
    )
    progress_bar = alive_bar
except ImportError:
    progress_bar = pylbm_progress_bar
