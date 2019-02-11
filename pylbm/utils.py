# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

# pylint: disable=invalid-name, missing-docstring

"""
utils module
"""


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


def header_string(title):
    # pylint: disable=unused-import
    from colorama import init, Fore, Style, Back
    init()
    barre = '+' + '-'*(len(title)+2) + '+'
    output = '\n| %s |\n' % title
    return Fore.BLUE + barre + output + barre + Fore.RESET


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
