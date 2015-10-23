# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import abc
from future.utils import with_metaclass

class Viewer(with_metaclass(abc.ABCMeta, object)):
    @abc.abstractmethod
    def text(self, t, pos, fontsize, color, **kwargs):
        """
        add text on the figure
        """
        return

    @abc.abstractmethod
    def line(self, pos, width, color, **kwargs):
        """
        add line on th figure
        """
        return

    @abc.abstractmethod
    def draw(self):
        """
        show figure
        """
        return

    @abc.abstractmethod
    def clear(self):
        return

    @abc.abstractmethod
    def axis(self, xmin, xmax, ymin, ymax, zmin, zmax):
        return

    @abc.abstractproperty
    def is3d(self):
        """
        3d support of the viewer
        """
        return
