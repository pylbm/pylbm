# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Base module that describes what we need to define a new viewer.
"""
from abc import ABC, abstractmethod

class Viewer(ABC):
    """
    Base class which describes the methods needed to
    define a new viewer
    """
    @abstractmethod
    def text(self, t, pos, fontsize, color, **kwargs):
        """
        add text on the figure
        """
        pass

    @abstractmethod
    def line(self, pos, width, color, **kwargs):
        """
        add line on th figure
        """
        pass

    @abstractmethod
    def draw(self):
        """
        show figure
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear the figure
        """
        pass

    @abstractmethod
    def axis(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """
        define the axis
        """
        pass

    @abstractmethod
    def is3d(self):
        """
        3d support of the viewer
        """
        pass
