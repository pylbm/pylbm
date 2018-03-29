# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
from .base import Element
from ..logs import setLogger
from .utils import *

class Circle(Element):
    """
    Class Circle

    Parameters
    ----------
    center : a list that contains the two coordinates of the center
    radius : a positive float for the radius
    label : list of one integer (default [0])
    isfluid : boolean
             - True if the circle is added
             - False if the circle is deleted

    Attributes
    ----------
    number_of_bounds : int
      1
    center : numpy array
      the coordinates of the center of the circle
    radius : double
      positive float for the radius of the circle
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the circle is added
      and False if the circle is deleted

    Examples
    --------

    the circle centered in (0, 0) with radius 1

    >>> center = [0., 0.]
    >>> radius = 1.
    >>> Circle(center, radius)
        Circle([0 0],1) (solid)

    """
    def __init__(self, center, radius, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 1 # number of edges
        self.center = np.asarray(center)
        if radius >= 0:
            self.radius = radius
        else:
            self.log.error('The radius of the circle should be positive')
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the circle.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the circle.

        Notes
        -----

        the edge of the circle is considered as inside.

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points

        Returns
        -------

        Array of boolean (True inside the circle, False otherwise)
        """
        x, y = grid
        v2 = np.asarray([x - self.center[0], y - self.center[1]])
        return (v2[0]**2 + v2[1]**2)<=self.radius**2

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between
        the circle and the points defined by (x, y).

        .. image:: ../figures/Circle.png
            :width: 100%

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points
        v : direction of interest

        Returns
        -------

        array of distances

        """
        x, y = grid
        v1 = self.radius*np.array([1,0])
        v2 = self.radius*np.array([0,1])
        return distance_ellipse(x, y, v, self.center, v1, v2, dmax, self.label)

    def __str__(self):
        s = 'Circle(' + self.center.__str__() + ',' + str(self.radius) + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(2), alpha=1.):
        viewer.ellipse(self.center*scale, tuple(self.radius*scale), color, alpha=alpha)
        if viewlabel:
            theta = self.center[0] + 2*self.center[1]+10*self.radius
            x, y = self.center[0] + self.radius*np.cos(theta), self.center[1] + self.radius*np.sin(theta)
            viewer.text(str(self.label[0]), [x*scale[0], y*scale[1]])
