from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
from six.moves import range

from .base import Element
from ..logs import setLogger
from .utils import *

class Triangle(Element):
    """
    Class Triangle

    Parameters
    ----------

    point : list
        the coordinates of the first point of the triangle
    vecta : list
        the coordinates of the first vector
    vectb : list
        the coordinates of the second vector
    label : list of three integers (default [0, 0, 0])
    isfluid : boolean
             - True if the triangle is added
             - False if the triangle is deleted

    Examples
    --------

    the bottom half square of [0,1]x[0,1]

    >>> point = [0., 0.]
    >>> vecta = [1., 0.]
    >>> vectb = [0., 1.]
    >>> Triangle(point, vecta, vectb)
        Triangle([0 0],[0 1],[1 0]) (solid)

    Attributes
    ----------

    point : numpy array
      the coordinates of the first point of the triangle
    vecta : numpy array
      the coordinates of the first vector
    vectb : numpy array
      the coordinates of the second vector
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the triangle is added
      and False if the triangle is deleted
    number_of_bounds : int
        number of edges

    """
    def __init__(self, point, vecta, vectb, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 3 # number of edges
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        return the smallest box where the triangle is.

        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the triangle.

        Notes
        -----

        the edges of the triangle are considered as inside.

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points

        Returns
        -------

        Array of boolean (True inside the triangle, False otherwise)

        """
        x, y = grid
        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]])
        invdelta = 1./(self.v0[0]*self.v1[1] - self.v0[1]*self.v1[0])
        u = (v2[0]*self.v1[1] - v2[1]*self.v1[0])*invdelta
        v = (v2[1]*self.v0[0] - v2[0]*self.v0[1])*invdelta
        return np.logical_and(np.logical_and(u>=0, v>=0), u + v<=1)

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between the triangle
        and the points defined by (x, y).

        .. image:: ../figures/Triangle.png
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
        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v0]
        vt = [self.v0, self.v1, self.v1 - self.v0]

        return distance_lines(x - self.point[0], y - self.point[1],
                              v, p, vt, dmax, self.label)

    def __str__(self):
        s = 'Triangle(' + self.point.__str__() + ','
        s += self.v0.__str__() + ',' + self.v1.__str__()  + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, viewer, color, viewlabel, alpha=1.):
        A = [self.point[k] for k in range(2)]
        B = [A[k] + self.v0[k] for k in range(2)]
        D = [A[k] + self.v1[k] for k in range(2)]
        viewer.polygon([A,B,D], color, alpha=alpha)
        if viewlabel:
            viewer.text(str(self.label[0]), [0.5*(A[0]+B[0]), 0.5*(A[1]+B[1])])
            viewer.text(str(self.label[1]), [0.5*(A[0]+D[0]), 0.5*(A[1]+D[1])])
            viewer.text(str(self.label[2]), [0.5*(B[0]+D[0]), 0.5*(B[1]+D[1])])
