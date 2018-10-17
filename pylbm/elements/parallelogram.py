# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Parallelogram element
"""

#pylint: disable=invalid-name

import logging
import numpy as np
from six.moves import range

from .base import Element
from .utils import distance_lines

log = logging.getLogger(__name__) #pylint: disable=invalid-name

class Parallelogram(Element):
    """
    Class Parallelogram

    Parameters
    ----------

    point : list
        the coordinates of the first point of the parallelogram
    vecta : list
        the coordinates of the first vector
    vectb : list
        the coordinates of the second vector
    label : list
        four integers (default [0, 0, 0, 0])
    isfluid : boolean
        - True if the parallelogram is added
        - False if the parallelogram is deleted

    Examples
    --------

    the square [0,1]x[0,1]

    >>> point = [0., 0.]
    >>> vecta = [1., 0.]
    >>> vectb = [0., 1.]
    >>> Parallelogram(point, vecta, vectb)
        Parallelogram([0 0],[0 1],[1 0]) (solid)

    Attributes
    ----------

    number_of_bounds : int
        4
    point : ndarray
        the coordinates of the first point of the parallelogram
    v0 : ndarray
        the coordinates of the first vector
    v1 : ndarray
        the coordinates of the second vector
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the parallelogram is added
        and False if the parallelogram is deleted

    """
    def __init__(self, point, vecta, vectb, label=0, isfluid=False):
        self.number_of_bounds = 4 # number of edges
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        super(Parallelogram, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        """
        return the bounds of the parallelogram.
        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the parallelogram.

        Notes
        -----

        the edges of the parallelogram are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the parallelogram, False otherwise)

        """
        x, y = grid
        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]])
        invdelta = 1./(self.v0[0]*self.v1[1] - self.v0[1]*self.v1[0])
        u = (v2[0]*self.v1[1] - v2[1]*self.v1[0])*invdelta
        v = (v2[1]*self.v0[0] - v2[0]*self.v0[1])*invdelta
        return np.logical_and(np.logical_and(u >= 0, v >= 0),
                              np.logical_and(u <= 1, v <= 1))

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between the parallelogram
        and the points defined by (x, y).

        .. image:: ../figures/Parallelogram.png
            :width: 100%

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max

        Returns
        -------

        ndarray
            array of distances

        """
        x, y = grid
        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v1, self.v0]
        vt = [self.v0, self.v1, self.v0, self.v1]

        return distance_lines(x - self.point[0], y - self.point[1],
                              v, p, vt, dmax, self.label)

    def __str__(self):
        s = 'Parallelogram(' + self.point.__str__() + ','
        s += self.v0.__str__() + ',' + self.v1.__str__()  + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def visualize(self, viewer, color, viewlabel=False, scale=np.ones(2), alpha=1.):
        A = [self.point[k] for k in range(2)]
        B = [A[k] + self.v0[k] for k in range(2)]
        C = [B[k] + self.v1[k] for k in range(2)]
        D = [A[k] + self.v1[k] for k in range(2)]
        viewer.polygon([A, B, C, D], color, alpha=alpha)
        if viewlabel:
            viewer.text(str(self.label[0]), [0.5*(A[0]+B[0]), 0.5*(A[1]+B[1])])
            viewer.text(str(self.label[1]), [0.5*(A[0]+D[0]), 0.5*(A[1]+D[1])])
            viewer.text(str(self.label[2]), [0.5*(C[0]+D[0]), 0.5*(C[1]+D[1])])
            viewer.text(str(self.label[3]), [0.5*(B[0]+C[0]), 0.5*(B[1]+C[1])])
