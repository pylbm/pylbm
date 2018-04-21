# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from .base import Element
from ..logs import setLogger
from .utils import *

class Ellipsoid(Element):
    """
    Class Ellipsoid

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v1 : a vector
    v2 : a vector
    v3 : a vector (v1, v2, and v3 have to be orthogonal)
    label : list of one integer (default [0])
    isfluid : boolean
             - True if the ellipsoid is added
             - False if the ellipsoid is deleted

    Attributes
    ----------
    number_of_bounds : int
      1
    center : numpy array
      the coordinates of the center of the sphere
    v1 : numpy array
      the coordinates of the first vector
    v2 : numpy array
      the coordinates of the second vector
    v3 : numpy array
      the coordinates of the third vector
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the ellipsoid is added
      and False if the ellipsoid is deleted
    number_of_bounds : int
        number of edges (1)

    Examples
    --------

    the ellipsoid centered in (0, 0, 0) with v1=[3,0,0], v2=[0,2,0], and v3=[0,0,1]

    >>> center = [0., 0., 0.]
    >>> v1, v2, v3 = [3,0,0], [0,2,0], [0,0,1]
    >>> Ellipsoid(center, v1, v2, v3)
        Ellipsoid([0 0 0], [3 0 0], [0 2 0], [0 0 1]) (solid)

    """
    def __init__(self, center, v1, v2, v3, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 1 # number of edges
        self.center = np.asarray(center)
        p12 = abs(v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2])
        p23 = abs(v2[0]*v3[0] + v2[1]*v3[1] + v2[2]*v3[2])
        p31 = abs(v3[0]*v1[0] + v3[1]*v1[1] + v3[2]*v1[2])
        if  max(p12, p23, p31)> 1.e-14:
            self.log.error('The vectors of the ellipsoid are not orthogonal')
        else:
            self.v1 = np.asarray(v1)
            self.v2 = np.asarray(v2)
            self.v3 = np.asarray(v3)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the ellipsoid.
        """
        r = max(np.linalg.norm(self.v1),
                np.linalg.norm(self.v2),
                np.linalg.norm(self.v3))
        return self.center - r, self.center + r

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the ellipsoid.

        Notes
        -----

        the edge of the ellipsoid is considered as inside.

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points
        z : z coordinates of the points

        Returns
        -------

        Array of boolean (True inside the ellipsoid, False otherwise)
        """
        x, y, z = grid

        X = x - self.center[0]
        Y = y - self.center[1]
        Z = z - self.center[2]
        v12 = np.cross(self.v1, self.v2)
        v23 = np.cross(self.v2, self.v3)
        v31 = np.cross(self.v3, self.v1)
        d = np.inner(self.v1, v23)**2
        # equation of the ellipsoid:
        # cxx XX + cyy YY + czz ZZ + cxy XY + cyz YZ + czx ZX = d
        cxx = v12[0]**2 + v23[0]**2 + v31[0]**2
        cyy = v12[1]**2 + v23[1]**2 + v31[1]**2
        czz = v12[2]**2 + v23[2]**2 + v31[2]**2
        cxy = 2 * (v12[0]*v12[1] + v23[0]*v23[1] + v31[0]*v31[1])
        cyz = 2 * (v12[1]*v12[2] + v23[1]*v23[2] + v31[1]*v31[2])
        czx = 2 * (v12[2]*v12[0] + v23[2]*v23[0] + v31[2]*v31[0])
        return cxx*X**2 + cyy*Y**2 + czz*Z**2 + \
               cxy*X*Y + cyz*Y*Z + czx*Z*X <= d

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between
        the ellipsoid and the points defined by (x, y, z).

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points
        z : z coordinates of the points
        v : direction of interest

        Returns
        -------

        array of distances
        """
        x, y, z = grid
        return distance_ellipsoid(x, y, z, v, self.center, self.v1, self.v2, self.v3, dmax, self.label)

    def __str__(self):
        s = 'Ellipsoid(' + self.center.__str__() + ',' + str(self.v1) + ',' + str(self.v2) + ',' + str(self.v3) + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(3), alpha=1.):
        v1 = scale*self.v1
        v2 = scale*self.v2
        v3 = scale*self.v3
        viewer.ellipse_3D(self.center*scale, v1, v2, v3 , color, alpha=alpha)
        if viewlabel:
            x, y, z = self.center[0], self.center[1], self.center[2]
            viewer.text(str(self.label[0]), [x, y, z])
