# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from .base import Element
from ..logs import setLogger
from .utils import *

class Sphere(Element):
    """
    Class Sphere

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    radius : a positive float for the radius
    label : list of one integer (default [0])
    isfluid : boolean
             - True if the sphere is added
             - False if the sphere is deleted

    Attributes
    ----------
    number_of_bounds : int
      1
    center : numpy array
      the coordinates of the center of the sphere
    radius : double
      positive float for the radius of the sphere
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the sphere is added
      and False if the sphere is deleted
    number_of_bounds : int
        number of edges (1)

    Examples
    --------

    the sphere centered in (0, 0, 0) with radius 1

    >>> center = [0., 0., 0.]
    >>> radius = 1.
    >>> Sphere(center, radius)
        Sphere([0 0 0],1) (solid)

    """
    def __init__(self, center, radius, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 1 # number of edges
        self.center = np.asarray(center)
        if radius>=0:
            self.radius = radius
        else:
            self.log.error('The radius of the sphere should be positive')
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the sphere.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the sphere.

        Notes
        -----

        the edge of the sphere is considered as inside.

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points
        z : z coordinates of the points

        Returns
        -------

        Array of boolean (True inside the sphere, False otherwise)
        """
        x, y, z = grid
        v2 = np.asarray([x - self.center[0], y - self.center[1], z - self.center[2]])
        return (v2[0]**2 + v2[1]**2 + v2[2]**2)<=self.radius**2

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between
        the sphere and the points defined by (x, y, z).

        .. image:: ../figures/Sphere.png
            :width: 100%

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
        v1 = self.radius*np.array([1,0,0])
        v2 = self.radius*np.array([0,1,0])
        v3 = self.radius*np.array([0,0,1])
        return distance_ellipsoid(x, y, z, v, self.center, v1, v2, v3, dmax, self.label)


    def __str__(self):
        s = 'Sphere(' + self.center.__str__() + ',' + str(self.radius) + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(3), alpha=1.):
        v1 = self.radius*np.array([1,0,0])*scale
        v2 = self.radius*np.array([0,1,0])*scale
        v3 = self.radius*np.array([0,0,1])*scale
        viewer.ellipse_3D(self.center*scale, v1, v2, v3, color, alpha=alpha)
        if viewlabel:
            x, y, z = self.center[0], self.center[1], self.center[2]
            viewer.text(str(self.label[0]), [x, y, z])
