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

    Examples
    --------

    the sphere centered in (0, 0, 0) with radius 1

    >>> center = [0., 0., 0]
    >>> radius = 1.
    >>> Sphere(center, radius)
        Sphere([0 0 0],1) (solid)

    Methods
    -------
    get_bounds :
      return the bounds of the sphere
    point_inside :
      return True or False if the points are in or out the sphere
    distance :
      get the distance of a point to the sphere
    """
    number_of_bounds = 1 # number of edges

    def __init__(self, center, radius, label = 0, isfluid = False):
        self.log = setLogger(__name__)
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

    def point_inside(self, x, y, z):
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
        v2 = np.asarray([x - self.center[0], y - self.center[1], z - self.center[2]])
        return (v2[0]**2 + v2[1]**2 + v2[2]**2)<=self.radius**2

    def distance(self, x, y, z, v, dmax=None):
        """
        Compute the distance in the v direction between
        the sphere and the points defined by (x, y, z).

        .. image:: figures/Sphere.png
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
        p = np.asarray([x - self.center[0], y - self.center[1], z - self.center[2]])
        v2 = v[0]**2 + v[1]**2 + v[2]**2
        delta = - (p[0]*v[1] - p[1]*v[0])**2 \
                - (p[0]*v[2] - p[2]*v[0])**2 \
                - (p[2]*v[1] - p[1]*v[2])**2 \
                + self.radius**2*v2
        ind = delta>=0

        delta[ind] = np.sqrt(delta[ind])/v2

        d = -np.ones(delta.shape)
        d1 = 1e16*np.ones(delta.shape)
        d2 = 1e16*np.ones(delta.shape)

        d1 = -v[0]/v2*p[0] - v[1]/v2*p[1] - v[2]/v2*p[2] - delta
        d2 = -v[0]/v2*p[0] - v[1]/v2*p[1] - v[2]/v2*p[2] + delta

        d1[d1<0] = 1e16
        d2[d2<0] = 1e16
        d[ind] = np.minimum(d1[ind], d2[ind])
        d[d==1e16] = -1
        alpha = -np.ones((x.size, y.size, z.size))
        border = -np.ones((x.size, y.size, z.size))

        if dmax is None:
            ind = d>0
        else:
            ind = np.logical_and(d>0, d<=dmax)
        alpha[ind] = d[ind]
        border[ind] = self.label[0]
        return alpha, border

    def __str__(self):
        s = 'Sphere(' + self.center.__str__() + ',' + str(self.radius) + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(3)):
        #viewer.ellipse_3D(self.center*scale, tuple(self.radius*scale), color)
        u = np.linspace(0, 2.*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.center[0]*scale[0] + self.radius*scale[0]*np.outer(np.cos(u), np.sin(v))
        y = self.center[1]*scale[1] + self.radius*scale[1]*np.outer(np.sin(u), np.sin(v))
        z = self.center[2]*scale[2] + self.radius*scale[2]*np.outer(np.ones(np.size(u)), np.cos(v))
        viewer.plot_surface(x, y, z, rstride=4, cstride=4, color=color)
        #if viewlabel:
        #    theta = self.center[0] + 2*self.center[1]+10*self.radius
        #    x, y = self.center[0] + self.radius*np.cos(theta), self.center[1] + self.radius*np.sin(theta)
        #    viewer.text(str(self.label[0]), [x, y])
