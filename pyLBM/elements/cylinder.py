# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
from .base import Element
from ..logs import setLogger
from .utils import *

class Cylinder(Element):
    """
    Class Cylinder

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    radius : a positive float for the radius of the basis
    v0 : a list of the three coordinates of the first vector that defines the circular section
    v1 : a list of the three coordinates of the second vector that defines the circular section
    w : a list of the three coordinates of the vector that defines the direction of the side
    label : list of three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
             - True if the cylinder is added
             - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
      3
    center : numpy array
      the coordinates of the center of the cylinder
    radius : double
      positive float for the radius of the cylinder
    v0 : list of doubles
      the three coordinates of the first vector that defines the circular section
    v1 : list of doubles
      the three coordinates of the second vector that defines the circular section
    w : list of doubles
      the three coordinates of the vector that defines the direction of the side
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the cylinder is added
      and False if the cylinder is deleted

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2) with radius 1

    >>> center = [0., 0., 0.5]
    >>> radius = 1.
    >>> v0, v1 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> Cylinder(center, radius, v0, v1, w)
        Cylinder([0 0 0.5], 1, [1 0 0], [0 1 0], [0 0 1]) (solid)

    Methods
    -------
    get_bounds :
      return the bounds of the cylinder
    point_inside :
      return True or False if the points are in or out the cylinder
    distance :
      get the distance of a point to the cylinder
    """
    number_of_bounds = 3 # number of edges

    def __init__(self, center, radius, v0, v1, w, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        if radius>=0:
            self.radius = radius
        else:
            self.log.error('The radius of the cylinder should be positive')
        nvO = np.linalg.norm(v0)
        if nv0 == 0:
            self.log.error('Error in the definition of the cylinder: the first vector is zero')
        else:
            self.v0 = v0 / nv0
        self.v1 = v1 - np.inner(v1,self.v0) * self.v0
        nv1 = np.linalg.norm(self.v1)
        if nv1 == 0:
            self.log.error('Error in the definition of the cylinder: the vectors are colinear')
        else:
            self.v1 /= nv1
        self.w = w
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        # matrix for the change of variables
        # used to write the coordinates in the basis of the cylinder
        A = np.empty((3,3))
        A[:,0] = self.v0
        A[:,1] = self.v1
        A[:,2] = self.w
        self.iA = A.inv()
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the cylinder.
        """
        dummy = max(self.w)
        dummy = max(dummy, self.radius)
        return self.center - dummy, self.center + dummy

    def point_inside(self, x, y, z):
        """
        return a boolean array which defines
        if a point is inside or outside of the cylinder.

        Notes
        -----

        the edge of the cylinder is considered as inside.

        Parameters
        ----------

        x : x coordinates of the points
        y : y coordinates of the points
        z : z coordinates of the points

        Returns
        -------

        Array of boolean (True inside the cylinder, False otherwise)
        """
        v_xyz = np.asarray([x - self.center[0], y - self.center[1], z - self.center[2]])
        v_cyl = self.iA.dot(v_xyz)
        return np.logical_and((v_cyl[0]**2 + v_cyl[1]**2)<=self.radius**2, np.abs(v_cyl[2])<=1.)

    def distance(self, x, y, z, v, dmax=None):
        """
        Compute the distance in the v direction between
        the cylinder and the points defined by (x, y, z).

        .. image:: figures/Cylinder.png
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
        p_xyz = np.asarray([x - self.center[0], y - self.center[1], z - self.center[2]])
        p_cyl = self.iA.dot(p_xyz)
        v_cyl = self.iA.dot(np.asarray(v))
        v2_cyl = v_cyl[0]**2 + v_cyl[1]**2
        delta = -(p_cyl[0]*v_cyl[1] - p_cyl[1]*v_cyl[0])**2 + self.radius**2*v2_cyl
        ind = delta >= 0
        delta[ind] = np.sqrt(delta[ind])/v2_cyl

        d = -np.ones(delta.shape)
        d1 = 1e16*np.ones(delta.shape)
        d2 = 1e16*np.ones(delta.shape)

        d1 = -v_cyl[0]/v2_cyl*p_cyl[0] - v_cyl[1]/v2_cyl*p_cyl[1] - delta
        d2 = -v_cyl[0]/v2_cyl*p_cyl[0] - v_cyl[1]/v2_cyl*p_cyl[1] + delta

        d1[d1<0] = 1e16
        d2[d2<0] = 1e16
        d[ind] = np.minimum(d1[ind], d2[ind])
        d[d==1e16] = -1
        alpha = -np.ones((x.size, y.size))
        border = -np.ones((x.size, y.size))

        if dmax is None:
            ind = d>0
        else:
            ind = np.logical_and(d>0, d<=dmax)
        alpha[ind] = d[ind]

        ###### WARNING LABELS !!!
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
        #print self.center*scale, self.radius*scale
        viewer.ellipse_3D(self.center*scale, tuple(self.radius*scale), color)
        #if viewlabel:
        #    theta = self.center[0] + 2*self.center[1]+10*self.radius
        #    x, y = self.center[0] + self.radius*np.cos(theta), self.center[1] + self.radius*np.sin(theta)
        #    viewer.text(str(self.label[0]), [x, y])
