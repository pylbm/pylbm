from __future__ import division
# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
from six.moves import range

from .base import *
from ..logs import setLogger
from .utils import *

class Cylinder(Element):
    """
    Class Cylinder

    generic class for the cylinders
    """

    def __init__(self, base, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        self.log.info(self.__str__())

    def change_of_variables(self):
        # matrix for the change of variables
        # used to write the coordinates in the basis of the cylinder
        self.A = np.empty((3,3))
        self.A[:,0] = self.v1
        self.A[:,1] = self.v2
        self.A[:,2] = self.w
        self.iA = np.linalg.inv(self.A)

    def get_bounds(self):
        """
        Get the bounds of the cylinder.
        """
        lw = [abs(self.w[k]) for k in range(len(self.w))]
        bounds_base = self.base.get_bounds()
        return bounds_base[0] - lw, bounds_base[1] + lw

    def point_inside(self, grid):
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
        x, y, z = grid
        xx = x - self.center[0]
        yy = y - self.center[1]
        zz = z - self.center[2]
        x_cyl = self.iA[0,0]*xx + self.iA[0,1]*yy + self.iA[0,2]*zz # the new x coordinates
        y_cyl = self.iA[1,0]*xx + self.iA[1,1]*yy + self.iA[1,2]*zz # the new y coordinates
        z_cyl = self.iA[2,0]*xx + self.iA[2,1]*yy + self.iA[2,2]*zz # the new z coordinates
        return np.logical_and(self.base.point_inside((x_cyl, y_cyl)), np.abs(z_cyl)<=1.)

    def distance(self, grid, v, dmax=None):
        """
        Compute the distance in the v direction between
        the cylinder and the points defined by (x, y, z).

        .. image:: ../figures/Cylinder.png
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

        # rewritte the coordinates in the frame of the cylinder
        v_cyl = self.iA.dot(np.asarray(v)) # the velocity
        xx = x - self.center[0]
        yy = y - self.center[1]
        zz = z - self.center[2]
        x_cyl = self.iA[0,0]*xx + self.iA[0,1]*yy + self.iA[0,2]*zz # the new x coordinates
        y_cyl = self.iA[1,0]*xx + self.iA[1,1]*yy + self.iA[1,2]*zz # the new y coordinates
        z_cyl = self.iA[2,0]*xx + self.iA[2,1]*yy + self.iA[2,2]*zz # the new z coordinates

        # considering the infinite cylinder
        alpha, border = self.base.distance((x_cyl, y_cyl), v_cyl[:-1], dmax, self.label[:-2])
        # indices where the intersection is too high or to low
        alpha[alpha<0] = 1.e16
        ind = np.logical_and(alpha>0, np.abs(z_cyl + alpha*v_cyl[2]) > 1.)
        alpha[ind] = 1.e16
        border[ind] = -1.

        # considering the two planes
        dummyf = self.base.point_inside
        if v_cyl[2] == 0: # to avoid vertical velocities
            decal = 1.e-16
        else:
            decal = 0.
        alpha_top = (1.-z_cyl)/(v_cyl[2] + decal)
        ind = np.logical_or( np.logical_or(alpha_top<0, alpha_top>dmax),
                             np.logical_not(
                                 dummyf((x_cyl + alpha_top*v_cyl[0],
                                        y_cyl + alpha_top*v_cyl[1]))))
        alpha_top[ind] = 1.e16
        alpha_bot = -(1.+z_cyl)/(v_cyl[2] + decal)
        ind = np.logical_or( np.logical_or(alpha_bot<0, alpha_bot>dmax),
                             np.logical_not(
                                 dummyf((x_cyl + alpha_bot*v_cyl[0],
                                        y_cyl + alpha_bot*v_cyl[1]))))
        alpha_bot[ind] = 1.e16

        # considering the first intersection point
        alpha = np.amin([alpha, alpha_top, alpha_bot], axis=0)
        border[alpha == alpha_top] = self.label[-1]
        border[alpha == alpha_bot] = self.label[-2]
        alpha[alpha==1.e16] = -1.

        return alpha, border

    def __str__(self):
        pass

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(3), alpha=1.):
        if isinstance(color, int):
            color = [color]*self.number_of_bounds
        lx_b, ly_b = self.base._visualize()
        c = self.center
        for k in range(len(lx_b)-2): # loop over the faces of the side
            x_b, y_b = lx_b[k], ly_b[k]
            z_b = [-1., 1.]
            X_cyl, Z_cyl = np.meshgrid(x_b, z_b)
            Y_cyl, Z_cyl = np.meshgrid(y_b, z_b)
            X = c[0] + self.A[0,0]*X_cyl + self.A[0,1]*Y_cyl + self.A[0,2]*Z_cyl
            Y = c[1] + self.A[1,0]*X_cyl + self.A[1,1]*Y_cyl + self.A[1,2]*Z_cyl
            Z = c[2] + self.A[2,0]*X_cyl + self.A[2,1]*Y_cyl + self.A[2,2]*Z_cyl
            viewer.surface(X, Y, Z, color[k], alpha=alpha)
        vv = np.sin(np.linspace(0, np.pi, 10))
        Xbase = np.outer(lx_b[-2], vv)
        Ybase = np.outer(ly_b[-2], vv)
        Zbase = np.ones(Xbase.shape)
        X = c[0] + self.A[0,0]*Xbase + self.A[0,1]*Ybase + self.A[0,2]*Zbase
        Y = c[1] + self.A[1,0]*Xbase + self.A[1,1]*Ybase + self.A[1,2]*Zbase
        Z = c[2] + self.A[2,0]*Xbase + self.A[2,1]*Ybase + self.A[2,2]*Zbase
        viewer.surface(X, Y, Z, color=color[-2], alpha=alpha)
        X = c[0] + self.A[0,0]*Xbase + self.A[0,1]*Ybase - self.A[0,2]*Zbase
        Y = c[1] + self.A[1,0]*Xbase + self.A[1,1]*Ybase - self.A[1,2]*Zbase
        Z = c[2] + self.A[2,0]*Xbase + self.A[2,1]*Ybase - self.A[2,2]*Zbase
        viewer.surface(X, Y, Z, color=color[-2], alpha=alpha)


class Cylinder_Circle(Cylinder):
    """
    Class Cylinder_Circle

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
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
    v0 : list of doubles
      the three coordinates of the first vector that defines the base section
    v1 : list of doubles
      the three coordinates of the second vector that defines the base section
    w : list of doubles
      the three coordinates of the vector that defines the direction of the side
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the cylinder is added
      and False if the cylinder is deleted
    number_of_bounds : int
        number of edges (3)

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2) with radius 1

    >>> center = [0., 0., 0.5]
    >>> v0, v1 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> Cylinder_Circle(center, v0, v1, w)
        Cylinder_Circle([0 0 0.5], [1 0 0], [0 1 0], [0 0 1]) (solid)

    """
    def __init__(self, center, v1, v2, w, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 3 # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = Base_Circle(self.center, self.v1, self.v2)
        self.isfluid = isfluid
        Cylinder.__init__(self, self.base, label=label, isfluid=isfluid)

    def __str__(self):
        s = 'Cylinder_Circle(' + self.center.__str__() + ', '
        s += self.v1.__str__() + ', ' + self.v2.__str__() + ', '
        s += self.w.__str__() +  ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s


class Cylinder_Ellipse(Cylinder):
    """
    Class Cylinder_Ellipse

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v0 : a list of the three coordinates of the first vector that defines the circular section
    v1 : a list of the three coordinates of the second vector that defines the circular section
    w : a list of the three coordinates of the vector that defines the direction of the side
    label : list of three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
             - True if the cylinder is added
             - False if the cylinder is deleted

    Warnings
    --------

    The vectors v1 and v2 have to be orthogonal.

    Attributes
    ----------
    number_of_bounds : int
      3
    center : numpy array
      the coordinates of the center of the cylinder
    v0 : list of doubles
      the three coordinates of the first vector that defines the base section
    v1 : list of doubles
      the three coordinates of the second vector that defines the base section
    w : list of doubles
      the three coordinates of the vector that defines the direction of the side
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the cylinder is added
      and False if the cylinder is deleted
    number_of_bounds : int
        number of edges (3)
    number_of_bounds : int
        number of edges (3)

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2) with radius 1

    >>> center = [0., 0., 0.5]
    >>> v0, v1 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> Cylinder_Ellipse(center, v0, v1, w)
        Cylinder_Ellipse([0 0 0.5], [1 0 0], [0 1 0], [0 0 1]) (solid)

    """
    def __init__(self, center, v1, v2, w, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 3 # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = Base_Ellipse(self.center, self.v1, self.v2)
        self.isfluid = isfluid
        Cylinder.__init__(self, self.base, label=label, isfluid=isfluid)

    def __str__(self):
        s = 'Cylinder_Ellipse(' + self.center.__str__() + ', '
        s += self.v1.__str__() + ', ' + self.v2.__str__() + ', '
        s += self.w.__str__() +  ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

class Cylinder_Triangle(Cylinder):
    """
    Class Cylinder_Triangle

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v0 : a list of the three coordinates of the first vector that defines the triangular section
    v1 : a list of the three coordinates of the second vector that defines the triangular section
    w : a list of the three coordinates of the vector that defines the direction of the side
    label : list of three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
             - True if the cylinder is added
             - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
      5
    center : numpy array
      the coordinates of the center of the cylinder
    v0 : list of doubles
      the three coordinates of the first vector that defines the base section
    v1 : list of doubles
      the three coordinates of the second vector that defines the base section
    w : list of doubles
      the three coordinates of the vector that defines the direction of the side
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the cylinder is added
      and False if the cylinder is deleted
    number_of_bounds : int
        number of edges (3)

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2)

    >>> center = [0., 0., 0.5]
    >>> v0, v1 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> Cylinder_Triangle(center, v0, v1, w)
        Cylinder_Triangle([0 0 0.5], [1 0 0], [0 1 0], [0 0 1]) (solid)

    """
    def __init__(self, center, v1, v2, w, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 5 # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = Base_Triangle(self.center, self.v1, self.v2)
        self.isfluid = isfluid
        Cylinder.__init__(self, self.base, label=label, isfluid=isfluid)

    def __str__(self):
        s = 'Cylinder_Triangle(' + self.center.__str__() + ', '
        s += self.v1.__str__() + ', ' + self.v2.__str__() + ', '
        s += self.w.__str__() +  ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

class Parallelepiped(Cylinder):
    """
    Class Parallelepiped

    Parameters
    ----------
    point : a list that contains the three coordinates of the first point
    v0 : a list of the three coordinates of the first vector that defines the edge
    v1 : a list of the three coordinates of the second vector that defines the edge
    v2 : a list of the three coordinates of the third vector that defines the edge
    label : list of three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
             - True if the cylinder is added
             - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
      6
    point : numpy array
      the coordinates of the first point of the parallelepiped
    v0 : list of doubles
      the three coordinates of the first vector
    v1 : list of doubles
      the three coordinates of the second vector
    v2 : list of doubles
      the three coordinates of the third vector
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the parallelepiped is added
      and False if the parallelepiped is deleted
    number_of_bounds : int
        number of edges (6)

    Examples
    --------

    the vertical canonical cube centered in (0, 0, 0)

    >>> center = [0., 0., 0.5]
    >>> v0, v1, v2 = [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
    >>> Parallelepiped(center, v0, v1, v2)
        Parallelepiped([0 0 0], [1 0 0], [0 1 0], [0 0 1]) (solid)

    """
    def __init__(self, point, v0, v1, v2, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.number_of_bounds = 6 # number of edges
        self.point = np.asarray(point)
        self.v1 = np.asarray(v0)
        self.v2 = np.asarray(v1)
        self.w = .5*np.asarray(v2)
        self.center = self.point + self.w
        self.change_of_variables()
        self.base = Base_Parallelogram(self.center, self.v1, self.v2)
        self.isfluid = isfluid
        Cylinder.__init__(self, self.base, label=label, isfluid=isfluid)

    def __str__(self):
        s = 'Parallelepiped(' + self.point.__str__() + ', '
        s += self.v1.__str__() + ', ' + self.v2.__str__() + ', '
        s += (2*self.w).__str__() +  ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s
