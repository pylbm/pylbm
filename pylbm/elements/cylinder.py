# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Cylinder element with
    - circle base
    - ellipse base
    - triangle base
    - parallelogram base
"""

# pylint: disable=invalid-name, no-member, attribute-defined-outside-init
# pylint: disable=wildcard-import, unused-wildcard-import

import logging
# from textwrap import dedent
import numpy as np

from .base import *  # pylint: disable=redefined-builtin

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Cylinder(Element):
    """
    Class Cylinder

    generic class for the cylinders
    """
    dim = 3

    def change_of_variables(self):
        """
        matrix for the change of variables
        used to write the coordinates in the basis of the cylinder

        After the change of variables, 
        the base of the cylinder is in the plane x, y
        and the other direction is limited by -1 < z < 1
        """
        self.A = np.empty((3, 3))
        self.A[:, 0] = self.v1
        self.A[:, 1] = self.v2
        self.A[:, 2] = self.w
        self.iA = np.linalg.inv(self.A)

    def get_bounds(self):
        """
        Get the bounds of the cylinder.

        Returns
        -------

        ndarray
            minimal box where the cylinder is included

        """
        lw = [0.5*abs(self.w[k]) for k in range(len(self.w))]
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

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the cylinder, False otherwise)

        """
        x, y, z = grid
        xx = x - self.center[0]
        yy = y - self.center[1]
        zz = z - self.center[2]
        # the new x coordinates
        x_cyl = self.iA[0, 0]*xx + self.iA[0, 1]*yy + self.iA[0, 2]*zz
        # the new y coordinates
        y_cyl = self.iA[1, 0]*xx + self.iA[1, 1]*yy + self.iA[1, 2]*zz
        # the new z coordinates
        z_cyl = self.iA[2, 0]*xx + self.iA[2, 1]*yy + self.iA[2, 2]*zz
        return np.logical_and(
            self.base.point_inside((x_cyl, y_cyl)),
            np.abs(z_cyl) <= 1.
        )

    # pylint: disable=too-many-locals
    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between
        the cylinder and the points defined by (x, y, z).

        .. image:: ../figures/Cylinder.png
            :width: 100%

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y, z = grid

        # rewritte the coordinates in the frame of the cylinder
        v_cyl = self.iA.dot(np.asarray(v))  # the velocity
        xx = x - self.center[0]
        yy = y - self.center[1]
        zz = z - self.center[2]
        # the new x coordinates
        x_cyl = self.iA[0, 0]*xx + self.iA[0, 1]*yy + self.iA[0, 2]*zz
        # the new y coordinates
        y_cyl = self.iA[1, 0]*xx + self.iA[1, 1]*yy + self.iA[1, 2]*zz
        # the new z coordinates
        z_cyl = self.iA[2, 0]*xx + self.iA[2, 1]*yy + self.iA[2, 2]*zz

        # normal vector in the frame of the cylinder
        normal_cyl = np.zeros(tuple(list(x_cyl.shape) + [3]))
        normal_cyl_x = normal_cyl[..., 0]
        normal_cyl_y = normal_cyl[..., 1]
        normal_cyl_z = normal_cyl[..., 2]
        # considering the infinite cylinder
        alpha, border, normal_cyl[..., :2] = self.base.distance(
            (x_cyl, y_cyl),
            v_cyl[:-1],
            dmax, self.label[:-2], normal
        )
        # indices where the intersection is too high or to low
        alpha[alpha < 0] = 1.e16
        border[alpha < 0] = -1
        normal_cyl_x[alpha < 0] = 0
        normal_cyl_y[alpha < 0] = 0
        normal_cyl_z[alpha < 0] = 0
        ind = np.logical_and(alpha > 0, np.abs(z_cyl + alpha*v_cyl[2]) > 1.)
        alpha[ind] = 1.e16
        border[ind] = -1.
        normal_cyl_x[ind] = 0
        normal_cyl_y[ind] = 0
        normal_cyl_z[ind] = 0

        # considering the two planes
        dummyf = self.base.point_inside
        if v_cyl[2] == 0:  # to avoid vertical velocities
            decal = 1.e-16
        else:
            decal = 0.
        alpha_top = (1.-z_cyl)/(v_cyl[2] + decal)
        ind = np.logical_or(
            np.logical_or(alpha_top < 0, alpha_top > dmax),
            np.logical_not(dummyf(
                (x_cyl + alpha_top*v_cyl[0], y_cyl + alpha_top*v_cyl[1])
            ))
        )
        alpha_top[ind] = 1.e16
        alpha_bot = -(1.+z_cyl)/(v_cyl[2] + decal)
        ind = np.logical_or(
            np.logical_or(alpha_bot < 0, alpha_bot > dmax),
            np.logical_not(dummyf(
                (x_cyl + alpha_bot*v_cyl[0], y_cyl + alpha_bot*v_cyl[1])
            ))
        )
        alpha_bot[ind] = 1.e16

        # considering the first intersection point
        alpha = np.amin([alpha, alpha_top, alpha_bot], axis=0)
        # fix the top
        ind = alpha == alpha_top
        border[ind] = self.label[-1]
        normal_cyl_x[ind] = 0.
        normal_cyl_y[ind] = 0.
        normal_cyl_z[ind] = 1.
        # fix the bottom
        ind = alpha == alpha_bot
        border[ind] = self.label[-2]
        normal_cyl_x[ind] = 0.
        normal_cyl_y[ind] = 0.
        normal_cyl_z[ind] = -1.
        # fix the distance
        alpha[alpha == 1.e16] = -1.
        # rewritte the normal vector in the initial frame
        normal_vect = np.zeros(normal_cyl.shape)
        for k in range(3):
            normal_vect[..., k] = self.A[k, 0] * normal_cyl_x + \
                self.A[k, 1] * normal_cyl_y + self.A[k, 2] * normal_cyl_z

        return alpha, border, normal_vect

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('cylinder.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        return template.render(header=header_string(self.__class__.__name__),
                               elem=self, type=elem_type)

    # pylint: disable=too-many-locals
    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(3), alpha=1.
                  ):
        if not isinstance(color, list):
            color = [color]*self.number_of_bounds
        lx_b, ly_b = self.base._visualize()  # pylint: disable=protected-access
        c = self.center
        for k in range(len(lx_b)-2):  # loop over the faces of the side
            x_b, y_b = lx_b[k], ly_b[k]
            z_b = [-1., 1.]
            X_cyl, Z_cyl = np.meshgrid(x_b, z_b)
            Y_cyl, Z_cyl = np.meshgrid(y_b, z_b)
            X = c[0] + self.A[0, 0]*X_cyl \
                + self.A[0, 1]*Y_cyl \
                + self.A[0, 2]*Z_cyl
            Y = c[1] + self.A[1, 0]*X_cyl \
                + self.A[1, 1]*Y_cyl \
                + self.A[1, 2]*Z_cyl
            Z = c[2] + self.A[2, 0]*X_cyl \
                + self.A[2, 1]*Y_cyl \
                + self.A[2, 2]*Z_cyl
            viewer.surface(X, Y, Z, color[k], alpha=alpha)
        vv = np.sin(np.linspace(0, np.pi, 10))
        Xbase = np.outer(lx_b[-2], vv)
        Ybase = np.outer(ly_b[-2], vv)
        Zbase = np.ones(Xbase.shape)
        X = c[0] + self.A[0, 0]*Xbase + self.A[0, 1]*Ybase + self.A[0, 2]*Zbase
        Y = c[1] + self.A[1, 0]*Xbase + self.A[1, 1]*Ybase + self.A[1, 2]*Zbase
        Z = c[2] + self.A[2, 0]*Xbase + self.A[2, 1]*Ybase + self.A[2, 2]*Zbase
        viewer.surface(X, Y, Z, color=color[-2], alpha=alpha)
        X = c[0] + self.A[0, 0]*Xbase + self.A[0, 1]*Ybase - self.A[0, 2]*Zbase
        Y = c[1] + self.A[1, 0]*Xbase + self.A[1, 1]*Ybase - self.A[1, 2]*Zbase
        Z = c[2] + self.A[2, 0]*Xbase + self.A[2, 1]*Ybase - self.A[2, 2]*Zbase
        viewer.surface(X, Y, Z, color=color[-2], alpha=alpha)


class CylinderCircle(Cylinder):
    """
    Class CylinderCircle

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the first vector that defines the circular section
    v2 : list
        the second vector that defines the circular section
    w : list
        the vector that defines the direction of the side
    label : list
        three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
        - True if the cylinder is added
        - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
        3
    dim: int
        3
    center : ndarray
        the coordinates of the center of the cylinder
    v1 : list
        the three coordinates of the first vector defining the base section
    v2 : list
        the three coordinates of the second vector defining the base section
    w : list
        the three coordinates of the vector defining the direction of the side
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the cylinder is added
        and False if the cylinder is deleted

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2) with radius 1

    >>> center = [0., 0., 0.5]
    >>> v1, v2 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> CylinderCircle(center, v1, v2, w)
    +----------------+
    | CylinderCircle |
    +----------------+
        - dimension: 3
        - center: [0.  0.  0.5]
        - v1: [1. 0. 0.]
        - v2: [0. 1. 0.]
        - w: [0. 0. 1.]
        - label: [0, 0, 0]
        - type: solid

    """
    def __init__(self, center, v1, v2, w, label=0, isfluid=False):
        self.number_of_bounds = 3  # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = BaseCircle(self.center, self.v1, self.v2)
        Cylinder.__init__(self, label, isfluid)


class CylinderEllipse(Cylinder):
    """
    Class CylinderEllipse

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the first vector that defines the circular section
    v2 : list
        the second vector that defines the circular section
    w : list
        the vector that defines the direction of the side
    label : list
        three integers (default [0,0,0] for the bottom, the top and the side)
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
    dim: int
        3
    center : ndarray
        the coordinates of the center of the cylinder
    v1 : list
        the three coordinates of the first vector defining the base section
    v2 : list
        the three coordinates of the second vector defining the base section
    w : list
        the three coordinates of the vector defining the direction of the side
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the cylinder is added
        and False if the cylinder is deleted

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2) with radius 1

    >>> center = [0., 0., 0.5]
    >>> v1, v2 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> CylinderEllipse(center, v1, v2, w)
    +-----------------+
    | CylinderEllipse |
    +-----------------+
        - dimension: 3
        - center: [0.  0.  0.5]
        - v1: [1. 0. 0.]
        - v2: [0. 1. 0.]
        - w: [0. 0. 1.]
        - label: [0, 0, 0]
        - type: solid

    """
    def __init__(self, center, v1, v2, w, label=0, isfluid=False):
        self.number_of_bounds = 3  # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = BaseEllipse(self.center, self.v1, self.v2)
        Cylinder.__init__(self, label, isfluid)


class CylinderTriangle(Cylinder):
    """
    Class CylinderTriangle

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the first vector that defines the triangular section
    v2 : list
        the second vector that defines the triangular section
    w : list
        the vector that defines the direction of the side
    label : list
        three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
        - True if the cylinder is added
        - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
        5
    dim: int
        3
    center : numpy array
        the coordinates of the center of the cylinder
    v1 : list of doubles
        the three coordinates of the first vector defining the base section
    v2 : list of doubles
        the three coordinates of the second vector defining the base section
    w : list of doubles
        the three coordinates of the vector defining the direction of the side
    label : list of integers
        the list of the label of the edge
    isfluid : boolean
        True if the cylinder is added
        and False if the cylinder is deleted

    Examples
    --------

    the vertical canonical cylinder centered in (0, 0, 1/2)

    >>> center = [0., 0., 0.5]
    >>> v1, v2 = [1., 0., 0.], [0., 1., 0.]
    >>> w = [0., 0., 1.]
    >>> CylinderTriangle(center, v1, v2, w)
    +------------------+
    | CylinderTriangle |
    +------------------+
        - dimension: 3
        - center: [0.  0.  0.5]
        - v1: [1. 0. 0.]
        - v2: [0. 1. 0.]
        - w: [0. 0. 1.]
        - label: [0, 0, 0, 0, 0]
        - type: solid

    """
    def __init__(self, center, v1, v2, w, label=0, isfluid=False):
        self.number_of_bounds = 5  # number of edges
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        self.w = np.asarray(w)
        self.change_of_variables()
        self.base = BaseTriangle(self.center, self.v1, self.v2)
        Cylinder.__init__(self, label, isfluid)


class Parallelepiped(Cylinder):
    """
    Class Parallelepiped

    Parameters
    ----------
    point : list
        the three coordinates of the first point
    v0 : list
        the three coordinates of the first vector that defines the edge
    v1 : list
        the three coordinates of the second vector that defines the edge
    v2 : list
        the three coordinates of the third vector that defines the edge
    label : list
        three integers (default [0,0,0] for the bottom, the top and the side)
    isfluid : boolean
        - True if the cylinder is added
        - False if the cylinder is deleted

    Attributes
    ----------
    number_of_bounds : int
        6
    dim: int
        3
    point : ndarray
        the coordinates of the first point of the parallelepiped
    v0 : list
        the three coordinates of the first vector
    v1 : list
        the three coordinates of the second vector
    v2 : list
        the three coordinates of the third vector
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the parallelepiped is added
        and False if the parallelepiped is deleted

    Examples
    --------

    the vertical canonical cube centered in (0, 0, 0)

    >>> center = [0., 0., 0.5]
    >>> v0, v1, v2 = [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]
    >>> Parallelepiped(center, v0, v1, v2)
    +----------------+
    | Parallelepiped |
    +----------------+
        - dimension: 3
        - center: [0. 0. 1.]
        - v1: [1. 0. 0.]
        - v2: [0. 1. 0.]
        - w: [0.  0.  0.5]
        - label: [0, 0, 0, 0, 0, 0]
        - type: solid

    """
    def __init__(self, point, v0, v1, v2, label=0, isfluid=False):
        self.number_of_bounds = 6  # number of edges
        self.point = np.asarray(point)
        self.v1 = np.asarray(v0)
        self.v2 = np.asarray(v1)
        self.w = .5*np.asarray(v2)
        self.center = self.point + self.w
        self.change_of_variables()
        self.base = BaseParallelogram(self.center, self.v1, self.v2)
        Cylinder.__init__(self, label, isfluid)
