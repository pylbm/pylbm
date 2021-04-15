# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Base element
"""

# pylint: disable=invalid-name

import logging
from abc import ABC, abstractmethod
from six.moves import range
import numpy as np

from .utils import distance_lines, distance_ellipse

__all__ = ['Element', 'BaseCircle', 'BaseEllipse',
           'BaseTriangle', 'BaseParallelogram']

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Element(ABC):
    """
    Class Element

    generic class for the elements

    """
    number_of_bounds = -1

    def __init__(self, label, isfluid):
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()

    @abstractmethod
    def get_bounds(self):
        """
        return the smallest box where the element is.
        """

    @abstractmethod
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the element.

        Notes
        -----

        the edges of the element are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the element, False otherwise)

        """

    @abstractmethod
    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(2), alpha=1.
                  ):
        """
        visualize the element

        Parameters
        ----------

        viewer : Viewer
            a viewer (default matplotlib_viewer)

        color : color
            color of the element

        viewlabel : bool
            activate the labels mark (default False)

        scale : ndarray
            scale the distance of the labels (default ones)

        alpha : double
            transparency of the element (default 1)

        """

    @abstractmethod
    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between the element
        and the points defined in grid by (x, y) or (x, y, z).

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

    def __repr__(self):
        return self.__str__()

    def test_label(self):
        """
        test if the number of labels is equal to the number of bounds.
        """
        return len(self.label) == self.number_of_bounds


class BaseCircle:
    """
    Class BaseCircle

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the three coordinates of the first vector defining the circular base
    v2 : list
        the three coordinates of the second vector defining the circular base

    """
    def __init__(self, center, v1, v2):
        self.center = np.asarray(center)
        radius = np.linalg.norm(v1)
        if radius != np.linalg.norm(v2):
            err_msg = "Error in BaseCircle: "
            err_msg += "vectors v1 and v2 must have the same norm"
            log.error(err_msg)
        self.radius = radius
        # orthogonalization of the two vectors
        self.v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        self.v2 = v2 - np.inner(v2, self.v1) * self.v1 \
            / np.inner(self.v1, self.v1)
        nv2 = np.linalg.norm(self.v2)
        if nv2 == 0:
            err_msg = "Error in the definition of the cylinder: "
            err_msg += "the vectors are colinear"
            log.error(err_msg)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        return self.center - self.radius, self.center + self.radius

    # pylint: disable=no-self-use
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the element.

        Notes
        -----

        the edges of the element are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the element, False otherwise)

        """
        x, y = grid
        return (x**2 + y**2) <= 1.

    @staticmethod
    def distance(grid, v, dmax, label, normal=False):
        """
        Compute the distance in the v direction between the element
        and the points defined by (x, y) for a given label.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        label : int
            the label of interest
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y = grid
        c = np.zeros((2,))
        v1 = np.asarray([1, 0])
        v2 = np.asarray([0, 1])
        return distance_ellipse(x, y, v, c, v1, v2, dmax, label, normal)

    @staticmethod
    def _visualize():
        t = np.linspace(0, 2.*np.pi, 100)
        lx_b = [np.cos(t), np.cos(t), np.cos(t[::-1])]
        ly_b = [np.sin(t), np.sin(t), np.sin(t[::-1])]
        return lx_b, ly_b

    def __str__(self):
        s = "Circular base with radius {} ".format(self.radius)
        s += "centered in " + str(self.center) + "\n"
        s += "     in the plane spanned by " + str(self.v1)
        s += " and " + str(self.v2) + "\n"
        return s


class BaseEllipse:
    """
    Class BaseEllipse

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the three coordinates of the first vector defining the ellipsoidal base
    v2 : list
        the three coordinates of the second vector
        defining the ellipsoidal base

    Warnings
    --------

    The vectors v1 and v2 have to be orthogonal.

    """

    def __init__(self, center, v1, v2):
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        if abs(np.inner(self.v1, self.v2)) > 1.e-10:
            err_msg = "Error in the definition of the cylinder: "
            err_msg += "the vectors have to be orthogonal"
            log.error(err_msg)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        r = max(np.linalg.norm(self.v1), np.linalg.norm(self.v2))
        return self.center - r, self.center + r

    # pylint: disable=no-self-use
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the element.

        Notes
        -----

        the edges of the element are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the element, False otherwise)

        """
        x, y = grid
        return (x**2 + y**2) <= 1.

    @staticmethod
    def distance(grid, v, dmax, label, normal=False):
        """
        Compute the distance in the v direction between the element
        and the points defined by (x, y) for a given label.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        label : int
            the label of interest
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y = grid
        c = np.zeros((2,))
        v1 = np.asarray([1, 0])
        v2 = np.asarray([0, 1])
        return distance_ellipse(x, y, v, c, v1, v2, dmax, label, normal)

    @staticmethod
    def _visualize():
        t = np.linspace(0, 2.*np.pi, 100)
        lx_b = [np.cos(t), np.cos(t), np.cos(t[::-1])]
        ly_b = [np.sin(t), np.sin(t), np.sin(t[::-1])]
        return lx_b, ly_b

    def __str__(self):
        s = 'Ellipsoidal base centered in ' + str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1)
        s += ' and ' + str(self.v2) + '\n'
        return s


class BaseTriangle:
    """
    Class BaseTriangle

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the three coordinates of the first vector defining the triangular base
    v2 : list
        the three coordinates of the second vector defining the triangular base

    """

    def __init__(self, center, v1, v2):
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        nv1 = np.linalg.norm(self.v1)
        nv2 = np.linalg.norm(self.v2)
        if np.allclose(nv1*self.v2, nv2*self.v1):
            err_msg = "Error in the definition of the cylinder: "
            err_msg += "the vectors are not free"
            log.error(err_msg)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        box = np.asarray(
            [
                self.center,
                self.center + self.v1,
                self.center + self.v1 + self.v2,
                self.center + self.v2
            ]
        )
        return np.min(box, axis=0), np.max(box, axis=0)

    # pylint: disable=no-self-use
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the element.

        Notes
        -----

        the edges of the element are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the element, False otherwise)

        """
        x, y = grid
        return np.logical_and(np.logical_and(x >= 0, y >= 0), x + y <= 1)

    @staticmethod
    def distance(grid, v, dmax, label, normal=False):
        """
        Compute the distance in the v direction between the element
        and the points defined by (x, y) for a given label.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        label : int
            the label of interest
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y = grid
        p = [[0, 0], [0, 0], [1, 0]]
        vt = [[1, 0], [0, 1], [-1, 1]]
        return distance_lines(x, y, v, p, vt, dmax, label, normal)

    @staticmethod
    def _visualize():
        p = np.asarray([[0, 0],
                        [1, 0],
                        [0, 1],
                        [0, 0],
                        [1, 0]]
                       ).T
        lx_b = []
        ly_b = []
        for k in range(3):
            lx_b.append(p[0, k:k+2])
            ly_b.append(p[1, k:k+2])
        lx_b.append(p[0, :4])
        ly_b.append(p[1, :4])
        lx_b.append(p[0, 3::-1])
        ly_b.append(p[1, 3::-1])
        return lx_b, ly_b

    def __str__(self):
        s = 'Triangular base centered in ' + str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1)
        s += ' and ' + str(self.v2) + '\n'
        return s


class BaseParallelogram:
    """
    Class BaseParallelogram

    Parameters
    ----------
    center : list
        the three coordinates of the center
    v1 : list
        the three coordinates of the first vector that defines the base
    v2 : list
        the three coordinates of the second vector that defines the base

    """
    def __init__(self, center, v1, v2):
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        nv1 = np.linalg.norm(self.v1)
        nv2 = np.linalg.norm(self.v2)
        if np.allclose(nv1*self.v2, nv2*self.v1):
            err_msg = "Error in the definition of the cylinder: "
            err_msg += "the vectors are not free"
            log.error(err_msg)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        box = np.asarray(
            [
                self.center,
                self.center + self.v1,
                self.center + self.v1 + self.v2,
                self.center + self.v2
            ]
        )
        return np.min(box, axis=0), np.max(box, axis=0)

    # pylint: disable=no-self-use
    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the element.

        Notes
        -----

        the edges of the element are considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the element, False otherwise)

        """
        x, y = grid
        return np.logical_and(np.logical_and(x >= 0, y >= 0),
                              np.logical_and(x <= 1, y <= 1))

    @staticmethod
    def distance(grid, v, dmax, label, normal=False):
        """
        Compute the distance in the v direction between the element
        and the points defined by (x, y) for a given label.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points
        v : ndarray
            direction of interest
        dmax : float
            distance max
        label : int
            the label of interest
        normal : bool
            return the normal vector if True (default False)

        Returns
        -------

        ndarray
            array of distances if normal is False and
            the coordinates of the normal vectors
            if normal is True
        """
        x, y = grid
        p = [[0, 0], [0, 0], [1, 0], [0, 1]]
        vt = [[1, 0], [0, 1], [0, 1], [1, 0]]
        return distance_lines(x, y, v, p, vt, dmax, label, normal)

    @staticmethod
    def _visualize():
        p = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0]]).T
        lx_b = []
        ly_b = []
        for k in range(4):
            lx_b.append(p[0, k:k+2])
            ly_b.append(p[1, k:k+2])
        lx_b.append(p[0, :5])
        ly_b.append(p[1, :5])
        lx_b.append(p[0, 4::-1])
        ly_b.append(p[1, 4::-1])
        return lx_b, ly_b

    def __str__(self):
        s = 'Parallelogram base centered in ' + str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1)
        s += ' and ' + str(self.v2) + '\n'
        return s
