# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np
from six.moves import range

from ..logs import setLogger
from .utils import *

class Element(object):
    """
    Class Element

    generic class for the elements

    Attributes
    ----------
    number_of_bounds : int
        number of edges
    """
    #number_of_bounds = 0
    def __init__(self):
        self.log = setLogger(__name__)
        self.isfluid = False
        self.label = []
        self.log.info(self.__str__())

    def get_bounds(self):
        return float('Inf'), -float('Inf')

    def point_inside(self, grid):
        if len(grid) == 2:
            x, y = grid
            return x**2 + y**2 < -1.
        elif len(grid) == 3:
            x, y, z = grid
            return x**2 + y**2 + z**2 < -1.

    def __str__(self):
        return 'Generic element'

    def __repr__(self):
        return self.__str__()

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(2), alpha=1.):
        pass

    def test_label(self):
        """
        test if the number of labels is equal to the number of bounds.
        """
        return len(self.label) == self.number_of_bounds


class Base2D(object):
    """
    Class Base2D

    generic class for the 2D base of a cylinder or a cone
    """

    def __init__(self):
        self.log = setLogger(__name__)
        self.log.info(self.__str__())

    def __str__(self):
        return 'Generic 2D base'

    def __repr__(self):
        return self.__str__()


class Base_Circle(Base2D):
    """
    Class Base_Circle

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v1 : a list of the three coordinates of the first vector that defines the circular base
    v2 : a list of the three coordinates of the second vector that defines the circular base

    """

    def __init__(self, center, v1, v2):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        radius = np.linalg.norm(v1)
        if radius != np.linalg.norm(v2):
            self.log.error("Error in Base_Circle: vectors v1 and v2 must have the same norm")
        self.radius = radius
        # orthogonalization of the two vectors
        self.v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        self.v2 = v2 - np.inner(v2,self.v1) * self.v1 / np.inner(self.v1, self.v1)
        nv2 = np.linalg.norm(self.v2)
        if nv2 == 0:
            self.log.error('Error in the definition of the cylinder: the vectors are colinear')
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, grid):
        x, y = grid
        return (x**2 + y**2) <= 1.

    def distance(self, grid, v, dmax, label):
        x, y = grid
        c = np.zeros((2,))
        v1 = np.asarray([1,0])
        v2 = np.asarray([0,1])
        return distance_ellipse(x, y, v, c, v1, v2, dmax, label)

    def _visualize(self):
        t = np.linspace(0, 2.*np.pi, 100)
        lx_b = [np.cos(t),np.cos(t), np.cos(t[::-1])]
        ly_b = [np.sin(t),np.sin(t), np.sin(t[::-1])]
        return lx_b, ly_b

    def __str__(self):
        s = 'Circular base with radius ' + str(self.radius) + ' centerd in '+ str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1) + ' and ' + str(self.v2) + '\n'
        return s


class Base_Ellipse(Base2D):
    """
    Class Base_Ellipse

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v1 : a list of the three coordinates of the first vector that defines the ellipsoidal base
    v2 : a list of the three coordinates of the second vector that defines the ellipsoidal base

    Warnings
    --------

    The vectors v1 and v2 have to be orthogonal.

    """

    def __init__(self, center, v1, v2):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        if abs(np.inner(self.v1, self.v2)) > 1.e-10:
            self.log.error('Error in the definition of the cylinder: the vectors have to be orthogonal')
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        r = max(np.linalg.norm(self.v1), np.linalg.norm(self.v2))
        return self.center - r, self.center + r

    def point_inside(self, grid):
        x, y = grid
        return (x**2 + y**2) <= 1.

    def distance(self, grid, v, dmax, label):
        x, y = grid
        c = np.zeros((2,))
        v1 = np.asarray([1,0])
        v2 = np.asarray([0,1])
        return distance_ellipse(x, y, v, c, v1, v2, dmax, label)

    def _visualize(self):
        t = np.linspace(0, 2.*np.pi, 100)
        lx_b = [np.cos(t),np.cos(t), np.cos(t[::-1])]
        ly_b = [np.sin(t),np.sin(t), np.sin(t[::-1])]
        return lx_b, ly_b

    def __str__(self):
        s = 'Ellipsoidal base centerd in '+ str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1) + ' and ' + str(self.v2) + '\n'
        return s


class Base_Triangle(Base2D):
    """
    Class Base_Triangle

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v1 : a list of the three coordinates of the first vector that defines the triangular base
    v2 : a list of the three coordinates of the second vector that defines the triangular base
    """

    def __init__(self, center, v1, v2):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        nv1 = np.linalg.norm(self.v1)
        nv2 = np.linalg.norm(self.v2)
        if np.allclose(nv1*self.v2, nv2*self.v1):
            self.log.error('Error in the definition of the cylinder: the vectors are not free')
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        box = np.asarray([self.center, self.center + self.v1,
                          self.center + self.v1 + self.v2, self.center + self.v2])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, grid):
        x, y = grid
        return np.logical_and(np.logical_and(x>=0, y>=0), x+y<=1)

    def distance(self, grid, v, dmax, label):
        x, y = grid
        p = [[0, 0], [0, 0], [1, 0]]
        vt = [[1, 0], [0, 1], [-1, 1]]
        return distance_lines(x, y, v, p, vt, dmax, label)

    def _visualize(self):
        p = np.asarray([[0,0], [1,0], [0,1], [0,0], [1,0]]).T
        lx_b = []
        ly_b = []
        for k in range(3):
            lx_b.append(p[0,k:k+2])
            ly_b.append(p[1,k:k+2])
        lx_b.append(p[0,:4])
        ly_b.append(p[1,:4])
        lx_b.append(p[0,3::-1])
        ly_b.append(p[1,3::-1])
        return lx_b, ly_b

    def __str__(self):
        s = 'Triangular base centerd in '+ str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1) + ' and ' + str(self.v2) + '\n'
        return s


class Base_Parallelogram(Base2D):
    """
    Class Base_Parallelogram

    Parameters
    ----------
    center : a list that contains the three coordinates of the center
    v1 : a list of the three coordinates of the first vector that defines the base
    v2 : a list of the three coordinates of the second vector that defines the base
    """

    def __init__(self, center, v1, v2):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        self.v1 = np.asarray(v1)
        self.v2 = np.asarray(v2)
        nv1 = np.linalg.norm(self.v1)
        nv2 = np.linalg.norm(self.v2)
        if np.allclose(nv1*self.v2, nv2*self.v1):
            self.log.error('Error in the definition of the cylinder: the vectors are not free')
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the base
        """
        box = np.asarray([self.center, self.center + self.v1,
                          self.center + self.v1 + self.v2, self.center + self.v2])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, grid):
        x, y = grid
        return np.logical_and(np.logical_and(x>=0, y>=0),
                              np.logical_and(x<=1, y<=1))

    def distance(self, grid, v, dmax, label):
        x, y = grid
        p = [[0, 0], [0, 0], [1, 0], [0, 1]]
        vt = [[1, 0], [0, 1], [0, 1], [1, 0]]
        return distance_lines(x, y, v, p, vt, dmax, label)

    def _visualize(self):
        p = np.asarray([[0,0], [1,0], [1,1], [0,1], [0,0], [1,0]]).T
        lx_b = []
        ly_b = []
        for k in range(4):
            lx_b.append(p[0,k:k+2])
            ly_b.append(p[1,k:k+2])
        lx_b.append(p[0,:5])
        ly_b.append(p[1,:5])
        lx_b.append(p[0,4::-1])
        ly_b.append(p[1,4::-1])
        return lx_b, ly_b

    def __str__(self):
        s = 'Parallelogram base centerd in '+ str(self.center) + '\n'
        s += '     in the plane spanned by ' + str(self.v1) + ' and ' + str(self.v2) + '\n'
        return s
