# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

from math import sin, cos

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Polygon

from logs import setLogger
log = setLogger(__name__)

class Circle:
    """
    Class Circle

    Parameters
    ----------
    center : the coordinates of the center of the circle
    radius : positive float for the radius of the circle
    label : list of one integer (default [0])
    isfluid : boolean
             - True if the circle is added
             - False if the circle is deleted

    Attributes
    ----------
    number_of_bounds : 1
    center : the coordinates of the center of the circle
    radius : positive float for the radius of the circle
    label : the list of the label of the edge
    isfluid : boolean
             - True if the circle is added
             - False if the circle is deleted
    description : a list that contains the description of the element

    Methods
    -------
    get_bounds
    point_inside
    distance
    """
    number_of_bounds = 1

    number_of_bounds = 1 # number of edges

    def __init__(self, center, radius, label = 0, isfluid = False):
        self.center = np.asarray(center)
        self.radius = radius
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        str = 'circle centered in '
        str += '({0:f},{1:f})'.format(self.center[0], self.center[1])
        str += ' with radius {0:f}'.format(self.radius)
        self.description = [str]
        log.info(self.__str__())

    def get_bounds(self):
        """
        return the bounds of the circle.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, x, y):
        """
        return a boolean array which defines
        if a point is inside or outside of the circle.
        the edge of the circle is considered as inside.

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points

        Returns
        -------
        Array of boolean (True inside the circle, False otherwise)
        """
        v2 = np.asarray([x - self.center[0], y - self.center[1]])
        return (v2[0]**2 + v2[1]**2)<=self.radius**2

    def distance(self, x, y, v, dmax=None):
        """
        Compute the distance in the v direction between
        the circle and the points defined by (x, y).

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points
        v : direction of interest

        Returns
        -------
        array of distances

        """
        p = np.asarray([x - self.center[0], y - self.center[1]])
        v2 = v[0]**2 + v[1]**2
        delta = -(p[0]*v[1] - p[1]*v[0])**2 + self.radius**2*v2
        ind = delta>=0

        delta[ind] = np.sqrt(delta[ind])/v2

        d = -np.ones(delta.shape)
        d1 = 1e16*np.ones(delta.shape)
        d2 = 1e16*np.ones(delta.shape)

        d1 = -v[0]/v2*p[0] - v[1]/v2*p[1] - delta
        d2 = -v[0]/v2*p[0] - v[1]/v2*p[1] + delta

        d1[d1<0] = 1e16
        d2[d2<0] = 1e16
        d[ind] = np.minimum(d1[ind], d2[ind])
        d[d==1e16] = -1
        alpha = -np.ones((y.size, x.size))
        border = -np.ones((y.size, x.size))

        if dmax is None:
            ind = d>0
        else:
            ind = np.logical_and(d>0, d<=dmax)
        alpha[ind] = d[ind]
        border[ind] = self.label[0]
        return alpha, border

    def __str__(self):
        s = 'Circle(' + self.center.__str__() + ',' + str(self.radius) + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, ax, coul, viewlabel):
        ax.add_patch(Ellipse(self.center, 2*self.radius, 2*self.radius, fill=True, color=coul))
        if viewlabel:
            theta = self.center[0] + 2*self.center[1]+10*self.radius
            x, y = self.center[0] + self.radius*cos(theta), self.center[1] + self.radius*sin(theta)
            plt.text(x, y, str(self.label[0]),
                fontsize=18, horizontalalignment='center',verticalalignment='center')


class Parallelogram:
    """
    Class Parallelogram

    Parameters
    ----------
    point : the coordinates of the first point of the parallelogram
    vecta : the coordinates of the first vector
    vectb : the coordinates of the second vector
    label : list of four integers (default [0, 0, 0, 0])
    isfluid : boolean
             - True if the parallelogram is added
             - False if the parallelogram is deleted

    Attributes
    ----------
    number_of_bounds : 4
    point : the coordinates of the first point of the parallelogram
    vecta : the coordinates of the first vector
    vectb : the coordinates of the second vector
    label : the list of the four labels of the edges
    isfluid : boolean
             - True if the parallelogram is added
             - False if the parallelogram is deleted
    description : a list that contains the description of the element

    Methods
    -------
    get_bounds
    point_inside
    distance
    """

    number_of_bounds = 4 # number of edges

    def __init__(self, point, vecta, vectb, label = 0, isfluid = False):
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        d = self.point + self.v0 + self.v1
        self.description = [
            'edge 0: ({0:f},{1:f})->({2:f},{3:f})'.format(a[0], a[1], b[0], b[1]),
            'edge 1: ({0:f},{1:f})->({2:f},{3:f})'.format(b[0], b[1], d[0], d[1]),
            'edge 2: ({0:f},{1:f})->({2:f},{3:f})'.format(d[0], d[1], c[0], c[1]),
            'edge 3: ({0:f},{1:f})->({2:f},{3:f})'.format(c[0], c[1], a[0], a[1])
            ]
        log.info(self.__str__())

    def get_bounds(self):
        """
        return the bounds of the parallelogram.
        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, x, y):
        """
        return a boolean array which defines
        if a point is inside or outside of the parallelogram.
        the edges of the parallelogram are considered as inside.

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points

        Returns
        -------
        Array of boolean (True inside the parallelogram, False otherwise)

        """

        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]])
        invdelta = 1./(self.v0[0]*self.v1[1] - self.v0[1]*self.v1[0])
        u = (v2[0]*self.v1[1] - v2[1]*self.v1[0])*invdelta
        v = (v2[1]*self.v0[0] - v2[0]*self.v0[1])*invdelta
        return np.logical_and(np.logical_and(u>=0, v>=0),
                              np.logical_and(u<=1, v<=1))

    def distance(self, x, y, v, dmax=None):
        """
        Compute the distance in the v direction between the parallelogram
        and the points defined by (x, y).

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points
        v : direction of interest

        Returns
        -------
        array of distances

        """

        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v0, self.v1]
        vt = [self.v0, self.v1, self.v1, self.v0]

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

    def _visualize(self, ax, coul, viewlabel):
        A = [self.point[k] for k in xrange(2)]
        B = [A[k] + self.v0[k] for k in xrange(2)]
        C = [B[k] + self.v1[k] for k in xrange(2)]
        D = [A[k] + self.v1[k] for k in xrange(2)]
        ax.add_patch(Polygon([A,B,C,D], closed=True, fill=True, color=coul))
        if viewlabel:
            plt.text(0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]), str(self.label[0]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')
            plt.text(0.5*(A[0]+D[0]), 0.5*(A[1]+D[1]), str(self.label[1]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')
            plt.text(0.5*(C[0]+D[0]), 0.5*(C[1]+D[1]), str(self.label[2]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')
            plt.text(0.5*(B[0]+C[0]), 0.5*(B[1]+C[1]), str(self.label[3]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')


class Triangle:
    """
    Class Triangle

    Parameters
    ----------
    point: the coordinates of the first point of the triangle
    vecta: the coordinates of the first vector
    vectb: the coordinates of the second vector
    label : list of three integers (default [0, 0, 0])
    isfluid : boolean
             - True if the triangle is added
             - False if the triangle is deleted

    Attributes
    ----------
    number_of_bounds : 3
    point : the coordinates of the first point of the triangle
    vecta : the coordinates of the first vector
    vectb : the coordinates of the second vector
    label : the list of the three labels of the edges
    isfluid : boolean
             - True if the triangle is added
             - False if the triangle is deleted
    description : a list that contains the description of the element

    Methods
    -------
    get_bounds
    point_inside
    distance
    """

    number_of_bounds = 3 # number of edges

    def __init__(self, point, vecta, vectb, label = 0, isfluid = False):
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        self.description = [
            'edge 0: ({0:f},{1:f})->({2:f},{3:f})'.format(a[0], a[1], b[0], b[1]),
            'edge 1: ({0:f},{1:f})->({2:f},{3:f})'.format(b[0], b[1], c[0], c[1]),
            'edge 2: ({0:f},{1:f})->({2:f},{3:f})'.format(c[0], c[1], a[0], a[1])
            ]
        log.info(self.__str__())

    def get_bounds(self):
        """
        return the smallest box where the triangle is.

        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, x, y):
        """
        return a boolean array which defines
        if a point is inside or outside of the triangle.
        the edges of the triangle are considered as inside.

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points

        Returns
        -------
        Array of boolean (True inside the triangle, False otherwise)

        """

        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]])
        invdelta = 1./(self.v0[0]*self.v1[1] - self.v0[1]*self.v1[0])
        u = (v2[0]*self.v1[1] - v2[1]*self.v1[0])*invdelta
        v = (v2[1]*self.v0[0] - v2[0]*self.v0[1])*invdelta
        return np.logical_and(np.logical_and(u>=0, v>=0), u + v<=1)

    def distance(self, x, y, v, dmax=None):
        """
        Compute the distance in the v direction between the triangle
        and the points defined by (x, y).

        Parameters
        ----------
        x : x coordinates of the points
        y : y coordinates of the points
        v : direction of interest

        Returns
        -------
        array of distances

        """

        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v0]
        vt = [self.v0, self.v1, self.v1 - self.v0]

        return distance_lines(x - self.point[0], y - self.point[1],
                              v, p, vt, dmax, self.label)

    # a priori unused function
    # def _get_minimum(self, a, b):
    #     """
    #     return the element-wise minimum between a and b
    #     if a is not None, b otherwise.
    #     """
    #     if a is None:
    #         return b
    #     else:
    #         ind = np.where(b < a)
    #         return np.minimum(a, b), ind

    def __str__(self):
        s = 'Triangle(' + self.point.__str__() + ','
        s += self.v0.__str__() + ',' + self.v1.__str__()  + ') '
        if self.isfluid:
            s += '(fluid)'
        else:
            s += '(solid)'
        return s

    def _visualize(self, ax, coul, viewlabel):
        A = [self.point[k] for k in xrange(2)]
        B = [A[k] + self.v0[k] for k in xrange(2)]
        D = [A[k] + self.v1[k] for k in xrange(2)]
        ax.add_patch(Polygon([A,B,D], closed=True, fill=True, color=coul))
        if viewlabel:
            plt.text(0.5*(A[0]+B[0]), 0.5*(A[1]+B[1]), str(self.label[0]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')
            plt.text(0.5*(A[0]+D[0]), 0.5*(A[1]+D[1]), str(self.label[1]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')
            plt.text(0.5*(B[0]+D[0]), 0.5*(B[1]+D[1]), str(self.label[2]),
                fontsize=18,
                horizontalalignment='center',verticalalignment='center')


def intersection_two_lines(p1, v1, p2, v2):
    """
    intersection between two lines defined by a point and a vector.
    """
    alpha = beta = None
    det = v1[1]*v2[0] - v1[0]*v2[1]
    if det != 0:
        invdet = 1./det
        c1 = p2[0] - p1[0]
        c2 = p2[1] - p1[1]
        alpha = (-v2[1]*c1 + v2[0]*c2)*invdet
        beta = (-v1[1]*c1 + v1[0]*c2)*invdet
    return alpha, beta

def distance_lines(x, y, v, p, vt, dmax, label):
    """
    return distance for several lines
    """
    v2 = np.asarray([x, y])
    alpha = 1e16*np.ones((y.size, x.size))
    border = -np.ones((y.size, x.size))
    for i in xrange(len(vt)):
        tmp1, tmp2 = intersection_two_lines(v2, v, p[i], vt[i])
        if tmp1 is not None:
            if dmax is None:
                ind = np.logical_and(tmp1>0, np.logical_and(tmp2>=0, tmp2<=1))
            else:
                ind = np.logical_and(np.logical_and(tmp1>0, tmp1<=dmax),
                                   np.logical_and(tmp2>=0, tmp2<=1))
            ind = np.where(np.logical_and(alpha>tmp1, ind))
            alpha[ind] = tmp1[ind]
            border[ind] = label[i]
    alpha[alpha == 1e16] = -1.
    return alpha, border
