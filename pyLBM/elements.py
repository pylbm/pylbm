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

from .logs import setLogger

class Element:
    """
    Class Element

    generic class for the elements
    """
    number_of_bounds = 0
    def __init__(self):
        self.log = setLogger(__name__)
        self.isfluid = False
        self.label = []
        self.log.info(self.__str__())

    def get_bounds(self):
        return float('Inf'), -float('Inf')

    def point_inside(self, x, y, z=None):
        if z is None:
            return x**2 + y**2 < -1.
        else:
            return x**2 + y**2 + z**2 < -1.

    def __str__(self):
        return 'Generic element'

    def __repr__(self):
        return self.__str__()

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(2)):
        pass

    def test_label(self):
        return len(self.label) == self.number_of_bounds

class Circle(Element):
    """
    Class Circle

    Parameters
    ----------
    center : a list that contains the two coordinates of the center
    radius : a positive float for the radius
    label : list of one integer (default [0])
    isfluid : boolean
             - True if the circle is added
             - False if the circle is deleted

    Attributes
    ----------
    number_of_bounds : int
      1
    center : numpy array
      the coordinates of the center of the circle
    radius : double
      positive float for the radius of the circle
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the circle is added
      and False if the circle is deleted

    Examples
    --------

    the circle centered in (0, 0) with radius 1

    >>> center = [0., 0.]
    >>> radius = 1.
    >>> Circle(center, radius)
        Circle([0 0],1) (solid)

    Methods
    -------
    get_bounds :
      return the bounds of the circle
    point_inside :
      return True or False if the points are in or out the circle
    distance :
      get the distance of a point to the circle
    """
    number_of_bounds = 1 # number of edges

    def __init__(self, center, radius, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.center = np.asarray(center)
        if radius>=0:
            self.radius = radius
        else:
            self.log.error('The radius of the circle should be positive')
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        #str = 'circle centered in '
        #str += '({0:f},{1:f})'.format(self.center[0], self.center[1])
        #str += ' with radius {0:f}'.format(self.radius)
        #self.description = [str]
        self.log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the circle.
        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, x, y):
        """
        return a boolean array which defines
        if a point is inside or outside of the circle.

        Notes
        -----

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

        .. image:: figures/Circle.png
            :width: 100%

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
        alpha = -np.ones((x.size, y.size))
        border = -np.ones((x.size, y.size))

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

    def _visualize(self, viewer, color, viewlabel=False, scale=np.ones(2)):
        viewer.ellipse(self.center*scale, tuple(self.radius*scale), color)
        if viewlabel:
            theta = self.center[0] + 2*self.center[1]+10*self.radius
            x, y = self.center[0] + self.radius*cos(theta), self.center[1] + self.radius*sin(theta)
            viewer.text(str(self.label[0]), [x, y])

class Parallelogram(Element):
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
    point : numpy array
      the coordinates of the first point of the parallelogram
    vecta : numpy array
      the coordinates of the first vector
    vectb : numpy array
      the coordinates of the second vector
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the parallelogram is added
      and False if the parallelogram is deleted

    Methods
    -------
    get_bounds :
      return the bounds of the parallelogram
    point_inside :
      return True or False if the points are in or out the parallelogram
    distance :
      get the distance of a point to the parallelogram
    """

    number_of_bounds = 4 # number of edges

    def __init__(self, point, vecta, vectb, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        d = self.point + self.v0 + self.v1
        self.log.info(self.__str__())

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

        Notes
        -----

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

        .. image:: figures/Parallelogram.png
            :width: 100%

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

    def _visualize(self, viewer, color, viewlabel):
        A = [self.point[k] for k in xrange(2)]
        B = [A[k] + self.v0[k] for k in xrange(2)]
        C = [B[k] + self.v1[k] for k in xrange(2)]
        D = [A[k] + self.v1[k] for k in xrange(2)]
        viewer.polygon([A,B,C,D], color)
        if viewlabel:
            viewer.text(str(self.label[0]), [0.5*(A[0]+B[0]), 0.5*(A[1]+B[1])])
            viewer.text(str(self.label[1]), [0.5*(A[0]+D[0]), 0.5*(A[1]+D[1])])
            viewer.text(str(self.label[2]), [0.5*(C[0]+D[0]), 0.5*(C[1]+D[1])])
            viewer.text(str(self.label[3]), [0.5*(B[0]+C[0]), 0.5*(B[1]+C[1])])

class Triangle(Element):
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

    Examples
    --------

    the bottom half square of [0,1]x[0,1]

    >>> point = [0., 0.]
    >>> vecta = [1., 0.]
    >>> vectb = [0., 1.]
    >>> Triangle(point, vecta, vectb)
        Triangle([0 0],[0 1],[1 0]) (solid)

    Attributes
    ----------

    number_of_bounds : int
      3
    point : numpy array
      the coordinates of the first point of the triangle
    vecta : numpy array
      the coordinates of the first vector
    vectb : numpy array
      the coordinates of the second vector
    label : list of integers
      the list of the label of the edge
    isfluid : boolean
      True if the triangle is added
      and False if the triangle is deleted

    Methods
    -------
    get_bounds :
      return the bounds of the triangle
    point_inside :
      return True or False if the points are in or out the triangle
    distance :
      get the distance of a point to the triangle
    """

    number_of_bounds = 3 # number of edges

    def __init__(self, point, vecta, vectb, label = 0, isfluid = False):
        self.log = setLogger(__name__)
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.isfluid = isfluid
        if isinstance(label, int):
            self.label = [label]*self.number_of_bounds
        else:
            self.label = label
        self.test_label()
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        # self.description = [
        #     'edge 0: ({0:f},{1:f})->({2:f},{3:f})'.format(a[0], a[1], b[0], b[1]),
        #     'edge 1: ({0:f},{1:f})->({2:f},{3:f})'.format(b[0], b[1], c[0], c[1]),
        #     'edge 2: ({0:f},{1:f})->({2:f},{3:f})'.format(c[0], c[1], a[0], a[1])
        #     ]
        self.log.info(self.__str__())

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

        Notes
        -----

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

        .. image:: figures/Triangle.png
            :width: 100%

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

    def _visualize(self, viewer, color, viewlabel):
        A = [self.point[k] for k in xrange(2)]
        B = [A[k] + self.v0[k] for k in xrange(2)]
        D = [A[k] + self.v1[k] for k in xrange(2)]
        viewer.polygon([A,B,D], color)
        if viewlabel:
            viewer.text(str(self.label[0]), [0.5*(A[0]+B[0]), 0.5*(A[1]+B[1])])
            viewer.text(str(self.label[1]), [0.5*(A[0]+D[0]), 0.5*(A[1]+D[1])])
            viewer.text(str(self.label[2]), [0.5*(B[0]+D[0]), 0.5*(B[1]+D[1])])


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
        #    x, y = self.center[0] + self.radius*cos(theta), self.center[1] + self.radius*sin(theta)
        #    viewer.text(str(self.label[0]), [x, y])


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
        #    x, y = self.center[0] + self.radius*cos(theta), self.center[1] + self.radius*sin(theta)
        #    viewer.text(str(self.label[0]), [x, y])


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
    alpha = 1e16*np.ones((x.size, y.size))
    border = -np.ones((x.size, y.size))
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
