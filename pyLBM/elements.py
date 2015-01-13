# Authors:
#     Loic Gouarin <loic.gouarin@math.u-psud.fr>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

import numpy as np

class Circle:
    """
    Class Circle

    * Arguments
       - geomtype: geometrical type ( = 'Circle')
       - center: 2-tuple for the coordinates of the center of the circle
       - radius: positive float for the radius of the circle

    * Attributs
       - geomtype: geometrical type ( = 'Circle')
       - Center: 2-tuple for the coordinates of the center of the circle
       - Radius: positive float for the radius of the circle
       - number_of_bounds: number of bounds = 1
       - description: a list that contains the description of the element
       - tag: a list that contains the tag of the edges
       - bw: integer
             - 1 if the circle is added
             - 0 if the circle is deleted
             - 2 else
    """

    def __init__(self, center, radius):
        self.geomtype = 'Circle'
        self.center = np.asarray(center)
        self.radius = radius
        self.bw = 2
        self.number_of_bounds = 1
        self.tag = []
        self.label = []
        self.description = ['circle centered in ({0:f},{1:f}) with radius {2:f}'.format(self.center[0], self.center[1], self.radius)]

    def get_bounds(self):
        """
        return the bounds of the circle.

        """
        return self.center - self.radius, self.center + self.radius

    def point_inside(self, x, y):
        """
        return a boolean array which defines if a point is inside or outside of the circle.

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points

        * Output

            Array of boolean (1 inside the circle, 0 otherwise)

        """
        v2 = np.asarray([x - self.center[0], y - self.center[1]])

        return (v2[0]**2 + v2[1]**2)<=self.radius**2

    def distance(self, x, y, v, dmax=None):
        """
        Compute the distance in the v direction between the circle and the points defined by (x, y).

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points
            - v: direction of interest

        * Output

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
        border[ind] = self.label[0]#self.tag[0]
        return alpha, border

    def __str__(self):
        s = 'Circle(' + self.center.__str__() + ',' + str(self.radius) + ') '
        if (self.bw == 1):
            s += '(Added)'
        elif (self.bw == 0):
            s += '(Deleted)'
        return s

class Parallelogram:
    """
    Class Parallelogram

    * Arguments
       - geomtype: geometrical type ( = 'Parallelogram')
       - point: 2-tuple for the coordinates of the first point of the parallelogram
       - vecta: 2-tuple for the coordinates of the first vector
       - vectb: 2-tuple for the coordinates of the second vector

    * Attributs
       - geomtype: geometrical type ( = 'Parallelogram')
       - Point: 2-tuple for the coordinates of the first point of the parallelogram
       - Vecta: 2-tuple for the coordinates of the first vector
       - Vectb: 2-tuple for the coordinates of the second vector
       - number_of_bounds: number of bounds = 4
       - description: a list that contains the description of the element
       - tag: a list that contains the tag of the edges
       - bw: integer
             - 1 if the parallelogram is added
             - 0 if the parallelogram is deleted
             - 2 else
    """

    def __init__(self, point, vecta, vectb):
        self.geomtype = 'Parallelogram'
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.bw = 2
        self.number_of_bounds = 4
        self.tag = []
        self.label = []
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        d = self.point + self.v0 + self.v1
        self.description = ['edge 0: ({0:f},{1:f})->({2:f},{3:f})'.format(a[0], a[1], b[0], b[1]),
                            'edge 1: ({0:f},{1:f})->({2:f},{3:f})'.format(b[0], b[1], d[0], d[1]),
                            'edge 2: ({0:f},{1:f})->({2:f},{3:f})'.format(d[0], d[1], c[0], c[1]),
                            'edge 3: ({0:f},{1:f})->({2:f},{3:f})'.format(c[0], c[1], a[0], a[1])
                            ]

    def get_bounds(self):
        """
        return the bounds of the parallelogram.

        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, x, y):
        """
        return a boolean array which defines if a point is inside or outside of the parallelogram.

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points

        * Output

            Array of boolean (1 inside the parallelogram, 0 otherwise)

        """

        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]])
        invdelta = 1./(self.v0[0]*self.v1[1] - self.v0[1]*self.v1[0])
        u = (v2[0]*self.v1[1] - v2[1]*self.v1[0])*invdelta
        v = (v2[1]*self.v0[0] - v2[0]*self.v0[1])*invdelta
        return np.logical_and(np.logical_and(u>=0, v>=0), np.logical_and(u<=1, v<=1))

    def distance(self, x, y, v, dmax=None):
        """
        Compute the distance in the v direction between the parallelogram
        and the points defined by (x, y).

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points
            - v: direction of interest

        * Output

            array of distances

        """

        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v0, self.v1]
        vt = [self.v0, self.v1, self.v1, self.v0]

        #return distance_lines(x - self.point[0], y - self.point[1], v, p, vt, dmax, self.tag)
        return distance_lines(x - self.point[0], y - self.point[1], v, p, vt, dmax, self.label)

    def __str__(self):
        s = 'Parallelogram(' + self.point.__str__() + ',' + self.v0.__str__() + ',' + self.v1.__str__()  + ') '
        if (self.bw == 1):
            s += '(Added)'
        elif (self.bw == 0):
            s += '(Deleted)'
        return s

class Triangle:
    """
    Class Triangle

    * Arguments
       - geomtype: geometrical type ( = 'Triangle')
       - point: 2-tuple for the coordinates of the first point of the triangle
       - vecta: 2-tuple for the coordinates of the first vector
       - vectb: 2-tuple for the coordinates of the second vector

    * Attributs
       - geomtype: geometrical type ( = 'Triangle')
       - Point: 2-tuple for the coordinates of the first point of the triangle
       - Vecta: 2-tuple for the coordinates of the first vector
       - Vectb: 2-tuple for the coordinates of the second vector
       - number_of_bounds: number of bounds = 1
       - description: a list that contains the description of the element
       - tag: a list that contains the tag of the edges
       - bw: integer
             - 1 if the circle is added
             - 0 if the circle is deleted
             - 2 else
    """

    def __init__(self, point, vecta, vectb):
        self.geomtype = 'Triangle'
        self.point = np.asarray(point)
        self.v0 = np.asarray(vecta)
        self.v1 = np.asarray(vectb)
        self.bw = 2
        self.number_of_bounds = 3
        self.tag = []
        self.label = []
        a = self.point
        b = self.point + self.v0
        c = self.point + self.v1
        self.description = ['edge 0: ({0:f},{1:f})->({2:f},{3:f})'.format(a[0], a[1], b[0], b[1]),
                            'edge 1: ({0:f},{1:f})->({2:f},{3:f})'.format(b[0], b[1], c[0], c[1]),
                            'edge 2: ({0:f},{1:f})->({2:f},{3:f})'.format(c[0], c[1], a[0], a[1])
                            ]

    def get_bounds(self):
        """
        return the smallest box where the triangle is.

        """
        box = np.asarray([self.point, self.point + self.v0,
                          self.point + self.v0 + self.v1, self.point + self.v1])
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, x, y):
        """
        return a boolean array which defines if a point is inside or outside of the triangle.

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points

        * Output

            Array of boolean (1 inside the triangle, 0 otherwise)

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

        * Parameters

            - x: x coordinates of the points
            - y: y coordinates of the points
            - v: direction of interest

        * Output

            array of distances

        """

        # points and triangle edges which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v0]
        vt = [self.v0, self.v1, self.v1 - self.v0]

        #return distance_lines(x - self.point[0], y - self.point[1], v, p, vt, dmax, self.tag)
        return distance_lines(x - self.point[0], y - self.point[1], v, p, vt, dmax, self.label)

    def _get_minimum(self, a, b):
        """
        return the element-wise minimum between a and b if a is not None, b otherwise.
        """
        if a is None:
            return b
        else:
            ind = np.where(b < a)
            return np.minimum(a, b), ind

    def __str__(self):
        s = 'Triangle(' + self.point.__str__() + ',' + self.v0.__str__() + ',' + self.v1.__str__()  + ') '
        if (self.bw == 1):
            s += '(Added)'
        elif (self.bw == 0):
            s += '(Deleted)'
        return s

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

def distance_lines(x, y, v, p, vt, dmax, tag):
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
            border[ind] = tag[i]

    alpha[alpha == 1e16] = -1.
    return alpha, border
