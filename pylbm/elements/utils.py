# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Useful functions to compute the distance
between a point and an object
"""
# pylint: disable=invalid-name

import logging
import numpy as np


log = logging.getLogger(__name__)  # pylint: disable=invalid-name
tgv = 1.e16  # very large value


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
    else:
        message = "Problem in intersection_two_lines\n"
        message += "The lines are parallel"
        log.info(message)
    return alpha, beta


def distance_lines(x, y, v, p, vt, dmax, label, normal):
    """
    return distance for several lines

    normal is a boolean
    if normal is True, the normal vector is also returned
    """
    v2 = np.asarray([x, y], dtype=object)
    alpha = tgv*np.ones((x.size, y.size))
    border = -np.ones((x.size, y.size))
    normal_x = np.zeros((x.size, y.size))
    normal_y = np.zeros((x.size, y.size))
    for i, vti in enumerate(vt):
        tmp1, tmp2 = intersection_two_lines(v2, v, p[i], vti)
        if tmp1 is not None:
            if dmax is None:
                ind = np.logical_and(
                    tmp1 > 0,
                    np.logical_and(
                        tmp2 >= 0,
                        tmp2 <= 1
                    )
                )
            else:
                ind = np.logical_and(
                    np.logical_and(
                        tmp1 > 0,
                        tmp1 <= dmax
                    ),
                    np.logical_and(
                        tmp2 >= 0,
                        tmp2 <= 1
                    )
                )
            ind = np.where(np.logical_and(alpha > tmp1, ind))
            alpha[ind] = tmp1[ind]
            border[ind] = label[i]
            if normal:
                # compute the normal vector
                # with a negative scalar product with v
                if v[0]*vti[1] < v[1]*vti[0]:
                    sign = 1
                else:
                    sign = -1
                nv = np.sqrt(vti[0]**2+vti[1]**2)
                normal_x[ind] = sign*vti[1]/nv
                normal_y[ind] = -sign*vti[0]/nv
    alpha[alpha == tgv] = -1.
    if normal:
        normal_vect = np.zeros((x.size, y.size, 2))
        normal_vect[:, :, 0] = normal_x
        normal_vect[:, :, 1] = normal_y
    else:
        normal_vect = None
    return alpha, border, normal_vect


# pylint: disable=too-many-locals
def distance_ellipse(x, y, v, center, v1, v2, dmax, label, normal):
    """
    return the distance according
    a line defined by a point x, y and a vector v
    to an ellipse defined by a point c and two vectors v1 and v2

    normal is a boolean
    if normal is True, the normal vector is also returned
    """
    # build the equation of the ellipse
    # then write the second order equation in d
    # a d**2 + b d + c = 0
    # delta = b**2-4ac
    X = x - center[0]
    Y = y - center[1]
    vx2 = v1[0]**2 + v2[0]**2
    vy2 = v1[1]**2 + v2[1]**2
    vxy = v1[0]*v1[1] + v2[0]*v2[1]
    a = v[0]**2*vy2 + v[1]**2*vx2 - 2*v[0]*v[1]*vxy
    b = 2*X*v[0]*vy2 + 2*Y*v[1]*vx2 - 2*(X*v[1]+Y*v[0])*vxy
    c = X**2*vy2 + Y**2*vx2 - 2*X*Y*vxy - (v1[0]*v2[1]-v1[1]*v2[0])**2
    delta = b**2 - 4*a*c
    ind = delta >= 0
    delta[ind] = np.sqrt(delta[ind])

    shape = delta.shape
    d1 = tgv*np.ones(shape)
    d2 = tgv*np.ones(shape)
    if a != 0:
        d1[ind] = (-b[ind] - delta[ind])/(2*a)
        d2[ind] = (-b[ind] + delta[ind])/(2*a)
    d1[d1 < 0] = tgv
    d2[d2 < 0] = tgv
    d = -np.ones(shape)
    d[ind] = np.minimum(d1[ind], d2[ind])
    d[d == tgv] = -1

    alpha = -np.ones(shape)
    border = -np.ones(shape)
    if dmax is None:
        ind = d > 0
    else:
        ind = np.logical_and(d > 0, d <= dmax)
    alpha[ind] = d[ind]
    border[ind] = label[0]
    if normal:
        # compute the normal vector
        normal_x = np.zeros(shape)
        normal_y = np.zeros(shape)
        norm_normal = np.ones(shape)
        Xe = X + alpha * v[0]
        Ye = Y + alpha * v[1]
        normal_x[ind] = Xe[ind] * vy2 - Ye[ind] * vxy
        normal_y[ind] = Ye[ind] * vx2 - Xe[ind] * vxy
        norm_normal[ind] = np.sqrt(normal_x[ind]**2 + normal_y[ind]**2)
        normal_vect = np.zeros(tuple(list(shape) + [2]))
        normal_vect[:, :, 0] = normal_x / norm_normal
        normal_vect[:, :, 1] = normal_y / norm_normal
    else:
        normal_vect = None
    return alpha, border, normal_vect


# pylint: disable=too-many-locals
def distance_ellipsoid(x, y, z, v, center, v1, v2, v3, dmax, label):
    """
    return the distance according
    a line defined by a point x, y, z and a vector v
    to an ellipsoid defined by a point c and three vectors v1, v2, and v3
    """
    # build the equation of the ellipsoid
    # then write the second order equation in d
    # a d**2 + b d + c = 0
    # delta = b**2-4ac
    X = x - center[0]
    Y = y - center[1]
    Z = z - center[2]
    v12 = np.cross(v1, v2)
    v23 = np.cross(v2, v3)
    v31 = np.cross(v3, v1)
    d = np.inner(v1, v23)**2
    # equation of the ellipsoid:
    # cxx XX + cyy YY + czz ZZ + cxy XY + cyz YZ + czx ZX = d
    cxx = v12[0]**2 + v23[0]**2 + v31[0]**2
    cyy = v12[1]**2 + v23[1]**2 + v31[1]**2
    czz = v12[2]**2 + v23[2]**2 + v31[2]**2
    cxy = 2 * (v12[0]*v12[1] + v23[0]*v23[1] + v31[0]*v31[1])
    cyz = 2 * (v12[1]*v12[2] + v23[1]*v23[2] + v31[1]*v31[2])
    czx = 2 * (v12[2]*v12[0] + v23[2]*v23[0] + v31[2]*v31[0])
    a = cxx*v[0]**2 + cyy*v[1]**2 + czz*v[2]**2 \
        + cxy*v[0]*v[1] + cyz*v[1]*v[2] + czx*v[2]*v[0]
    b = (2*cxx*v[0]+cxy*v[1]+czx*v[2])*X \
        + (2*cyy*v[1]+cyz*v[2]+cxy*v[0])*Y \
        + (2*czz*v[2]+czx*v[0]+cyz*v[1])*Z
    c = cxx*X**2 + cyy*Y**2 + czz*Z**2 \
        + cxy*X*Y + cyz*Y*Z + czx*Z*X - d
    delta = b**2 - 4*a*c
    ind = delta >= 0  # wird but it works
    delta[ind] = np.sqrt(delta[ind])
    d1 = 1e16*np.ones(delta.shape)
    d2 = 1e16*np.ones(delta.shape)
    d1[ind] = (-b[ind] - delta[ind])/(2*a)
    d2[ind] = (-b[ind] + delta[ind])/(2*a)
    d1[d1 < 0] = 1e16
    d2[d2 < 0] = 1e16
    d = -np.ones(d1.shape)
    d[ind] = np.minimum(d1[ind], d2[ind])
    d[d == 1e16] = -1

    alpha = -np.ones((x.size, y.size, z.size))
    border = -np.ones((x.size, y.size, z.size))
    if dmax is None:
        ind = d > 0
    else:
        ind = np.logical_and(d > 0, d <= dmax)
    alpha[ind] = d[ind]
    border[ind] = label[0]
    return alpha, border
