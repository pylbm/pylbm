# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Triangle element
"""

# pylint: disable=invalid-name

import logging
# from textwrap import dedent
import numpy as np

from .base import Element
from .utils import distance_lines

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Triangle(Element):
    """
    Class Triangle

    Parameters
    ----------

    point : list
        the coordinates of the first point of the triangle
    vecta : list
        the coordinates of the first vector
    vectb : list
        the coordinates of the second vector
    label : list
        three integers (default [0, 0, 0])
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
    +----------+
    | Triangle |
    +----------+
        - dimension: 2
        - start point: [0. 0.]
        - v1: [1. 0.]
        - v2: [0. 1.]
        - label: [0, 0, 0]
        - type: solid

    Attributes
    ----------

    point : ndarray
        the coordinates of the first point of the triangle
    v1 : ndarray
        the coordinates of the first vector
    v2 : ndarray
        the coordinates of the second vector
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the triangle is added
        and False if the triangle is deleted
    number_of_bounds : int
        number of edges: 3
    dim: int
        2

    """
    def __init__(self, point, vecta, vectb, label=0, isfluid=False):
        self.number_of_bounds = 3  # number of edges
        self.dim = 2
        self.point = np.asarray(point)
        self.v1 = np.asarray(vecta)
        self.v2 = np.asarray(vectb)
        super(Triangle, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        box = np.asarray(
            [
                self.point,
                self.point + self.v1,
                self.point + self.v1 + self.v2,
                self.point + self.v2
            ]
        )
        return np.min(box, axis=0), np.max(box, axis=0)

    def point_inside(self, grid):
        x, y = grid
        # Barycentric coordinates
        v2 = np.asarray([x - self.point[0], y - self.point[1]], dtype=object)
        invdelta = 1./(self.v1[0]*self.v2[1] - self.v1[1]*self.v2[0])
        u = (v2[0]*self.v2[1] - v2[1]*self.v2[0])*invdelta
        v = (v2[1]*self.v1[0] - v2[0]*self.v1[1])*invdelta
        return np.logical_and(np.logical_and(u >= 0, v >= 0), u + v <= 1)

    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between the triangle
        and the points defined by (x, y).

        .. image:: ../figures/Triangle.png
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
        x, y = grid
        # points and triangle edges
        # which define the lines for the intersections
        # with the lines defined by (x, y) and v
        p = [[0, 0], [0, 0], self.v1]
        vt = [self.v1, self.v2, self.v2 - self.v1]

        return distance_lines(
            x - self.point[0],
            y - self.point[1],
            v, p, vt,
            dmax, self.label, normal
        )

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('square.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        return template.render(
            header=header_string(self.__class__.__name__),
            elem=self, type=elem_type
        )

    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(3), alpha=1.
                  ):
        A = [self.point[k] for k in range(2)]
        B = [A[k] + self.v1[k] for k in range(2)]
        D = [A[k] + self.v2[k] for k in range(2)]
        viewer.polygon([A, B, D], color, alpha=alpha)
        if viewlabel:
            viewer.text(str(self.label[0]), [0.5*(A[0]+B[0]), 0.5*(A[1]+B[1])])
            viewer.text(str(self.label[1]), [0.5*(A[0]+D[0]), 0.5*(A[1]+D[1])])
            viewer.text(str(self.label[2]), [0.5*(B[0]+D[0]), 0.5*(B[1]+D[1])])
