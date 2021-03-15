# Authors:
#     Loic Gouarin <loic.gouarin@polytechnique.edu>
#     Benjamin Graille <benjamin.graille@math.u-psud.fr>
#
# License: BSD 3 clause

"""
Ellipse element
"""
# pylint: disable=invalid-name

import logging
# from textwrap import dedent
import numpy as np

from .base import Element
from .utils import distance_ellipse

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Ellipse(Element):
    """
    Class Ellipse

    Parameters
    ----------
    center : list
        the two coordinates of the center
    v1 : list
        a vector
    v2 : list
        a second vector (v1 and v2 have to be othogonal)
    label : list
        one integer (default [0])
    isfluid : boolean
        - True if the ellipse is added
        - False if the ellipse is deleted

    Attributes
    ----------
    number_of_bounds : int
        1
    dim: int
        2
    center : ndarray
        the coordinates of the center of the ellipse
    v1 : ndarray
        the coordinates of the first vector
    v2 : ndarray
        the coordinates of the second vector
    label : list
        the list of the label of the edge
    isfluid : boolean
        True if the ellipse is added
        and False if the ellipse is deleted

    Examples
    --------

    the ellipse centered in (0, 0) with v1=[2,0], v2=[0,1]

    >>> center = [0., 0.]
    >>> v1 = [2., 0.]
    >>> v2 = [0., 1.]
    >>> Ellipse(center, v1, v2)
    +---------+
    | Ellipse |
    +---------+
        - dimension: 2
        - center: [0. 0.]
        - v1: [2. 0.]
        - v2: [0. 1.]
        - label: [0]
        - type: solid

    """
    def __init__(self, center, v1, v2, label=0, isfluid=False):
        self.number_of_bounds = 1  # number of edges
        self.dim = 2
        self.center = np.asarray(center)
        if abs(v1[0]*v2[0] + v1[1]*v2[1]) > 1.e-14:
            log.error('The vectors of the ellipse are not orthogonal')
        else:
            self.v1 = np.asarray(v1)
            self.v2 = np.asarray(v2)
        super(Ellipse, self).__init__(label, isfluid)
        log.info(self.__str__())

    def get_bounds(self):
        """
        Get the bounds of the ellipse.
        """
        r = max(np.linalg.norm(self.v1), np.linalg.norm(self.v2))
        return self.center - r, self.center + r

    def point_inside(self, grid):
        """
        return a boolean array which defines
        if a point is inside or outside of the ellipse.

        Notes
        -----

        the edge of the ellipse is considered as inside.

        Parameters
        ----------

        grid : ndarray
            coordinates of the points

        Returns
        -------

        ndarray
            Array of boolean (True inside the ellipse, False otherwise)

        """
        x, y = grid

        X = x - self.center[0]
        Y = y - self.center[1]
        vx2 = self.v1[0]**2 + self.v2[0]**2
        vy2 = self.v1[1]**2 + self.v2[1]**2
        vxy = 2 * (self.v1[0]*self.v1[1] + self.v2[0]*self.v2[1])
        tv = self.v1[0]*self.v2[1]-self.v1[1]*self.v2[0]
        return X**2*vy2 + Y**2*vx2 - X*Y*vxy <= tv**2

    def distance(self, grid, v, dmax=None, normal=False):
        """
        Compute the distance in the v direction between
        the ellipse and the points defined by (x, y).

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
        return distance_ellipse(
            x, y, v,
            self.center, self.v1, self.v2,
            dmax, self.label, normal
        )

    def __str__(self):
        from ..utils import header_string
        from ..jinja_env import env
        template = env.get_template('ellipse.tpl')
        elem_type = 'fluid' if self.isfluid else 'solid'
        return template.render(
            header=header_string(self.__class__.__name__),
            elem=self,
            type=elem_type
        )

    def visualize(self,
                  viewer, color, viewlabel=False,
                  scale=np.ones(2), alpha=1.
                  ):
        nv1 = np.linalg.norm(self.v1)
        nv2 = np.linalg.norm(self.v2)
        if nv1 > nv2:
            r1, r2 = nv1, nv2
            v = self.v1
        else:
            r1, r2 = nv2, nv2
            v = self.v2
        if v[0] == 0:
            theta = .5*np.pi
        else:
            theta = np.arctan(v[1]/v[0])

        viewer.ellipse(
            self.center*scale, (r1*scale[0], r2*scale[1]),
            color, angle=theta, alpha=alpha
        )
        if viewlabel:
            x = self.center[0] + r1*np.cos(theta)
            y = self.center[1] + r1*np.sin(theta)
            viewer.text(str(self.label[0]), [x*scale[0], y*scale[1]])
